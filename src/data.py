"""Sequence cache and windowed sampler.

Parquet row-group reads are too slow to sit in a training loop, so we
materialise a chosen set of sequences once into memmapped .npy files and sample
windows out of them. Nothing is ever fully resident: the builder streams one
sequence at a time and the sampler copies only the batch it returns.

Layout per cache directory:
    X.npy         float32 (N, 20000, 112)
    Y.npy         float32 (N, 20000, 2)
    M.npy         bool    (N, 20000)      is_scored; valid only
    manifest.json which row groups, which source, which seed
"""
from __future__ import annotations
import json, sys
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, WARMUP, N_FEATURES

BYTES_PER_SEQ = SEQUENCE_LENGTH * (N_FEATURES + 2) * 4


def build_cache(source: str, out_dir: Path, n_seqs: int, seed: int = 0,
                row_groups: list[int] | None = None) -> Path:
    """Materialise `n_seqs` sequences from datasets/<source>.parquet into out_dir."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    f = pq.ParquetFile(ROOT / "datasets" / f"{source}.parquet")
    total = f.metadata.num_row_groups
    if row_groups is None:
        row_groups = sorted(np.random.default_rng(seed).choice(total, min(n_seqs, total), replace=False).tolist())
    n = len(row_groups)
    has_mask = "is_scored" in f.schema_arrow.names
    print(f"building cache: {source} -> {out_dir}  {n} sequences, {n*BYTES_PER_SEQ/1e9:.2f} GB")

    X = np.lib.format.open_memmap(out_dir/"X.npy", mode="w+", dtype=np.float32,
                                  shape=(n, SEQUENCE_LENGTH, N_FEATURES))
    Y = np.lib.format.open_memmap(out_dir/"Y.npy", mode="w+", dtype=np.float32,
                                  shape=(n, SEQUENCE_LENGTH, 2))
    M = (np.lib.format.open_memmap(out_dir/"M.npy", mode="w+", dtype=bool,
                                   shape=(n, SEQUENCE_LENGTH)) if has_mask else None)
    cols = ["seq_ix"] + (["is_scored"] if has_mask else []) + list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)
    seq_ids = []
    for i, g in enumerate(row_groups):
        t = f.read_row_group(g, columns=cols, use_threads=False)
        X[i] = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        Y[i] = np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        if M is not None:
            M[i] = t["is_scored"].to_numpy().astype(bool)
        seq_ids.append(int(t["seq_ix"][0].as_py()))
        del t
        if (i+1) % 50 == 0 or i+1 == n:
            print(f"  {i+1}/{n}", flush=True)
    X.flush(); Y.flush()
    if M is not None: M.flush()
    json.dump({"source": source, "n": n, "seed": seed, "row_groups": row_groups,
               "seq_ix": seq_ids, "has_mask": has_mask},
              open(out_dir/"manifest.json", "w"))
    return out_dir


class SequenceCache:
    """Read-only view over a built cache, with a windowed batch sampler."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.load(open(self.path/"manifest.json"))
        self.X = np.load(self.path/"X.npy", mmap_mode="r")
        self.Y = np.load(self.path/"Y.npy", mmap_mode="r")
        self.M = np.load(self.path/"M.npy", mmap_mode="r") if self.meta["has_mask"] else None
        self.n = self.X.shape[0]

    def __len__(self): return self.n

    def feature_stats(self, k: int = 64):
        """Per-feature mean/std over the first k sequences, streamed."""
        s1 = np.zeros(N_FEATURES, np.float64); s2 = np.zeros(N_FEATURES, np.float64); c = 0
        for i in range(min(k, self.n)):
            x = np.asarray(self.X[i], np.float64)
            s1 += x.sum(0); s2 += (x*x).sum(0); c += len(x)
        mu = s1/c; sd = np.sqrt(np.maximum(s2/c - mu*mu, 1e-12))
        return mu.astype(np.float32), sd.astype(np.float32)

    def batch(self, rng, batch: int, body: int, burn: int):
        """Random windows. Returns X (B, burn+body, 112), Y (B, body, 2), Mask (B, body).

        `burn` rows warm the recurrent state (no loss); `body` rows carry the loss.
        Windows never start before WARMUP - burn, so the state the model sees always
        begins where a real sequence's context would.
        """
        L = burn + body
        seqs = rng.integers(0, self.n, batch)
        lo = max(0, WARMUP - burn)
        starts = rng.integers(lo, SEQUENCE_LENGTH - L, batch)
        xb = np.empty((batch, L, N_FEATURES), np.float32)
        yb = np.empty((batch, body, 2), np.float32)
        mb = np.ones((batch, body), np.float32)
        for k, (i, s) in enumerate(zip(seqs, starts)):
            xb[k] = self.X[i, s:s+L]
            yb[k] = self.Y[i, s+burn:s+L]
            if self.M is not None:
                mb[k] = self.M[i, s+burn:s+L].astype(np.float32)
        return xb, yb, mb

    def sequence(self, i: int):
        """One full sequence, for evaluation."""
        m = np.asarray(self.M[i]) if self.M is not None else None
        return np.asarray(self.X[i]), np.asarray(self.Y[i]), m


class _ConcatField:
    """Presents two memmapped arrays as one along axis 0.

    Must support BOTH access patterns the trainer uses:
        cache.X[i]            whole sequence   (evaluate, feature_stats)
        cache.X[i, s:s+L]     a window          (sample_grouped)
    """

    def __init__(self, parent, name):
        self.p, self.name = parent, name

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, rest = key[0], key[1:]
            c, j = self.p._pick(int(i))
            return getattr(c, self.name)[(j,) + rest]
        c, j = self.p._pick(int(key))
        return getattr(c, self.name)[j]

    @property
    def shape(self):
        a = getattr(self.p.a, self.name)
        return (len(self.p),) + tuple(a.shape[1:])


class ConcatCache:
    """Two caches as one, for training on train + the reserved valid retrain split.

    The rules permit training on valid.parquet (only EXTERNAL data is banned). The
    reason is not the extra rows - data volume saturated long ago - but DISTRIBUTION
    COVERAGE: a classifier separates train from valid on input-only summaries at
    AUC 0.954, so they are different periods and test is a third. A model that has
    seen two periods should transfer to a third better than one that has seen one.

    Only the reserved retrain split may go here; dev-300 and holdout-500 stay clean.
    """

    def __init__(self, primary, extra, extra_weight: float = 1.0):
        # `extra` may be one cache or a list of caches (each a SequenceCache or a
        # SliceCache); they are presented as ONE extra block `self.b` so that
        # SequenceStream's per-source row factor still sees exactly two sources.
        self.a = primary
        self.b = extra if not isinstance(extra, (list, tuple)) else (
            extra[0] if len(extra) == 1 else _MultiCache(list(extra)))
        self.extra_weight = extra_weight
        self.X = _ConcatField(self, "X")
        self.Y = _ConcatField(self, "Y")
        self.M = _ConcatField(self, "M") if getattr(primary, "M", None) is not None else None

    def __len__(self):
        return len(self.a) + len(self.b)

    def _pick(self, i):
        return (self.a, i) if i < len(self.a) else (self.b, i - len(self.a))

    def feature_stats(self, k: int = 64):
        return self.a.feature_stats(k)          # primary defines the normalisation

    def sequence(self, i: int):
        c, j = self._pick(i)
        return c.sequence(j)

    def batch(self, rng, batch: int, body: int, burn: int):
        raise NotImplementedError("ConcatCache is used with sample_grouped only")


class SliceCache:
    """A contiguous [lo, hi) slice of a SequenceCache, e.g. the 300 holdout sequences
    (positions 200-499) that no holdout-200 comparison ever scores."""

    def __init__(self, cache, lo: int, hi: int):
        self.c, self.lo, self.hi = cache, lo, min(hi, len(cache))
        self.X = _OffsetField(cache.X, lo); self.Y = _OffsetField(cache.Y, lo)
        self.M = _OffsetField(cache.M, lo) if cache.M is not None else None

    def __len__(self): return self.hi - self.lo
    def sequence(self, i): return self.c.sequence(self.lo + i)
    def feature_stats(self, k=64): return self.c.feature_stats(k)


class _OffsetField:
    def __init__(self, arr, off): self.arr, self.off = arr, off
    def __getitem__(self, key):
        if isinstance(key, tuple): return self.arr[(int(key[0]) + self.off,) + key[1:]]
        return self.arr[int(key) + self.off]
    @property
    def shape(self): return self.arr.shape


class _MultiCache:
    """Several caches presented as one block (used for the extra side of ConcatCache)."""

    def __init__(self, parts):
        self.parts = parts
        self.X = _ConcatField(self, "X"); self.Y = _ConcatField(self, "Y")
        self.M = _ConcatField(self, "M") if all(getattr(p, "M", None) is not None for p in parts) else None
        self.a = parts[0]                       # _ConcatField.shape reads .a

    def __len__(self): return sum(len(p) for p in self.parts)

    def _pick(self, i):
        for p in self.parts:
            if i < len(p): return p, i
            i -= len(p)
        raise IndexError(i)

    def sequence(self, i):
        p, j = self._pick(i); return p.sequence(j)


def open_extra(spec: str, root):
    """'cache/valid_retrain,cache/holdout:200:500' -> list of caches (sliced where asked)."""
    out = []
    for part in spec.split(","):
        bits = part.split(":")
        c = SequenceCache(root/bits[0])
        if len(bits) == 3: c = SliceCache(c, int(bits[1]), int(bits[2]))
        out.append(c)
    return out
