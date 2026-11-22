"""Per-family input normalisation.

train.py bakes one (mu, sd) per column, measured over 64 train sequences, into the
model as buffers. fam_scale_diag.py measures the cost: after that single transform,
family 0's price columns reach the GRU with std 0.62 where family 3's have 1.84 -- a
2.9x spread the same weights have to cover. Family 0 is also the family sitting at 60%
of its perfect-sign ceiling while families 1-3 all reach 86%.

Five earlier family experiments -- conditioning, resampling, specialists, fine-tuning,
residual heads -- all changed WHICH DATA the model sees, and all came back null. None
of them could reach this, because the normalisation is fixed before any of them apply.
This replaces it with one (mu, sd) per family, which equalises by construction rather
than by proxy (per-sequence warm-up statistics were tried first and inflate instead:
f0 3.92 vs f1 1.48, with 18% of columns hitting the variance floor).

Deployable as-is: 4x112 extra constants, and the family comes from the warm-up
classifier that already reads it at 99.8% from rows 0..98 -- before the first scored
row, with no recurrent work and no extra parameters.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent
for p in (str(R), str(R/"src")): sys.path.insert(0, p)

NFAM = 4


def stats(cache, fam, tag: str, g_mu, g_sd, max_seq: int = 400, force: bool = False):
    """(4, F) mean and sd per family, streamed, cached to runs/famnorm_<tag>.npz.

    Families with too few sequences fall back to the global statistics rather than to a
    noisy estimate.
    """
    out = R/"runs"/f"famnorm_{tag}.npz"
    if out.exists() and not force:
        z = np.load(out)
        if z["mu"].shape[-1] == cache.X.shape[-1]: return z["mu"], z["sd"]
    F = cache.X.shape[-1]
    n = np.zeros(NFAM); s1 = np.zeros((NFAM, F)); s2 = np.zeros((NFAM, F))
    per = np.zeros(NFAM, int)
    for i in range(len(cache)):
        f = int(fam[i])
        if f < 0 or per[f] >= max_seq: continue
        per[f] += 1
        X = np.asarray(cache.X[i], np.float64)
        n[f] += len(X); s1[f] += X.sum(0); s2[f] += (X*X).sum(0)
        if per.sum() >= NFAM*max_seq: break
    mu = np.where(n[:, None] > 0, s1/np.maximum(n[:, None], 1), np.asarray(g_mu)[None, :])
    var = np.maximum(s2/np.maximum(n[:, None], 1) - mu**2, 0.0)
    sd = np.where(n[:, None] > 0, np.sqrt(var), np.asarray(g_sd)[None, :])
    thin = n < 50_000
    mu[thin] = np.asarray(g_mu)[None, :]; sd[thin] = np.asarray(g_sd)[None, :]
    sd = np.maximum(sd, 1e-6)
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, mu=mu.astype(np.float32), sd=sd.astype(np.float32), n=n)
    print(f"famnorm[{tag}]: rows per family {n.astype(int).tolist()}"
          + ("  (some fell back to global)" if thin.any() else ""))
    return mu.astype(np.float32), sd.astype(np.float32)


def apply(X, fam_ix, mu, sd):
    """X (B, T, F) from sequences with family labels `fam_ix` (B,) -> normalised."""
    f = np.asarray(fam_ix, np.int64)
    return ((X - mu[f][:, None, :]) / sd[f][:, None, :]).astype(np.float32)
