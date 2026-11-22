"""Clip-aware output scale, selected per target channel on MASKED TRAINING data.

The metric clips predictions to [-2, 2], so output scale is a real parameter with an
interior optimum. The trainer's `calibrate_scale` targets a fixed prediction std of 1.0
on 16 unmasked train sequences; on the sequential refgru models that lands eval
predictions at std ~1.8 and costs 0.02-0.04 WP against the clip-aware optimum
(A_s801/final.pt on eval-300: 0.6228 at the trainer's scale, 0.6603 at half of it).

Selection here uses cache/valid_retrain: real is_scored masks, legal to train on, never
part of dev or holdout. Weighted Pearson is computed per target, so the two channels'
multiples are chosen independently on a grid. `ensemble_mult` does the same for the
averaged output of several members, which has a smaller std than any one of them.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np, torch
R = Path(__file__).resolve().parent.parent
for p in (str(R), str(R/"src")): sys.path.insert(0, p)
from utils import weighted_pearson, WARMUP, SEQUENCE_LENGTH, METRIC_CLIP
from data import SequenceCache

GRID = (0.04, 0.05, 0.06, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5)
N_CAL = 200


@torch.no_grad()
def forward(core, cfg, cache, n, chunk=None, device=None, reset_every=0):
    chunk = chunk or int(os.environ.get("SCORE_CHUNK", "16"))
    """Per-sequence (p0, p1) at the checkpoint's OWN scale; None where ineligible."""
    from score_ckpt import emit, DEV, _lat_for, _z43_extra
    dev = device or DEV
    common = bool(cfg.get("common_head")); sc = float(cfg["_scale"])
    _lat = _lat_for(cfg, cache)          # None unless the run used --lattice-head
    steps = np.arange(SEQUENCE_LENGTH); out = []; N = min(n, len(cache))
    for a in range(0, N, chunk):
        b = min(a+chunk, N)
        X = np.stack([np.asarray(cache.X[i], np.float32) for i in range(a, b)])
        if getattr(core, "keep_cols", None) is not None: X = X[..., core.keep_cols]
        _z4e = _z43_extra(core, cache, a, b)            # add_z43_score.py
        if _z4e is not None: X = np.concatenate([X, _z4e], -1)
        xt = torch.from_numpy(X).permute(1, 0, 2).to(dev)
        if reset_every and reset_every < SEQUENCE_LENGTH:
            # --state-reset N: zero state every N rows, as train.py (fixed counter) and the served driver would
            segs = [core(xt[t0:t0 + reset_every], core.init_state(b-a, dev))[0] for t0 in range(0, SEQUENCE_LENGTH, reset_every)]
            o = torch.cat(segs, 0)
        else:
            o, _ = core(xt, core.init_state(b-a, dev))
        _l = None if _lat is None else (_lat[0], torch.as_tensor(_lat[1][a:b], device=dev, dtype=torch.float32))
        p0, p1 = emit(o.permute(1, 0, 2), sc, cfg, common, lat=_l)
        p0 = p0.cpu().numpy(); p1 = p1.cpu().numpy()
        for k in range(b-a):
            i = a+k
            m = np.asarray(cache.M[i]) & (steps >= WARMUP); y = np.asarray(cache.Y[i])
            ok = m.any() and all(np.any(y[m, j] > 0) and np.any(y[m, j] < 0) for j in (0, 1))
            out.append((p0[k], p1[k]) if ok else None)
    return out


def per_target(P, cache, m0, m1):
    """Per-sequence WP for each target at channel multiples (m0, m1)."""
    steps = np.arange(SEQUENCE_LENGTH); s0, s1 = [], []
    for i, pr in enumerate(P):
        if pr is None: continue
        m = np.asarray(cache.M[i]) & (steps >= WARMUP); y = np.asarray(cache.Y[i])
        s0.append(weighted_pearson(y[m, 0], pr[0][m]*m0)); s1.append(weighted_pearson(y[m, 1], pr[1][m]*m1))
    return np.array(s0), np.array(s1)


def _global_wp(P, cache, k, mult):
    """Globally POOLED weighted Pearson for one target at one multiple -- the live metric."""
    steps = np.arange(SEQUENCE_LENGTH)
    s = sy = sp = syy = spp = syp = 0.0
    for i, pr in enumerate(P):
        if pr is None: continue
        m = np.asarray(cache.M[i]) & (steps >= WARMUP)
        y = np.clip(np.asarray(cache.Y[i])[m, k], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        q = np.clip(mult * pr[k][m].astype(np.float64), -METRIC_CLIP, METRIC_CLIP)
        w = np.abs(y)
        s += w.sum(); sy += (w*y).sum(); sp += (w*q).sum()
        syy += (w*y*y).sum(); spp += (w*q*q).sum(); syp += (w*y*q).sum()
    if s <= 0: return 0.0
    my, mp = sy/s, sp/s
    cov, vy, vp = syp/s - my*mp, syy/s - my*my, spp/s - mp*mp
    return float(cov/np.sqrt(vy*vp)) if vy > 1e-12 and vp > 1e-12 else 0.0


def select(P, cache, grid=GRID):
    """Best multiple per channel, on the GLOBALLY POOLED weighted Pearson.

    This used to maximise the MEAN of PER-SEQUENCE WP, which is the pre-15-Sep metric. That
    objective is degenerate here -- it walks channel 1 to whatever the bottom of the grid is
    (every h256/h384 arm pinned at 0.04, and at 0.004 when the grid was extended). Measured
    2026-09-16, the live global metric is insensitive across that whole range (h384p128 scores
    the same at x0.004 and x0.5, G_pw50_s801 gains +0.000044), so no past screened number is
    materially wrong -- but the exported scale should still be picked on the metric we are
    actually scored by, and this picks sane values (x0.5) instead of degenerate ones.

    The targets are separable under both objectives: channel k only enters target k's WP.
    """
    best = [None, None]; tab = {}
    for g in grid:
        v0, v1 = _global_wp(P, cache, 0, g), _global_wp(P, cache, 1, g)
        tab[g] = (v0, v1)
        for j, v in enumerate((v0, v1)):
            if best[j] is None or v > best[j][1]: best[j] = (g, v)
    return best[0][0], best[1][0], tab


def edge_warning(m0, m1, grid=GRID):
    return [f"channel {j} multiple {m} is at the grid edge" for j, m in enumerate((m0, m1))
            if m in (grid[0], grid[-1])]


def family_filter(cache, n, fams):
    """Sequence positions (< n) whose target-lattice family is in `fams` (training-side label)."""
    from family import base_of
    return [i for i in range(min(n, len(cache))) if base_of(np.asarray(cache.Y[i])) in fams]


def calib_ckpt(tag, ckpt, n=N_CAL, cache=None, verbose=True, fams=None):
    """(m0, m1) for runs/<tag>/<ckpt>, selected on valid_retrain[:n] (optionally only the
    sequences of the given tick families -- for a graph that only serves those families)."""
    from score_ckpt import load
    core, sc, cfg = load(tag, ckpt); cfg = dict(cfg, _scale=sc)
    cache = cache or SequenceCache(R/"cache/valid_retrain")
    if fams is not None:
        n_scan = max(n * 3, 600)
        keep = set(family_filter(cache, n_scan, set(fams)))
        P = forward(core, cfg, cache, n_scan)
        P = [p if i in keep else None for i, p in enumerate(P)]
        if verbose: print(f"  calibrating on {len(keep)} valid_retrain sequences of families {sorted(fams)}")
    else:
        P = forward(core, cfg, cache, n)
    m0, m1, tab = select(P, cache)
    if verbose:
        print(f"  clip-aware scale for {tag}/{ckpt} on valid_retrain[:{n}]: channel multiples "
              f"x{m0}, x{m1} (at x1.0: {tab[1.0][0]:+.4f}/{tab[1.0][1]:+.4f}; chosen: "
              f"{tab[m0][0]:+.4f}/{tab[m1][1]:+.4f})" + "".join("  WARNING " + w for w in edge_warning(m0, m1)))
    return m0, m1


def apply(P, m0, m1):
    return [None if p is None else (p[0]*m0, p[1]*m1) for p in P]


def average(members):
    out = []
    for parts in zip(*members):
        if any(p is None for p in parts): out.append(None); continue
        out.append((np.mean([p[0] for p in parts], 0), np.mean([p[1] for p in parts], 0)))
    return out


def ensemble_mult(members_calibrated, cache, verbose=True):
    """Extra per-channel multiple for the equal-weight average of already-calibrated members."""
    m0, m1, tab = select(average(members_calibrated), cache)
    if verbose:
        print(f"  clip-aware scale for the ensemble on valid_retrain: x{m0}, x{m1} "
              f"(at x1.0: {tab[1.0][0]:+.4f}/{tab[1.0][1]:+.4f}; chosen {tab[m0][0]:+.4f}/{tab[m1][1]:+.4f})")
    return m0, m1
