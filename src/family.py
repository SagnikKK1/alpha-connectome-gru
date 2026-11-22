"""Identify a sequence's tick FAMILY from its warm-up rows, and cache it per sequence.

The target is a trading P&L in ticks: t0 = 2q*n0, t1 = 2q*n1 with (n0,n1) = (P-ask, bid-P)
and the losing side pinned at -3. s = (t0-t1)/2 = P - mid in half-ticks. The base q takes
exactly FOUR values, one per sequence, constant to 7 decimals:

    0.328431   42.5% of train, 36.4% of holdout
    0.234268   33.9% / 24.8%
    0.120664   11.5% / 17.8%
    0.062717   12.1% / 21.0%

Per-family WP is 0.359 / 0.659 / 0.699 / 0.728 -- the large-tick family is 36-42% of all
sequences and is where essentially all of the score is lost.

The family is recoverable from rows 0-98 (the forced warm-up, before scoring begins), so
it is available CAUSALLY at inference: classify once at sequence start, then hold it
constant. Ground truth for training comes from the target lattice; the classifier is what
deploys.
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

R = Path(__file__).resolve().parent.parent
for p in ("", "src"): sys.path.insert(0, str(R/p) if p else str(R))
from utils import WARMUP, SEQUENCE_LENGTH, METRIC_CLIP
from data import SequenceCache

BASES = np.array([0.328431, 0.234268, 0.120664, 0.062717])


def base_of(Y):
    """Ground-truth family from the target lattice: the base whose grid the jumps sit on."""
    t0 = np.clip(Y[:, 0], -METRIC_CLIP, METRIC_CLIP); t1 = np.clip(Y[:, 1], -METRIC_CLIP, METRIC_CLIP)
    s = (t0 - t1) / 2
    d = np.abs(np.diff(s)); d = d[d > 1e-9]
    if len(d) < 50: return -1
    cov = [(np.abs(d/q - np.round(d/q)) < 0.02).mean() for q in BASES]
    # the COARSEST base that covers: finer lattices trivially cover coarser ones
    ok = [i for i, c in enumerate(cov) if c > 0.90]
    return min(ok) if ok else int(np.argmax(cov))


def warmup_feats(X):
    """Descriptors from rows 0..98 only -- available before the first scored row."""
    P = np.asarray(X[:WARMUP], np.float64)
    d = np.abs(np.diff(P, axis=0))
    return np.concatenate([P.mean(0), P.std(0), d.mean(0), (d > 1e-12).mean(0)])


def fit(cache, n=1500):
    F, y = [], []
    for i in range(min(n, len(cache))):
        b = base_of(np.asarray(cache.Y[i]))
        if b < 0: continue
        F.append(warmup_feats(cache.X[i])); y.append(b)
    F = np.array(F); y = np.array(y)
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = np.c_[(F - mu)/sd, np.ones(len(F))]
    W = np.zeros((4, Z.shape[1]))
    for k in range(4):                       # one-vs-rest ridge
        t = (y == k).astype(np.float64)*2 - 1
        G = Z.T@Z
        W[k] = np.linalg.solve(G + 1e-2*np.trace(G)/G.shape[0]*np.eye(G.shape[0]), Z.T@t)
    return W, mu, sd


def predict(X, W, mu, sd):
    z = np.r_[(warmup_feats(X) - mu)/sd, 1.0]
    return int(np.argmax(W @ z))


if __name__ == "__main__":
    tr = SequenceCache(R/"cache/train_full")
    W, mu, sd = fit(tr, 1500)
    np.savez(R/"runs/family_clf.npz", W=W, mu=mu, sd=sd, bases=BASES)
    for nm in ("train_full", "eval", "holdout", "valid_retrain"):
        c = SequenceCache(R/f"cache/{nm}")
        n = min(400, len(c))
        truth, pred = [], []
        for i in range(n):
            b = base_of(np.asarray(c.Y[i]))
            if b < 0: continue
            truth.append(b); pred.append(predict(c.X[i], W, mu, sd))
        truth = np.array(truth); pred = np.array(pred)
        share = np.bincount(truth, minlength=4)/len(truth)
        print(f"  {nm:14s} acc {100*(truth==pred).mean():5.1f}%   family shares "
              f"{np.array2string(share, precision=3)}", flush=True)
    # cache the PREDICTED family for every training sequence, for use as a model input
    fam = np.array([predict(tr.X[i], W, mu, sd) for i in range(len(tr))], np.int64)
    np.save(R/"runs/family_train.npy", fam)
    print(f"\n  cached predicted family for {len(fam)} train sequences: "
          f"{np.bincount(fam, minlength=4)}")
