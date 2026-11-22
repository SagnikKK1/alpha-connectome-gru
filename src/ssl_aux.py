"""Self-supervised auxiliary targets, computed from FEATURES only (no target leakage).

Motivation: the label is information-poor. Per sequence s takes a median of 22 distinct
values over 20,000 rows, changes on ~17.8% of rows, and has lag-1 autocorrelation 0.97.
The features are 112 dimensions changing constantly. So there is far more structure to
learn from than the target alone supplies.

The pretext tasks are the EVENTS the target is built from, not generic masked features
(squared error on an unknown mixing is not mixing-invariant -- the rotation decides which
error directions get penalised):

    ev1    will the i0 price block change at t+1?            dense binary
    dir    sign of the next change vector along its
           dominant direction                                 binary, carried between events
    wait   log rows until the next i0 price event             regression

All are causal and derived from X alone. Heads are dropped at export.
"""
from __future__ import annotations
import numpy as np

I0P = slice(0, 22)


def aux_targets(X: np.ndarray, w_dir: np.ndarray) -> np.ndarray:
    """X: (T,F) raw. Returns (T,6): 3 targets then 3 VALIDITY flags.

    Validity matters: a sampled window is a truncated view. If the next event falls
    outside it we do not know its direction or its distance, and the final row does not
    know whether an event occurs at t+1. Emitting a default there would teach the model
    a false "no event / downward / exactly this far" label on every window edge. Those
    rows are marked invalid and excluded from the loss instead.
    """
    T = X.shape[0]
    d = np.zeros((T, 22), np.float64); d[1:] = np.diff(X[:, I0P], axis=0)
    ev = np.zeros(T, bool); ev[1:] = (np.abs(d[1:]) > 1e-12).any(1)

    nxt = np.zeros(T, np.float32)          # does an event occur at t+1
    nxt[:-1] = ev[1:]
    v_nxt = np.ones(T, np.float32); v_nxt[-1] = 0.0        # t+1 is outside the window

    proj = d @ w_dir                        # signed projection at event rows
    idx = np.where(ev)[0]
    sign_next = np.zeros(T, np.float32)
    wait = np.zeros(T, np.float32)
    v_fut = np.zeros(T, np.float32)         # is there a NEXT event inside this window
    if len(idx):
        pos = np.searchsorted(idx, np.arange(T), side="right")
        has = pos < len(idx)
        j = idx[np.clip(pos, 0, len(idx)-1)]
        sign_next[has] = np.sign(proj[j])[has]
        wait[has] = np.log1p((j - np.arange(T))[has])
        v_fut[has] = 1.0
        v_fut[np.sign(proj[j]) == 0] = 0.0  # a zero projection has no direction to learn
    return np.stack([nxt, sign_next, wait, v_nxt, v_fut, v_fut], 1)


def fit_direction(cache, n=6, warmup=99):
    """Dominant direction of the i0 price-block change vector, from train only."""
    acc = None
    for i in range(n):
        X = np.asarray(cache.X[i], np.float64)
        T = X.shape[0]
        d = np.zeros((T, 22)); d[1:] = np.diff(X[:, I0P], axis=0)
        ev = np.zeros(T, bool); ev[1:] = (np.abs(d[1:]) > 1e-12).any(1)
        V = d[ev & (np.arange(T) >= warmup)]
        nrm = np.linalg.norm(V, axis=1); V = V[nrm > 1e-9] / nrm[nrm > 1e-9, None]
        G = V.T @ V; acc = G if acc is None else acc + G
    return np.linalg.eigh(acc)[1][:, -1]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    R = Path(__file__).resolve().parent.parent
    for p in ("", "src"): sys.path.insert(0, str(R/p) if p else str(R))
    from data import SequenceCache
    tr = SequenceCache(R/"cache/train_full")
    w = fit_direction(tr)
    np.save(R/"runs/aux_direction.npy", w)
    a = aux_targets(np.asarray(tr.X[0], np.float64), w)
    print(f"direction saved. sample targets from sequence 0:")
    print(f"  P(event at t+1)      mean {a[:,0].mean():.4f}")
    print(f"  next-change sign     +1 {100*(a[:,1]>0).mean():.1f}%  -1 {100*(a[:,1]<0).mean():.1f}%")
    print(f"  log wait to event    mean {a[:,2].mean():.3f}  p90 {np.percentile(a[:,2],90):.3f}")
