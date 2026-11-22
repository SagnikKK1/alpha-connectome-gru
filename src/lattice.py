"""Categorical head over the integer tick lattice, with the metric's own readout.

Two facts make this a target REPARAMETERISATION rather than another family
intervention (all eight of which are closed):

1. On RAW targets, s/q is an integer to 5.45e-04 -- every family lives on the same
   integer lattice, so one categorical head serves all four. (On CLIPPED targets it is
   not: the +-2 clip destroys the lattice, which is easy to measure and get wrong.)

2. Family 0's lattice is bimodal with a hole. 57% of its rows sit at k=+-1 and ~25% at
   |k|>=4, while k=+-2 and +-3 hold 0.008 and 0.000. A scalar head must interpolate
   through a region the data never occupies, and |k|>=4 carries 55% of family 0's metric
   weight -- precisely where sign accuracy falls to 0.677 while other families reach 0.87.

**Readout.** Under weighted Pearson with w = |clip(y,+-2)|, the correlation-maximising
predictor is NOT the conditional mean. Maximising corr_w(y, p) over p(x) gives the
w-TILTED conditional mean:

    p*(x) = E[w y | x] / E[w | x],    w = |clip(y, +-2)|

so the readout weights each lattice point by the metric weight it would carry. Reading
out a plain expectation instead -- as disthead.py does -- optimises the wrong functional.
"""
from __future__ import annotations
import numpy as np
import torch

CLIP = 2.0


def k_of(Y, q):
    """Integer lattice index from RAW targets. Y (T,2), q scalar -> (T,) float."""
    return (Y[..., 0] - Y[..., 1]) * 0.5 / q


def weight_of(k, q):
    """The metric weight a row at lattice point k carries: |clip(2q k, +-2)|."""
    return np.minimum(np.abs(2.0 * q * k), CLIP)


def bins(K):
    return np.arange(-K, K + 1, dtype=np.float64)


def readout(logits, q, K):
    """(.., 2K+1) logits -> predicted signal s in TARGET units, w-tilted.

    s = q * sum_k P(k) w(k) k / sum_k P(k) w(k)
    """
    dev = logits.device
    kk = torch.arange(-K, K + 1, device=dev, dtype=logits.dtype)
    q = torch.as_tensor(q, device=dev, dtype=logits.dtype)
    while q.dim() < logits.dim() - 1:
        q = q.unsqueeze(-1)
    w = torch.clamp((2.0 * q.unsqueeze(-1) * kk).abs(), max=CLIP)
    p = torch.softmax(logits, -1) * w
    return q * (p * kk).sum(-1) / p.sum(-1).clamp_min(1e-12)


def labels(Y, q, K):
    """Lattice class index in [0, 2K], with everything beyond +-K folded into the ends."""
    k = np.round(k_of(np.asarray(Y, np.float64), q))
    return np.clip(k, -K, K).astype(np.int64) + K
