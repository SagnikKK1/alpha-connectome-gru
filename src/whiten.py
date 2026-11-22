"""PCA-whitened input in front of the GRU.

EDA finding that motivates this: the top 3 principal components hold 40% of the
input variance and ~ZERO predictive signal (ridge on them alone: -0.0075 WP;
removing them costs 0.0020). The signal lives in the low-variance directions.

Per-column standardisation - what every model so far has used - preserves that
variance hierarchy, so the network is handed a representation whose dominant
directions are noise and must learn to amplify the rest against its initialisation.
Whitening hands it over for free:

    z = diag(1/sqrt(lam + eps)) Vᵀ (x - mu)

A fixed linear map folded into a buffer, so the recurrent state and the export path
are unchanged and it costs ~0.25 us/row. `shrink` optionally soft-caps the
amplification of the very smallest eigendirections, which are the ones most likely
to be noise that whitening would blow up.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

N_FEATURES = 112


def fit_whitener(cache, n_seq: int = 60, stride: int = 7, warmup: int = 99,
                 shrink: float = 0.0):
    """Streamed covariance -> (W, b) with z = x @ W.T + b."""
    C = np.zeros((N_FEATURES, N_FEATURES)); mu = np.zeros(N_FEATURES); n = 0
    for i in range(min(n_seq, len(cache))):
        X = np.asarray(cache.X[i][warmup::stride], np.float64)
        C += X.T @ X; mu += X.sum(0); n += len(X)
    mu /= n; C = C / n - np.outer(mu, mu)
    lam, V = np.linalg.eigh(C)
    lam = np.maximum(lam, 1e-8)[::-1]; V = V[:, ::-1]
    # shrink>0 pulls the smallest eigenvalues up toward the mean, damping the gain
    # applied to directions that are probably noise rather than weak signal.
    if shrink > 0:
        lam = (1 - shrink) * lam + shrink * lam.mean()
    W = (V / np.sqrt(lam)).T                      # (112,112): z = W @ (x - mu)
    return W.astype(np.float32), (-W @ mu).astype(np.float32)


class WhitenGRUCore(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, n_out: int = 1,
                 n_features: int = N_FEATURES, dropout: float = 0.0):
        super().__init__()
        self.hidden, self.layers, self.n_out = hidden, layers, n_out
        self.register_buffer("W", torch.eye(n_features))
        self.register_buffer("b", torch.zeros(n_features))
        self.register_buffer("mu", torch.zeros(n_features))    # kept for interface parity
        self.register_buffer("sd", torch.ones(n_features))
        self.gru = nn.GRU(n_features, hidden, layers, batch_first=False,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_out)
        nn.init.zeros_(self.head.bias); nn.init.normal_(self.head.weight, std=0.02)

    def set_whitener(self, W, b):
        with torch.no_grad():
            self.W.copy_(torch.as_tensor(W)); self.b.copy_(torch.as_tensor(b))

    def set_norm(self, mu, sd):        # trainer calls this; whitening supersedes it
        pass

    def forward(self, x, h=None):
        y, h = self.gru(torch.nn.functional.linear(x, self.W, self.b), h)
        return self.head(y), h

    def init_state(self, batch: int, device=None):
        return torch.zeros(self.layers, batch, self.hidden,
                           device=device or next(self.parameters()).device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        return 17.2 + 19.9 * ((self.n_params() + self.W.numel()) / 1e6)
