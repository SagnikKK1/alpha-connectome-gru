"""GRU with a DISTRIBUTIONAL output head.

EDA: the targets live on a tick ladder — t0 takes 133 distinct values with a quantum
of ~0.0627 (min nonzero |t0| = 0.1254 = exactly two ticks), and s = (t0−t1)/2 takes
~230. So the target is closer to a count than to a continuous quantity.

Instead of regressing a scalar, predict a distribution over bins of s and emit its
EXPECTATION. Two reasons this can beat a point estimate under a correlation metric:
the expectation of a softmax is a smoother function of the features than a direct
linear read-out, and it can express uncertainty (a wide distribution shrinks toward
the mean) rather than being forced to commit.

Costs K−1 extra head parameters and one softmax per row. The output stays a single
scalar, so the (s, −s) emission and the export path are unchanged.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

N_FEATURES = 112


class DistGRUCore(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, bins: int = 33,
                 n_out: int = 1, n_features: int = N_FEATURES, span: float = 2.0):
        super().__init__()
        assert n_out == 1, "distributional head emits one scalar"
        self.hidden, self.layers, self.n_out, self.bins = hidden, layers, 1, bins
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        # centres span the clipped range; the metric clips at +-2 so nothing beyond matters
        self.register_buffer("centres", torch.linspace(-span, span, bins))
        self.gru = nn.GRU(n_features, hidden, layers, batch_first=False)
        self.head = nn.Linear(hidden, bins)
        nn.init.zeros_(self.head.bias); nn.init.normal_(self.head.weight, std=0.02)

    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, x, h=None):
        y, h = self.gru((x - self.mu) / self.sd, h)
        p = torch.softmax(self.head(y), dim=-1)          # (T,B,bins)
        return (p * self.centres).sum(-1, keepdim=True), h

    def init_state(self, batch: int, device=None):
        return torch.zeros(self.layers, batch, self.hidden,
                           device=device or next(self.parameters()).device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        return 17.2 + 19.9 * (self.n_params() / 1e6)
