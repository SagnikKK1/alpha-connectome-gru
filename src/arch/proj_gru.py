"""Bottlenecked GRU: a learned 112 -> k projection in front of the fused nn.GRU.

Why: in the baseline, W_ih of layer 1 (112x384) is 43k of the 93k layer-1 params, and the
memoryless ridge (rank ONE in the row) already scores 0.534, so the signal-carrying
subspace of the row is low-dimensional. Factorising W_ih as (112xk)(kx3H) keeps the fused
ONNX GRU op (1 node, MLAS sgemv) and spends the saved parameters on hidden width. This is
the lowest-risk candidate: identical export path to the baseline plus one MatMul.
"""
from __future__ import annotations
import torch
import torch.nn as nn

N_FEATURES = 112


class ProjGRUCore(nn.Module):
    def __init__(self, k: int = 32, hidden: int = 192, layers: int = 1, n_out: int = 1,
                 n_features: int = N_FEATURES, dropout: float = 0.0):
        super().__init__()
        self.k, self.hidden, self.layers, self.n_out = k, hidden, layers, n_out
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.proj = nn.Linear(n_features, k, bias=False)
        self.gru = nn.GRU(k, hidden, layers, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_out)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)

    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def init_state(self, batch: int, device=None):
        return torch.zeros(self.layers, batch, self.hidden, device=device or self.mu.device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        return 12.0 + 97.8 * self.n_params() / 1e6 + (5.0 if self.layers > 1 else 0.0)

    def forward(self, x, h=None):
        x = (x - self.mu) / self.sd
        y, h = self.gru(self.proj(x), h)
        return self.head(y), h
