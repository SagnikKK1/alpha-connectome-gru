"""Nonlinear per-timestep encoder in front of the GRU.

Only a nonlinearity adds EXPRESSIVITY: a linear encoder folds straight into the GRU's
W_ih, so it spans no new function class. It is not, however, a training no-op —
factorising W_ih as two matrices changes the optimisation path, and enc < n_features
imposes a rank constraint on the input map. `act="none"` is exactly that control: the
gap between it and plain `gru` is optimisation and rank; the gap between a nonlinear
encoder and it is nonlinearity. The encoder is stateless, so the recurrent state and the
ONNX export path are unchanged from GRUCore — it is just a few extra nodes per row.
"""
from __future__ import annotations
import torch
import torch.nn as nn

N_FEATURES = 112


class EncGRUCore(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, enc: int = 160, enc_layers: int = 2,
                 n_out: int = 1, n_features: int = N_FEATURES, act: str = "gelu"):
        super().__init__()
        self.hidden, self.layers, self.n_out = hidden, layers, n_out
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        A = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh, "none": nn.Identity}[act]
        dims = [n_features] + [enc] * enc_layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), A()]
        self.enc = nn.Sequential(*mods)
        self.gru = nn.GRU(enc, hidden, layers, batch_first=False)
        self.head = nn.Linear(hidden, n_out)
        nn.init.zeros_(self.head.bias); nn.init.normal_(self.head.weight, std=0.02)

    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, x, h=None):
        x = (x - self.mu) / self.sd
        y, h = self.gru(self.enc(x), h)
        return self.head(y), h

    def init_state(self, batch: int, device=None):
        return torch.zeros(self.layers, batch, self.hidden,
                           device=device or next(self.parameters()).device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        # architecture.md section 2 cost law, the one every other core uses:
        # 12 + 97.8 x M(params) us, +5 for a second layer, +0.1 us per extra node.
        # (Was 17.2 + 19.9 x M, a ~5x too-small slope that under-reported the budget.)
        n_enc_nodes = 2 * sum(1 for m in self.enc if isinstance(m, nn.Linear))
        return (12.0 + 97.8 * self.n_params() / 1e6 + (5.0 if self.layers > 1 else 0.0)
                + 0.1 * n_enc_nodes)
