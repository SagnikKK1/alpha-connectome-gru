"""minGRU-style selective diagonal core: the gate depends on the INPUT only, never on h.

    z_t = sigmoid(W_z x_t + b_z)          per-channel update rate, input-dependent
    u_t = W_h x_t + b_h                   candidate
    h_t = (1 - z_t) * h_{t-1} + z_t * u_t

This is DiagSSMCore with an input-dependent decay (a Mamba-style "selective" SSM without the
dt/B/C machinery): the model can forget faster when the book changes regime (F4: a3 is a
slow regime variable) and integrate longer when nothing happens. Because z does not depend
on h there is no h-to-h matrix (no W_hh term, the GRU's dominant cost at H=128), the step
is elementwise + two MatMuls, and training uses Heinsen's log-space parallel scan (scan.py).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .scan import chunked_scan
from .ssm import _decay_logit, N_FEATURES, NODE_US


class MinGRUCore(nn.Module):
    def __init__(self, d: int = 256, mlp: int = 96, n_out: int = 1, n_features: int = N_FEATURES,
                 hl_min: float = 2.0, hl_max: float = 2048.0, chunk: int | None = 128):
        super().__init__()
        self.d, self.n_out, self.layers, self.hidden, self.chunk = d, n_out, 1, d, chunk
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.gate = nn.Linear(n_features, d)
        self.cand = nn.Linear(n_features, d)
        with torch.no_grad():                      # start as the fixed-half-life EMA bank
            self.gate.bias.copy_(-_decay_logit(d, hl_min, hl_max))   # z = 1 - a0
            self.gate.weight.mul_(0.1)
        self.read = nn.Sequential(nn.Linear(d + n_features, mlp), nn.SiLU(), nn.Linear(mlp, n_out))
        nn.init.zeros_(self.read[-1].bias)
        nn.init.normal_(self.read[-1].weight, std=0.02)

    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def init_state(self, batch: int, device=None):
        return torch.zeros(1, batch, self.d, device=device or self.mu.device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_nodes(self):
        return 24          # after ORT optimisation, measured

    def est_us_per_row(self):
        return 12.0 + 97.8 * self.n_params() / 1e6 + NODE_US * self.n_nodes()

    def forward(self, x, h=None):
        x = (x - self.mu) / self.sd
        T, B, _ = x.shape
        z = torch.sigmoid(self.gate(x))
        v = z * self.cand(x)
        a = 1.0 - z
        h0 = h[0] if h is not None else x.new_zeros(B, self.d)
        if T == 1:
            hn = a[0] * h0 + v[0]
            H = hn.unsqueeze(0)
        else:
            H, hn = chunked_scan(a, v, h0, self.chunk)
        out = self.read(torch.cat([H, x], dim=-1))
        return out, hn.unsqueeze(0)

    # `export.py`, `train.py` and `score_ckpt.py` all key on `core.head`; these cores
    # named their readout `read`, so they silently bypassed the exporter's head-slicing
    # and every aux/common/lattice-head path. A property rather than a rename, so
    # checkpoints saved with `read.*` keys still load.
    @property
    def head(self):
        return self.read

    @head.setter
    def head(self, v):
        self.read = v
