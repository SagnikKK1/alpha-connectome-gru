"""TCN front-end feeding a GRU: local temporal patterns + unbounded event memory.

Why this shape, given what the target is:
  - s is a near-flat sum of the next ~64 i0 EVENTS, and the price mean-reverts hard
    (variance ratio 0.0425 at 128 events). So the job is estimating the current
    displacement from fair value, not forecasting.
  - A pure TCN scored 0.6314 vs the GRU's 0.6386 and, blended as a separate predictor,
    contributed nothing (optimal alpha = 0). Its dilations are uniform in ROW space
    while the target's clock is EVENTS, so a fixed receptive field covers a variable
    and unknown number of events.
  - So use the TCN as a learned local FEATURE EXTRACTOR and let the GRU keep the
    adaptive, unbounded memory. The raw row is concatenated through so the dominant
    i0_v direction is never lost to a low-dimensional projection.

Gated activation (tanh * sigmoid, WaveNet-style) rather than GELU: multiplicative
gating lets a stage suppress its own output when nothing happened, which matters when
most rows carry no i0 price event.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNGRUCore(nn.Module):
    def __init__(self, hidden: int = 192, layers: int = 2, n_out: int = 1,
                 n_features: int = 112, tcn_ch: int = 64, tcn_dilations=(1, 2, 4, 8, 16, 32),
                 kernel: int = 3):
        super().__init__()
        self.hidden, self.layers, self.n_out = hidden, layers, n_out
        self.tcn_ch, self.dils, self.kernel = tcn_ch, tuple(tcn_dilations), kernel
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.proj = nn.Conv1d(n_features, tcn_ch, 1)
        self.dw = nn.ModuleList([nn.Conv1d(tcn_ch, tcn_ch, kernel, dilation=d, groups=tcn_ch)
                                 for d in self.dils])
        self.pw = nn.ModuleList([nn.Conv1d(tcn_ch, 2*tcn_ch, 1) for _ in self.dils])   # filter+gate
        self.res = nn.ModuleList([nn.Conv1d(tcn_ch, tcn_ch, 1) for _ in self.dils])
        self.norm = nn.ModuleList([nn.LayerNorm(tcn_ch) for _ in self.dils])
        self.gru = nn.GRU(tcn_ch + n_features, hidden, layers)
        self.head = nn.Linear(hidden, n_out)

    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu)); self.sd.copy_(torch.as_tensor(sd))

    @property
    def receptive_field(self):
        return 1 + (self.kernel - 1) * sum(self.dils)

    def forward(self, x, h=None):
        xn = (x - self.mu) / self.sd                    # (T,B,F)
        T, B, _ = xn.shape
        z = xn.permute(1, 2, 0)                         # (B,F,T)
        y = self.proj(z)
        for dil, dw, pw, res, nm in zip(self.dils, self.dw, self.pw, self.res, self.norm):
            u = F.pad(y, ((self.kernel - 1) * dil, 0))  # causal: left pad only
            u = pw(dw(u))
            f, g = u.chunk(2, dim=1)
            u = torch.tanh(f) * torch.sigmoid(g)        # gated activation
            u = nm(u.transpose(1, 2)).transpose(1, 2)   # norm over CHANNELS at each t
            y = y + res(u)
        feat = y.permute(2, 0, 1)                       # (T,B,C)
        out, h = self.gru(torch.cat([feat, xn], dim=-1), h)
        return self.head(out), h

    def init_state(self, batch: int, device=None):
        d = device or next(self.parameters()).device
        return torch.zeros(self.layers, batch, self.hidden, device=d)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        """Placeholder on the GRU cost law. A TCN front-end needs ring buffers and extra
        per-stage dispatch at streaming time; this MUST be replaced by a measured
        full-callback benchmark before any deployment decision."""
        return 16.8 + 18.0 * (self.n_params() / 1e6)
