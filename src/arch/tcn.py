"""Cached causal temporal convolution.

Replaces the learned recurrent transition with finite causal filters. Dilation widens
the receptive field without touching every past position at inference, and the raw row
bypasses the stack so current volume coordinates are never lost to a low-dimensional
history projection.

  proj      112 -> C pointwise
  7 stages  depthwise conv(k=3, dilation 1,2,4,8,16,32,64) -> pointwise C->C -> GELU
            -> residual
  head      [stage output C | raw normalised 112] -> 1

Receptive field 1 + (3-1)*(1+2+4+8+16+32+64) = 255 rows, which fits inside the existing
256-row burn-in, so a TCN is never compared against a GRU with missing history.

Training uses one padded conv1d per stage over the whole window (fully parallel).
Streaming needs a per-stage ring buffer of 2*dilation rows; total state is
C * 2 * 127 floats = 63.5 KiB at C=64.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

DILATIONS = (1, 2, 4, 8, 16, 32, 64)


class TCNCore(nn.Module):
    """Same interface as GRUCore: forward((T,B,F), state) -> ((T,B,n_out), state)."""

    def __init__(self, hidden: int = 64, layers: int = 7, n_out: int = 1,
                 n_features: int = 112, kernel: int = 3):
        super().__init__()
        self.hidden, self.layers, self.n_out = hidden, len(DILATIONS[:layers]), n_out
        self.kernel = kernel
        self.dils = DILATIONS[:self.layers]
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.proj = nn.Conv1d(n_features, hidden, 1)
        self.dw = nn.ModuleList([nn.Conv1d(hidden, hidden, kernel, dilation=d, groups=hidden)
                                 for d in self.dils])
        self.pw = nn.ModuleList([nn.Conv1d(hidden, hidden, 1) for _ in self.dils])
        # LayerNorm over CHANNELS ONLY, applied independently at each timestep.
        # GroupNorm(1,C) on (B,C,T) normalises over C *and T*, which makes every
        # output depend on the whole sequence -- non-causal, and it leaks the future.
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in self.dils])
        self.head = nn.Linear(hidden + n_features, n_out)   # raw-row bypass

    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu)); self.sd.copy_(torch.as_tensor(sd))

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel - 1) * sum(self.dils)

    def forward(self, x, h=None):
        xn = (x - self.mu) / self.sd                    # (T,B,F)
        T, B, Fdim = xn.shape
        z = xn.permute(1, 2, 0)                         # (B,F,T)
        y = self.proj(z)
        for dil, dw, pw, nm in zip(self.dils, self.dw, self.pw, self.norm):
            pad = (self.kernel - 1) * dil               # causal: pad left only
            u = F.pad(y, (pad, 0))
            u = pw(dw(u))                               # (B,C,T)
            u = nm(u.transpose(1, 2)).transpose(1, 2)   # norm over C at each t
            y = y + F.gelu(u)
        feat = y.permute(2, 0, 1)                       # (T,B,C)
        out = self.head(torch.cat([feat, xn], dim=-1))
        return out, None

    def init_state(self, batch: int, device=None):
        return None                                     # training path is stateless

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        """Rough placeholder on the GRU cost law. A TCN's real streaming cost depends on
        ring-buffer handling and per-stage dispatch, so this MUST be replaced by a
        measured full-callback benchmark before any deployment decision."""
        return 16.8 + 18.0 * (self.n_params() / 1e6)
