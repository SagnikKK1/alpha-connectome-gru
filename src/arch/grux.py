"""GRU cores that wire the CURRENT ROW straight into the readout, plus two priors.

architecture.md measurement 1: smoothing the GRU's prediction hurts even at half-life 1, so the
predictable part of the target is fast-varying (the book now) and memory is context for
interpreting it. In GRUCore the row reaches the output only through two tanh-bounded
recurrent layers. Here the readout sees [h_top, x] directly:

    skip = "mlp"  : out = MLP([h, x])          (d+112 -> mlp -> 1, SiLU)   the SSM's readout
    skip = "lin"  : out = W h + w.x            (a linear skip; a memoryless ridge is 0.534 on its own)
    skip = "none" : out = W h                  (GRUCore, for the two priors below)

Two further inductive biases, orthogonal to the readout:

    chrono=True   : chrono initialisation (Tallec & Ollivier 2018) of the update-gate bias so the
                    units START with retention time-scales log-uniform in [2, chrono_max] rows
                    instead of ~2 rows. Zero cost, identical graph, a prior on memory length.
    ssm_d > 0     : a parallel learned-decay EMA bank (DiagSSM, bias-corrected) on the same input,
                    concatenated into the readout: GRU for fast nonlinear dynamics, SSM for long
                    memory ("hybrid", multi-rate).

State: (layers, B, hidden) when ssm_d == 0 (GRUCore layout), else one flat (1, B, S) tensor
[gru layers*hidden | ssm h | ssm c]. Every op is a MatMul or elementwise, so the row graph is the
fused GRU op (or the unrolled MatMulNBits cell) plus a Concat and the readout MatMuls.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .scan import chunked_scan
from .ssm import _decay_logit, N_FEATURES, NODE_US


class GRUXCore(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, mlp: int = 96, skip: str = "mlp",
                 chrono: bool = False, chrono_max: float = 1024.0, ssm_d: int = 0, n_out: int = 1,
                 n_features: int = N_FEATURES, hl_min: float = 2.0, hl_max: float = 2048.0):
        super().__init__()
        assert skip in ("mlp", "lin", "none")
        self.hidden, self.layers, self.n_out, self.skip, self.ssm_d = hidden, layers, n_out, skip, ssm_d
        self.F = n_features
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.gru = nn.GRU(n_features, hidden, layers)
        if chrono:
            with torch.no_grad():
                for l in range(layers):
                    tau = torch.logspace(math.log10(2.0), math.log10(chrono_max), hidden)
                    getattr(self.gru, f"bias_hh_l{l}")[hidden:2 * hidden] = torch.log(tau - 1.0)
                    getattr(self.gru, f"bias_ih_l{l}")[hidden:2 * hidden] = 0.0
        if ssm_d:
            self.inp = nn.Linear(n_features, ssm_d)
            self.a_logit = nn.Parameter(_decay_logit(ssm_d, hl_min, hl_max))
        feat = hidden + ssm_d
        if skip == "mlp":
            self.read = nn.Sequential(nn.Linear(feat + n_features, mlp), nn.SiLU(), nn.Linear(mlp, n_out))
            last = self.read[-1]
        else:
            self.head = nn.Linear(feat, n_out); last = self.head
            if skip == "lin":
                self.lin = nn.Linear(n_features, n_out, bias=False)
                nn.init.normal_(self.lin.weight, std=0.02)
        nn.init.zeros_(last.bias); nn.init.normal_(last.weight, std=0.02)
        self.state_dim = layers * hidden + 2 * ssm_d

    # --- GRUCore interface -------------------------------------------------
    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def init_state(self, batch: int, device=None):
        device = device or self.mu.device
        if self.ssm_d == 0:
            return torch.zeros(self.layers, batch, self.hidden, device=device)
        return torch.zeros(1, batch, self.state_dim, device=device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        extra = (3 if self.skip != "none" else 0) + (24 if self.ssm_d else 0)
        return 12.0 + 97.8 * self.n_params() / 1e6 + (5.0 if self.layers > 1 else 0.0) + NODE_US * extra

    # --- forward ------------------------------------------------------------
    def forward(self, x, h=None):
        x = (x - self.mu) / self.sd
        T, B, _ = x.shape
        if h is None:
            h = self.init_state(B, x.device)
        if self.ssm_d == 0:
            y, hn = self.gru(x, h)
            feat, state = y, hn
        else:
            LH = self.layers * self.hidden
            hg = h[0, :, :LH].reshape(B, self.layers, self.hidden).permute(1, 0, 2).contiguous()
            hs, cs = h[0, :, LH:LH + self.ssm_d], h[0, :, LH + self.ssm_d:]
            y, hgn = self.gru(x, hg)
            a = torch.sigmoid(self.a_logit)
            v = (1.0 - a) * self.inp(x)
            if T == 1:
                hsn = a * hs + v[0]; csn = a * cs + (1.0 - a)
                Hs, Cs = hsn.unsqueeze(0), csn.unsqueeze(0)
            else:
                Hs, hsn = chunked_scan(a.view(1, 1, -1), v, hs)
                t = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
                P = torch.exp(t[:, None] * torch.log(a)[None, :])
                Cs = 1.0 - P[:, None, :] * (1.0 - cs)[None]
                csn = Cs[-1]
            feat = torch.cat([y, Hs / Cs.clamp_min(1e-4)], -1)
            state = torch.cat([hgn.permute(1, 0, 2).reshape(1, B, -1), hsn.unsqueeze(0), csn.unsqueeze(0)], -1)
        if self.skip == "mlp":
            out = self.read(torch.cat([feat, x], -1))
        elif self.skip == "lin":
            out = self.head(feat) + self.lin(x)
        else:
            out = self.head(feat)
        return out, state

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
