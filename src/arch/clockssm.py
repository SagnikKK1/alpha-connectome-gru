"""Clock-aware diagonal SSM: per-channel decay driven by ELAPSED TIME, not row count.

The market is not uniformly sampled: a0 is the standardised log of the inter-row time gap
(features.md section 3; on train, exp(a0_std) has median 0.91, mean 1.51, 99th pct 9.6). Every
recurrent model so far decays its state by a fixed factor per ROW, so "the last 500 rows" means
seconds in a burst and minutes in a lull. Here a channel's log-decay on row t is

    la_t[i] = -rate_i * g_t          g_t = exp(k * a0_t + c)   (wall-clock channels)
    la_t[i] = -rate_i                                          (row-clock channels)

with rate_i = ln2 / half-life_i learned per channel (log-uniform init over 2..2048), and k, c
learned scalars (k=1, c=-0.4 so that g averages ~1 on train: the same initial half-lives as the
row-clock bank, measured in "typical rows"). `frac_wall` sets the share of wall-clock channels;
0 reproduces DiagSSMCore through the new scan (the code-path control), 1 makes every channel a
wall-clock EMA. Half-lives are interleaved so both banks span the whole range.

Everything else is DiagSSMCore: u = W_in x, h_t = a_t h_{t-1} + (1-a_t) u_t, Adam-style bias
correction c_t = a_t c_{t-1} + (1-a_t) as a second state vector (closed form over a window:
c_t = 1 - exp(sum_{j<=t} la_j) (1 - c_0), one fp64 cumsum), readout MLP([h/c, x]). Step cost:
one extra Exp/Mul/Clip chain on a d-vector (~6 nodes).

Bounds (identical in the scan and the T==1 step, so export matches training):
    g <= 20 (clamp of the exponent at 3), rate <= 0.46 (half-life >= 1.5 rows), la >= -4.5.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .scan64 import scan_two_level, LA_MAX
from .ssm import N_FEATURES, NODE_US

CLOCK_COL = 104          # a0
RATE_MAX = math.log(2) / 1.5
G_LOG_MAX = 3.0          # g <= e^3 = 20
SEG = 2048               # no-grad forward over longer windows runs in segments of this length


class ClockSSMCore(nn.Module):
    def __init__(self, d: int = 512, mlp: int = 128, n_out: int = 1, n_features: int = N_FEATURES,
                 hl_min: float = 2.0, hl_max: float = 2048.0, frac_wall: float = 0.5,
                 k_init: float = 1.0, c_init: float = -0.4, learn_clock: bool = True, chunk: int = 16):
        super().__init__()
        self.d, self.n_out, self.hidden, self.layers, self.chunk = d, n_out, d, 2, chunk
        self.frac_wall = frac_wall
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.inp = nn.Linear(n_features, d)
        hl = torch.logspace(math.log10(hl_min), math.log10(hl_max), d)
        self.log_rate = nn.Parameter(torch.log(math.log(2) / hl))
        n_wall = int(round(frac_wall * d))
        wall = torch.zeros(d)
        if n_wall > 0:                                   # interleave: every (d/n_wall)-th channel is wall-clock
            idx = torch.linspace(0, d - 1, n_wall).round().long()
            wall[idx] = 1.0
        self.register_buffer("wall", wall)
        k = torch.tensor(float(k_init)); c = torch.tensor(float(c_init))
        if learn_clock:
            self.clock_k = nn.Parameter(k); self.clock_c = nn.Parameter(c)
        else:
            self.register_buffer("clock_k", k); self.register_buffer("clock_c", c)
        self.read = nn.Sequential(nn.Linear(d + n_features, mlp), nn.SiLU(), nn.Linear(mlp, n_out))
        nn.init.zeros_(self.read[-1].bias)
        nn.init.normal_(self.read[-1].weight, std=0.02)

    # --- GRUCore interface -------------------------------------------------
    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def init_state(self, batch: int, device=None):
        return torch.zeros(2, batch, self.d, device=device or self.mu.device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_nodes(self):
        return 24 + 6

    def est_us_per_row(self):
        return 12.0 + 97.8 * self.n_params() / 1e6 + NODE_US * self.n_nodes()

    def half_lives(self):
        return (math.log(2) / torch.exp(self.log_rate).clamp(max=RATE_MAX).detach()).cpu()

    # --- pieces -------------------------------------------------------------
    def log_decay(self, xn):
        """xn (T,B,F) standardised -> la (T,B,d) in [-LA_MAX, 0]."""
        z = xn[..., CLOCK_COL:CLOCK_COL + 1]                                  # (T,B,1)
        g = torch.exp((self.clock_k * z + self.clock_c).clamp(max=G_LOG_MAX))  # (T,B,1)
        geff = self.wall * g + (1.0 - self.wall)                              # (T,B,d)
        rate = torch.exp(self.log_rate).clamp(max=RATE_MAX)
        return -(rate * geff).clamp(max=LA_MAX)

    def forward(self, x, h=None):
        T, B, _ = x.shape
        if h is None:
            h = self.init_state(B, x.device)
        if T > SEG and not torch.is_grad_enabled():
            # whole-sequence eval: bound memory by carrying the state across segments
            outs = []
            for s in range(0, T, SEG):
                o, h = self.forward(x[s:s + SEG], h); outs.append(o)
            return torch.cat(outs, 0), h
        x = (x - self.mu) / self.sd
        la = self.log_decay(x)
        a = torch.exp(la)
        one_m_a = 1.0 - a
        v = one_m_a * self.inp(x)
        if T == 1:
            hn = a[0] * h[0] + v[0]
            cn = a[0] * h[1] + one_m_a[0]
            H, C = hn.unsqueeze(0), cn.unsqueeze(0)
        else:
            H, hn = scan_two_level(la, v, h[0], self.chunk)
            S = torch.cumsum(la.double(), 0)                       # sum_{j<=t} la_j  (fp64: |S| up to ~3500)
            C = (1.0 - torch.exp(S) * (1.0 - h[1].double())[None]).to(x.dtype)
            cn = C[-1]
        feat = H / C.clamp_min(1e-4)
        out = self.read(torch.cat([feat, x], dim=-1))
        return out, torch.stack([hn, cn])

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
