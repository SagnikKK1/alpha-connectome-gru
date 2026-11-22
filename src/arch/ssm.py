"""Diagonal linear state-space core (LRU / S4D-style), the learned generalisation of the
EDA's fixed EMA bank (F5: a 6-timescale EMA bank recovers half the GRU's gain over a
memoryless ridge).

Per channel i:   h_t[i] = a_i * h_{t-1}[i] + (1 - a_i) * u_t[i],   u_t = W_in x_t + b
so each channel is an exponential moving average of a LEARNED linear combination of the
112 features, with a LEARNED half-life (initialised log-uniformly over 2..2048 rows, the
range F3 says the targets live on). The readout is a small MLP over [h_t, x_t], so the
only nonlinearity is between smoothed context and the current row. Optional `gate` adds a
Mamba-style multiplicative gate h * silu(W_g x).

`bias_correct` keeps a second state vector c_t = 1 - a^(t+1) (closed form, no scan) and
reads h_t / c_t: an Adam-style correction so a half-life-2048 channel is an average of the
rows seen so far rather than a value shrunk toward zero for thousands of rows after the
step-0 reset. The EDA's EMA bank got the same effect by initialising at the first row.

Cost: every weight is a plain MatMul (no fused-GRU op). After ORT's optimiser the row
graph is ~21 nodes, measured to sit ON the GRU cost law (node tax < 1 us), and it is
int8-quantisable with onnxruntime.quantization.quantize_dynamic (-20% at d=512).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import chunked_scan

N_FEATURES = 112
# Per-node ONNX Runtime dispatch tax at ~30-node graphs, measured by bench_latency.py (architecture.md).
NODE_US = 0.10


def _decay_logit(d: int, hl_min: float, hl_max: float) -> torch.Tensor:
    hl = torch.logspace(math.log10(hl_min), math.log10(hl_max), d)
    a0 = 0.5 ** (1.0 / hl)                       # per-step retention for that half-life
    return torch.log(a0 / (1.0 - a0))


class DiagSSMCore(nn.Module):
    def __init__(self, d: int = 256, mlp: int = 96, n_out: int = 1, n_features: int = N_FEATURES,
                 hl_min: float = 2.0, hl_max: float = 2048.0, learn_decay: bool = True,
                 gate: bool = False, pre_act: str = "none", chunk: int | None = None,
                 bias_correct: bool = True):
        super().__init__()
        self.d, self.n_out, self.hidden = d, n_out, d
        self.pre_act, self.chunk, self.use_gate, self.bias_correct = pre_act, chunk, gate, bias_correct
        self.layers = 2 if bias_correct else 1          # state slots: h, and c if bias-corrected
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.inp = nn.Linear(n_features, d)
        logit = _decay_logit(d, hl_min, hl_max)
        if learn_decay:
            self.a_logit = nn.Parameter(logit)
        else:
            self.register_buffer("a_logit", logit)
        if gate:
            self.gate = nn.Linear(n_features, d)
        self.read = nn.Sequential(nn.Linear(d + n_features, mlp), nn.SiLU(), nn.Linear(mlp, n_out))
        nn.init.zeros_(self.read[-1].bias)
        nn.init.normal_(self.read[-1].weight, std=0.02)

    # --- GRUCore interface -------------------------------------------------
    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def init_state(self, batch: int, device=None):
        return torch.zeros(self.layers, batch, self.d, device=device or self.mu.device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_nodes(self):
        """ONNX nodes after ORT optimisation (measured: 21 plain, +3 gate, +1 tanh, +3 bias-corr)."""
        return 21 + (3 if self.use_gate else 0) + (1 if self.pre_act != "none" else 0) + (3 if self.bias_correct else 0)

    def est_us_per_row(self):
        return 12.0 + 97.8 * self.n_params() / 1e6 + NODE_US * self.n_nodes()

    def decay(self):
        return torch.sigmoid(self.a_logit)

    def half_lives(self):
        return (math.log(0.5) / torch.log(self.decay().detach())).cpu()

    # --- forward ------------------------------------------------------------
    def forward(self, x, h=None):
        """x: (T,B,F) -> out (T,B,n_out), state (layers,B,d). T==1 takes the plain step (ONNX path)."""
        x = (x - self.mu) / self.sd
        T, B, _ = x.shape
        a = self.decay()
        u = self.inp(x)
        if self.pre_act == "tanh":
            u = torch.tanh(u)
        v = (1.0 - a) * u
        if h is None:
            h = self.init_state(B, x.device)
        h0 = h[0]
        if T == 1:
            hn = a * h0 + v[0]
            H = hn.unsqueeze(0)
        else:
            H, hn = chunked_scan(a.view(1, 1, -1), v, h0, self.chunk)
        if self.bias_correct:                     # c_t = 1 - a^(t+1) (1 - c_0), closed form
            c0 = h[1]
            if T == 1:
                cn = a * c0 + (1.0 - a)
                C = cn.unsqueeze(0)
            else:
                t = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
                P = torch.exp(t[:, None] * torch.log(a)[None, :])           # (T,D): a^(t+1)
                C = 1.0 - P[:, None, :] * (1.0 - c0)[None]
                cn = C[-1]
            feat = H / C.clamp_min(1e-4)
            state = torch.stack([hn, cn])
        else:
            feat = H
            state = hn.unsqueeze(0)
        if self.use_gate:
            feat = feat * F.silu(self.gate(x))
        out = self.read(torch.cat([feat, x], dim=-1))
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
