"""GRU + streaming softmax attention pooling: content-weighted history at O(1) per row.

Query-free attention pooling is exactly computable in a streaming setting. If each hidden
state is scored on its own, s_t = f(h_t), then the softmax-weighted summary of history

    c_t = sum_{j<=t} e^{s_j} h_j / sum_{j<=t} e^{s_j}

is a ratio of two running sums, so it needs constant work per row and fixed-size state --
no window buffer and no rescoring of the past. Only a *query-dependent* score
s_j = f(h_j, h_t), which rescores history against the current row, costs O(window).

Two departures from the plain form above, both deliberate:

* **Decay.** Each head keeps lambda_k in (0,1):

      N_t = lambda_k N_{t-1} + e^{s_t} h_t        Z_t = lambda_k Z_{t-1} + e^{s_t}

  Undecayed pooling averages over the whole sequence, so by row 20,000 the denominator is
  so large that c_t barely moves -- a near-static summary. With decay, a head pools over an
  exponential window of ~lambda's half-life, and K heads initialised at different half-lives
  give multi-timescale *content-weighted* summaries. lambda_k = 1 recovers the global form.

* **Bounded scores.** s_t = score_scale * tanh(w_k . h_t + b_k) keeps e^s in
  [e^-c, e^c], so both accumulators stay in fp32 range over 20,000 rows with no running-max
  tracking (the usual online-softmax trick) and no log-space scan. The ratio is invariant to
  a constant shift in s, so bounding costs only the sharpness of the weighting.

**The ablation that isolates attention.** `score=False` fixes s_t = 0, so every row gets equal
weight and c_t collapses to a plain EMA of the hidden state at the same half-lives -- the same
parameters, the same state, the same cost, minus the content weighting. Comparing `attnpool`
against `emapool` is therefore a paired test of whether attention *itself* buys anything over
the decayed summaries the SSM bank already provides.

Both recurrences are linear with a time-invariant decay, so training uses the same chunked
parallel scan as DiagSSMCore; the T==1 path is the two-line update above and is what exports.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .scan import chunked_scan
from .ssm import N_FEATURES, NODE_US


class AttnPoolCore(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 1, heads: int = 2, mlp: int = 96,
                 n_out: int = 1, n_features: int = N_FEATURES, score: bool = True,
                 score_scale: float = 4.0, hl_min: float = 16.0, hl_max: float = 2048.0):
        super().__init__()
        self.hidden, self.layers, self.heads = hidden, layers, heads
        self.n_out, self.use_score, self.score_scale = n_out, score, score_scale
        self.F = n_features
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.gru = nn.GRU(n_features, hidden, layers)
        # one decay per head, log-uniform half-lives over [hl_min, hl_max]
        hl = torch.logspace(math.log10(hl_min), math.log10(hl_max), heads)
        self.lam_logit = nn.Parameter(torch.log(torch.expm1(math.log(2.0) / hl).clamp_min(1e-6)))
        if score:
            self.score = nn.Linear(hidden, heads)
            nn.init.normal_(self.score.weight, std=0.02); nn.init.zeros_(self.score.bias)
        self.read = nn.Sequential(nn.Linear(hidden * (1 + heads) + n_features, mlp),
                                  nn.SiLU(), nn.Linear(mlp, n_out))
        nn.init.zeros_(self.read[-1].bias); nn.init.normal_(self.read[-1].weight, std=0.02)
        # state: [gru layers*hidden | N (heads*hidden) | Z (heads)]
        self.state_dim = layers * hidden + heads * hidden + heads

    # --- GRUCore interface -------------------------------------------------
    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def init_state(self, batch: int, device=None):
        device = device or self.mu.device
        return torch.zeros(1, batch, self.state_dim, device=device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def lam(self):
        """decay per head in (0,1); softplus-parameterised so it cannot leave the range."""
        return torch.exp(-nn.functional.softplus(self.lam_logit))

    def half_lives(self):
        return (math.log(2) / (-torch.log(self.lam())).clamp_min(1e-9)).detach().cpu()

    def est_us_per_row(self):
        # +2 matmuls (score, readout) and ~10 elementwise nodes for the two accumulators
        return (12.0 + 97.8 * self.n_params() / 1e6 + (5.0 if self.layers > 1 else 0.0)
                + NODE_US * (12 + 2 * self.heads))

    # --- forward ------------------------------------------------------------
    def forward(self, x, h=None):
        x = (x - self.mu) / self.sd
        T, B, _ = x.shape
        if h is None:
            h = self.init_state(B, x.device)
        LH, K, H = self.layers * self.hidden, self.heads, self.hidden
        hg = h[0, :, :LH].reshape(B, self.layers, H).transpose(0, 1).contiguous()
        N0 = h[0, :, LH:LH + K * H]                                  # (B, K*H)
        Z0 = h[0, :, LH + K * H:]                                    # (B, K)

        y, hgn = self.gru(x, hg)                                     # y (T,B,H)
        if self.use_score:
            s = self.score_scale * torch.tanh(self.score(y))         # (T,B,K), bounded
        else:
            s = y.new_zeros(T, B, K)                                 # ablation: uniform weights
        w = torch.exp(s)                                             # (T,B,K) in [e^-c, e^c]

        lam = self.lam()                                             # (K,)
        vN = (w.unsqueeze(-1) * y.unsqueeze(2)).reshape(T, B, K * H)  # e^s * h
        aN = lam.repeat_interleave(H).view(1, 1, K * H)
        aZ = lam.view(1, 1, K)
        if T == 1:
            Nn = aN[0, 0] * N0 + vN[0]
            Zn = aZ[0, 0] * Z0 + w[0]
            Nt, Zt = Nn.unsqueeze(0), Zn.unsqueeze(0)
        else:
            Nt, Nn = chunked_scan(aN, vN, N0)
            Zt, Zn = chunked_scan(aZ, w, Z0)
        c = Nt.view(T, B, K, H) / Zt.clamp_min(1e-6).unsqueeze(-1)   # (T,B,K,H)

        out = self.read(torch.cat([y, c.reshape(T, B, K * H), x], dim=-1))
        state = torch.cat([hgn.transpose(0, 1).reshape(B, LH), Nn, Zn], dim=-1).unsqueeze(0)
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
