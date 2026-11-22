"""GRU + STREAMING softmax attention pooling, O(1) work per row.

I had wrongly excluded attention pooling as O(W)-per-row. That is only true when a
changing query rescores history. If the score depends on the hidden state alone,
s_t = f(h_t), softmax pooling has an exact recursive form:

    N_t = lam * N_{t-1} + exp(s_t) * h_t
    Z_t = lam * Z_{t-1} + exp(s_t)
    c_t = N_t / Z_t

lam = 1 is exact cumulative softmax pooling over all history; lam < 1 decays it.
Constant work, fixed-size state, so it fits the row-at-a-time budget.

Two choices specific to this task:
* `s_t` is bounded as `alpha * tanh(.)`, so exp(s_t) stays in [e^-alpha, e^alpha].
  No running-max renormalisation is needed, which keeps the exported graph free of
  control flow.
* Several decay rates run in parallel. At lam = 1 a 20,000-row sequence is dominated
  by its own distant past, while the targets here decorrelate over ~500 rows.

Training uses the shared linear scan (arch.scan); inference steps one row.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arch.scan import chunked_scan

N_FEATURES = 112


class AttnPoolGRUCore(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, n_out: int = 1,
                 halflives=(64, 512, 4096), alpha: float = 4.0,
                 n_features: int = N_FEATURES, cumulative: bool = False):
        super().__init__()
        self.hidden, self.layers, self.n_out, self.alpha = hidden, layers, n_out, alpha
        hl = torch.tensor([float(h) for h in halflives])
        lam = torch.ones_like(hl) * (1.0 - 1e-7) if cumulative else 0.5 ** (1.0 / hl)
        self.register_buffer("lam", lam)
        self.K = len(lam)
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.gru = nn.GRU(n_features, hidden, layers, batch_first=False)
        self.score = nn.Linear(hidden, 1)
        self.head = nn.Linear(hidden * (1 + self.K), n_out)
        self.state_dim = layers * hidden + self.K * hidden + self.K
        nn.init.zeros_(self.head.bias); nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.score.bias); nn.init.normal_(self.score.weight, std=0.02)

    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def init_state(self, batch: int, device=None):
        device = device or next(self.parameters()).device
        return torch.zeros(1, batch, self.state_dim, device=device)

    def _split(self, st):
        B = st.shape[1]; LH = self.layers * self.hidden; KH = self.K * self.hidden
        h = st[0, :, :LH].reshape(B, self.layers, self.hidden).permute(1, 0, 2).contiguous()
        return h, st[0, :, LH:LH + KH], st[0, :, LH + KH:]      # h, N(B,K*H), Z(B,K)

    def forward(self, x, st=None):
        T, B, _ = x.shape
        if st is None:
            st = self.init_state(B, x.device)
        h0, N0, Z0 = self._split(st)
        ys, hn = self.gru((x - self.mu) / self.sd, h0)          # (T,B,H)
        e = torch.exp(self.alpha * torch.tanh(self.score(ys)))  # (T,B,1) bounded
        H, K = self.hidden, self.K
        vN = (e.unsqueeze(2) * ys.unsqueeze(2)).expand(T, B, K, H).reshape(T, B, K * H)
        aN = self.lam.repeat_interleave(H).view(1, 1, K * H)
        Nt, Nl = chunked_scan(aN, vN, N0)                       # (T,B,K*H)
        vZ = e.expand(T, B, K)
        Zt, Zl = chunked_scan(self.lam.view(1, 1, K), vZ, Z0)   # (T,B,K)
        c = Nt.view(T, B, K, H) / Zt.unsqueeze(-1).clamp_min(1e-6)
        out = self.head(torch.cat([ys, c.reshape(T, B, K * H)], -1))
        nst = torch.cat([hn.permute(1, 0, 2).reshape(B, -1), Nl, Zl], -1).unsqueeze(0)
        return out, nst

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        return 17.2 + 19.9 * (self.n_params() / 1e6)
