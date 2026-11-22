"""sLSTM (xLSTM, Beck et al. 2024) as a drop-in recurrent core.

WHY THIS ONE. The target is P&L in ticks - an INTEGRATED quantity - and this repo has repeatedly
found that a tanh GRU cannot hold an unbounded running sum (src/arch/integrator.py exists for
exactly that reason). sLSTM's distinguishing feature is a NORMALISER state n_t running alongside
the cell state c_t, with the output taken as c_t / n_t. That lets the cell accumulate without
saturating, which is the mechanism a GRU lacks. Reported best-in-class on MASE/CRPS for
time-series at 1M-40M parameters, which is our range.

NOT a rerun of the SSM probe. arch/ssm.py is a POSITIVE-diagonal linear recurrence and scored
0.6242 against a matched GRU's 0.634. sLSTM is gated and nonlinear with exponential input gating;
the only shared property is "not a GRU".

    c_t = f_t * c_{t-1} + i_t * z_t          cell
    n_t = f_t * n_{t-1} + i_t                normaliser   <- the point
    h_t = o_t * (c_t / max(n_t, 1))          normalised output

Exponential input gating overflows without care, so the standard stabiliser m_t is carried:
    m_t  = max(log f_t + m_{t-1}, log i_t)
    i'_t = exp(log i_t - m_t)
    f'_t = exp(log f_t + m_{t-1} - m_t)
which is algebraically identical and bounded. STATE is (h, c, n, m) = 4 slots of width H, so it
packs into the single (S,B,D) tensor the core interface requires.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn
for _p in (str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])):
    if _p not in sys.path: sys.path.insert(0, _p)
from utils import N_FEATURES


def _slstm_scan(gi, Rw, ht, ct, nt, mt):
    """One layer's recurrence over a chunk. Pure-tensor so torch.compile can fuse the ~12
    elementwise ops per timestep into a couple of kernels. Eager costs 3635 ms/step at T=1000,
    B=64 (14.7 h per run) because it launches ~12k tiny kernels; compiled it is 313 ms (1.3 h),
    an 11.6x speedup, which is what makes this architecture affordable at all."""
    T = gi.shape[0]
    outs = []
    for t in range(T):
        g = gi[t] + torch.nn.functional.linear(ht, Rw)
        zt, fg, ig, og = g.chunk(4, -1)
        logf = torch.nn.functional.logsigmoid(fg)
        mn = torch.maximum(logf + mt, ig)
        i_ = torch.exp(ig - mn)
        f_ = torch.exp(logf + mt - mn)
        ct = f_ * ct + i_ * torch.tanh(zt)
        nt = f_ * nt + i_
        mt = mn
        ht = torch.sigmoid(og) * (ct / nt.clamp_min(1.0))
        outs.append(ht)
    return torch.stack(outs, 0), ht, ct, nt, mt


_BLOCK = 250          # timesteps per compiled graph; T must be driven in these
_SCAN = None
def _get_scan():
    global _SCAN
    if _SCAN is None:
        try:
            _SCAN = torch.compile(_slstm_scan, dynamic=False)
        except Exception:
            _SCAN = _slstm_scan          # eager fallback: correct, just slow
    return _SCAN


class SLSTMCore(nn.Module):
    def __init__(self, hidden=256, layers=1, proj=192, n_out=1, n_features=N_FEATURES,
                 clamp=8.0, dropout=0.0, head_hidden=192, proj_layers=1):
        super().__init__()
        self.hidden, self.layers, self.clamp = hidden, layers, clamp
        self.proj = proj
        blocks = [nn.Linear(n_features, proj), nn.GELU(), nn.LayerNorm(proj)]
        for _ in range(proj_layers - 1):
            blocks += [nn.Linear(proj, proj), nn.GELU(), nn.LayerNorm(proj)]
        self.in_proj = nn.Sequential(*blocks)
        # one fused input matmul and one fused recurrent matmul per layer, 4 gates each
        self.W = nn.ModuleList([nn.Linear(proj if l == 0 else hidden, 4*hidden) for l in range(layers)])
        self.R = nn.ModuleList([nn.Linear(hidden, 4*hidden, bias=False) for l in range(layers)])
        self.post_ln = nn.LayerNorm(hidden + proj)
        self.head = nn.Sequential(nn.Linear(hidden + proj, head_hidden), nn.GELU(),
                                  nn.Linear(head_hidden, n_out))
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        for l in range(layers):                      # forget-gate bias high: remember by default
            with torch.no_grad():
                self.W[l].bias[hidden:2*hidden].fill_(3.0)

    def set_norm(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=self.mu.dtype))
        self.sd.copy_(torch.as_tensor(sd, dtype=self.sd.dtype))

    def init_state(self, batch: int, device=None):
        dev = device or next(self.parameters()).device
        # 4 slots per layer: h, c, n, m.  n starts at 1 and m at 0 so the first step is well defined.
        s = torch.zeros(4*self.layers, batch, self.hidden, device=dev)
        for l in range(self.layers):
            s[4*l + 2] = 1.0
        return s

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        return 9999.0

    def forward(self, x, h=None):
        if h is None: h = self.init_state(x.shape[1], x.device)
        T, B, _ = x.shape
        z0 = ((x - self.mu) / self.sd).clamp(-self.clamp, self.clamp)
        z0 = self.in_proj(z0)
        inp = z0
        new = []
        for l in range(self.layers):
            ht, ct, nt, mt = (h[4*l+k] for k in range(4))
            gi = self.W[l](inp)                       # (T,B,4H) precomputed for the whole chunk
            # Drive the compiled scan in FIXED-SIZE blocks. The scan is a Python loop, so
            # torch.compile unrolls it: feeding it T directly means the graph size follows the
            # input length. Training uses T=1000, but evaluate() feeds whole 20,000-row
            # sequences, which recompiled a 20,000-step unrolled graph -- 64 GB of RSS and
            # still climbing when it was killed. A fixed block keeps ONE graph for every
            # caller: training is 4 blocks, evaluation 80.
            scan = _get_scan()
            outs = []
            for s0 in range(0, T, _BLOCK):
                o, ht, ct, nt, mt = scan(gi[s0:s0+_BLOCK], self.R[l].weight, ht, ct, nt, mt)
                outs.append(o)
            inp = outs[0] if len(outs) == 1 else torch.cat(outs, 0)
            new += [ht, ct, nt, mt]
        out = self.head(self.post_ln(torch.cat([inp, z0], -1)))
        return out, torch.stack(new, 0).detach()
