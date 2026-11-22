"""Linear scans  h_t = a_t * h_{t-1} + v_t  shared by the diagonal-SSM cores.

Training needs the whole (T, B, D) trajectory at once; a Python loop over T=768 steps costs
~7 s per training step on CPU (autograd overhead), so two parallel forms are used:

* time-invariant decay (DiagSSMCore, the LRU / S4D case): chunks of L rows, sequential
  across chunks; inside a chunk the recurrence is a lower-triangular matrix
      M[t, k] = a^(t-k),   t >= k,
  shared over the batch, so one chunk is a single einsum. L=64 measured fastest.
* input-dependent decay (MinGRUCore, the selective case): Heinsen's log-space parallel
  scan. With S_t = sum_{j<=t} log a_j,  h_t = exp(S_t) * sum_{k<=t} exp(-S_k) v_k; the
  sum is split into positive and negative parts and each is a logcumsumexp, so nothing
  overflows. Chunks of L=128 rows keep the log-domain values small enough for float32
  (set HEINSEN_DTYPE = torch.float64 for a reference-precision run).
"""
from __future__ import annotations
import torch

_A_MIN = 1e-7
_NEG = -1e30
HEINSEN_DTYPE = torch.float32


def chunked_scan(a: torch.Tensor, v: torch.Tensor, h0: torch.Tensor,
                 L: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """a: (1,1,D) time-invariant or (T,B,D) time-varying decays in (0,1); v: (T,B,D);
    h0: (B,D). Returns (H (T,B,D), h_T (B,D))."""
    tv = a.dim() == 3 and a.shape[0] == 1 and a.shape[1] == 1
    a = a.clamp(_A_MIN, 1.0 - 1e-7)
    return _scan_tiv(a[0, 0], v, h0, L or 64) if tv else _scan_heinsen(a, v, h0, L or 128)


def _scan_tiv(a, v, h0, L):
    T = v.shape[0]
    L = min(L, T)
    la = torch.log(a)                                                   # (D,)
    t = torch.arange(L, device=v.device, dtype=v.dtype)
    diff = (t[:, None] - t[None, :])[:, :, None] * la[None, None, :]    # (L,L,D): (t-k) log a
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=v.device))
    M = torch.exp(diff.masked_fill(~mask[:, :, None], -1e4))           # built once, shared by all chunks
    P = torch.exp((t[:, None] + 1.0) * la[None, :])                     # (L,D): a^(t+1)
    outs, h = [], h0
    for s in range(0, T, L):
        vc = v[s:s + L]; Lc = vc.shape[0]
        Hc = torch.einsum("tkd,kbd->tbd", M[:Lc, :Lc], vc) + P[:Lc, None, :] * h[None]
        outs.append(Hc); h = Hc[-1]
    return torch.cat(outs, 0), h


def _scan_heinsen(a, v, h0, L):
    T, B, D = v.shape
    dt = HEINSEN_DTYPE
    outs, h = [], h0
    for s in range(0, T, L):
        ac, vc = a[s:s + L].to(dt), v[s:s + L].to(dt)
        la = torch.cat([torch.zeros(1, B, D, dtype=dt, device=v.device), torch.log(ac)], 0)
        S = torch.cumsum(la, 0)                                         # S_{-1}=0, then S_t
        vv = torch.cat([h.to(dt).unsqueeze(0), vc], 0)                  # h0 enters as step -1
        lv = torch.log(vv.abs().clamp_min(1e-30))
        lp = torch.where(vv > 0, lv, torch.full_like(lv, _NEG)) - S
        ln = torch.where(vv < 0, lv, torch.full_like(lv, _NEG)) - S
        Hc = (torch.exp(S + torch.logcumsumexp(lp, 0)) - torch.exp(S + torch.logcumsumexp(ln, 0)))[1:]
        Hc = Hc.to(v.dtype)
        outs.append(Hc); h = Hc[-1]
    return torch.cat(outs, 0), h


def naive_scan(a, v, h0):
    """Reference loop, for tests only."""
    T = v.shape[0]
    a = a.expand(T, *v.shape[1:]) if a.shape[0] == 1 else a
    h, outs = h0, []
    for t in range(T):
        h = a[t] * h + v[t]; outs.append(h)
    return torch.stack(outs), h
