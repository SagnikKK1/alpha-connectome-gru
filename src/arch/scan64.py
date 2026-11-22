"""Exact scans for time-VARYING diagonal decays.

    h_t = a_t * h_{t-1} + v_t,     a_t = exp(la_t),  la_t in [-LA_MAX, 0]

Why not scan.py: the fp32 log-space (Heinsen) scan loses ~2e-4 at saturated gates and the
(L,L,B,D) matrix form is 10x slower on a shared GPU. Both scans here use the exponentially
weighted cumsum: within a chunk of L rows, with S_t = -sum_{j<=t} la_j >= 0 (the accumulated
"age" since the chunk start),

    h_t = exp(-S_t) * ( h_start + sum_{k<=t} exp(S_k) v_k )

The cumsum is dominated by its latest term, so there is no cancellation problem; the only
constraint is that exp(S) must not overflow, i.e. L * LA_MAX must stay below the dtype's
exponent range. Callers MUST clamp la at -LA_MAX (a = exp(-4.5) = 0.011 is a dead state
anyway) and apply the same clamp in the row-at-a-time path, so the stepwise fp32 recurrence
and the scan compute the same function (measured: ~3e-7 relative to max|h| over 768 rows,
i.e. fp32 rounding).

scan_two_level  (default) all T/L chunks in parallel in fp32 (L=16: S <= 72, e^72 = 2e31),
                one tiny fp64 loop over the T/L chunk carries, then one broadcast. ~25 kernel
                launches per forward regardless of T; 91 ms fwd+bwd at (768,32,512) on the
                shared L4 vs 158 ms for the sequential fp64 version.
scan_exact64    sequential fp64 chunks (L=64: S <= 288), the reference.
"""
from __future__ import annotations
import torch

LA_MAX = 4.5
CHUNK = 16


def scan_two_level(la: torch.Tensor, v: torch.Tensor, h0: torch.Tensor,
                   L: int = CHUNK) -> tuple[torch.Tensor, torch.Tensor]:
    """la: (T,B,D) log-decays in [-LA_MAX, 0]; v: (T,B,D); h0: (B,D). Returns (H (T,B,D), h_T)."""
    T, B, D = v.shape
    pad = (-T) % L
    if pad:
        la = torch.cat([la, torch.zeros(pad, B, D, dtype=la.dtype, device=la.device)], 0)
        v = torch.cat([v, torch.zeros(pad, B, D, dtype=v.dtype, device=v.device)], 0)
    nC = (T + pad) // L
    lac = la.reshape(nC, L, B, D); vc = v.reshape(nC, L, B, D)
    S = torch.cumsum(-lac, 1)                        # (nC,L,B,D) in [0, L*LA_MAX]
    E = torch.exp(S)
    Hin = torch.cumsum(E * vc, 1) / E                # each chunk with zero carry-in
    A = torch.exp(-S[:, -1].double())                # chunk decay  (nC,B,D)
    b = Hin[:, -1].double()                          # chunk-end value with zero carry-in
    h = h0.double(); cin = [h]
    for c in range(nC - 1):                          # carries: nC-1 tiny fp64 steps
        h = A[c] * h + b[c]; cin.append(h)
    Cin = torch.stack(cin).to(v.dtype)               # (nC,B,D) carry into each chunk
    H = (Hin + Cin[:, None] / E).reshape(-1, B, D)[:T]
    return H, H[-1]


def scan_exact64(la: torch.Tensor, v: torch.Tensor, h0: torch.Tensor,
                 L: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential fp64 chunks; same contract as scan_two_level. Reference implementation."""
    T = v.shape[0]
    dt = torch.float64
    outs, h = [], h0.to(dt)
    for s in range(0, T, L):
        S = torch.cumsum(-la[s:s + L].to(dt), 0)
        E = torch.exp(S)
        c = torch.cumsum(E * v[s:s + L].to(dt), 0)
        Hc = (c + h[None]) / E
        outs.append(Hc); h = Hc[-1]
    H = torch.cat(outs, 0).to(v.dtype)
    return H, H[-1]


if __name__ == "__main__":
    import math, sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from arch.scan import naive_scan
    torch.manual_seed(0)
    T, B, D = 768, 4, 64
    hl = torch.logspace(math.log10(2), math.log10(2048), D)
    g = torch.exp(torch.randn(T, B, 1) - 0.4).clamp(max=20)
    la = (-(math.log(2) / hl).view(1, 1, -1) * g).clamp(min=-LA_MAX)
    v = torch.randn(T, B, D); h0 = torch.randn(B, D)
    ref64 = naive_scan(torch.exp(la.double()), v.double(), h0.double())[0]
    ref32 = naive_scan(torch.exp(la), v, h0)[0]
    for name, fn in (("scan_two_level", scan_two_level), ("scan_exact64", scan_exact64)):
        H, hT = fn(la, v, h0)
        print(f"{name:15s} vs float64 loop: {(H.double() - ref64).abs().max():.2e}   h_T {(hT.double() - ref64[-1]).abs().max():.2e}"
              f"   (fp32 loop vs float64 loop: {(ref32.double() - ref64).abs().max():.2e}, max|h| {ref64.abs().max():.1f})")
    H, _ = scan_two_level(la[:700], v[:700], h0)      # T not a multiple of L
    print(f"padding path: {(H.double() - ref64[:700]).abs().max():.2e}")
    la_ = la[:64, :2, :8].double().requires_grad_(True); v_ = v[:64, :2, :8].double().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: scan_two_level(a, b, h0[:2, :8].double(), L=16)[0], (la_, v_))
    print("gradcheck ok")
