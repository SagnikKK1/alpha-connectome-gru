"""GRU over [x, EMA_hl1(x), ..., EMA_hlK(x)]: explicit fixed multi-timescale memory as input.

Why: docs/research/features.md measured a fixed EMA bank at +0.036 WP on a LINEAR model; a GRU
must otherwise discover 500-row timescales by saturating its update gate. Here the timescales are
handed to it: each EMA channel is a fixed diagonal recurrence (Mul+Add) on the STANDARDISED row,
so it exports as a handful of elementwise nodes in front of the GRU.

State: ONE flat tensor (1, B, layers*hidden + K*F). GRU hidden first, then the K*F EMA values.
Convention for the EMA start (matches src/features.py::Ema): the state *before* row 0 is x_0
itself, so EMA(row 0) == x_0 (no shrink toward zero). `init_state(batch, x0=...)` builds that; the
training sampler (data_io.sample_windows) supplies the exact state at a mid-sequence window start.

forward(x, h): x (T, B, F) raw features. T == 1 is the row-at-a-time path (what ONNX traces);
T > 1 uses the batch-shared chunked scan from src/arch/scan.py (no gradient flows through it:
alpha is fixed and x is data). verify_ema.py checks the two paths agree to ~1e-6.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from arch.scan import chunked_scan

N_FEATURES = 112


def alphas_for(hls) -> torch.Tensor:
    return torch.tensor([1.0 - 0.5 ** (1.0 / float(h)) for h in hls], dtype=torch.float32)


class EmaGRUCore(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2, hls=(8, 64, 512), n_out: int = 1,
                 n_features: int = N_FEATURES, diff: bool = False, match_init: bool = True, feat: str | None = None):
        super().__init__()
        self.hidden, self.layers, self.n_out, self.F = hidden, layers, n_out, n_features
        self.hls = tuple(float(h) for h in hls); self.K = len(self.hls)
        # feat: 'ema'  -> [x, e1..eK]            (levels; collinear with x for sticky columns)
        #       'diff' -> [x, x-e1..x-eK]        (innovations relative to each timescale)
        #       'band' -> [x, x-e1, e1-e2, ..]   (multi-resolution bands; decorrelated)
        self.feat = feat or ("diff" if diff else "ema"); self.diff = self.feat == "diff"
        assert self.feat in ("ema", "diff", "band")
        self.n_in = n_features * (1 + self.K)
        self.state_dim = layers * hidden + self.K * n_features
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.register_buffer("alpha", alphas_for(self.hls).view(self.K, 1))          # (K,1)
        self.gru = nn.GRU(self.n_in, hidden, layers)
        self.head = nn.Linear(hidden, n_out)
        nn.init.zeros_(self.head.bias); nn.init.normal_(self.head.weight, std=0.02)
        if match_init and self.K > 0:
            # nn.GRU inits W_ih ~ U(-1/sqrt(H), 1/sqrt(H)) regardless of fan-in: a (1+K)x wider input
            # would start (1+K)x higher pre-activation variance than the 112-input control. Rescale so
            # the two arms start at the same operating point (an init choice, not a capacity change).
            with torch.no_grad():
                self.gru.weight_ih_l0.mul_(1.0 / math.sqrt(1.0 + self.K))

    # --- GRUCore interface -------------------------------------------------------------------
    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        return 17.2 + 19.9 * self.n_params() / 1e6 + 0.5 * 7      # int8 law + ~7 elementwise nodes

    def normalise(self, x):
        return (x - self.mu) / self.sd

    def init_state(self, batch: int, device=None, x0: torch.Tensor | None = None, ema0=None):
        """Zeros for the GRU part. EMA part: `ema0` (B,K,F) in standardised units if given, else
        x0 (B,F) RAW row 0 -> standardised and tiled (the V[0] convention), else zeros."""
        device = device or self.mu.device
        h = torch.zeros(1, batch, self.state_dim, device=device)
        if self.K == 0:
            return h
        LH = self.layers * self.hidden
        if ema0 is not None:
            h[0, :, LH:] = torch.as_tensor(ema0, dtype=torch.float32, device=device).reshape(batch, -1)
        elif x0 is not None:
            xn0 = self.normalise(torch.as_tensor(x0, dtype=torch.float32, device=device))
            h[0, :, LH:] = xn0.repeat(1, self.K)
        return h

    def features(self, xn, E):
        """xn (T,B,F), E (T,B,K,F) -> the K derived channels (T,B,K,F) for the chosen feat mode."""
        if self.feat == "ema":
            return E
        if self.feat == "diff":
            return xn[:, :, None, :] - E
        prev = torch.cat([xn[:, :, None, :], E[:, :, :-1, :]], dim=2)      # x, e1, ..., e_{K-1}
        return prev - E                                                     # x-e1, e1-e2, ..., e_{K-1}-e_K

    def split_state(self, h):
        B = h.shape[1]; LH = self.layers * self.hidden
        hg = h[0, :, :LH].reshape(B, self.layers, self.hidden).permute(1, 0, 2).contiguous()
        e = h[0, :, LH:].reshape(B, self.K, self.F)
        return hg, e

    # --- forward -------------------------------------------------------------------------------
    def forward(self, x, h=None):
        T, B, _ = x.shape
        xn = self.normalise(x)
        if h is None:
            h = self.init_state(B, x.device, x0=x[0])
        hg, e0 = self.split_state(h)
        if self.K > 0:
            with torch.no_grad():
                if T == 1:
                    e1 = e0 + self.alpha * (xn[0][:, None, :] - e0)                # (B,K,F)
                    E = e1.unsqueeze(0); eT = e1
                else:
                    a = (1.0 - self.alpha).expand(self.K, self.F).reshape(1, 1, -1)   # (1,1,K*F)
                    v = (self.alpha * xn[:, :, None, :]).reshape(T, B, self.K * self.F)
                    E, eT = chunked_scan(a, v, e0.reshape(B, self.K * self.F))
                    E = E.reshape(T, B, self.K, self.F); eT = eT.reshape(B, self.K, self.F)
                feats = self.features(xn, E)
            inp = torch.cat([xn, feats.reshape(T, B, self.K * self.F)], dim=-1)
        else:
            inp = xn; eT = e0
        y, hgn = self.gru(inp, hg)
        out = self.head(y)
        hn = torch.cat([hgn.permute(1, 0, 2).reshape(1, B, -1), eT.reshape(1, B, -1)], dim=-1)
        return out, hn


def build_boost(preset: str, **kw):
    """'gru' -> the unchanged src/model.GRUCore control; 'ema' -> EmaGRUCore."""
    if preset == "ema":
        return EmaGRUCore(**kw)
    from model import GRUCore
    for k in ("hls", "diff", "match_init", "feat"): kw.pop(k, None)
    return GRUCore(**kw)


if __name__ == "__main__":
    torch.manual_seed(0); torch.set_num_threads(1)
    for feat in ("ema", "diff", "band"):
        core = EmaGRUCore(hidden=32, layers=2, hls=(8, 64, 512), feat=feat).eval()
        core.set_norm(torch.randn(112), torch.rand(112) + 0.5)
        x = torch.randn(300, 3, 112) * 2 + 1
        with torch.no_grad():
            o_scan, h_scan = core(x); h = core.init_state(3, x0=x[0]); outs = []
            for t in range(300):
                o, h = core(x[t:t + 1], h); outs.append(o)
        print(f"feat={feat:4s} scan vs stepwise max|out| diff {(o_scan - torch.cat(outs)).abs().max().item():.2e}")
    core = EmaGRUCore(hidden=32, layers=2, hls=(8, 64, 512)).eval()
    core.set_norm(torch.randn(112), torch.rand(112) + 0.5)
    T, B = 300, 3
    x = torch.randn(T, B, 112) * 2 + 1
    with torch.no_grad():
        out_scan, h_scan = core(x)                      # h=None -> V[0]-init from x[0]
        h = core.init_state(B, x0=x[0]); outs = []
        for t in range(T):
            o, h = core(x[t:t + 1], h); outs.append(o)
        out_step = torch.cat(outs)
    print("scan vs stepwise: max|out| diff %.2e  max|state| diff %.2e" %
          ((out_scan - out_step).abs().max().item(), (h_scan - h).abs().max().item()))
    # EMA at row 0 equals x_0 (standardised)
    _, e = core.split_state(core(x[:1])[1])
    print("EMA(row0) == xn0:", torch.allclose(e, core.normalise(x[0])[:, None, :].expand(B, 3, 112), atol=1e-6))
    print("params", core.n_params(), "state_dim", core.state_dim)
