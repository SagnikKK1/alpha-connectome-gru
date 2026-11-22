"""Models for the Alpha Connectome task.

Parameterisation follows F1: t0 and t1 correlate at -0.91 and are never both
positive, so the network predicts ONE scalar s and the solution emits (s, -s).
That halves the head and puts every gradient on the same latent signal. A
two-head variant is kept for comparison.

Everything is written so the same weights serve two call patterns:
  * forward(x, h)      whole window, (T, B, F) - what training uses
  * the identical nn.GRU, exported at T=1 for the row-by-row scorer
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

N_FEATURES = 112


class GRUCore(nn.Module):
    """Stacked GRU with an optional fixed input standardisation.

    `mu`/`sd` are registered buffers, not parameters: they travel into the ONNX
    graph so the submission needs no separate normalisation step.
    """

    def __init__(self, hidden: int = 128, layers: int = 2, n_out: int = 1,
                 n_features: int = N_FEATURES, dropout: float = 0.0, proj: int = 0):
        super().__init__()
        self.hidden, self.layers, self.n_out = hidden, layers, n_out
        self.common_head = False      # set by train.py for the (s, c) parameterisation
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.gru = nn.GRU(n_features, hidden, layers, batch_first=False,
                          dropout=dropout if layers > 1 else 0.0)
        # Optional bottleneck between the recurrence and the readout. It saves head
        # parameters and, more importantly at batch 1, shrinks the final GEMM -- the
        # competitor's latency table quotes every model as "+ proj 64". It is NOT a
        # recurrent projection: the GRU state stays `hidden` wide, so the recurrence is
        # unchanged and the export path needs no new state.
        self.proj = nn.Linear(hidden, proj) if proj else None
        self.head = nn.Linear(proj if proj else hidden, n_out)
        # Start near zero so early training does not fight a large random output.
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)

    def set_norm(self, mu, sd):
        with torch.no_grad():
            self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
            self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32).clamp_min(1e-6))

    def forward(self, x, h=None):
        """x: (T, B, F) -> out (T, B, n_out), h (layers, B, hidden)."""
        x = (x - self.mu) / self.sd
        y, h = self.gru(x, h)
        if self.proj is not None:
            y = torch.tanh(self.proj(y))
        return self.head(y), h

    def init_state(self, batch: int, device=None):
        return torch.zeros(self.layers, batch, self.hidden,
                           device=device or next(self.parameters()).device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        """Measured cost law: us/row = 16.8 + 18.0 x Mparams, single-threaded ORT int8.

        The previous constants (12.0 + 97.8*M + 5) were superseded and over-predicted by
        3.5-4x: 159.9 us/row for the 1.46M fusion that MEASURES 46.1, and 265.2 for the
        2.54M quad that measures 65.6. train.py prints "% of budget" from this, so every
        sizing decision taken against the old law was made against a 4x pessimistic
        number. At the 90 us/row target the old law allows 0.75M parameters; the measured
        law allows 4.07M. The largest single model ever trained here is 679k.
        """
        return 16.8 + 18.0 * (self.n_params() / 1e6)


class ExportWrapper(nn.Module):
    """Row-at-a-time graph for ONNX: (1,1,112) + state -> (1,1,2) + state.

    Emits the (s, -s) pair directly so solution.py stays trivial, and folds the
    output scale in here rather than leaving it to inference code.
    """

    def __init__(self, core: GRUCore, out_scale: float = 1.0):
        super().__init__()
        self.core = core
        self.register_buffer("scale", torch.tensor(float(out_scale)))

    def forward(self, features, hidden):
        out, h = self.core(features, hidden)          # (1,1,n_out)
        s = out[..., :1] * self.scale
        if self.core.n_out == 1:
            pred = torch.cat([s, -s], dim=-1)
        elif getattr(self.core, "common_head", False):
            c = out[..., 1:2] * self.scale            # shared component, not a 2nd target
            pred = torch.cat([s + c, -s + c], dim=-1)
        else:
            pred = torch.cat([s, out[..., 1:2] * self.scale], dim=-1)
        return pred, h


class LSTMCore(GRUCore):
    """Same interface as GRUCore, LSTM cell. State is (h, c) flattened to one tensor
    so the export path and the row-at-a-time wrapper stay unchanged."""

    def __init__(self, hidden=128, layers=2, n_out=1, n_features=112):
        super().__init__(hidden=hidden, layers=layers, n_out=n_out, n_features=n_features)
        import torch.nn as nn
        self.gru = nn.LSTM(n_features, hidden, layers)

    def forward(self, x, h=None):
        x = (x - self.mu) / self.sd
        y, h = self.gru(x, h)
        return self.head(y), h

    def init_state(self, batch, device=None):
        d = device or next(self.parameters()).device
        z = torch.zeros(self.layers, batch, self.hidden, device=d)
        return (z, z.clone())


class CfCCore(GRUCore):
    """GRU whose forget path decays with REAL elapsed time, not row count.

    Rows are wildly irregular in time: dt spans 5 orders of magnitude (p1 1e-5,
    p50 5.8e-3, p99 1.01). a5 is a log-age that resets on events, so
    dt = exp(a5_t) - exp(a5_{t-1}) recovers genuine elapsed time.

        d_t = exp(-softplus(tau) * dt_t)                  per-unit decay
        h_t = (1-z)*d_t (*) h_{t-1} + (1 - (1-z)*d_t) (*) n_t

    dt -> 0 reduces exactly to a standard GRU; dt large forgets the state. This is
    the closed-form continuous-time form: no ODE solver, one pass per row.
    Unlike the EMA banks and the diagonal SSM (both -0.002 / -0.015), the decay is
    per unit TIME, which is a different model on irregularly sampled rows.
    """
    A5 = 109                                   # a5 column index

    def __init__(self, hidden=128, layers=2, n_out=1, n_features=112):
        super().__init__(hidden=hidden, layers=layers, n_out=n_out, n_features=n_features)
        import torch.nn as nn
        self.cells = nn.ModuleList([nn.GRUCell(n_features if l == 0 else hidden, hidden)
                                    for l in range(layers)])
        self.tau = nn.Parameter(torch.zeros(layers, hidden))
        del self.gru

    @staticmethod
    def dt_from(x_raw):
        """x_raw: (T,B,F) unnormalised. Returns (T,B,1) elapsed time, >= 0."""
        age = torch.exp(x_raw[..., CfCCore.A5].clamp(max=3.0))
        d = age[1:] - age[:-1]
        d = torch.where(d < 0, age[1:], d)          # reset: the new age IS the elapsed time
        return torch.cat([torch.zeros_like(age[:1]), d], 0).clamp(min=0.0).unsqueeze(-1)

    def forward(self, x, h=None):
        # nn.GRUCell is ONE fused CUDA kernel; hand-writing the gate math and
        # torch.jit.script-ing it was 11x SLOWER (693 -> 7970 ms/step) because it
        # unfuses that kernel into ~12 launches. Keep the fused cell.
        import torch.nn.functional as F
        dt = self.dt_from(x)
        xn = (x - self.mu) / self.sd
        T, B, _ = xn.shape
        if h is None:
            h = self.init_state(B, xn.device)
        hs = [h[l] for l in range(self.layers)]
        rates = [F.softplus(self.tau[l]).unsqueeze(0) for l in range(self.layers)]
        outs = []
        for t in range(T):
            inp = xn[t]
            dtt = dt[t]
            for l, cell in enumerate(self.cells):
                prev = hs[l]
                cand = cell(inp, prev)
                d = torch.exp(-rates[l] * dtt)
                hs[l] = d * prev + (1.0 - d) * cand
                inp = hs[l]
            outs.append(inp)
        return self.head(torch.stack(outs)), torch.stack(hs)


def build(name: str = "gru", **kw) -> GRUCore:
    # These cores had NO routing here, so they were unreachable from train.py -- which
    # is why arch/mingru.py has never been trained despite being implemented and
    # ONNX-verified. They take `d` (state width) and `mlp`, not `hidden`/`layers`.
    _dcore = {"mingru": ("arch.mingru", "MinGRUCore"),
              "diagssm": ("arch.ssm", "DiagSSMCore"),
              "clockssm": ("arch.clockssm", "ClockSSMCore")}
    for _pfx, (_mod, _cls) in _dcore.items():
        if name.startswith(_pfx):
            import importlib
            C = getattr(importlib.import_module(_mod), _cls)
            # A NAMED PRESET must win over these defaults. This block used to ignore
            # arch.PRESETS entirely, so `mingru384` and `mingru_m` both silently built the
            # class default d=256/mlp=96 (93,474 params) instead of the preset's shape --
            # an architecture probe against a 428k refgru would have been measuring a model
            # 4.5x smaller and the null would have been meaningless. Caught 16 Sep.
            kw2 = {}
            try:
                from arch import PRESETS as _AP
                if name in _AP and _AP[name][0] is C:
                    kw2.update(_AP[name][1])
            except Exception:
                pass
            kw2.update({k: v for k, v in kw.items() if k in ("d", "mlp", "n_out", "n_features")})
            if "hidden" in kw and "d" not in kw2: kw2["d"] = kw["hidden"]
            return C(**kw2)
    if name.startswith("grux"):
        from arch.grux import GRUXCore
        return GRUXCore(**{k: v for k, v in kw.items()
                           if k in ("hidden", "layers", "mlp", "skip", "n_out", "n_features")})
    if name.startswith("slstm"):
        from arch.xlstm import SLSTMCore
        kw2={}
        try:
            from arch import PRESETS as _AP
            if name in _AP and _AP[name][0] is SLSTMCore: kw2.update(_AP[name][1])
        except Exception:
            pass
        kw2.update({k:v for k,v in kw.items()
                    if k in ("hidden","layers","proj","proj_layers","n_out","n_features",
                             "dropout","clamp","head_hidden")})
        return SLSTMCore(**kw2)
    if name.startswith("refgru"):
        from arch.refgru import RefGRUCore
        _ALLOWED = ("quad", "bookconv", "ofi", "adanorm", "proj_layers", "proj_gated", "fastw", "fw_groups", "fw_hl_min", "fw_hl_max", "fw_to_gru", "bidir", "dualmode", "hint", "aux_next", "mask_aux", "hetero", "resnorm", "resnorm_sd0", "in_drop", "out_drop", "grid", "hidden", "layers", "n_out", "n_features", "proj", "head_hidden",
                    "dropout", "clamp", "cell", "diff", "ewma", "posfeat", "split", "nlfeat", "learn_h0",
                    "adanorm_cols",                    # add_z48.py: --adanorm-cols (research Z48)
                    "hlg_bins", "mh_heads",            # add_z59.py: --hlg / --mh (research Z59)
                    "bookconv_taps", "bookconv_unshare", "tradeconv", "instshare",   # add_s1.py (research S1): linear input re-reads
                    "inproj_init_mult", "densein",     # add_s1b.py (research S1 controls)
                    "xinst", "bookconv_act", "hh_rank",   # add_struct.py (research ST): structures
                    "ple_bins", "ple_ch",                 # add_h4_ple.py (research H4): piecewise-linear embedding of the price codes
                    "bookconv_split", "bookconv_side")    # add_h7_bcsplit.py (research H7): restricted bookconv kernels
        # A kwarg dropped silently here produces a BASELINE run wearing the tag of a feature
        # experiment, which then gets written up as "that feature is null". --nlfeat was
        # dropped exactly this way on 16 Sep and was caught only by a build-path smoke test.
        _bad = [k for k, v in kw.items()
                if k not in _ALLOWED and v not in (None, "", (), False, 0)]
        if _bad:
            raise ValueError("refgru: unsupported non-default kwargs %s -- they would be "
                             "silently dropped; add them to _ALLOWED or stop passing them" % _bad)
        return RefGRUCore(**{k: v for k, v in kw.items() if k in _ALLOWED})
    if name.startswith("intgru"):
        from arch.integrator import IntGRUCore
        kw2 = {k: v for k, v in kw.items()
               if k in ("hidden", "layers", "n_out", "n_features", "n_int")}
        return IntGRUCore(**kw2)
    if name.startswith("tcngru"):
        from arch.tcngru import TCNGRUCore
        return TCNGRUCore(**{k: v for k, v in kw.items()
                             if k in ("hidden", "layers", "n_out", "n_features")})
    if name.startswith("tcn"):
        from arch.tcn import TCNCore
        return TCNCore(**{k: v for k, v in kw.items()
                          if k in ("hidden", "layers", "n_out", "n_features")})
    if name.startswith("lstm"):
        return LSTMCore(**{k: v for k, v in kw.items() if k in ("hidden", "layers", "n_out", "n_features")})
    if name.startswith("cfc"):
        return CfCCore(**{k: v for k, v in kw.items() if k in ("hidden", "layers", "n_out", "n_features")})
    presets = {
        "tiny":     dict(hidden=64,  layers=1),
        "gru":      dict(hidden=128, layers=2),   # baseline-shaped
        "wide":     dict(hidden=192, layers=1),
        "wide2":    dict(hidden=256, layers=1),
    }
    if name.startswith("dist"):
        # distributional head over a tick ladder (targets take ~133-230 distinct values)
        from disthead import DistGRUCore
        spec = {"dist": dict(bins=33), "dist65": dict(bins=65), "dist17": dict(bins=17)}[name]
        cfg = dict(hidden=128, layers=2); cfg.update(spec); cfg.update(kw)
        cfg.pop("n_out", None)
        return DistGRUCore(**cfg)
    if name.startswith("whiten"):
        # PCA-whitened input: EDA shows the top 3 PCs hold 40% of variance and ~0
        # signal, so per-column standardisation buries the useful directions
        from whiten import WhitenGRUCore
        spec = {"whiten": dict(hidden=128, layers=2),
                "whiten_h384": dict(hidden=384, layers=1),
                "whiten_h256": dict(hidden=256, layers=2)}[name]
        cfg = dict(spec); cfg.update(kw)
        return WhitenGRUCore(**cfg)
    _ATTN = {"attn": dict(halflives=(64, 512, 4096)),
             "attn_cum": dict(halflives=(1,), cumulative=True),
             "attn_short": dict(halflives=(32, 256))}
    if name in _ATTN:
        # Membership, not prefix. A bare startswith("attn") also claims "attnpool", which is an
        # arch.PRESETS entry living in a DIFFERENT module, and then dies on the dict lookup -- the
        # same shadowing that once let mingru384 and mingru_m silently build the wrong core. Any
        # prefix rule here must check its own table before claiming a name away from arch.PRESETS.
        # streaming softmax attention pooling over hidden states, O(1) per row
        from attnpool import AttnPoolGRUCore
        spec = _ATTN[name]
        cfg = dict(hidden=128, layers=2); cfg.update(spec); cfg.update(kw)
        return AttnPoolGRUCore(**cfg)
    if name.startswith("enc"):
        # nonlinear per-timestep encoder -> GRU (stateless encoder, export unchanged)
        from encgru import EncGRUCore
        spec = {"enc": dict(enc=160, enc_layers=2), "enc_small": dict(enc=128, enc_layers=1),
                "enc_deep": dict(enc=192, enc_layers=3),
                # linear twins: identical shape, no nonlinearity -> isolates rank/optimisation
                "enc_lin": dict(enc=160, enc_layers=2, act="none"),
                "enc_small_lin": dict(enc=128, enc_layers=1, act="none")}.get(name)
        if spec is None:
            raise KeyError(f"unknown enc* preset {name!r}")
        cfg = dict(hidden=128, layers=2); cfg.update(spec); cfg.update(kw)
        return EncGRUCore(**cfg)
    if name.startswith("ema"):
        # Fixed multi-timescale causal EMAs concatenated to the raw row, then a GRU.
        # A GRU has to learn 500-row memory by saturating its update gate; handing it
        # the timescales directly is the cheapest version of that. Never yet tested.
        from boost.ema_gru import EmaGRUCore
        hls = {"ema": (8, 64, 512), "ema_short": (4, 16, 64),
               "ema_wide": (2, 8, 32, 128, 512), "ema_long": (32, 256, 2048)}[name]
        cfg = dict(hidden=128, layers=2, hls=hls); cfg.update(kw)
        return EmaGRUCore(**cfg)
    if name not in presets:
        from arch import PRESETS as ARCH_PRESETS, build as arch_build   # src/arch candidates
        if name in ARCH_PRESETS:
            return arch_build(name, **kw)
    cfg = dict(presets.get(name, presets["gru"])); cfg.update(kw)
    return GRUCore(**cfg)
