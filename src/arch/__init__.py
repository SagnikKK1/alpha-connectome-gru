"""Candidate recurrent cores that aim to beat GRU(2x128) at lower latency.
See docs/research/architecture.md (round 1) and architecture2.md (round 2) for the analysis.

Every core shares GRUCore's interface -- forward(x, h) over (T, B, F) -> (out, h),
init_state(batch), n_params(), set_norm(mu, sd), est_us_per_row() -- and keeps its state
in ONE tensor of shape (S, B, D), so src/train.py and model.ExportWrapper work unchanged.
"""
from .ssm import DiagSSMCore
from .mingru import MinGRUCore
from .proj_gru import ProjGRUCore
from .clockssm import ClockSSMCore
from .grux import GRUXCore
from .attnpool import AttnPoolCore

from .xlstm import SLSTMCore


def _z19xf(**kw):
    """research Z19: transformer-over-window lookahead TEACHER (src/arch/z19xf.py), imported on first use only.
    A function in place of a class keeps the default path's import graph unchanged (offline/patches/add_z19.py)."""
    from .z19xf import Z19XFCore
    return Z19XFCore(**kw)

PRESETS = {
    # sLSTM (xLSTM, Beck et al. 2024). Its NORMALISER state n_t, with output c_t/n_t, lets a cell
    # accumulate an unbounded running sum without saturating - the exact capability a tanh GRU
    # lacks and that this repo built arch/integrator.py to chase. Distinct from arch/ssm.py, which
    # is a POSITIVE-diagonal linear recurrence and scored 0.6242 against a matched GRU's 0.634.
    "slstm":    (SLSTMCore, dict(hidden=256, layers=1, proj=192)),
    "slstm_pl2":(SLSTMCore, dict(hidden=256, layers=1, proj=192, proj_layers=2)),
    "slstm_b":  (SLSTMCore, dict(hidden=384, layers=1, proj=192)),
    # round 1 -------------------------------------------------------------------------
    "ssm256":    (DiagSSMCore, dict(d=256, mlp=96)),
    "ssm512":    (DiagSSMCore, dict(d=512, mlp=128)),
    "ssm256g":   (DiagSSMCore, dict(d=256, mlp=96, gate=True)),        # Mamba-style gate
    "ssm256t":   (DiagSSMCore, dict(d=256, mlp=96, pre_act="tanh")),   # nonlinear before smoothing
    "ema_fixed": (DiagSSMCore, dict(d=256, mlp=96, learn_decay=False)),  # F5 bank, learned mixing only
    "mingru256": (MinGRUCore, dict(d=256, mlp=96)),
    "mingru384": (MinGRUCore, dict(d=384, mlp=128)),
    "projgru":   (ProjGRUCore, dict(k=32, hidden=192, layers=1)),
    "projgru2":  (ProjGRUCore, dict(k=32, hidden=128, layers=2)),
    "projgru48": (ProjGRUCore, dict(k=48, hidden=160, layers=1)),
    # round 2: the row wired into the readout, priors on memory, clock-aware decay ------
    "grux":        (GRUXCore, dict(hidden=128, layers=2, mlp=96, skip="mlp")),
    "grulin":      (GRUXCore, dict(hidden=128, layers=2, skip="lin")),
    "gru_chrono":  (GRUXCore, dict(hidden=128, layers=2, skip="none", chrono=True)),
    "grux_chrono": (GRUXCore, dict(hidden=128, layers=2, mlp=96, skip="mlp", chrono=True)),
    "grux1":       (GRUXCore, dict(hidden=192, layers=1, mlp=96, skip="mlp")),
    "hybrid":      (GRUXCore, dict(hidden=128, layers=1, mlp=96, skip="mlp", ssm_d=256)),
    "rowssm512":   (ClockSSMCore, dict(d=512, mlp=128, frac_wall=0.0)),   # DiagSSM through the new scan
    "clockssm512": (ClockSSMCore, dict(d=512, mlp=128, frac_wall=0.5)),
    "wallssm512":  (ClockSSMCore, dict(d=512, mlp=128, frac_wall=1.0)),
    # round 3: streaming softmax attention pooling ------------------------------------
    "attnpool":   (AttnPoolCore, dict(hidden=128, layers=1, heads=4, mlp=96, score=True)),
    "attnpool_u": (AttnPoolCore, dict(hidden=128, layers=1, heads=4, mlp=96, score=False)),  # ablation
    # round 4 (16 Sep): first architecture probes under the seqrun recipe, sized to refgru's
    # 428,546 params so a null is interpretable. Both are ELEMENTWISE recurrences, so they
    # unroll to int8 the way refgru does if one of them wins.
    "mingru_m":   (MinGRUCore, dict(d=1152, mlp=128)),      # ~422k
    "ssm_m":      (DiagSSMCore, dict(d=1728, mlp=128)),     # ~437k
    # research Z19 (5 Oct): lookahead TEACHERS of a different class -- a transformer encoder over the W-row window on
    # top of a causal GRU (src/arch/z19xf.py). Offline only, never a member. The window is intrinsic to the preset:
    # dump with `teacher_dump.py --window W --raw` for the preset's W. hidden/layers/proj may be overridden from the
    # CLI (train.py forwards exactly those three); everything else is fixed here.
    "z19xf":      (_z19xf, dict(hidden=256, layers=1, proj=192, proj_layers=2, window=100, d_model=256, heads=4, ff=512, xf_layers=4)),
    "z19xf_l2":   (_z19xf, dict(hidden=256, layers=1, proj=192, proj_layers=2, window=100, d_model=256, heads=4, ff=512, xf_layers=2)),
    "z19xf_w50":  (_z19xf, dict(hidden=256, layers=1, proj=192, proj_layers=2, window=50,  d_model=256, heads=4, ff=512, xf_layers=4)),
    "z19xf_w200": (_z19xf, dict(hidden=256, layers=1, proj=192, proj_layers=2, window=200, d_model=256, heads=4, ff=512, xf_layers=4)),
}


def build(name: str, **kw):
    cls, cfg = PRESETS[name]
    cfg = dict(cfg); cfg.update(kw)
    return cls(**cfg)