"""Per-sequence holdout WP for a raw checkpoint, in the format paired_zip.py reads.

Exporting to a zip and scoring row-at-a-time takes ~10 minutes; this runs the same
sequences batched on the GPU in well under one, so a run can be screened against v8
the moment a checkpoint lands. It is a SCREEN, not the decision: it scores fp32
weights, where the submission scores int8 MatMulNBits. Confirm anything promising
with score_zip.py before believing it.

The skip rule and the output convention are copied from score_zip.py / model.Wrapper
so the vectors stay aligned element-for-element with the saved submission vectors.

    python src/score_ckpt.py SP_skew_s301 [--ckpt best.pt] [--n 200] [--cache holdout]
    python src/paired_zip.py v8 SP_skew_s301
"""
from __future__ import annotations
import argparse, os, re, sys
from pathlib import Path
import numpy as np, torch

R = Path(__file__).resolve().parent.parent
for p in (str(R), str(R/"src")): sys.path.insert(0, p)
from utils import weighted_pearson, WARMUP, SEQUENCE_LENGTH
from data import SequenceCache
from model import build

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(tag, ckpt):
    ck = torch.load(R/"runs"/tag/ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    kw = {k: cfg[k] for k in ("hidden", "layers") if cfg.get(k)}
    preset = cfg.get("preset", "gru")
    if preset.startswith("z19") and cfg.get("proj"):
        kw["proj"] = cfg["proj"]        # add_z19.py: train.py forwards --proj to every preset; mirror it for the rebuild
    if preset.startswith("slstm"):
        # Same failure the refgru branch already documents, one preset over: slstm inherited
        # only hidden/layers, so --proj was dropped and the rebuilt core came back at the
        # default width. DS_slstm_p256_s801 died exactly this way -- post_ln.weight was [512]
        # in the checkpoint against [448] rebuilt (192 hidden + 256 proj vs + 192 default) --
        # and every dump attempt failed. Derive the list from the constructor so it stays in
        # step with arch/xlstm.py automatically.
        import inspect as _insp
        from arch.xlstm import SLSTMCore as _SL
        _acc = set(_insp.signature(_SL.__init__).parameters)
        for _k, _v in cfg.items():
            if _k in kw or _k not in _acc or _v in (None, "", False, 0):
                continue
            kw[_k] = _v
        # cfg CANNOT be trusted for slstm geometry: it records the CLI flag, not what the
        # model was actually built with. DS_slstm_p256_s801's cfg says proj_layers=2 while its
        # state_dict has only in_proj.0/2 -- one block -- because train.py never forwarded
        # --proj-layers to the slstm core. Infer the geometry from the WEIGHTS, which are the
        # only record of what was really trained.
        _sd = ck["state"]
        _nblk = sum(1 for k in _sd if re.fullmatch(r"in_proj\.\d+\.weight", k)
                    and _sd[k].ndim == 2)
        if _nblk: kw["proj_layers"] = _nblk
        if "in_proj.0.weight" in _sd: kw["proj"] = int(_sd["in_proj.0.weight"].shape[0])
        if "W.0.weight" in _sd:       kw["hidden"] = int(_sd["W.0.weight"].shape[0]) // 4
        if cfg.get("head_hidden"): kw["head_hidden"] = cfg["head_hidden"]
    if preset.startswith("refgru"):
        if cfg.get("proj"): kw["proj"] = cfg["proj"]
        kw["cell"] = cfg.get("cell", "gru"); kw["diff"] = bool(cfg.get("diff")); kw["posfeat"] = bool(cfg.get("posfeat")); kw["split"] = bool(cfg.get("split_towers"))
        kw["ewma"] = tuple(float(v) for v in str(cfg.get("ewma", "")).split(",") if v)
        kw["nlfeat"] = tuple(float(v) for v in str(cfg.get("nlfeat", "")).split(",") if v)
        if cfg.get("quad"): kw["quad"] = True
        if cfg.get("learn_h0"): kw["learn_h0"] = True   # add_learn_h0.py
        if cfg.get("fastw"):
            kw["fastw"] = int(cfg["fastw"]); kw["fw_groups"] = int(cfg.get("fastw_groups", 4))
            _hl = [float(v) for v in str(cfg.get("fastw_hl", "32,4096")).split(",")]
            kw["fw_hl_min"], kw["fw_hl_max"] = _hl[0], _hl[-1]
            kw["fw_to_gru"] = bool(cfg.get("fastw_gru"))
        if cfg.get("head_hidden"): kw["head_hidden"] = cfg["head_hidden"]
        if cfg.get("bidir"): kw["bidir"] = True      # non-causal teacher (add_bidir.py)
        if cfg.get("dualmode"): kw["dualmode"] = int(cfg["dualmode"])   # train-only branch (add_dualmode.py)
        if "aux_next_head.weight" in ck["state"]: kw["aux_next"] = 1     # train-only head (add_aux_next.py)
        if "mask_head.weight" in ck["state"]: kw["mask_aux"] = 1          # train-only head (add_signals.py)
        if "hetero_head.0.weight" in ck["state"]: kw["hetero"] = 1      # train-only log-variance head (add_losses2.py)
        if "grid_v" in ck["state"]: kw["grid"] = int(ck["state"]["grid_v"].shape[0])   # --grid-head (research P)
        if "hint_adapter.weight" in ck["state"]:                          # train-only adapter (add_hint.py)
            kw["hint"] = int(ck["state"]["hint_adapter.weight"].shape[0])
        # FORWARD EVERY ARCHITECTURE FLAG THE CORE ACCEPTS. This function rebuilds the model from
        # cfg in order to load a state_dict into it, so any constructor kwarg missing here yields
        # a DIFFERENT model and the load fails (or, worse, silently succeeds with a smaller one).
        # --ofi and --proj-layers were added to train.py and refgru.py without reaching here, so
        # every dump of those checkpoints died: in_proj.0.weight was [192,136] in the checkpoint
        # against [192,112] in the rebuilt core (--ofi adds 24 input channels) plus unexpected
        # in_proj.3/5 keys (--proj-layers 2 adds a second block). Deriving the list from the
        # constructor keeps this in step with refgru.py automatically.
        import inspect as _insp
        from arch.refgru import RefGRUCore as _RG
        _accepts = set(_insp.signature(_RG.__init__).parameters)
        for _k, _v in cfg.items():
            if _k in kw or _k not in _accepts or _v in (None, "", False, 0):
                continue
            kw[_k] = _v
        if cfg.get("clamp") not in (None, 10.0): kw["clamp"] = float(cfg["clamp"])
        if cfg.get("bookconv"): kw["bookconv"] = int(cfg["bookconv"])
        if cfg.get("ofi"): kw["ofi"] = True
        if cfg.get("adanorm"): kw["adanorm"] = float(cfg["adanorm"])
        if cfg.get("proj_layers", 1) != 1: kw["proj_layers"] = int(cfg["proj_layers"])
        if cfg.get("proj_gated"): kw["proj_gated"] = True
    # the readout may be a Linear (head.weight) or an MLP (head.0.weight ... head.3.weight);
    # the deployed width is the LAST linear's rows
    keep = None
    if cfg.get("drop_blocks"):
        BLOCKS = {"i0_p": (0, 22), "i0_v": (22, 44), "i0_dp": (44, 48), "i0_dv": (48, 52), "i1_p": (52, 74),
                  "i1_v": (74, 96), "i1_dp": (96, 100), "i1_dv": (100, 104), "a": (104, 112)}
        drop = set()
        for nm in cfg["drop_blocks"].split(","):
            lo, hi = BLOCKS[nm.strip()]; drop.update(range(lo, hi))
        keep = np.array([c for c in range(112) if c not in drop]); kw["n_features"] = len(keep)
    hkeys = [k for k in ck["state"] if k.startswith("head") and k.endswith("weight")]
    hw = ck["state"][sorted(hkeys)[-1]] if hkeys else None
    if hw is None:
        # Cores from src/arch do not call their readout "head" -- DiagSSMCore's is "read.2.weight" --
        # so the refgru-shaped lookup finds nothing and n_out silently defaults to 1, which then dies
        # in load_state_dict with a size mismatch. Fall back to the LAST 2-D weight in the state dict,
        # which is the final linear for every core here, and take its row count.
        twod = [k for k, v in ck["state"].items() if k.endswith("weight") and v.dim() == 2]
        if twod:
            hw = ck["state"][twod[-1]]
    _nout = int(hw.shape[0]) if hw is not None else 1
    if kw.get("grid"): _nout = _nout // int(kw["grid"])          # grid head emits n_out x K logits
    if cfg.get("extra_in"):                          # add_z43_score.py: --extra-in checkpoints read 112 + K columns
        kw["n_features"] = int(ck["state"]["mu"].shape[0])
    core = build(preset, n_out=_nout, **kw)
    core.load_state_dict(ck["state"])
    core.z43_dir = cfg.get("extra_in") or None       # add_z43_score.py
    core.common_head = bool(cfg.get("common_head"))
    core.keep_cols = keep                       # consumers slice X[..., keep_cols] when set
    return core.to(DEV).eval(), float(ck.get("scale", 1.0)), cfg


def _z43_extra(core, cache, a, b):
    """add_z43_score.py: the K decoded channels of cache rows a..b (NaN -> fill) for an --extra-in checkpoint, else None."""
    d = getattr(core, "z43_dir", None)
    if not d: return None
    import json as _json
    p = Path(d) if Path(d).is_absolute() else R / d
    name = Path(cache.path).name
    f, mf = p / f"{name}.npy", p / f"{name}_meta.json"
    if not (f.exists() and mf.exists()):
        raise SystemExit(f"--extra-in checkpoint: {f} or {mf.name} is missing (feature dump builder, not included in this repo)")
    meta = _json.load(open(mf))
    if meta.get("seq_ix") != list(cache.meta["seq_ix"]):
        raise SystemExit(f"--extra-in checkpoint: {mf} seq_ix differs from the {name} cache manifest")
    e = np.asarray(np.load(f, mmap_mode="r")[a:b], np.float32)
    return np.where(np.isnan(e), np.float32(meta.get("fill", {}).get("nan", 0.0)), e)


def emit(o, sc, cfg, common, lat=None):
    """(T,B,n_out) raw -> (p0, p1), matching model.Wrapper.

    `lat=(K, q)` applies the LATTICE readout. Without it a lattice checkpoint is scored by
    reading logits 0 and 1 as if they were predictions, which produced holdout -0.35 (an
    ANTI-correlation) for a model whose own in-training eval read +0.6363. Same train/serve
    mismatch class as --fam-norm, which `_famnorm_for` below exists to prevent.
    """
    if lat is not None:
        import lattice as _lat
        K, q = lat
        sig = _lat.readout(o, q, K) * sc
        return sig, -sig
    s = o[..., 0] * sc
    if o.shape[-1] == 1 or (not common and not cfg.get("two_head")):
        return s, -s
    c = o[..., 1] * sc
    return (s + c, -s + c) if common else (s, c)


def _famnorm_for(cfg, cache, cache_name):
    """(fnorm, family labels) when the run used --fam-norm, else (None, None).

    The per-family transform lives OUTSIDE the checkpoint -- train.py sets the model's
    own (mu, sd) buffers to identity and normalises the batch before it reaches the
    core. Scoring such a checkpoint without reapplying it feeds the model raw features
    and reads as a catastrophic loss that is entirely an artefact.

    Family comes from the warm-up classifier, exactly as it would at inference.
    """
    if not cfg.get("fam_norm"): return None, None
    import famnorm
    from family import predict as fpredict
    # cfg["train_cache"] is already repo-relative ("cache/train_full"), not a bare name
    tr = SequenceCache(R/cfg.get("train_cache", "cache/train_full"))
    g_mu, g_sd = tr.feature_stats(64)
    fnorm = famnorm.stats(tr, np.load(R/"runs/family_train.npy"),
                          f"train{len(tr)}", g_mu, g_sd)
    z = np.load(R/"runs/family_clf.npz")
    fam = np.array([fpredict(cache.X[i], z["W"], z["mu"], z["sd"])
                    for i in range(len(cache))], np.int64)
    print(f"  {cache_name}: per-family normalisation active, mix "
          f"{np.bincount(fam, minlength=4)/len(fam)}")
    return fnorm, fam


def _lat_for(cfg, cache):
    """(K, q_per_sequence) when the run used --lattice-head, else None.

    Family comes from the warm-up classifier, exactly as it would at inference -- the
    lattice readout needs the sequence's tick base q, and s/q is an integer only on RAW
    targets, so q must be right or the readout is meaningless.
    """
    K = int(cfg.get("lattice_head") or 0)
    if not K: return None
    from family import predict as fpredict, BASES
    z = np.load(R/"runs/family_clf.npz")
    fam = np.array([fpredict(cache.X[i], z["W"], z["mu"], z["sd"])
                    for i in range(len(cache))], np.int64)
    return K, np.asarray(BASES, np.float64)[fam]


@torch.no_grad()
def run(tag, ckpt, cache_name, n, chunk=None, reset_every=0):
    chunk = chunk or int(os.environ.get("SCORE_CHUNK", "16"))
    """reset_every > 0 zeroes the recurrent state every K rows.

    This is R2D2's history-length sweep and Buitrago Ruiz & Gu's Effective Remembrance,
    which two unrelated literatures arrived at independently. If weighted Pearson is FLAT
    in K, then training on 768-row windows from a zero state has taught the model to
    ignore its own state beyond a few hundred rows -- the 20,000-row carried state at
    deployment is decorative, and a periodic reset is free. If the score CLIMBS with K,
    there is real long-range signal and state-passing has headroom.
    """
    core, sc, cfg = load(tag, ckpt)
    common = bool(cfg.get("common_head"))
    cache = SequenceCache(R/"cache"/cache_name)
    fnorm, fam = _famnorm_for(cfg, cache, cache_name)

    # A lattice head emits 2K+1 class logits, not a signal: reading logit 0 as the
    # prediction would score noise. Rebuild the metric's own w-tilted readout, with q
    # from the warm-up classifier exactly as inference would.
    lat = None
    if cfg.get("lattice_head"):
        import lattice as _lat
        from family import predict as _fp, BASES as _B
        z = np.load(R/"runs/family_clf.npz")
        qv = _B[np.array([_fp(cache.X[i], z["W"], z["mu"], z["sd"])
                          for i in range(min(n, len(cache)))], np.int64)]
        lat = (int(cfg["lattice_head"]), qv, _lat)
        print(f"  lattice head K={lat[0]}: w-tilted readout, q from the warm-up classifier")
    steps = np.arange(SEQUENCE_LENGTH); out = []
    N = min(n, len(cache))
    for a in range(0, N, chunk):
        b = min(a+chunk, N)
        X = np.stack([np.asarray(cache.X[i], np.float32) for i in range(a, b)])
        if getattr(core, "keep_cols", None) is not None: X = X[..., core.keep_cols]
        _z4e = _z43_extra(core, cache, a, b)            # add_z43_score.py
        if _z4e is not None: X = np.concatenate([X, _z4e], -1)
        if fnorm is not None:
            import famnorm
            X = famnorm.apply(X, fam[a:b], *fnorm)
        xt = torch.from_numpy(X).permute(1, 0, 2).to(DEV)
        if reset_every and reset_every < SEQUENCE_LENGTH:
            chunks = []
            for t0 in range(0, SEQUENCE_LENGTH, reset_every):
                seg, _ = core(xt[t0:t0+reset_every], core.init_state(b-a, DEV))
                chunks.append(seg)
            o = torch.cat(chunks, 0)
        else:
            o, _ = core(xt, core.init_state(b-a, DEV))
        ot = o.permute(1, 0, 2)
        if lat is not None:
            K, qv, _lat = lat
            sig = _lat.readout(ot, qv[a:b], K) * sc
            p0, p1 = sig, -sig
        else:
            p0, p1 = emit(ot, sc, cfg, common)
        p0 = p0.cpu().numpy(); p1 = p1.cpu().numpy()
        for k in range(b-a):
            i = a+k
            m = np.asarray(cache.M[i]) & (steps >= WARMUP)
            if not m.any(): continue
            y = np.asarray(cache.Y[i])
            if not all(np.any(y[m, j] > 0) and np.any(y[m, j] < 0) for j in (0, 1)): continue
            out.append(0.5*(weighted_pearson(y[m, 0], p0[k][m]) +
                            weighted_pearson(y[m, 1], p1[k][m])))
    return np.array(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--cache", default="holdout"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--calib", choices=("ckpt", "retrain"), default="ckpt",
                    help="output scale: the checkpoint's own train-std calibration, or a "
                         "clip-aware per-channel multiple selected on valid_retrain (clipcal.py)")
    ap.add_argument("--calib-fams", default="", help="with --calib retrain: select the scale on these families only")
    ap.add_argument("--reset-every", type=int, default=0,
                    help="zero the recurrent state every K rows (0 = never). Sweep this "
                         "to measure how much of the 20,000-row history the model uses.")
    a = ap.parse_args()
    if a.calib == "retrain":
        import clipcal
        _f = [int(v) for v in a.calib_fams.split(",") if v] or None
        m0, m1 = clipcal.calib_ckpt(a.tag, a.ckpt, fams=_f)
        core, sc, cfg = load(a.tag, a.ckpt); cfg = dict(cfg, _scale=sc)
        cache = SequenceCache(R/"cache"/a.cache)
        P = clipcal.apply(clipcal.forward(core, cfg, cache, a.n), m0, m1)
        s0, s1 = clipcal.per_target(P, cache, 1.0, 1.0); s = 0.5*(s0 + s1)
    else:
        s = run(a.tag, a.ckpt, a.cache, a.n, reset_every=a.reset_every)
    name = a.tag if a.ckpt == "best.pt" else f"{a.tag}@{Path(a.ckpt).stem}"
    if a.calib == "retrain": name += "_cal" + (("f" + a.calib_fams.replace(",", "")) if a.calib_fams else "")
    if a.reset_every: name += f"_reset{a.reset_every}"
    np.save(R/"runs"/f"score_{name}_{a.cache}.npy", s)
    print(f"{name} on {a.cache}: WP {s.mean():+.4f} +/- {s.std(ddof=1)/np.sqrt(len(s)):.4f} (n={len(s)})")
    print(f"saved runs/score_{name}_{a.cache}.npy  ->  python src/paired_zip.py v8 {name}")
