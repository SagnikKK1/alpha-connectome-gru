"""Teacher predictions for distillation: average of several checkpoints' CALIBRATED outputs
(model output x export scale x multiplier, i.e. exactly what the merged ensemble graph emits),
over every row of a cache, stored as float16 (N, T, 2).

    python src/teacher_dump.py --tags G_pw50_s801,GP_d999_s801 --mults 0.5 --caches train_full,valid_retrain \
        --out runs/teacher_pair --batch 32 --device cuda

Deterministic (no dropout, eval mode); the same tags always give the same file.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

R = Path(__file__).resolve().parents[1]
for p in (str(R), str(R/"src"), str(R/"src"/"arch")): sys.path.insert(0, p)
from data import SequenceCache
from score_ckpt import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True); ap.add_argument("--ckpt", default="final.pt")
    ap.add_argument("--mults", default="0.5", help="per-target multiplier (scalar or m0,m1) applied on top of the export scale")
    ap.add_argument("--raw", action="store_true",
                    help="dump the teacher on the RAW output surface -- no export scale, no multiplier.\n                         The student's own anchor pulls its raw output toward the clipped target, so a\n                         teacher carrying the calibration sits on a different surface (export scale ~2.19\n                         times the default 0.5 multiplier is ~1.085x) and the two loss terms then fight\n                         over the student's scale. Since the anchor exists precisely to FIX that scale,\n                         matching on the calibrated surface undoes its job. Use --raw for distillation;\n                         the calibrated surface is only right if you are reproducing a deployed graph.")
    ap.add_argument("--caches", default="train_full,valid_retrain"); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=0,
                    help="bidir teachers: bound the LOOKAHEAD to W rows -- run the sequence in W-row windows, "
                         "carrying the forward state and restarting the backward direction at each window, "
                         "the condition chunked training imposed. 0 = whole sequence (unbounded lookahead).")
    ap.add_argument("--aux-head", action="store_true",
                    help="dual-mode students (--dualmode): dump the train-only AUX (non-causal, windowed) head "
                         "instead of the causal head -- a frozen window-matched teacher (research J2a, round 4). "
                         "Runs the core in train mode with every Dropout switched to eval.")
    ap.add_argument("--min-look", type=int, default=0,
                    help="with --window W: uniform lookahead -- run overlapping windows of W+min_look rows "
                         "starting every W rows and keep the first W outputs, so every row sees at least "
                         "min_look future rows (the plain window dump is a sawtooth: row k of a window sees "
                         "only W-1-k rows; research J, round 4)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    mults = [float(v) for v in a.mults.split(",")]; mults = mults * 2 if len(mults) == 1 else mults
    models = []
    for t in a.tags.split(","):
        core, sc, cfg = load(t, a.ckpt); core = core.to(dev).eval()
        if a.aux_head:
            if not getattr(core, "dualmode", 0): raise SystemExit(f"--aux-head: {t} is not a --dualmode checkpoint")
            core.train()
            for m in core.modules():
                if isinstance(m, torch.nn.Dropout): m.eval()
        k = torch.tensor([1.0, 1.0] if a.raw else [sc * mults[0], sc * mults[1]], device=dev)
        models.append((core, k))
        print(f"{t}: " + ("RAW surface (no scale, no mult)" if a.raw
                          else f"export scale {sc:.4f} x mults {mults}"), flush=True)
    out = R/a.out; out.mkdir(parents=True, exist_ok=True)
    for cname in a.caches.split(","):
        c = SequenceCache(R/"cache"/cname); N, T = c.X.shape[0], c.X.shape[1]
        res = np.lib.format.open_memmap(out/f"{cname}.npy", mode="w+", dtype=np.float16, shape=(N, T, 2))
        t0 = time.time()
        for s in range(0, N, a.batch):
            X = torch.from_numpy(np.nan_to_num(np.asarray(c.X[s:s+a.batch], dtype=np.float32))).permute(1, 0, 2).to(dev)  # (T,B,F)
            acc = None
            with torch.no_grad():
                for core, k in models:
                    if a.window > 0 and getattr(core, "bidir", False):
                        h = core.init_state(X.shape[1], dev); outs = []
                        for t0 in range(0, X.shape[0], a.window):
                            h = h.clone(); h[1:2 * core.layers:2] = 0.0      # backward direction restarts
                            if a.min_look > 0:
                                # uniform lookahead: outputs from a longer window, forward state from W rows
                                o_, _ = core(X[t0:t0 + a.window + a.min_look], h)
                                _, h = core(X[t0:t0 + a.window], h)
                                outs.append(o_[:a.window])
                            else:
                                o_, h = core(X[t0:t0 + a.window], h)
                                outs.append(o_)
                        o = torch.cat(outs, 0)
                    else:
                        o, _ = core(X, core.init_state(X.shape[1], dev))
                        if a.aux_head: o = core._aux_out
                    o = o[..., :2] if o.shape[-1] >= 2 else torch.cat([o, -o], -1)
                    o = o * k
                    acc = o if acc is None else acc + o
            res[s:s+a.batch] = (acc / len(models)).permute(1, 0, 2).cpu().numpy().astype(np.float16)
            if (s // a.batch) % 20 == 0:
                print(f"  {cname} {s+a.batch}/{N}  {time.time()-t0:.0f}s", flush=True)
        res.flush(); del res
        print(f"wrote {out/f'{cname}.npy'}  ({N}x{T}x2 float16)", flush=True)


if __name__ == "__main__":
    main()
