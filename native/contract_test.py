"""research H11: solution.py CONTRACT test of a built zip through the official row-by-row DataPoint protocol (laptop or box).
  a) normal backend (native): rows 0..WARMUP-1 return None; a new seq_ix resets the state AND the posfeat counter: the protocol
     predictions of every sequence == the emulation (ref_run) from a fresh reset, bit for bit; the state is really carried
     (second sequence != first sequence re-run from the second's row 0 only if the data differ -- trivially; the strong check is
     the bit-identity to a from-reset emulation, which an un-reset counter or hidden state cannot pass);
  b) WNN_FORCE=crash in a child process: the ONNX fallback (model.onnx, patched pcnt state) produces real predictions; reported
     vs the native ones (corr, max|d|, WP per target) and its pcnt state is listed.
usage: python native/contract_test.py ZIP [N=2]   (needs datasets/valid.parquet and utils.py at the repo root)"""
import sys, os, zipfile, tempfile, importlib.util, subprocess, json, numpy as np, pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
from utils import DataPoint, FEATURE_COLUMNS, TARGET_COLUMNS, weighted_pearson, WARMUP
zp = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2
d = tempfile.mkdtemp(); zipfile.ZipFile(zp).extractall(d)


def protocol_run(force=""):
    """Run the zip's PredictionModel over the first N valid sequences; returns (preds (N, 20000, 2) with None rows = NaN, backend)."""
    code = f"""
import sys, os, json, importlib.util, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, {ROOT!r}); from utils import DataPoint, FEATURE_COLUMNS, WARMUP
spec = importlib.util.spec_from_file_location("sol_zip", {d!r} + "/solution.py"); m = importlib.util.module_from_spec(spec)
os.chdir({d!r}); spec.loader.exec_module(m); model = m.PredictionModel()
f = pq.ParquetFile({os.path.join(ROOT, "datasets/valid.parquet")!r}); out = []
for g in range({N}):
    t = f.read_row_group(g, columns=["seq_ix"] + list(FEATURE_COLUMNS))
    X = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32); sq = int(t["seq_ix"][0].as_py())
    o = np.full((20000, 2), np.nan, np.float32)
    for s in range(20000):
        r = model.predict(DataPoint(sq, s, s >= WARMUP, X[s]))
        if r is not None: o[s] = r
        elif s >= WARMUP: raise SystemExit("None where a prediction was required")
    out.append(o)
np.save({d!r} + "/proto_" + {force!r} + ".npy", np.stack(out)); print(json.dumps({{"backend": model.backend, "log": model.log}}))
"""
    env = dict(os.environ); env["WNN_FORCE"] = force
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=d)
    if r.returncode != 0: print(r.stdout, r.stderr); raise SystemExit("protocol run failed (force=%r)" % force)
    info = json.loads(r.stdout.strip().splitlines()[-1])
    return np.load(f"{d}/proto_{force}.npy"), info


spec = importlib.util.spec_from_file_location("sol_zip_main", f"{d}/solution.py"); sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
blob = open(f"{d}/weights.wnn", "rb").read(); B = sol.Blob(blob)
print(f"{os.path.basename(zp)}: blob v{B.ver} N={B.N} flags={B.flags} posfeat={B.posfeat}; files {sorted(zipfile.ZipFile(zp).namelist())}")
f = pq.ParquetFile(os.path.join(ROOT, "datasets/valid.parquet")); Xs, Ys, Ms = [], [], []
for g in range(N):
    t = f.read_row_group(g, columns=["seq_ix", "is_scored"] + list(FEATURE_COLUMNS) + list(TARGET_COLUMNS))
    Xs.append(np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32))
    Ys.append(np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32))
    ms = t["is_scored"].to_numpy().astype(bool); ms[:WARMUP] = False; Ms.append(ms)
ok = True
pn, info_n = protocol_run("")
print("normal backend:", info_n["backend"], "|", " | ".join(info_n["log"]))
ok &= info_n["backend"].startswith("native")
warm_none = all(np.isnan(pn[g, :WARMUP]).all() and np.isfinite(pn[g, WARMUP:]).all() for g in range(N))
print(f"  warm-up rows 0..{WARMUP - 1} None and every later row finite: {warm_none}"); ok &= warm_none
for g in range(N):
    with np.errstate(all="ignore"): em = sol.ref_run(B, Xs[g])[0]                   # emulation from a fresh reset (counter 0)
    same = bool(np.array_equal(pn[g, WARMUP:], em[WARMUP:]))
    print(f"  sequence {g}: protocol native == emulation from reset (bit-identical) {same}; max|d| {np.abs(pn[g, WARMUP:] - em[WARMUP:]).max():.1e}")
    ok &= same
# a wrong reset would show as a mismatch above for g >= 1; make the failure mode visible: the emulation CONTINUED from sequence 0's
# state (no reset) differs from the protocol output of sequence 1
if N >= 2:
    with np.errstate(all="ignore"):
        _, h = sol.ref_run(B, Xs[0]); cont = sol.ref_run(B, Xs[1], h)[0]
    print(f"  (control) emulation of sequence 1 WITHOUT a reset vs protocol: max|d| {np.abs(pn[1, WARMUP:] - cont[WARMUP:]).max():.3f} (must be > 0)")
    ok &= not np.array_equal(pn[1, WARMUP:], cont[WARMUP:])
pc, info_c = protocol_run("crash")
print("WNN_FORCE=crash backend:", info_c["backend"], "|", " | ".join(info_c["log"]))
ok &= info_c["backend"] == "onnx"
import onnx
m = onnx.load(f"{d}/model.onnx"); ins = [i.name for i in m.graph.input]; outs = [o.name for o in m.graph.output]
print(f"  model.onnx inputs {ins} outputs {outs}; posfeat state present: {'pcnt' in ins and 'pcnt_out' in outs}")
y = np.concatenate([Ys[g][Ms[g]] for g in range(N)]); a = np.concatenate([pn[g][Ms[g]] for g in range(N)]); b = np.concatenate([pc[g][Ms[g]] for g in range(N)])
c = [float(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in (0, 1)]
wn = [weighted_pearson(y[:, k], a[:, k]) for k in (0, 1)]; wc = [weighted_pearson(y[:, k], b[:, k]) for k in (0, 1)]
print(f"  onnx fallback vs native on {len(a)} scored rows: corr t0 {c[0]:.5f} t1 {c[1]:.5f}, max|d| {np.abs(a - b).max():.4f}, rms d {np.sqrt(((a - b) ** 2).mean()):.5f}; "
      f"WP native {wn[0]:.4f}/{wn[1]:.4f} onnx {wc[0]:.4f}/{wc[1]:.4f}; onnx zero rows {float(np.mean((b == 0).all(1))):.3f}")
ok &= min(c) > 0.99 and min(wc) > 0.5 and float(np.mean((b == 0).all(1))) < 0.01
print("CONTRACT", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
