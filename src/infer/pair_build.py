"""Fused two-member refgru export: ONE graph whose activations carry a member axis.

A merged pair of separate graphs pays the ~40-node dispatch cost twice (v37: 71.8 us/row, v38 timed
out on the platform). Here every elementwise op (Split/Sigmoid/Tanh/Add/Mul/Sub/Gelu/LayerNorm) runs
once on tensors shaped (1, 2, D) -- batch 1, member 2 -- and only the weight matmuls are per member
(Split -> two MatMulNBits -> Concat). The two heads are averaged by a ReduceMean over the member axis.
LayerNorm gains/biases differ per member, so LN runs with unit scale and a (2, D) Mul/Add follows.

    python src/infer/pair_build.py runs/A/final.pt runs/B/final.pt --out submissions/PAIR_X --mults 0.5,0.5 \
        [--time submissions/submission_v37.zip]

Parity: the graph is checked against the mean of the two torch models (calibrated the same way) on 400
random rows and must match to 1e-4 in fp32 mode; the int8 graph is then timed in a bare loop.
"""
from __future__ import annotations
import argparse, sys, time, zipfile
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto

R = Path(__file__).resolve().parents[2]
for p in (str(R), str(R/"src"), str(R/"src"/"infer"), str(R/"src"/"arch")): sys.path.insert(0, p)
from onnx_build import _T, _io, _quant_nbits
from refgru_build import weights, DRIVER

def assert_single_member_only(ws):
    """Refuse a model whose input or head carries extra state this builder does not build.

    The fused and pair graphs assume the projection reads exactly the 112 raw columns and the head
    reads exactly [hidden | proj]. A model trained with --ewma widens the projection input and a
    model trained with --fastw widens the head input, and both carry extra state slots. Without this
    check the export SUCCEEDS and produces a graph that quietly computes something else, which is the
    one failure mode that has cost a real submission before. The single-member builder
    (refgru_build.py) handles both; multi-member does not yet.
    """
    for i, w in enumerate(ws if isinstance(ws, (list, tuple)) else [ws]):
        if w.get("ewma"):
            raise SystemExit("member %d was trained with --ewma; only refgru_build.py exports that" % i)
        if w.get("fastw"):
            raise SystemExit("member %d was trained with --fastw; only refgru_build.py exports that" % i)



def build_pair(wa: dict, wb: dict, mode: str = "nbits", bits: int = 8, block: int = 128, ln_eps: float = 1e-5, lean: bool = False):
    assert_single_member_only([wa, wb])
    H, L, P = wa["hidden"], wa["layers"], wa["proj"]
    assert (wb["hidden"], wb["layers"], wb["proj"]) == (H, L, P), "members must share the architecture"
    assert np.allclose(wa["mu"], wb["mu"]) and np.allclose(wa["sd"], wb["sd"]), "members must share input statistics"
    nodes, inits = [], []
    inputs = [_io("features", [1, 1, 112])]; outputs = [_io("prediction", [1, 1, 2])]
    ms = mode == "nbits"

    def mm_single(x, W, name, bias=None):
        if mode == "nbits":
            K, N = W.shape
            qi, _ = _quant_nbits(W, bits, block, name); inits.extend(qi)
            ins = [x, f"{name}_q", f"{name}_s"]
            if bias is not None:
                inits.append(_T(f"{name}_b", bias)); ins += ["", "", f"{name}_b"]
            nodes.append(helper.make_node("MatMulNBits", ins, [f"{name}_y"], domain="com.microsoft",
                                          K=K, N=N, bits=bits, block_size=block, accuracy_level=4, name=name))
            return f"{name}_y"
        inits.append(_T(f"{name}_W", W))
        nodes.append(helper.make_node("MatMul", [x, f"{name}_W"], [f"{name}_mm"], name=name))
        if bias is None: return f"{name}_mm"
        inits.append(_T(f"{name}_b", bias))
        nodes.append(helper.make_node("Add", [f"{name}_mm", f"{name}_b"], [f"{name}_y"]))
        return f"{name}_y"

    def mm_pair(x2, Wa, Wb, name, ba=None, bb=None, shared_input=False):
        """x2: (1,2,K) per-member rows (or (1,1,K) shared when shared_input). Returns (1,2,N)."""
        if shared_input:
            xa = xb = x2
        else:
            nodes.append(helper.make_node("Split", [x2], [f"{name}_xa", f"{name}_xb"], axis=1, num_outputs=2))
            xa, xb = f"{name}_xa", f"{name}_xb"
        ya = mm_single(xa, Wa, name + "_A", ba); yb = mm_single(xb, Wb, name + "_B", bb)
        nodes.append(helper.make_node("Concat", [ya, yb], [f"{name}_y2"], axis=1))
        return f"{name}_y2"

    def gelu(x, name):
        nodes.append(helper.make_node("Gelu", [x], [f"{name}_g"], domain="com.microsoft", name=f"{name}_gelu"))
        return f"{name}_g"

    def ln_pair(x, ga, ba_, gb, bb_, name):
        D = len(ga)
        inits.extend([_T(f"{name}_one", np.ones(D, np.float32)), _T(f"{name}_zero", np.zeros(D, np.float32)),
                      _T(f"{name}_g2", np.stack([ga, gb])[None].astype(np.float32)),
                      _T(f"{name}_b2", np.stack([ba_, bb_])[None].astype(np.float32))])
        nodes.append(helper.make_node("LayerNormalization", [x, f"{name}_one", f"{name}_zero"], [f"{name}_n"],
                                      axis=-1, epsilon=ln_eps, name=name))
        nodes.append(helper.make_node("Mul", [f"{name}_n", f"{name}_g2"], [f"{name}_ng"]))
        nodes.append(helper.make_node("Add", [f"{name}_ng", f"{name}_b2"], [f"{name}_y"]))
        return f"{name}_y"

    inv = (1.0 / wa["sd"]).astype(np.float32)
    inits.extend([_T("norm_mul", inv), _T("norm_add", (-wa["mu"] * inv).astype(np.float32)),
                  _T("clip_lo", np.array(-wa["clamp"], np.float32)), _T("clip_hi", np.array(wa["clamp"], np.float32))])
    nodes.append(helper.make_node("Mul", ["features", "norm_mul"], ["x_scaled"]))
    nodes.append(helper.make_node("Add", ["x_scaled", "norm_add"], ["x_norm"]))
    nodes.append(helper.make_node("Clip", ["x_norm", "clip_lo", "clip_hi"], ["x_clip"]))
    z = ln_pair(gelu(mm_pair("x_clip", wa["proj_W"], wb["proj_W"], "proj", wa["proj_b"], wb["proj_b"], shared_input=True), "proj"),
                wa["ln1_g"], wa["ln1_b"], wb["ln1_g"], wb["ln1_b"], "ln1")
    x = z
    for l in range(L):
        ga, gb = wa["gru"][l], wb["gru"][l]; p = f"l{l}_"; h = f"h{l}"
        inputs.append(_io(h, [1, 2, H])); outputs.append(_io(f"{h}_out", [1, 2, H]))
        gi = mm_pair(x, np.ascontiguousarray(ga["W_ih"].T), np.ascontiguousarray(gb["W_ih"].T), p + "ih", ga["b_ih"], gb["b_ih"])
        gh = mm_pair(h, np.ascontiguousarray(ga["W_hh"].T), np.ascontiguousarray(gb["W_hh"].T), p + "hh", ga["b_hh"], gb["b_hh"])
        nodes.append(helper.make_node("Split", [gi], [p + "gi_r", p + "gi_z", p + "gi_n"], axis=2, num_outputs=3))
        nodes.append(helper.make_node("Split", [gh], [p + "gh_r", p + "gh_z", p + "gh_n"], axis=2, num_outputs=3))
        nodes.append(helper.make_node("Add", [p + "gi_r", p + "gh_r"], [p + "r_pre"]))
        nodes.append(helper.make_node("Sigmoid", [p + "r_pre"], [p + "r"]))
        nodes.append(helper.make_node("Add", [p + "gi_z", p + "gh_z"], [p + "z_pre"]))
        nodes.append(helper.make_node("Sigmoid", [p + "z_pre"], [p + "z"]))
        nodes.append(helper.make_node("Mul", [p + "r", p + "gh_n"], [p + "rgh"]))
        nodes.append(helper.make_node("Add", [p + "gi_n", p + "rgh"], [p + "n_pre"]))
        nodes.append(helper.make_node("Tanh", [p + "n_pre"], [p + "n"]))
        nodes.append(helper.make_node("Sub", [h, p + "n"], [p + "hmn"]))
        nodes.append(helper.make_node("Mul", [p + "z", p + "hmn"], [p + "zhmn"]))
        nodes.append(helper.make_node("Add", [p + "n", p + "zhmn"], [f"{h}_out"]))
        x = f"{h}_out"
    nodes.append(helper.make_node("Concat", [x, z], ["cat"], axis=2))
    if lean:
        # LN2 with unit scale; its per-member gamma/beta are folded into head1: (n*g + b) W = n (diag(g) W) + b W
        D2 = len(wa["ln2_g"])
        inits.extend([_T("ln2_one", np.ones(D2, np.float32)), _T("ln2_zero", np.zeros(D2, np.float32))])
        nodes.append(helper.make_node("LayerNormalization", ["cat", "ln2_one", "ln2_zero"], ["ln2_n"], axis=-1, epsilon=ln_eps, name="ln2"))
        Wa1 = (wa["ln2_g"][:, None] * wa["h1_W"]).astype(np.float32); ba1 = (wa["h1_b"] + wa["ln2_b"] @ wa["h1_W"]).astype(np.float32)
        Wb1 = (wb["ln2_g"][:, None] * wb["h1_W"]).astype(np.float32); bb1 = (wb["h1_b"] + wb["ln2_b"] @ wb["h1_W"]).astype(np.float32)
        hdn = gelu(mm_pair("ln2_n", Wa1, Wb1, "head1", ba1, bb1), "head1")               # (1,2,HH)
        # stacked final head: reshape (1,2,HH) -> (1,1,2*HH), one matmul with [Wa/2; Wb/2] gives the member mean directly
        HH = wa["h2_W"].shape[0]
        inits.append(_T("flat_shape", np.array([1, 1, 2 * HH], np.int64)))
        nodes.append(helper.make_node("Reshape", [hdn, "flat_shape"], ["hdn_flat"]))
        W2 = np.concatenate([wa["h2_W"], wb["h2_W"]], 0).astype(np.float32) * 0.5
        b2 = ((wa["h2_b"] + wb["h2_b"]) * 0.5).astype(np.float32)
        y = mm_single("hdn_flat", W2, "head2", b2)
        nodes.append(helper.make_node("Identity", [y], ["prediction"]))
    else:
        c = ln_pair("cat", wa["ln2_g"], wa["ln2_b"], wb["ln2_g"], wb["ln2_b"], "ln2")
        hdn = gelu(mm_pair(c, wa["h1_W"], wb["h1_W"], "head1", wa["h1_b"], wb["h1_b"]), "head1")
        out2 = mm_pair(hdn, wa["h2_W"], wb["h2_W"], "head2", wa["h2_b"], wb["h2_b"])      # (1,2,2)
        inits.append(_T("mean_axes", np.array([1], np.int64)))
        nodes.append(helper.make_node("ReduceMean", [out2, "mean_axes"], ["prediction"], keepdims=1))
    graph = helper.make_graph(nodes, "refgru_pair_step", inputs, outputs, initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
                          producer_name="alpha-connectome")
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def run_graph(path, X, H, L):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    st = [np.zeros((1, 2, H), np.float32) for _ in range(L)]
    out = np.zeros((len(X), 2), np.float32); names = ["prediction"] + [f"h{l}_out" for l in range(L)]
    for i, r in enumerate(X):
        feed = {"features": r.reshape(1, 1, -1)}
        for l in range(L): feed[f"h{l}"] = st[l]
        res = s.run(names, feed); out[i] = res[0].reshape(-1)[:2]; st = [res[1 + l] for l in range(L)]
    return out, st


def bare_loop_us(path, n=20000):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    ins = {i.name: i for i in s.get_inputs()}; outs = [o.name for o in s.get_outputs()]
    feed = {nm: np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32) for nm, i in ins.items()}
    state_in = [nm for nm in ins if nm != "features"]; state_out = [o for o in outs if "pred" not in o.lower()]
    rows = np.random.default_rng(0).standard_normal((n, 112)).astype(np.float32); fshape = feed["features"].shape
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for r in rows:
            feed["features"] = r.reshape(fshape); res = s.run(outs, feed)
            for k, nm in enumerate(state_out): feed[state_in[k]] = res[outs.index(nm)]
        best = min(best, (time.perf_counter() - t0) / n * 1e6)
    return best


if __name__ == "__main__":
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a"); ap.add_argument("ckpt_b"); ap.add_argument("--out", required=True)
    ap.add_argument("--mults", default="0.5,0.5"); ap.add_argument("--time", default="")
    ap.add_argument("--bits", type=int, default=8); ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--lean", action="store_true", help="fold LN2 into head1 and stack the final head (-5 nodes)")
    a = ap.parse_args()
    from score_ckpt import load
    mults = tuple(float(v) for v in a.mults.split(","))
    ws, refs = [], []
    for ck in (a.ckpt_a, a.ckpt_b):
        tag, name = Path(ck).parent.name, Path(ck).name
        core, sc, cfg = load(tag, name); core = core.cpu().eval()
        ws.append(weights(core, sc, mults)); refs.append((core, torch.tensor([sc * mults[0], sc * mults[1]])))
    out = R/a.out; out.mkdir(parents=True, exist_ok=True)
    Xr = np.random.default_rng(0).standard_normal((400, 112)).astype(np.float32)
    ref = 0
    with torch.no_grad():
        for core, k in refs:
            o, _ = core(torch.from_numpy(Xr).reshape(len(Xr), 1, -1), core.init_state(1, torch.device("cpu")))
            o = o.reshape(len(Xr), -1); o = o if o.shape[1] >= 2 else torch.cat([o, -o], 1)
            ref = ref + (o[:, :2] * k).numpy() / 2
    for mode in ("fp32", "nbits"):
        m = build_pair(ws[0], ws[1], mode=mode, bits=a.bits, block=a.block, lean=a.lean); path = out/("model_fp32.onnx" if mode == "fp32" else "model.onnx"); onnx.save(m, str(path))
        got, _ = run_graph(path, Xr, ws[0]["hidden"], ws[0]["layers"]); d = np.abs(got - ref).max()
        print(f"{mode}: {len(m.graph.node)} nodes; vs mean of torch members on 400 rows: max |diff| {d:.2e}", flush=True)
        if mode == "fp32" and d > 1e-4: raise SystemExit("fused pair does not match the member average -- structural bug")
    print(f"bare loop: fused pair {bare_loop_us(out/'model.onnx'):.1f} us/row", flush=True)
    if a.time:
        import tempfile
        t = Path(tempfile.mkdtemp()); zipfile.ZipFile(a.time).extractall(t)
        for g in sorted(t.glob("*.onnx")): print(f"bare loop: {Path(a.time).name}/{g.name} {bare_loop_us(g):.1f} us/row", flush=True)
    (out/"solution.py").write_text(DRIVER)
    zp = R/"submissions"/f"submission_{out.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out/"solution.py", "solution.py"); z.write(out/"model.onnx", "model.onnx")
    print(f"packaged {zp} ({zp.stat().st_size//1024} KB)")
