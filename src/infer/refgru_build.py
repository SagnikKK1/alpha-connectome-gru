"""Row-at-a-time ONNX graph for RefGRUCore on the unrolled MatMulNBits path.

torch.onnx.export gives this architecture a 52-node graph with two fused fp32 `GRU` ops
that costs ~52 us/row (his own export of the same model: 47). The hand-built unrolled
int8 path prices a 200k-param GRU at 26 us/row, so the same treatment here should land
near 25-30. Structure, mirroring arch/refgru.py exactly:

    x -> (x-mu)/sd -> Clip(+-clamp)
      -> in_proj: MatMulNBits(+b) -> Gelu -> LayerNormalization            (width proj)
      -> unrolled GRU cells (MatMulNBits W_ih / W_hh), state h0, h1
      -> Concat(y, z) -> LayerNormalization
      -> head: MatMulNBits(+b) -> Gelu -> MatMul(2) + b                      (fp32, 2 columns)
    the checkpoint's train scale and the clip-aware per-channel multiples are folded
    into the last Linear.

    python src/infer/refgru_build.py runs/C_s801/final.pt --out submissions/RCP_C_s801_i8 --calib retrain
"""
from __future__ import annotations
import argparse, sys, zipfile
from pathlib import Path
import numpy as np, onnx
from onnx import helper, TensorProto

R = Path(__file__).resolve().parents[2]
for p in (str(R), str(R/"src"), str(R/"src"/"infer"), str(R/"src"/"arch")): sys.path.insert(0, p)
from onnx_build import _T, _io, _quant_nbits

DRIVER = '''"""Single graph, one session; state tensors discovered from the graph."""
from __future__ import annotations
import os
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
_ZERO2 = np.zeros(2, dtype=np.float32)


class PredictionModel:
    def __init__(self):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(str(next(HERE.glob("*.onnx"))), sess_options=so,
                                      providers=["CPUExecutionProvider"])
        ins, outs = self.s.get_inputs(), self.s.get_outputs()
        shp = lambda t: [d if isinstance(d, int) else 1 for d in t.shape]
        self.x = np.zeros(shp(next(i for i in ins if i.name == "features")), np.float32)
        self.xrow = self.x.reshape(-1)
        self.sin = [i.name for i in ins if i.name != "features"]
        self.sout = [o.name for o in outs if o.name != "prediction"]
        self.st = [np.zeros(shp(next(i for i in ins if i.name == n)), np.float32) for n in self.sin]
        self.want = ["prediction"] + self.sout
        self.seq = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq:
            self.seq = data_point.seq_ix
            for v in self.st: v[:] = 0.0
        self.xrow[:] = data_point.state
        feed = {"features": self.x}
        for n, v in zip(self.sin, self.st): feed[n] = v
        r = self.s.run(self.want, feed)
        for i, v in enumerate(r[1:]): self.st[i][:] = v
        if not data_point.need_prediction:      # state advances on every row; warm-up returns None
            return None
        p = r[0].reshape(-1)[:2]
        if not np.isfinite(p).all():
            for v in self.st: v[:] = 0.0
            return _ZERO2
        return p
'''


def weights(core, scale: float, mults=(1.0, 1.0)) -> dict:
    sd = core.state_dict(); f = lambda k: sd[k].detach().cpu().numpy().astype(np.float32)
    H, L = core.hidden, core.layers
    w = {"hidden": H, "layers": L, "clamp": float(core.clamp), "mu": f("mu"), "sd": f("sd"),
         "proj_W": f("in_proj.0.weight").T.copy(), "proj_b": f("in_proj.0.bias"),
         "ln1_g": f("in_proj.2.weight"), "ln1_b": f("in_proj.2.bias"),
         "ln2_g": f("post_ln.weight"), "ln2_b": f("post_ln.bias"),
         "h1_W": f("head.0.weight").T.copy(), "h1_b": f("head.0.bias"),
         "gru": [{"W_ih": f(f"gru.weight_ih_l{l}"), "W_hh": f(f"gru.weight_hh_l{l}"),
                  "b_ih": f(f"gru.bias_ih_l{l}"), "b_hh": f(f"gru.bias_hh_l{l}")} for l in range(L)]}
    W2, b2 = f("head.3.weight"), f("head.3.bias")            # (n_out, hh)
    if W2.shape[0] == 1:                                      # single signal -> (s, -s)
        W2 = np.concatenate([W2, -W2], 0); b2 = np.concatenate([b2, -b2])
    k = np.array([scale * mults[0], scale * mults[1]], np.float32)
    w["h2_W"] = (W2[:2] * k[:, None]).T.copy(); w["h2_b"] = (b2[:2] * k).astype(np.float32)
    w["proj"] = w["proj_W"].shape[1]
    # Extra projection blocks, if the model was trained with --proj-layers > 1. Each is another
    # Linear -> GELU -> LayerNorm at the same width, sitting between the first block and the GRU.
    # Input capacity is the axis that pays here, so these have to export or a win is unshippable.
    extra = []
    i = 3
    while ("in_proj.%d.weight" % i) in sd:
        extra.append({"W": f("in_proj.%d.weight" % i).T.copy(), "b": f("in_proj.%d.bias" % i),
                      "g": f("in_proj.%d.weight" % (i + 2)), "lb": f("in_proj.%d.bias" % (i + 2))})
        i += 3
    w["proj_extra"] = extra
    # extra causal state. Both of these are recurrences the GRU cannot express, so they carry
    # their own state slots and have to be built explicitly rather than folded into a matmul.
    # --- fold the book-depth convolution into the projection --------------------------------
    # The convolution is LINEAR in the clipped inputs, and the projection that consumes it is linear
    # too, so their composition is a single linear map of the original 112 columns. Materialising it
    # here costs the graph nothing: `--bookconv` is a structural prior on the projection matrix at
    # training time and is entirely free at inference -- no extra nodes, no extra state, no extra
    # microseconds. The guard that used to refuse these models is therefore unnecessary.
    _bc = int(getattr(core, "bookconv", 0) or 0)
    if _bc:
        bw = f("book_w")                                   # (3, 4, C)
        bb = f("book_b")                                   # (C,)
        nb = 2 * _bc * 11
        A = np.zeros((112, nb), np.float32)                # book features as a linear map of x
        c0 = np.zeros(nb, np.float32)
        for inst, base in enumerate((0, 52)):
            for ch in range(4):
                for l in range(11):
                    for j in range(3):
                        src = l + j - 1
                        if 0 <= src <= 10:
                            col = base + ch * 11 + src
                            for o in range(_bc):
                                A[col, inst * _bc * 11 + o * 11 + l] += bw[j, ch, o]
            for o in range(_bc):
                c0[inst * _bc * 11 + o * 11:inst * _bc * 11 + (o + 1) * 11] = bb[o]
        P_ = w["proj_W"]                                   # (n_in, proj), raw columns first
        W_raw, W_book = P_[:112], P_[112:112 + nb]
        w["proj_W"] = (W_raw + A @ W_book).astype(np.float32)
        w["proj_b"] = (w["proj_b"] + c0 @ W_book).astype(np.float32)
    w["adanorm"] = float(getattr(core, "adanorm", 0.0) or 0.0)
    w["ewma"] = tuple(float(v) for v in getattr(core, "ewma", ()) or ())
    w["fastw"] = int(getattr(core, "fastw", 0) or 0)
    w["fw_to_gru"] = bool(getattr(core, "fw_to_gru", False))
    if w["fastw"]:
        w["fw_q"] = f("fw_q.weight").T.copy(); w["fw_k"] = f("fw_k.weight").T.copy()
        w["fw_v"] = f("fw_v.weight").T.copy()
        lam = 1.0 / (1.0 + np.exp(-f("fw_decay")))                       # sigmoid, one per group
        w["fw_lam"] = np.repeat(lam, w["fastw"] // core.fw_groups).astype(np.float32)
    w["quad"] = bool(getattr(core, "quad", False))
    for attr in ("diff", "posfeat", "split"):
        if getattr(core, attr, False):
            raise SystemExit("refgru_build has no path for --%s yet" % attr)
    if getattr(core, "nlfeat", ()):
        raise SystemExit("refgru_build has no path for --nlfeat yet")
    return w


def build(w: dict, mode: str = "nbits", bits: int = 8, block: int = 128, ln_eps: float = 1e-5):
    H, L, P = w["hidden"], w["layers"], w["proj"]
    nodes, inits = [], []
    inputs = [_io("features", [1, 1, 112])]; outputs = [_io("prediction", [1, 1, 2])]
    ms = mode == "nbits"

    def matmul(x, W, name, bias=None):
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

    def gelu(x, name):
        if ms:
            nodes.append(helper.make_node("Gelu", [x], [f"{name}_g"], domain="com.microsoft", name=f"{name}_gelu"))
        else:   # exact erf GELU in plain ONNX
            inits.append(_T(f"{name}_c", np.array(1/np.sqrt(2), np.float32)))
            inits.append(_T(f"{name}_half", np.array(0.5, np.float32)))
            inits.append(_T(f"{name}_one", np.array(1.0, np.float32)))
            nodes.append(helper.make_node("Mul", [x, f"{name}_c"], [f"{name}_xc"]))
            nodes.append(helper.make_node("Erf", [f"{name}_xc"], [f"{name}_erf"]))
            nodes.append(helper.make_node("Add", [f"{name}_erf", f"{name}_one"], [f"{name}_1e"]))
            nodes.append(helper.make_node("Mul", [x, f"{name}_1e"], [f"{name}_x1e"]))
            nodes.append(helper.make_node("Mul", [f"{name}_x1e", f"{name}_half"], [f"{name}_g"]))
        return f"{name}_g"

    def ln(x, g, b, name):
        inits.extend([_T(f"{name}_g", g), _T(f"{name}_b", b)])
        nodes.append(helper.make_node("LayerNormalization", [x, f"{name}_g", f"{name}_b"], [f"{name}_y"],
                                      axis=-1, epsilon=ln_eps, name=name))
        return f"{name}_y"

    inv = (1.0 / w["sd"]).astype(np.float32)
    inits.extend([_T("norm_mul", inv), _T("norm_add", (-w["mu"] * inv).astype(np.float32)),
                  _T("clip_lo", np.array(-w["clamp"], np.float32)), _T("clip_hi", np.array(w["clamp"], np.float32))])
    nodes.append(helper.make_node("Mul", ["features", "norm_mul"], ["x_scaled"]))
    nodes.append(helper.make_node("Add", ["x_scaled", "norm_add"], ["x_norm"]))
    nodes.append(helper.make_node("Clip", ["x_norm", "clip_lo", "clip_hi"], ["x_clip"]))
    # EWMA inputs: e_t = e_{t-1} + alpha*(x_t - e_{t-1}), one state slot per half-life, appended to
    # the projection input in the same order arch/refgru.py appends them.
    # adanorm: per-column running standardisation, on the SAME surface arch/refgru.py uses --
    #     m += al*(x - m);  v += al*((x - m)^2 - v);  out = (x - m)/sqrt(v + 1)
    # with al = 1 - 2^(-1/half_life). The +1 keeps an all-quiet column from being amplified.
    # Two state slots, all elementwise: the inference graph is streaming per row, so unlike the
    # training loop there is no scan to unroll. Note v is updated with the ALREADY-UPDATED m, which
    # is what refgru does; using the previous m here would be a silently different model.
    xsrc = "x_clip"
    if w.get("adanorm"):
        al = np.float32(1.0 - 2.0 ** (-1.0 / w["adanorm"]))
        inputs += [_io("ada_m", [1, 1, 112]), _io("ada_v", [1, 1, 112])]
        outputs += [_io("ada_m_out", [1, 1, 112]), _io("ada_v_out", [1, 1, 112])]
        inits.extend([_T("ada_a", np.array(al, np.float32)), _T("ada_one", np.array(1.0, np.float32))])
        nodes.append(helper.make_node("Sub", ["x_clip", "ada_m"], ["ada_dm"]))
        nodes.append(helper.make_node("Mul", ["ada_dm", "ada_a"], ["ada_dma"]))
        nodes.append(helper.make_node("Add", ["ada_m", "ada_dma"], ["ada_m_out"]))
        nodes.append(helper.make_node("Sub", ["x_clip", "ada_m_out"], ["ada_d"]))
        nodes.append(helper.make_node("Mul", ["ada_d", "ada_d"], ["ada_d2"]))
        nodes.append(helper.make_node("Sub", ["ada_d2", "ada_v"], ["ada_vd"]))
        nodes.append(helper.make_node("Mul", ["ada_vd", "ada_a"], ["ada_vda"]))
        nodes.append(helper.make_node("Add", ["ada_v", "ada_vda"], ["ada_v_out"]))
        nodes.append(helper.make_node("Add", ["ada_v_out", "ada_one"], ["ada_v1"]))
        nodes.append(helper.make_node("Sqrt", ["ada_v1"], ["ada_s"]))
        nodes.append(helper.make_node("Div", ["ada_d", "ada_s"], ["x_ada"]))
        xsrc = "x_ada"
    proj_in = xsrc
    if w.get("ewma"):
        parts = [xsrc]
        for j, hl in enumerate(w["ewma"]):
            al = np.float32(1.0 - 2.0 ** (-1.0 / hl))
            inputs.append(_io(f"ew{j}", [1, 1, 112])); outputs.append(_io(f"ew{j}_out", [1, 1, 112]))
            inits.append(_T(f"ew{j}_a", np.array(al, np.float32)))
            nodes.append(helper.make_node("Sub", [xsrc, f"ew{j}"], [f"ew{j}_d"]))
            nodes.append(helper.make_node("Mul", [f"ew{j}_d", f"ew{j}_a"], [f"ew{j}_ad"]))
            nodes.append(helper.make_node("Add", [f"ew{j}", f"ew{j}_ad"], [f"ew{j}_out"]))
            parts.append(f"ew{j}_out")
        nodes.append(helper.make_node("Concat", parts, ["proj_in"], axis=2))
        proj_in = "proj_in"
    z = ln(gelu(matmul(proj_in, w["proj_W"], "proj", w["proj_b"]), "proj"), w["ln1_g"], w["ln1_b"], "ln1")
    for _i, blk in enumerate(w.get("proj_extra", [])):
        nm = "projx%d" % _i
        z = ln(gelu(matmul(z, blk["W"], nm, blk["b"]), nm), blk["g"], blk["lb"], nm + "_ln")
    # the memory only depends on z and its own state, so it is emitted before the recurrence;
    # with --fastw-gru its readout joins the GRU input, otherwise it joins the head input
    if w.get("fastw"):
        D = w["fastw"]
        inputs += [_io("zprev", [1, 1, P]), _io("fwA", [1, D, D]), _io("fwN", [1, 1, D])]
        outputs += [_io("zprev_out", [1, 1, P]), _io("fwA_out", [1, D, D]), _io("fwN_out", [1, 1, D])]
        nodes.append(helper.make_node("Identity", [z], ["zprev_out"]))
        # q and v read the current row, k reads the previous one, so the memory accumulates the
        # (previous -> current) map of THIS sequence. phi = elu + 1 keeps the keys positive.
        inits.append(_T("fw_one", np.array(1.0, np.float32)))
        qv = matmul(z, np.concatenate([w["fw_q"], w["fw_v"]], 1), "fw_qv")
        nodes.append(helper.make_node("Split", [qv], ["fw_qraw", "fw_v"], axis=2, num_outputs=2))
        nodes.append(helper.make_node("Elu", ["fw_qraw"], ["fw_qe"], alpha=1.0))
        nodes.append(helper.make_node("Add", ["fw_qe", "fw_one"], ["fw_q"]))
        kraw = matmul("zprev", w["fw_k"], "fw_kk")
        nodes.append(helper.make_node("Elu", [kraw], ["fw_ke"], alpha=1.0))
        nodes.append(helper.make_node("Add", ["fw_ke", "fw_one"], ["fw_k"]))
        # A_out = lam_row * A + k^T (x) v ; lam decays each KEY row of the memory at its own rate
        inits.append(_T("fw_lam_col", w["fw_lam"].reshape(1, D, 1)))
        inits.append(_T("fw_lam_row", w["fw_lam"].reshape(1, 1, D)))
        inits.append(_T("fw_kshape", np.array([1, D, 1], np.int64)))
        nodes.append(helper.make_node("Reshape", ["fw_k", "fw_kshape"], ["fw_kcol"]))
        nodes.append(helper.make_node("Mul", ["fwA", "fw_lam_col"], ["fw_Ad"]))
        nodes.append(helper.make_node("Mul", ["fw_kcol", "fw_v"], ["fw_kv"]))
        nodes.append(helper.make_node("Add", ["fw_Ad", "fw_kv"], ["fwA_out"]))
        nodes.append(helper.make_node("Mul", ["fwN", "fw_lam_row"], ["fw_Nd"]))
        nodes.append(helper.make_node("Add", ["fw_Nd", "fw_k"], ["fwN_out"]))
        nodes.append(helper.make_node("MatMul", ["fw_q", "fwA_out"], ["fw_num"]))
        nodes.append(helper.make_node("Mul", ["fw_q", "fwN_out"], ["fw_qn"]))
        nodes.append(helper.make_node("ReduceSum", ["fw_qn", "fw_axm1"], ["fw_den0"], keepdims=1))
        inits.append(_T("fw_axm1", np.array([-1], np.int64)))
        inits.append(_T("fw_dmin", np.array(1e-4, np.float32)))
        nodes.append(helper.make_node("Max", ["fw_den0", "fw_dmin"], ["fw_den"]))
        nodes.append(helper.make_node("Div", ["fw_num", "fw_den"], ["fw_r"]))
    gru_in = z
    if w.get("quad"):
        # level x change in the projection basis: the input path is Linear -> GELU -> LayerNorm, so
        # the only products the model can otherwise form are whatever one narrow layer approximates.
        # One state slot holds the previous projection; the GRU input doubles to 2*proj.
        inputs.append(_io("zprev_q", [1, 1, P])); outputs.append(_io("zprev_q_out", [1, 1, P]))
        nodes.append(helper.make_node("Identity", [z], ["zprev_q_out"]))
        nodes.append(helper.make_node("Sub", [z, "zprev_q"], ["q_dz"]))
        nodes.append(helper.make_node("Mul", [z, "q_dz"], ["q_zq"]))
        nodes.append(helper.make_node("Concat", [z, "q_zq"], ["q_gru_in"], axis=2))
        gru_in = "q_gru_in"
    if w.get("fastw") and w.get("fw_to_gru"):
        nodes.append(helper.make_node("Concat", [gru_in, "fw_r"], ["fw_gru_in"], axis=2))
        gru_in = "fw_gru_in"
    x = gru_in
    for l, g in enumerate(w["gru"]):
        p = f"l{l}_"; h = f"h{l}"
        inputs.append(_io(f"h{l}", [1, 1, H])); outputs.append(_io(f"h{l}_out", [1, 1, H]))
        gi = matmul(x, np.ascontiguousarray(g["W_ih"].T), p + "mm_ih", g["b_ih"])
        gh = matmul(h, np.ascontiguousarray(g["W_hh"].T), p + "mm_hh", g["b_hh"])
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
        nodes.append(helper.make_node("Add", [p + "n", p + "zhmn"], [f"h{l}_out"]))
        x = f"h{l}_out"
    head_parts = [x, z] + (["fw_r"] if (w.get("fastw") and not w.get("fw_to_gru")) else [])
    nodes.append(helper.make_node("Concat", head_parts, ["cat"], axis=2))
    c = ln("cat", w["ln2_g"], w["ln2_b"], "ln2")
    hdn = gelu(matmul(c, w["h1_W"], "head1", w["h1_b"]), "head1")
    inits.extend([_T("h2_W", w["h2_W"]), _T("h2_b", w["h2_b"])])
    nodes.append(helper.make_node("MatMul", [hdn, "h2_W"], ["h2_mm"], name="head2"))
    nodes.append(helper.make_node("Add", ["h2_mm", "h2_b"], ["prediction"]))
    graph = helper.make_graph(nodes, "refgru_step", inputs, outputs, initializer=inits)
    imports = [helper.make_opsetid("", 18)] + ([helper.make_opsetid("com.microsoft", 1)] if ms else [])
    m = helper.make_model(graph, opset_imports=imports, producer_name="alpha-connectome")
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def run_graph(path, X, H, L):
    """Row-by-row reference run. State tensors are read off the graph, not assumed to be h0..hL:
    the EWMA and fast-weight paths add slots of their own and a hard-coded list silently drops them."""
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    shp = lambda t: [d if isinstance(d, int) else 1 for d in t.shape]
    sin = [i.name for i in s.get_inputs() if i.name != "features"]
    sout = [o.name for o in s.get_outputs() if o.name != "prediction"]
    st = [np.zeros(shp(next(i for i in s.get_inputs() if i.name == n)), np.float32) for n in sin]
    names = ["prediction"] + sout
    out = np.zeros((len(X), 2), np.float32)
    for t, r in enumerate(X):
        feed = {"features": r.reshape(1, 1, -1)}
        feed.update(dict(zip(sin, st)))
        res = s.run(names, feed); out[t] = res[0].reshape(-1); st = list(res[1:])
    return out, dict(zip(sout, st))


if __name__ == "__main__":
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=8); ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--calib", choices=("ckpt", "retrain"), default="retrain")
    ap.add_argument("--check-seqs", type=int, default=8, help="real holdout sequences for the WP parity check")
    ap.add_argument("--calib-fams", default="", help="restrict the clip-aware calibration to these tick families, e.g. 0 or 1,2,3")
    a = ap.parse_args()
    from score_ckpt import load
    tag, name = Path(a.ckpt).parent.name, Path(a.ckpt).name
    core, sc, cfg = load(tag, name); core = core.cpu().eval()
    mults = (1.0, 1.0)
    if a.calib == "retrain":
        import clipcal
        fams = [int(v) for v in a.calib_fams.split(",") if v] or None
        mults = clipcal.calib_ckpt(tag, name, fams=fams)
    w = weights(core, sc, mults)
    out = R/a.out; out.mkdir(parents=True, exist_ok=True)

    # torch reference, row by row through the same scale/multiples
    def torch_ref(X):
        k = torch.tensor([sc * mults[0], sc * mults[1]])
        with torch.no_grad():
            o, h = core(torch.from_numpy(X).reshape(len(X), 1, -1), core.init_state(1, torch.device("cpu")))
        o = o.reshape(len(X), -1)
        if o.shape[1] == 1: o = torch.cat([o, -o], 1)
        return (o[:, :2] * k).numpy(), h.numpy()

    Xr = np.random.default_rng(0).standard_normal((400, 112)).astype(np.float32)
    for mode in ("fp32", "nbits"):
        m = build(w, mode=mode, bits=a.bits, block=a.block)
        path = out/("model_fp32.onnx" if mode == "fp32" else "model.onnx"); onnx.save(m, str(path))
        got, st = run_graph(path, Xr, w["hidden"], w["layers"])
        ref, href = torch_ref(Xr)
        # the recurrent slots are looked up BY NAME: the fast-weight block adds state inputs
        # ahead of h0 in the graph, so positional indexing silently compares the wrong tensors
        d = np.abs(got - ref).max()
        ds = max(np.abs(st[f"h{l}_out"].reshape(-1) - href[l].reshape(-1)).max() for l in range(w["layers"]))
        print(f"{mode}: {len(m.graph.node)} nodes; vs torch on 400 random rows: max |diff| pred {d:.2e}, state {ds:.2e}")
        if mode == "fp32" and (d > 1e-4 or ds > 1e-4):
            raise SystemExit("fp32 unrolled graph does not match torch -- structural bug")
    # WP parity on real sequences: int8 graph vs fp32 torch, per sequence
    from data import SequenceCache
    from utils import weighted_pearson, WARMUP, SEQUENCE_LENGTH
    ho = SequenceCache(R/"cache/holdout"); steps = np.arange(SEQUENCE_LENGTH); diffs = []
    for i in range(a.check_seqs):
        X = np.asarray(ho.X[i], np.float32); m_ = np.asarray(ho.M[i]) & (steps >= WARMUP); y = np.asarray(ho.Y[i])
        g, _ = run_graph(out/"model.onnx", X, w["hidden"], w["layers"]); r, _ = torch_ref(X)
        wg = 0.5*(weighted_pearson(y[m_, 0], g[m_, 0]) + weighted_pearson(y[m_, 1], g[m_, 1]))
        wr = 0.5*(weighted_pearson(y[m_, 0], r[m_, 0]) + weighted_pearson(y[m_, 1], r[m_, 1]))
        diffs.append(wg - wr)
    print(f"int8 graph vs fp32 torch WP on {a.check_seqs} holdout sequences: mean diff {np.mean(diffs):+.5f}, "
          f"max |diff| {np.abs(diffs).max():.5f}")
    (out/"solution.py").write_text(DRIVER)
    zp = R/"submissions"/f"submission_{out.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out/"solution.py", "solution.py"); z.write(out/"model.onnx", "model.onnx")
    print(f"packaged {zp} ({zp.stat().st_size//1024} KB)")
