"""Build row-at-a-time ONNX graphs for a stacked GRU + linear head.

The weights are plain numpy arrays (see `random_weights` / `weights_from_core`)
in PyTorch layout (gate order r, z, n; linear_before_reset semantics).  Input
standardisation and the (s, -s) output pair are folded into the first-layer
input weights and the head, so every graph is: features (1,1,112) + one state
tensor per layer -> prediction (1,1,2) + next states.

Graph variants:
  fused     : one ONNX `GRU` node per layer (what torch.onnx.export emits).
  unrolled  : explicit MatMul/Sigmoid/Tanh cell; the MatMuls can then be
              replaced by weight-only quantised `MatMulNBits` (4/8-bit) or
              dynamic int8 `MatMulInteger`.
"""
from __future__ import annotations
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

N_FEATURES = 112


# --------------------------------------------------------------------------- weights
def random_weights(hidden: int, layers: int, n_in: int = N_FEATURES, n_out: int = 1,
                   seed: int = 0, out_scale: float = 1.0) -> dict:
    """PyTorch-shaped random GRU weights, uniform(-1/sqrt(H), 1/sqrt(H)) like nn.GRU."""
    rng = np.random.default_rng(seed)
    k = 1.0 / np.sqrt(hidden)
    u = lambda *s: rng.uniform(-k, k, size=s).astype(np.float32)
    w = {"hidden": hidden, "layers": layers, "n_in": n_in, "n_out": n_out,
         "mu": np.zeros(n_in, np.float32), "sd": np.ones(n_in, np.float32),
         "out_scale": float(out_scale), "gru": []}
    for l in range(layers):
        i = n_in if l == 0 else hidden
        w["gru"].append({"W_ih": u(3 * hidden, i), "W_hh": u(3 * hidden, hidden),
                         "b_ih": u(3 * hidden), "b_hh": u(3 * hidden)})
    w["head_W"] = (rng.normal(0, 0.02, size=(n_out, hidden))).astype(np.float32)
    w["head_b"] = np.zeros(n_out, np.float32)
    return w


def weights_from_core(core, out_scale: float = 1.0) -> dict:
    """Extract numpy weights from src/model.py's GRUCore (torch imported lazily)."""
    sd = core.state_dict()
    f = lambda k: sd[k].detach().cpu().numpy().astype(np.float32)
    w = {"hidden": core.hidden, "layers": core.layers, "n_in": int(sd["mu"].numel()),
         "n_out": core.n_out, "mu": f("mu"), "sd": f("sd"), "out_scale": float(out_scale), "gru": []}
    for l in range(core.layers):
        w["gru"].append({"W_ih": f(f"gru.weight_ih_l{l}"), "W_hh": f(f"gru.weight_hh_l{l}"),
                         "b_ih": f(f"gru.bias_ih_l{l}"), "b_hh": f(f"gru.bias_hh_l{l}")})
    w["head_W"] = f("head.weight"); w["head_b"] = f("head.bias")
    # Optional bottleneck between the recurrence and the head. Without this the exporter
    # would read only head.*, whose input width is `proj` while the GRU emits `hidden` --
    # a silent shape error, or worse a silently wrong graph.
    if getattr(core, "proj", None) is not None:
        w["proj_W"] = f("proj.weight"); w["proj_b"] = f("proj.bias")
    return w


def fold(w: dict, keep_norm: bool = False) -> dict:
    """Fold (x-mu)/sd into layer-0 input weights and (s,-s)*scale into the head.

    Returns a dict with per-layer W_ih (3H,in), W_hh (3H,H), b_ih, b_hh (3H)
    [PyTorch gate order r,z,n] and head_W (H,2), head_b (2) so that
    pred = h_last @ head_W + head_b.

    keep_norm=True leaves layer 0 untouched and returns "norm_mul"/"norm_add"
    (x_norm = x*norm_mul + norm_add) for the graph to apply explicitly.  Needed
    for the int8-activation graphs: MatMulNBits quantises its input per row, so
    it must see the standardised x, not raw features of wildly different scale.
    """
    inv = (1.0 / w["sd"]).astype(np.float32)
    layers = []
    for l, g in enumerate(w["gru"]):
        W_ih, b_ih = g["W_ih"], g["b_ih"]
        if l == 0 and not keep_norm:
            W_ih = (W_ih * inv[None, :]).astype(np.float32)
            b_ih = (b_ih - (g["W_ih"] * inv[None, :]) @ w["mu"]).astype(np.float32)
        layers.append({"W_ih": W_ih, "W_hh": g["W_hh"], "b_ih": b_ih, "b_hh": g["b_hh"]})
    s = np.float32(w["out_scale"])
    hw, hb = w["head_W"] * s, w["head_b"] * s     # (n_out, P) if projected else (n_out,H)
    if w["n_out"] == 1:
        head_W = np.concatenate([hw[0:1], -hw[0:1]], 0).T     # (H,2)
        head_b = np.array([hb[0], -hb[0]], np.float32)
    else:
        head_W = hw[:2].T.copy(); head_b = hb[:2].copy()
    out = {"hidden": w["hidden"], "layers": w["layers"], "n_in": w["n_in"],
           "gru": layers, "head_W": np.ascontiguousarray(head_W, np.float32),
           "head_b": head_b.astype(np.float32)}
    if "proj_W" in w:
        # the output scale is folded into the head, so proj passes through unscaled
        out["proj_W"] = np.ascontiguousarray(w["proj_W"].T, np.float32)   # (H, P)
        out["proj_b"] = w["proj_b"].astype(np.float32)
    if keep_norm:
        out["norm_mul"] = inv; out["norm_add"] = (-w["mu"] * inv).astype(np.float32)
    return out


def n_params(w: dict) -> int:
    fw = fold(w) if "mu" in w else w
    n = fw["head_W"].size + fw["head_b"].size
    for g in fw["gru"]:
        n += sum(g[k].size for k in ("W_ih", "W_hh", "b_ih", "b_hh"))
    return int(n)


# --------------------------------------------------------------------------- helpers
def _T(name, arr):
    return numpy_helper.from_array(np.ascontiguousarray(arr), name)


def _io(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, inits, opset=17, ms=False):
    graph = helper.make_graph(nodes, "gru_step", inputs, outputs, initializer=inits)
    imports = [helper.make_opsetid("", opset)]
    if ms:
        imports.append(helper.make_opsetid("com.microsoft", 1))
    m = helper.make_model(graph, opset_imports=imports, producer_name="alpha-connectome")
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def _rzn_to_zrn(a, H, axis=0):
    """PyTorch (r,z,n) -> ONNX (z,r,n) gate order along `axis`."""
    r, z, n = np.split(a, 3, axis=axis)
    return np.concatenate([z, r, n], axis=axis)


# --------------------------------------------------------------------------- fused GRU
def build_fused(w: dict) -> onnx.ModelProto:
    fw = fold(w); H, L = fw["hidden"], fw["layers"]
    nodes, inits = [], []
    x = "features"
    inputs = [_io("features", [1, 1, fw["n_in"]])]
    outputs = [_io("prediction", [1, 1, 2])]
    for l, g in enumerate(fw["gru"]):
        W = _rzn_to_zrn(g["W_ih"], H)[None]            # (1,3H,in)
        R = _rzn_to_zrn(g["W_hh"], H)[None]            # (1,3H,H)
        B = np.concatenate([_rzn_to_zrn(g["b_ih"], H), _rzn_to_zrn(g["b_hh"], H)])[None]  # (1,6H)
        inits += [_T(f"W{l}", W), _T(f"R{l}", R), _T(f"B{l}", B)]
        inputs.append(_io(f"h{l}", [1, 1, H])); outputs.append(_io(f"h{l}_out", [1, 1, H]))
        nodes.append(helper.make_node("GRU", [x, f"W{l}", f"R{l}", f"B{l}", "", f"h{l}"],
                                      [f"Y{l}", f"h{l}_out"], hidden_size=H,
                                      linear_before_reset=1, name=f"gru{l}"))
        x = f"h{l}_out"                                  # (1,1,H) == Y[0] since seq=1
    if "proj_W" in fw:
        inits += [_T("projW", fw["proj_W"]), _T("projb", fw["proj_b"])]
        nodes.append(helper.make_node("MatMul", [x, "projW"], ["pm"], name="proj_mm"))
        nodes.append(helper.make_node("Add", ["pm", "projb"], ["pa"], name="proj_add"))
        nodes.append(helper.make_node("Tanh", ["pa"], ["pj"], name="proj_act"))
        x = "pj"
    inits += [_T("headW", fw["head_W"]), _T("headb", fw["head_b"])]
    nodes.append(helper.make_node("MatMul", [x, "headW"], ["hm"], name="head_mm"))
    nodes.append(helper.make_node("Add", ["hm", "headb"], ["prediction"], name="head_add"))
    return _model(nodes, inputs, outputs, inits)


# --------------------------------------------------------------------------- unrolled cell
def _quant_nbits(W: np.ndarray, bits: int, block: int, name: str):
    """Blockwise symmetric weight-only quantisation for com.microsoft MatMulNBits.

    W is (K,N) (activation @ W).  Returns (nodes-inputs, initializers, dequantised W)
    following the MatMulNBits packing: B is (N, n_blocks, blob) uint8 where each
    block of `block` weights along K is quantised with its own scale (and zero point 2^(bits-1)).
    """
    K, N = W.shape
    nb = (K + block - 1) // block
    Kp = nb * block
    Wp = np.zeros((Kp, N), np.float32); Wp[:K] = W
    Wb = Wp.reshape(nb, block, N)                        # (nb, block, N)
    qmax = 2 ** (bits - 1) - 1                           # symmetric: 7 or 127
    amax = np.abs(Wb).max(axis=1)                        # (nb, N)
    scale = (amax / qmax).astype(np.float32); scale[scale == 0] = 1.0
    q = np.clip(np.rint(Wb / scale[:, None, :]), -qmax - 1, qmax).astype(np.int32)
    zp = 2 ** (bits - 1)
    u = (q + zp).astype(np.uint8)                        # unsigned with zero point 2^(bits-1)
    deq = ((u.astype(np.float32) - zp) * scale[:, None, :]).reshape(Kp, N)[:K]
    u = np.transpose(u, (2, 0, 1))                       # (N, nb, block)
    if bits == 4:
        lo, hi = u[..., 0::2], u[..., 1::2]              # pack pairs: low nibble first
        blob = (lo | (hi << 4)).astype(np.uint8)         # (N, nb, block/2)
    elif bits == 8:
        blob = u.astype(np.uint8)                        # (N, nb, block)
    else:
        raise ValueError(bits)
    inits = [_T(f"{name}_q", blob), _T(f"{name}_s", np.transpose(scale, (1, 0)).reshape(-1))]
    return inits, deq.astype(np.float32)


def build_unrolled(w: dict, mode: str = "fp32", bits: int = 4, block: int = 32,
                   explicit_norm: bool | None = None) -> onnx.ModelProto:
    """Explicit GRU cell.  mode: fp32 | nbits (MatMulNBits, needs `bits`,`block`) | int8dyn.

    explicit_norm: apply (x-mu)/sd as Mul+Add nodes instead of folding it into
    W_ih of layer 0.  Defaults to True for `nbits` (int8 activations need the
    standardised input) and False otherwise (folding is free in fp32).
    For `int8dyn` the fp32 graph is returned; quantise afterwards with
    onnxruntime.quantization.quantize_dynamic (it targets the MatMul nodes).
    """
    if explicit_norm is None:
        explicit_norm = mode == "nbits"
    fw = fold(w, keep_norm=explicit_norm); H, L = fw["hidden"], fw["layers"]
    nodes, inits = [], []
    inputs = [_io("features", [1, 1, fw["n_in"]])]
    outputs = [_io("prediction", [1, 1, 2])]
    ms = mode == "nbits"

    def matmul(x, W, name, bias=None):                 # W (K,N), x (1,1,K); returns name of x@W(+bias)
        if mode == "nbits":
            K, N = W.shape
            qi, _ = _quant_nbits(W, bits, block, name)
            inits.extend(qi)
            ins = [x, f"{name}_q", f"{name}_s"]
            if bias is not None:                          # MatMulNBits has an optional bias input: saves an Add node
                inits.append(_T(f"{name}_b", bias)); ins += ["", "", f"{name}_b"]
            nodes.append(helper.make_node("MatMulNBits", ins, [f"{name}_y"],
                                          domain="com.microsoft", K=K, N=N, bits=bits,
                                          block_size=block, accuracy_level=4, name=name))
            return f"{name}_y"
        inits.append(_T(f"{name}_W", W))
        nodes.append(helper.make_node("MatMul", [x, f"{name}_W"], [f"{name}_mm"], name=name))
        if bias is None:
            return f"{name}_mm"
        inits.append(_T(f"{name}_b", bias))
        nodes.append(helper.make_node("Add", [f"{name}_mm", f"{name}_b"], [f"{name}_y"]))
        return f"{name}_y"

    x = "features"
    if explicit_norm:
        inits += [_T("norm_mul", fw["norm_mul"]), _T("norm_add", fw["norm_add"])]
        nodes.append(helper.make_node("Mul", ["features", "norm_mul"], ["x_scaled"]))
        nodes.append(helper.make_node("Add", ["x_scaled", "norm_add"], ["x_norm"]))
        x = "x_norm"
    for l, g in enumerate(fw["gru"]):
        p = f"l{l}_"
        inputs.append(_io(f"h{l}", [1, 1, H])); outputs.append(_io(f"h{l}_out", [1, 1, H]))
        h = f"h{l}"
        Wih_T = np.ascontiguousarray(g["W_ih"].T)      # (in,3H)
        Whh_T = np.ascontiguousarray(g["W_hh"].T)      # (H,3H)
        gi = matmul(x, Wih_T, p + "mm_ih", g["b_ih"]); gh = matmul(h, Whh_T, p + "mm_hh", g["b_hh"])
        nodes.append(helper.make_node("Split", [gi], [p + "gi_r", p + "gi_z", p + "gi_n"], axis=2, num_outputs=3))
        nodes.append(helper.make_node("Split", [gh], [p + "gh_r", p + "gh_z", p + "gh_n"], axis=2, num_outputs=3))
        nodes.append(helper.make_node("Add", [p + "gi_r", p + "gh_r"], [p + "r_pre"]))
        nodes.append(helper.make_node("Sigmoid", [p + "r_pre"], [p + "r"]))
        nodes.append(helper.make_node("Add", [p + "gi_z", p + "gh_z"], [p + "z_pre"]))
        nodes.append(helper.make_node("Sigmoid", [p + "z_pre"], [p + "z"]))
        nodes.append(helper.make_node("Mul", [p + "r", p + "gh_n"], [p + "rgh"]))
        nodes.append(helper.make_node("Add", [p + "gi_n", p + "rgh"], [p + "n_pre"]))
        nodes.append(helper.make_node("Tanh", [p + "n_pre"], [p + "n"]))
        # h' = n + z*(h - n)
        nodes.append(helper.make_node("Sub", [h, p + "n"], [p + "hmn"]))
        nodes.append(helper.make_node("Mul", [p + "z", p + "hmn"], [p + "zhmn"]))
        nodes.append(helper.make_node("Add", [p + "n", p + "zhmn"], [f"h{l}_out"]))
        x = f"h{l}_out"
    # head stays fp32 (2 columns; quantising it saves nothing) and writes the output directly
    if "proj_W" in fw:
        inits += [_T("projW", fw["proj_W"]), _T("projb", fw["proj_b"])]
        nodes.append(helper.make_node("MatMul", [x, "projW"], ["pm"], name="proj_mm"))
        nodes.append(helper.make_node("Add", ["pm", "projb"], ["pa"], name="proj_add"))
        nodes.append(helper.make_node("Tanh", ["pa"], ["pj"], name="proj_act"))
        x = "pj"
    inits += [_T("headW", fw["head_W"]), _T("headb", fw["head_b"])]
    nodes.append(helper.make_node("MatMul", [x, "headW"], ["head_mm"], name="head"))
    nodes.append(helper.make_node("Add", ["head_mm", "headb"], ["prediction"]))
    return _model(nodes, inputs, outputs, inits, opset=18, ms=ms)


def dequantised_weights(w: dict, bits: int, block: int, keep_norm: bool = False) -> dict:
    """The folded weights as the nbits graph actually sees them (for reference checks)."""
    fw = fold(w, keep_norm=keep_norm)
    out = {"hidden": fw["hidden"], "layers": fw["layers"], "n_in": fw["n_in"], "gru": [],
           "head_W": fw["head_W"], "head_b": fw["head_b"]}
    for g in fw["gru"]:
        out["gru"].append({"W_ih": _quant_nbits(np.ascontiguousarray(g["W_ih"].T), bits, block, "a")[1].T.copy(),
                           "W_hh": _quant_nbits(np.ascontiguousarray(g["W_hh"].T), bits, block, "b")[1].T.copy(),
                           "b_ih": g["b_ih"], "b_hh": g["b_hh"]})
    return out


def save(m: onnx.ModelProto, path):
    onnx.save(m, str(path))
    return path
