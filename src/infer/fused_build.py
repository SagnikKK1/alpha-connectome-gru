"""Fused N-member refgru export: ONE graph whose activations carry a member axis of size N.

Generalises src/infer/pair_build.py (which is hardcoded to exactly 2) after the 16 Sep result
that "ensembling saturates at 2 members" was a RUNTIME artifact, not a statistical one: on
pooled-800, a 4-member ensemble beats v37's pair by +0.00097 [+0.0005, +0.0015], and the
ordering 4 > 5 > 3 > 2 holds on a holdout-500 / eval-300 selection-validation split.

Every elementwise op (Split/Sigmoid/Tanh/Add/Mul/Sub/Gelu/LayerNorm) runs ONCE on tensors
shaped (1, N, D); only the weight matmuls are per member (Split -> N x MatMulNBits -> Concat).
The N heads are averaged by a ReduceMean over the member axis. LayerNorm gains/biases differ
per member, so LN runs with unit scale and an (N, D) Mul/Add follows.

    python src/infer/fused_build.py runs/A/final.pt runs/B/final.pt runs/C/final.pt runs/D/final.pt \
        --out submissions/FUSED4 --mults 0.5,0.5 [--time submissions/submission_v40.zip]

`--mults M0,M1` are the per-TARGET output multipliers (not per member), matching pair_build.
Parity: the fp32 graph is checked against the MEAN of the N torch members on 400 random rows
and must match to 1e-4, or the build is refused.
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



def build_fused(ws: list, mode: str = "nbits", bits: int = 8, block: int = 128, ln_eps: float = 1e-5,
                fold_ln2: bool = False, merge_gru: bool = False):
    """Two EXACT node-count rewrites are on by default (the graph is ~85% dispatch-bound, so nodes
    are the only runtime lever). Both are algebraic identities on the fp32 weights; neither changes
    the IO contract (features / h{l} / h{l}_out / prediction and their shapes are untouched).

    fold_ln2  (R1): LN_unit(cat) * g_i + b_i @ W1_i + b1_i == LN_unit(cat) @ (diag(g_i) W1_i) + (b_i @ W1_i + b1_i),
              so ln2's per-member Mul/Add fold into head1's weight and bias. -2 nodes per ln2.
    merge_gru (R3): the per-layer ih and hh matmul groups become ONE group over Concat([h, x]) with the
              (H+Kx, 4H) block weight [[W_hr W_hz 0 W_hn], [W_ir W_iz W_in 0]] and bias
              [b_ir+b_hr, b_iz+b_hz, b_in, b_hn], whose output is [rz_pre | gi_n | gh_n]. The two gate
              Adds and one whole Split/N x MatMulNBits/Concat group disappear and r,z share one Sigmoid.
              h goes FIRST so that, with H a multiple of `block`, every quantisation block (weights AND
              accuracy_level-4 activations) covers exactly the rows it covered before: the int8 packing
              is bit-identical to the unmerged graph's, only the fp32 accumulation order differs. If H is
              not block-aligned a constant zero pad is inserted between h and x. -6 nodes per layer.
    Both default OFF so every recorded build reproduces byte-for-byte; opt in with --fold-ln2 / --merge-gru
    (--lean = both). Laptop A/B (24 Sep, M3, ORT 1.23.2 and 1.30): fold_ln2 -1.2 us/row, merge_gru +1.1
    us/row (the merged MatMulNBits costs as much as the two it replaces); box A/B still owed.
    """
    N = len(ws)
    assert N >= 2, "need at least two members"
    w0 = ws[0]
    H, L, P = w0["hidden"], w0["layers"], w0["proj"]
    for i, w in enumerate(ws[1:], 1):
        assert (w["hidden"], w["layers"]) == (H, L), f"member {i} architecture differs"
        assert np.allclose(w["mu"], w0["mu"]) and np.allclose(w["sd"], w0["sd"]), f"member {i} input statistics differ"
    # Members may differ in projection WIDTH. Everything downstream of ln1 that consumes z is a
    # matmul, so a narrow member can be zero-padded to the widest projection and its weight rows
    # zero-padded to match -- that is exact. What is NOT exact is padding ACROSS a LayerNorm: refgru
    # has two, ln1 over the projection and ln2 over [hidden | proj], and padding either would fold
    # zeros into its mean and variance and change the member's output. So width groups stay separate
    # through ln1 and the projection blocks, pad only afterwards, and ln2/head1 slice the padding
    # back off. The GRU in between is shared, because it only ever multiplies.
    PMAX = max(w["proj"] for w in ws)
    MIXED_W = len({w["proj"] for w in ws}) > 1
    if MIXED_W:
        assert len({w["h1_W"].shape[1] for w in ws}) == 1, "members disagree on head width"
    nodes, inits = [], []
    inputs = [_io("features", [1, 1, 112])]; outputs = [_io("prediction", [1, 1, 2])]

    def mm_single(x, W, name, bias=None):
        if mode == "nbits":
            K, Nc = W.shape
            qi, _ = _quant_nbits(W, bits, block, name); inits.extend(qi)
            ins = [x, f"{name}_q", f"{name}_s"]
            if bias is not None:
                inits.append(_T(f"{name}_b", bias)); ins += ["", "", f"{name}_b"]
            nodes.append(helper.make_node("MatMulNBits", ins, [f"{name}_y"], domain="com.microsoft",
                                          K=K, N=Nc, bits=bits, block_size=block, accuracy_level=4, name=name))
            return f"{name}_y"
        inits.append(_T(f"{name}_W", W))
        nodes.append(helper.make_node("MatMul", [x, f"{name}_W"], [f"{name}_mm"], name=name))
        if bias is None: return f"{name}_mm"
        inits.append(_T(f"{name}_b", bias))
        nodes.append(helper.make_node("Add", [f"{name}_mm", f"{name}_b"], [f"{name}_y"]))
        return f"{name}_y"

    def mm_fused(xN, Ws, name, bs=None, shared_input=False, n_over=None):
        """xN: (1,n,K) per-member rows, or (1,1,K) shared. Returns (1,n,Nc).

        n_over covers the mixed-depth case, where a block applies to a SUBSET of the members and the
        tensor on the member axis is narrower than the graph's N."""
        n = N if n_over is None else n_over
        if shared_input:
            parts = [xN] * n
        else:
            outs = [f"{name}_x{i}" for i in range(n)]
            nodes.append(helper.make_node("Split", [xN], outs, axis=1, num_outputs=n))
            parts = outs
        ys = [mm_single(parts[i], Ws[i], f"{name}_M{i}", None if bs is None else bs[i]) for i in range(n)]
        nodes.append(helper.make_node("Concat", ys, [f"{name}_yN"], axis=1))
        return f"{name}_yN"

    def gelu(x, name):
        nodes.append(helper.make_node("Gelu", [x], [f"{name}_g"], domain="com.microsoft", name=f"{name}_gelu"))
        return f"{name}_g"

    def ln_fused(x, gs, bs, name):
        D = len(gs[0])
        inits.extend([_T(f"{name}_one", np.ones(D, np.float32)), _T(f"{name}_zero", np.zeros(D, np.float32)),
                      _T(f"{name}_gN", np.stack(gs)[None].astype(np.float32)),
                      _T(f"{name}_bN", np.stack(bs)[None].astype(np.float32))])
        nodes.append(helper.make_node("LayerNormalization", [x, f"{name}_one", f"{name}_zero"], [f"{name}_n"],
                                      axis=-1, epsilon=ln_eps, name=name))
        nodes.append(helper.make_node("Mul", [f"{name}_n", f"{name}_gN"], [f"{name}_ng"]))
        nodes.append(helper.make_node("Add", [f"{name}_ng", f"{name}_bN"], [f"{name}_y"]))
        return f"{name}_y"

    def ln_unit(x, D, name):
        """LayerNormalization with unit gain / zero bias and NO per-member affine: the affine has been
        folded into the consuming matmul by head_folded (R1)."""
        inits.extend([_T(f"{name}_one", np.ones(D, np.float32)), _T(f"{name}_zero", np.zeros(D, np.float32))])
        nodes.append(helper.make_node("LayerNormalization", [x, f"{name}_one", f"{name}_zero"], [f"{name}_n"],
                                      axis=-1, epsilon=ln_eps, name=name))
        return f"{name}_n"

    def head_folded(members):
        """(W1', b1') per member with ln2's affine folded in:  (n*g + b) @ W1 + b1 == n @ (g[:,None]*W1) + (b @ W1 + b1)."""
        Ws = [(w["ln2_g"][:, None] * w["h1_W"]).astype(np.float32) for w in members]
        bs = [(w["ln2_b"] @ w["h1_W"] + w["h1_b"]).astype(np.float32) for w in members]
        return Ws, bs

    inv = (1.0 / w0["sd"]).astype(np.float32)
    inits.extend([_T("norm_mul", inv), _T("norm_add", (-w0["mu"] * inv).astype(np.float32)),
                  _T("clip_lo", np.array(-w0["clamp"], np.float32)), _T("clip_hi", np.array(w0["clamp"], np.float32))])
    nodes.append(helper.make_node("Mul", ["features", "norm_mul"], ["x_scaled"]))
    nodes.append(helper.make_node("Add", ["x_scaled", "norm_add"], ["x_norm"]))
    nodes.append(helper.make_node("Clip", ["x_norm", "clip_lo", "clip_hi"], ["x_clip"]))
    # COLLAPSE 1: members that read the SAME input surface collapse their N matmuls x @ Pw_i into ONE
    # matmul against [Pw_0 | Pw_1 | ...] of shape (112, g*P), reshaped to (1,g,P). Exact, and saves
    # g-1 MatMulNBits dispatches (~1.4 us each).
    #
    # An adanorm member does NOT read x_clip: it reads its own running standardisation of it, so it
    # cannot share the collapsed matmul. Members are therefore grouped by input surface and each group
    # collapses separately, costing one extra dispatch per distinct surface rather than per member.
    # ws is sorted by surface first so each group is contiguous on the member axis; the members are
    # averaged at the output, so their order carries no meaning and reordering is safe.
    assert_single_member_only(ws)
    # width is the OUTERMOST key when mixed, so each width group is contiguous on the member axis and
    # ln2/head1 can address it with a single slice. Unmixed graphs keep the old key and stay identical.
    if MIXED_W:
        ws = sorted(ws, key=lambda w: (w["proj"], len(w.get("proj_extra", [])), float(w.get("adanorm", 0.0) or 0.0)))
    else:
        ws = sorted(ws, key=lambda w: (len(w.get("proj_extra", [])), float(w.get("adanorm", 0.0) or 0.0)))
    surfaces = []                       # [(half_life, tensor_name, [members])] in member-axis order
    for w in ws:
        hl = float(w.get("adanorm", 0.0) or 0.0)
        if surfaces and surfaces[-1][0] == hl:
            surfaces[-1][2].append(w)
        else:
            surfaces.append((hl, None, [w]))
    SURFACE = {}
    for gi, (hl, _, members) in enumerate(surfaces):
        if hl <= 0:
            surfaces[gi] = (hl, "x_clip", members)
            for _m in members:
                SURFACE[id(_m)] = "x_clip"
            continue
        if hl in SURFACE:                      # another group already built this half-life
            surfaces[gi] = (hl, SURFACE[hl], members)
            for _m in members:
                SURFACE[id(_m)] = SURFACE[hl]
            continue
        # m += al*(x-m);  v += al*((x-m)^2 - v);  out = (x-m)/sqrt(v+1), matching arch/refgru.py.
        # v uses the ALREADY-UPDATED m, which is what refgru does; the previous m is a different model.
        al = np.float32(1.0 - 2.0 ** (-1.0 / hl))
        nm = "ada%d" % gi
        inputs.extend([_io(f"{nm}_m", [1, 1, 112]), _io(f"{nm}_v", [1, 1, 112])])
        outputs.extend([_io(f"{nm}_m_out", [1, 1, 112]), _io(f"{nm}_v_out", [1, 1, 112])])
        inits.extend([_T(f"{nm}_a", np.array(al, np.float32)), _T(f"{nm}_one", np.array(1.0, np.float32))])
        nodes.append(helper.make_node("Sub", ["x_clip", f"{nm}_m"], [f"{nm}_dm"]))
        nodes.append(helper.make_node("Mul", [f"{nm}_dm", f"{nm}_a"], [f"{nm}_dma"]))
        nodes.append(helper.make_node("Add", [f"{nm}_m", f"{nm}_dma"], [f"{nm}_m_out"]))
        nodes.append(helper.make_node("Sub", ["x_clip", f"{nm}_m_out"], [f"{nm}_d"]))
        nodes.append(helper.make_node("Mul", [f"{nm}_d", f"{nm}_d"], [f"{nm}_d2"]))
        nodes.append(helper.make_node("Sub", [f"{nm}_d2", f"{nm}_v"], [f"{nm}_vd"]))
        nodes.append(helper.make_node("Mul", [f"{nm}_vd", f"{nm}_a"], [f"{nm}_vda"]))
        nodes.append(helper.make_node("Add", [f"{nm}_v", f"{nm}_vda"], [f"{nm}_v_out"]))
        nodes.append(helper.make_node("Add", [f"{nm}_v_out", f"{nm}_one"], [f"{nm}_v1"]))
        nodes.append(helper.make_node("Sqrt", [f"{nm}_v1"], [f"{nm}_s"]))
        nodes.append(helper.make_node("Div", [f"{nm}_d", f"{nm}_s"], [f"{nm}_x"]))
        surfaces[gi] = (hl, f"{nm}_x", members)
        SURFACE[hl] = f"{nm}_x"
        for _m in members:
            SURFACE[id(_m)] = f"{nm}_x"
    if MIXED_W:
        # --- mixed projection widths ---------------------------------------------------------
        # One homogeneous group per (width, depth, surface). Each runs its own collapsed matmul,
        # its own ln1 and its own projection blocks at its OWN width, and is zero-padded to PMAX
        # only at the very end -- after the last LayerNorm, where every remaining consumer is a
        # matmul and padding is therefore exact.
        gidx, key = [], None
        for j, w in enumerate(ws):
            k = (w["proj"], len(w.get("proj_extra", [])), float(w.get("adanorm", 0.0) or 0.0))
            if k != key:
                gidx.append([]); key = k
            gidx[-1].append(j)
        zparts = []
        for gi, idxs in enumerate(gidx):
            members = [ws[j] for j in idxs]
            g, Pg = len(members), members[0]["proj"]
            nm = "w%d" % gi
            gW = np.concatenate([w["proj_W"] for w in members], axis=1)
            gb = np.concatenate([w["proj_b"] for w in members], axis=0)
            pj = mm_single(SURFACE[id(members[0])], gW, nm + "_proj", gb)
            inits.append(_T(nm + "_shape", np.array([1, g, Pg], np.int64)))
            nodes.append(helper.make_node("Reshape", [pj, nm + "_shape"], [nm + "_yN"]))
            zg = ln_fused(gelu(nm + "_yN", nm + "_p"),
                          [w["ln1_g"] for w in members], [w["ln1_b"] for w in members], nm + "_ln1")
            for _i in range(len(members[0].get("proj_extra", []))):
                bn = "%s_x%d" % (nm, _i)
                zz = mm_fused(zg, [w["proj_extra"][_i]["W"] for w in members], bn,
                              [w["proj_extra"][_i]["b"] for w in members], n_over=g)
                zg = ln_fused(gelu(zz, bn), [w["proj_extra"][_i]["g"] for w in members],
                              [w["proj_extra"][_i]["lb"] for w in members], bn + "_ln")
            if Pg < PMAX:
                # pad AFTER the last LayerNorm; W_ih and h1_W get matching zero rows below
                inits.append(_T(nm + "_pad", np.zeros((1, g, PMAX - Pg), np.float32)))
                nodes.append(helper.make_node("Concat", [zg, nm + "_pad"], [nm + "_zp"], axis=2))
                zg = nm + "_zp"
            zparts.append(zg)
        if len(zparts) == 1:
            nodes.append(helper.make_node("Identity", zparts, ["z_all"]))
        else:
            nodes.append(helper.make_node("Concat", zparts, ["z_all"], axis=1))
        z = "z_all"
    else:
        parts = []
        for gi, (hl, src, members) in enumerate(surfaces):
            g = len(members)
            # With a single surface keep the ORIGINAL tensor names, so a graph that needs no grouping is
            # byte-identical to what this builder produced before mixed surfaces existed. Renaming them
            # would change every shipped artifact's bytes for no behavioural reason.
            sfx = "" if len(surfaces) == 1 else str(gi)
            gW = np.concatenate([w["proj_W"] for w in members], axis=1)    # (112, g*P)
            gb = np.concatenate([w["proj_b"] for w in members], axis=0)    # (g*P,)
            pj = mm_single(src, gW, "proj_all" + sfx, gb)                  # (1,1,g*P)
            inits.append(_T("proj_shape" + sfx, np.array([1, g, P], np.int64)))
            nodes.append(helper.make_node("Reshape", [pj, "proj_shape" + sfx], ["proj_yN" + sfx]))
            parts.append("proj_yN" + sfx)
        if len(parts) == 1:
            # Rename the Reshape output rather than appending an Identity: with one input surface this
            # graph must stay BYTE-IDENTICAL to what the builder produced before mixed surfaces existed,
            # or reproduce/v48.sh --check fails on a shipped artifact.
            nodes[-1].output[0] = "proj_yN"
        else:
            nodes.append(helper.make_node("Concat", parts, ["proj_yN"], axis=1))
        z = ln_fused(gelu("proj_yN", "proj"),
                     [w["ln1_g"] for w in ws], [w["ln1_b"] for w in ws], "ln1")
        # Deeper input projections (--proj-layers > 1). Unlike the first block, these cannot be collapsed
        # into one matmul: each member's second block reads that member's OWN first-block output, so there
        # is no shared input to share a matmul against. They go through the per-member helpers instead.
        # Confirmed at three seeds: two blocks is +0.0016 (3.0 sigma, rank p 0.018) where one is +0.0010.
        # Members may differ in projection DEPTH. A two-block member and a one-block member both emit
        # width P, so they can share the member axis; the deeper member simply has extra blocks applied
        # to its slice. ws is sorted by depth, so the members owning block i are a contiguous SUFFIX and
        # the block is applied to a slice rather than to everything.
        #
        # This is what lets a graph mix the two-block recipe with the p192 members and with an adanorm
        # member, which is the only composition that survives split validation on BOTH halves.
        depths = [len(w.get("proj_extra", [])) for w in ws]
        for _i in range(max(depths) if depths else 0):
            have = [j for j, dd in enumerate(depths) if dd > _i]
            nm = "projx%d" % _i
            if len(have) == N:
                zz = mm_fused(z, [w["proj_extra"][_i]["W"] for w in ws], nm,
                              [w["proj_extra"][_i]["b"] for w in ws])
                z = ln_fused(gelu(zz, nm), [w["proj_extra"][_i]["g"] for w in ws],
                             [w["proj_extra"][_i]["lb"] for w in ws], nm + "_ln")
                continue
            assert have == list(range(have[0], N)), "depth groups must be contiguous; ws sort is wrong"
            k = have[0]
            inits.append(_T(f"{nm}_split", np.array([k, N - k], np.int64)))
            nodes.append(helper.make_node("Split", [z, f"{nm}_split"], [f"{nm}_keep", f"{nm}_in"], axis=1))
            sub = [ws[j] for j in have]
            zz = mm_fused(f"{nm}_in", [w["proj_extra"][_i]["W"] for w in sub], nm,
                          [w["proj_extra"][_i]["b"] for w in sub], n_over=len(sub))
            # ln_fused takes its shape from the arrays it is given, so the subset needs no override
            zz = ln_fused(gelu(zz, nm), [w["proj_extra"][_i]["g"] for w in sub],
                          [w["proj_extra"][_i]["lb"] for w in sub], nm + "_ln")
            nodes.append(helper.make_node("Concat", [f"{nm}_keep", zz], [f"{nm}_out"], axis=1))
            z = f"{nm}_out"
    x = z
    for l in range(L):
        gs = [w["gru"][l] for w in ws]; p = f"l{l}_"; h = f"h{l}"
        inputs.append(_io(h, [1, N, H])); outputs.append(_io(f"{h}_out", [1, N, H]))
        # Layer 0 reads z, which is padded to PMAX for narrow members; zero rows on W_ih make that
        # exact (a zero input times any weight is zero, so the pad contributes nothing). Later layers
        # read the hidden state, which is the same width for everyone.
        def _ih(g, w):
            A = np.ascontiguousarray(g["W_ih"].T)
            if l == 0 and A.shape[0] < PMAX:
                A = np.pad(A, ((0, PMAX - A.shape[0]), (0, 0)))
            return A
        if merge_gru:
            # R3: one matmul group over [h | (pad) | x] instead of two over x and h. h goes first so its
            # rows start at 0 and x's rows start at Hp, a block boundary, so every weight block and every
            # accuracy_level-4 activation block covers exactly the rows it covered in the split graph.
            # The pad (only when H is not block-aligned) is a constant initializer, never a node.
            Hp = ((H + block - 1) // block) * block
            cat_in = [h]
            if Hp > H:
                inits.append(_T(p + "hpad", np.zeros((1, N, Hp - H), np.float32)))
                cat_in.append(p + "hpad")
            cat_in.append(x)
            nodes.append(helper.make_node("Concat", cat_in, [p + "hx"], axis=2))

            def _merged(g, w):
                A = _ih(g, w); B = np.ascontiguousarray(g["W_hh"].T)         # (Kx, 3H), (H, 3H)
                W = np.zeros((Hp + A.shape[0], 4 * H), np.float32)          # cols: r | z | gi_n | gh_n
                W[:H, :2 * H] = B[:, :2 * H]; W[:H, 3 * H:] = B[:, 2 * H:]
                W[Hp:, :3 * H] = A
                return W

            def _mbias(g):
                bi, bh = g["b_ih"], g["b_hh"]
                return np.concatenate([bi[:2 * H] + bh[:2 * H], bi[2 * H:], bh[2 * H:]]).astype(np.float32)
            gg = mm_fused(p + "hx", [_merged(g, w) for g, w in zip(gs, ws)], p + "ihh", [_mbias(g) for g in gs])
            inits.append(_T(p + "gsplit", np.array([2 * H, H, H], np.int64)))
            nodes.append(helper.make_node("Split", [gg, p + "gsplit"], [p + "rz_pre", p + "gi_n", p + "gh_n"], axis=2))
            nodes.append(helper.make_node("Sigmoid", [p + "rz_pre"], [p + "rz"]))
            nodes.append(helper.make_node("Split", [p + "rz"], [p + "r", p + "z"], axis=2, num_outputs=2))
        else:
            gi = mm_fused(x, [_ih(g, w) for g, w in zip(gs, ws)], p + "ih", [g["b_ih"] for g in gs])
            gh = mm_fused(h, [np.ascontiguousarray(g["W_hh"].T) for g in gs], p + "hh", [g["b_hh"] for g in gs])
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
    if not MIXED_W:
        if fold_ln2:        # R1: ln2's per-member affine lives inside head1's weight and bias
            c = ln_unit("cat", H + P, "ln2"); W1, b1 = head_folded(ws)
        else:
            c = ln_fused("cat", [w["ln2_g"] for w in ws], [w["ln2_b"] for w in ws], "ln2")
            W1, b1 = [w["h1_W"] for w in ws], [w["h1_b"] for w in ws]
        hdn = gelu(mm_fused(c, W1, "head1", b1), "head1")
    else:
        # ln2 normalises over [hidden | proj], so the zero padding would enter its mean and variance
        # and change the member's output. Slice each width group back to its REAL width first -- the
        # padding sits at the end of the projection, so [0, H+Pg) is exactly the real channels --
        # then run that group's ln2 and head1 at its own width. Only the final prediction merges, and
        # COLLAPSE 2 already sums over members, so nothing downstream cares about the grouping.
        hparts, off = [], 0
        for gi, idxs in enumerate(gidx):
            members = [ws[j] for j in idxs]
            g, Pg = len(members), members[0]["proj"]
            nm = "hw%d" % gi
            inits.extend([_T(nm + "_s", np.array([off, 0], np.int64)),
                          _T(nm + "_e", np.array([off + g, H + Pg], np.int64)),
                          _T(nm + "_a", np.array([1, 2], np.int64))])
            nodes.append(helper.make_node("Slice", ["cat", nm + "_s", nm + "_e", nm + "_a"], [nm + "_cat"]))
            if fold_ln2:    # R1, per width group
                cg = ln_unit(nm + "_cat", H + Pg, nm + "_ln2"); W1, b1 = head_folded(members)
            else:
                cg = ln_fused(nm + "_cat", [w["ln2_g"] for w in members],
                              [w["ln2_b"] for w in members], nm + "_ln2")
                W1, b1 = [w["h1_W"] for w in members], [w["h1_b"] for w in members]
            hparts.append(mm_fused(cg, W1, nm + "_head1", b1, n_over=g))
            off += g
        if len(hparts) == 1:
            nodes.append(helper.make_node("Identity", hparts, ["head1_all"]))
        else:
            nodes.append(helper.make_node("Concat", hparts, ["head1_all"], axis=1))
        hdn = gelu("head1_all", "head1")
    # COLLAPSE 2: prediction = mean_i(hdn_i @ W2_i + b2_i). Because we SUM over members, the
    # whole thing is one matmul: flatten hdn to (1,1,N*Hh) and stack W2 vertically to (N*Hh,2),
    # scaled by 1/N. Exact, saves N-1 dispatches AND removes the ReduceMean node.
    Hh = ws[0]["h2_W"].shape[0]
    inits.append(_T("hdn_shape", np.array([1, 1, N * Hh], np.int64)))
    nodes.append(helper.make_node("Reshape", [hdn, "hdn_shape"], ["hdn_flat"]))
    W2 = np.concatenate([w["h2_W"] for w in ws], axis=0) / float(N)    # (N*Hh, 2)
    b2 = np.sum([w["h2_b"] for w in ws], axis=0) / float(N)            # (2,)
    mm_single("hdn_flat", W2, "head2_all", b2)
    nodes[-1].output[0] = "prediction"
    graph = helper.make_graph(nodes, f"refgru_fused{N}_step", inputs, outputs, initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
                          producer_name="alpha-connectome")
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def run_graph(path, X, H, L, N):
    """Row-by-row reference run.

    State tensors are discovered BY NAME rather than assumed to be h0..hL. An adanorm member adds
    ada*_m / ada*_v inputs, and a hard-coded h-only list silently drops them -- which surfaces as
    ONNX Runtime refusing the feed if you are lucky, and as a graph verified against the wrong state
    if you are not. refgru_build.py already had to learn this.
    """
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    outs = {o.name for o in s.get_outputs()}
    st_names = [i.name for i in s.get_inputs()
                if i.name != "features" and f"{i.name}_out" in outs]
    shapes = {i.name: [1 if (d is None or isinstance(d, str)) else d for d in i.shape]
              for i in s.get_inputs()}
    st = {n: np.zeros(shapes[n], np.float32) for n in st_names}
    names = ["prediction"] + [f"{n}_out" for n in st_names]
    out = np.zeros((len(X), 2), np.float32)
    for i, r in enumerate(X):
        feed = {"features": r.reshape(1, 1, -1)}
        feed.update(st)
        res = s.run(names, feed)
        out[i] = res[0].reshape(-1)[:2]
        st = {n: res[1 + j] for j, n in enumerate(st_names)}
    return out, [st[n] for n in st_names]


def iob_loop_us(path, n=20000):
    """Time the graph the way the SHIPPED driver runs it: IOBinding, two state banks, pointer
    ping-pong, no per-row dict build and no output copies. `bare_loop_us` (plain session.run
    with dict feeds) overstates cost -- solution_iob.py is what actually ships."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    s_ = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    ins, outs = s_.get_inputs(), s_.get_outputs()
    shp = lambda t: [d if isinstance(d, int) else 1 for d in t.shape]
    x = np.zeros(shp(next(i for i in ins if i.name == "features")), np.float32); xrow = x.reshape(-1)
    sin = [i.name for i in ins if i.name != "features"]
    sout = [o.name for o in outs if o.name != "prediction"]
    st_shapes = [shp(next(i for i in ins if i.name == nm)) for nm in sin]
    pr = np.zeros(shp(next(o for o in outs if o.name == "prediction")), np.float32)
    bank = [[np.zeros(sh, np.float32) for sh in st_shapes] for _ in range(2)]
    binds = []
    for k in range(2):
        b = s_.io_binding()
        b.bind_input("features", "cpu", 0, np.float32, x.shape, x.ctypes.data)
        for nm, arr in zip(sin, bank[k]):
            b.bind_input(nm, "cpu", 0, np.float32, arr.shape, arr.ctypes.data)
        b.bind_output("prediction", "cpu", 0, np.float32, pr.shape, pr.ctypes.data)
        for nm, arr in zip(sout, bank[1 - k]):
            b.bind_output(nm, "cpu", 0, np.float32, arr.shape, arr.ctypes.data)
        binds.append(b)
    rows = np.random.default_rng(0).standard_normal((n, 112)).astype(np.float32)
    best = 1e9
    for _ in range(3):
        kk = 0; t0 = time.perf_counter()
        for r in rows:
            xrow[:] = r
            s_.run_with_iobinding(binds[kk]); kk ^= 1
        best = min(best, (time.perf_counter() - t0) / n * 1e6)
    return best


if __name__ == "__main__":
    import torch
    from pair_build import bare_loop_us
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="two or more runs/<tag>/final.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mults", default="0.5,0.5"); ap.add_argument("--time", default="")
    ap.add_argument("--bits", type=int, default=8); ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--lean", action="store_true",
                    help="enable BOTH exact node-count rewrites (R1 fold_ln2, R3 merge_gru); default builds the recorded graph")
    ap.add_argument("--fold-ln2", action="store_true", help="R1 only: fold ln2's affine into head1 (-2 nodes, ~-1.2 us/row on the laptop)")
    ap.add_argument("--merge-gru", action="store_true", help="R3 only: one ih+hh MatMulNBits group per layer (-7 nodes, +1.1 us/row on the laptop)")
    a = ap.parse_args()
    LEAN = dict(fold_ln2=(a.lean or a.fold_ln2), merge_gru=(a.lean or a.merge_gru))
    from score_ckpt import load
    mults = tuple(float(v) for v in a.mults.split(","))
    N = len(a.ckpts)
    ws, refs = [], []
    for ck in a.ckpts:
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
            ref = ref + (o[:, :2] * k).numpy() / N
    for mode in ("fp32", "nbits"):
        m = build_fused(ws, mode=mode, bits=a.bits, block=a.block, **LEAN)
        path = out/("model_fp32.onnx" if mode == "fp32" else "model.onnx"); onnx.save(m, str(path))
        got, _ = run_graph(path, Xr, ws[0]["hidden"], ws[0]["layers"], N)
        d = np.abs(got - ref).max()
        print(f"{mode}: N={N}, {len(m.graph.node)} nodes; vs mean of {N} torch members on 400 rows: max |diff| {d:.2e}", flush=True)
        if mode == "fp32" and d > 1e-4:
            raise SystemExit(f"fused {N}-member graph does not match the member average -- structural bug")
    print(f"bare loop (dict feeds): fused {N}-member {bare_loop_us(out/'model.onnx'):.1f} us/row", flush=True)
    print(f"IOBinding (as SHIPPED): fused {N}-member {iob_loop_us(out/'model.onnx'):.1f} us/row", flush=True)
    if a.time:
        import tempfile
        t = Path(tempfile.mkdtemp()); zipfile.ZipFile(a.time).extractall(t)
        for g in sorted(t.glob("*.onnx")): print(f"bare loop: {Path(a.time).name}/{g.name} {bare_loop_us(g):.1f} us/row", flush=True)
    if not a.no_zip:
        # ship solution_iob.py, not DRIVER: DRIVER uses plain session.run with dict feeds.
        (out/"solution.py").write_text((R/"src"/"infer"/"solution_iob.py").read_text())
        zp = R/"submissions"/f"submission_{out.name}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out/"solution.py", "solution.py"); z.write(out/"model.onnx", "model.onnx")
        print(f"packaged {zp} ({zp.stat().st_size//1024} KB)")
