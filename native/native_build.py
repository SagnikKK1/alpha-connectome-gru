"""native_build.py -- research H11 dev copy (12 Oct 2026) of the H10 builder. Adds EXACT support for --posfeat members (blob v5):
  * refgru.py --posfeat feeds `row index / 20000` (rows since the sequence start, warm-up rows included, 0 on the first row, no
    clamp, no standardisation) as in_proj.0's input column 112 (after the 112 raw features, before the bookconv read, which
    never touches it). load_members() strips that column before the usual bookconv fold (the fold is linear in the 112 raw
    columns only, so it is unchanged) and keeps it as w["pos_W"] (P,) = in_proj.0.weight[:, 112].
  * write_blob(): if any member has posfeat -> blob version 5, header flags bit 2, one fp32 section wpos[sum P] right after p0
    (a non-posfeat member's slice is zeros -> y0 + 0: mixing is exact). Blobs without posfeat members are written exactly as the
    H10 builder writes them (same version, same bytes).
  * the kernel adds (rows_seen / 20000) * wpos to the projection pre-activation in fp32 (one mul + one add per column, every ISA
    and the emulation bit-identical); the row counter is reset by wnn_reset (= new seq_ix in solution.py).
  * ONNX fallback: build_fused() (src/infer/fused_build.py, read-only) knows nothing of posfeat; add_posfeat_onnx() patches the
    fused graph with a (1,1,1) state `pcnt` / `pcnt_out` (+1 per row, zeroed on reset by the shipped Onnx class) and an fp32
    Add of (pcnt / 20000) * pos_wN (1,N,P) in front of the projection GELU. Verified by the usual 400-row fp32 gate (1e-4) against
    the torch member mean. Alternatively `--onnx-ckpts CK..` builds the fallback graph from OTHER (e.g. non-posfeat) members.

native_build.py -- research H10 dev copy (12 Oct 2026) of the H5 builder (offline/research_H5/native_build.py, built v118i4 /
v120 / v121). Adds blob v4: an EXACT fp32-activation path for the collapsed input projection p0 (`--p0 a32w8 | a32w16`):
  q8     (default) today's path: the 112 standardised inputs quantised to int8 per 128-block, int8 weights (lay 2) -- v2/v3 blobs;
  a32w8  lay 6: the inputs stay fp32 (no activation quantisation); weights = the SAME int8 codes and per-(block, column) scales as
         lay 2, stored k-major; identical bytes to q8; header flags bit 1 set, blob version 4;
  a32w16 lay 7: the inputs stay fp32; weights fp16 (2x p0's bytes, no scales); blob version 4.
  a16w8  lay 8 (the cheap alternative): the inputs quantised to int16 per 128-block (256x finer than int8), the SAME int8 weights
         and scales as lay 2 (k-pair interleaved), exact int32 dots; identical bytes to q8; blob version 4.
Every other matrix is written exactly as the H5 builder writes it, so a build with --p0 q8 is byte-identical to the H5 builder's.

research H5 (12 Oct 2026): blob v3 = per-matrix INT4 storage (`--int4 ih,head1 [--int4-block 64] [--int4-scale f16|f32]`) with exactly
research F1's asymmetric min/max quantiser (f1_fwd.fq_asym: s = (max - min) / 15 per (block, column), q = rint((w - min) / s) in
0..15); the stored scale/min are fp16 (default, 0.77x bytes at 8 members) or fp32 (bit-for-bit F1's reconstruction grid). With
--int4 the 400-row gate is also run against the fake-quantised torch twin (the F1 harness's weights), which must be closer to the
emulation than fp32 is.

research RT (11 Oct 2026) of the BOX native_build.py (md5 10d11bf6). Adds blob v2: per-member
head width and a FACTORED recurrence for --hh-rank members (W_hh = hh_A @ hh_B streamed as two int8 matrices, B^T then A^T).
A blob with no factored member and one head width is written as v1, byte-identical to the Z52 builder. --hh-dense forces the
materialised W_hh. WNN_ROOT overrides the repo root (for dev copies outside offline/).

native_build.py (Z52, 8 Oct 2026): N member checkpoints -> native submission zip.

    .venv-dev/bin/python native/native_build.py CK1 CK2 CK3 --out out/my_build \
        [--prebuilt path/to/wnnfast_x86_64.so] [--mults 0.5,0.5]

Writes into --out: weights.wnn (int8 blob read by wnnfast.c), wnnfast.c, solution.py (native/solution_native.py),
model.onnx (fused_build.build_fused, nbits 8 / block 128 -- the shipped ONNX graph, used as the last fallback),
wnnfast_x86_64.so (if --prebuilt), and the zip <out>.zip. Refuses to package unless:
  * the fp32 fused graph matches the torch member mean (fused_build's own 1e-4 parity gate),
  * the blob's NumPy emulation is within int8 noise of the torch member mean on 400 random rows,
  * (with --ref-onnx-zip) model.onnx is byte-identical to that zip's model.onnx.
"""
from __future__ import annotations
import argparse, os, sys, zipfile, hashlib, importlib.util, shutil
from pathlib import Path
import numpy as np

R = Path(os.environ.get("WNN_ROOT") or Path(__file__).resolve().parents[1])
HERE = Path(__file__).resolve().parent
for p in (str(R), str(R/"src"), str(R/"src"/"infer"), str(R/"src"/"arch")): sys.path.insert(0, p)
MAGIC, END, BLK = 0x464E4E57, 0x444E4557, 128


def load_sol():
    spec = importlib.util.spec_from_file_location("wnn_solution_native", HERE/"solution_native.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Writer:
    def __init__(self): self.b = bytearray()
    def put(self, raw: bytes):
        self.b += b"\0" * ((-len(self.b)) % 64); self.b += raw
    def i32(self, vals): self.put(np.asarray(vals, "<i4").tobytes())
    def f32(self, arr): self.put(np.ascontiguousarray(arr, "<f4").tobytes())
    def qmat(self, W, bias):
        W = np.asarray(W, np.float32); K, Nc = W.shape
        assert K % 16 == 0 and Nc % 64 == 0, (K, Nc)
        nb = (K + BLK - 1) // BLK; Kp = nb * BLK
        Wp = np.zeros((Kp, Nc), np.float32); Wp[:K] = W
        Wb = Wp.reshape(nb, BLK, Nc)
        amax = np.abs(Wb).max(axis=1)
        scale = (amax / 127).astype(np.float32); scale[scale == 0] = 1.0       # = onnx_build._quant_nbits(bits=8)
        q = np.clip(np.rint(Wb / scale[:, None, :]), -127, 127).astype(np.int32).reshape(Kp, Nc)[:K]
        corr = np.stack([128 * q[b * BLK:(b + 1) * BLK].sum(0) for b in range(nb)]).astype(np.int32)
        packed = q.astype(np.int8).reshape(K // 4, 4, Nc // 64, 4, 16).transpose(2, 0, 3, 4, 1)
        self.i32([K, Nc, nb, 0]); self.put(np.ascontiguousarray(packed).tobytes())
        self.f32(scale); self.i32(corr.reshape(-1)); self.f32(bias)
        return (q.astype(np.float32).reshape(K, Nc) * np.repeat(scale, BLK, axis=0)[:K])   # dequantised, for reports

    def qmat2(self, W, bias):
        """lay 2 (blob v2): the same int8 codes and scales; per 64-column group one contiguous record
        [u8 weights q+128 packed as lay 1 | sw (nb, 64) f32 | bias (64) f32]; no corr array."""
        W = np.asarray(W, np.float32); K, Nc = W.shape
        assert K % 16 == 0 and Nc % 64 == 0, (K, Nc)
        nb = (K + BLK - 1) // BLK; Kp = nb * BLK
        Wp = np.zeros((Kp, Nc), np.float32); Wp[:K] = W
        Wb = Wp.reshape(nb, BLK, Nc)
        amax = np.abs(Wb).max(axis=1)
        scale = (amax / 127).astype(np.float32); scale[scale == 0] = 1.0
        q = np.clip(np.rint(Wb / scale[:, None, :]), -127, 127).astype(np.int32).reshape(Kp, Nc)[:K]
        G = Nc // 64
        packed = (q + 128).astype(np.uint8).reshape(K // 4, 4, G, 4, 16).transpose(2, 0, 3, 4, 1).reshape(G, K * 64)
        sw = np.ascontiguousarray(scale.astype("<f4").reshape(nb, G, 64).transpose(1, 0, 2)).reshape(G, nb * 64)
        bb = np.ascontiguousarray(np.asarray(bias, "<f4").reshape(G, 64))
        rec = np.concatenate([packed, sw.view(np.uint8), bb.view(np.uint8)], axis=1)
        self.i32([K, Nc, nb, 2]); self.put(np.ascontiguousarray(rec).tobytes())

    def qmat4(self, W, bias, blk=64, sdt="f16"):
        """H5 (blob v3, lay 4 = fp32 scale/min, lay 5 = fp16): int4 ASYMMETRIC min/max per (blk-row K block, column), the codes
        computed EXACTLY as research F1's f1_fwd.fq_asym (same float32 expressions, same order), stored as nibbles: per 4-k step,
        vector v (64 bytes) = tile 2v in the low nibbles and tile 2v+1 in the high nibbles, byte = col*4 + k; then s (nb4, 64),
        min (nb4, 64) in sdt, bias (64) f32. Returns the dequantised matrix (K, Nc) as the kernel reconstructs it (stored dtype)."""
        W = np.asarray(W, np.float32); K, Nc = W.shape
        assert blk in (32, 64, 128) and K % blk == 0 and Nc % 64 == 0, (K, Nc, blk)
        nb4 = K // blk
        Wb = W.reshape(nb4, blk, Nc); mn = Wb.min(1); mx = Wb.max(1)
        s = ((mx - mn) / (2 ** 4 - 1)).astype(np.float32); s[s == 0] = 1                     # = f1_fwd.fq_asym(bits=4)
        q = np.clip(np.rint((Wb - mn[:, None, :]) / s[:, None, :]), 0, 2 ** 4 - 1).astype(np.int32)
        G = Nc // 64
        t = q.reshape(K // 4, 4, G, 4, 16).transpose(2, 0, 3, 4, 1)                            # (G, K/4, tile, col, k)
        packed = (t[:, :, 0::2] | (t[:, :, 1::2] << 4)).astype(np.uint8).reshape(G, K * 32)
        dt = "<f2" if sdt == "f16" else "<f4"
        sc = np.ascontiguousarray(s.astype(dt).reshape(nb4, G, 64).transpose(1, 0, 2)).reshape(G, -1).view(np.uint8)
        mm = np.ascontiguousarray(mn.astype(dt).reshape(nb4, G, 64).transpose(1, 0, 2)).reshape(G, -1).view(np.uint8)
        bb = np.ascontiguousarray(np.asarray(bias, "<f4").reshape(G, 64)).view(np.uint8)
        rec = np.concatenate([packed, sc, mm, bb], axis=1)
        self.i32([K, Nc, nb4, 4 if sdt == "f32" else 5]); self.put(np.ascontiguousarray(rec).tobytes())
        s_st = s.astype(dt).astype(np.float32); mn_st = mn.astype(dt).astype(np.float32)
        return (q.astype(np.float32) * s_st[:, None, :] + mn_st[:, None, :]).reshape(K, Nc)

    def qmat6(self, W, bias):
        """H10 lay 6 (blob v4, p0 only): the SAME int8 codes and per-(128-row block, column) scales as lay 2 (absmax/127), stored
        for the fp32-activation path: per 64-column group [q int8 [K][64] k-major (signed) | sw (nb, 64) f32 | bias (64) f32]
        (record size = lay 2's, so a32w8 costs no bytes). Returns the dequantised matrix (K, Nc)."""
        W = np.asarray(W, np.float32); K, Nc = W.shape
        assert K % 16 == 0 and Nc % 64 == 0, (K, Nc)
        nb = (K + BLK - 1) // BLK; Kp = nb * BLK
        Wp = np.zeros((Kp, Nc), np.float32); Wp[:K] = W
        Wb = Wp.reshape(nb, BLK, Nc)
        amax = np.abs(Wb).max(axis=1)
        scale = (amax / 127).astype(np.float32); scale[scale == 0] = 1.0
        q = np.clip(np.rint(Wb / scale[:, None, :]), -127, 127).astype(np.int32).reshape(Kp, Nc)[:K]
        G = Nc // 64
        packed = np.ascontiguousarray(q.astype(np.int8).reshape(K, G, 64).transpose(1, 0, 2)).reshape(G, K * 64).view(np.uint8)
        sw = np.ascontiguousarray(scale.astype("<f4").reshape(nb, G, 64).transpose(1, 0, 2)).reshape(G, nb * 64)
        bb = np.ascontiguousarray(np.asarray(bias, "<f4").reshape(G, 64))
        rec = np.concatenate([packed, sw.view(np.uint8), bb.view(np.uint8)], axis=1)
        self.i32([K, Nc, nb, 6]); self.put(np.ascontiguousarray(rec).tobytes())
        return (q.astype(np.float32) * np.repeat(scale, BLK, axis=0)[:K])

    def qmat7(self, W, bias):
        """H10 lay 7 (blob v4, p0 only): fp16 weights (round-to-nearest-even) for the fp32-activation path: per 64-column group
        [w f16 [K][64] k-major | bias (64) f32]. Returns the stored matrix as fp32 (K, Nc)."""
        W = np.asarray(W, np.float32); K, Nc = W.shape
        assert K % 16 == 0 and Nc % 64 == 0, (K, Nc)
        nb = (K + BLK - 1) // BLK; G = Nc // 64
        W16 = W.astype("<f2")
        packed = np.ascontiguousarray(W16.reshape(K, G, 64).transpose(1, 0, 2)).reshape(G, K * 64).view(np.uint8)
        bb = np.ascontiguousarray(np.asarray(bias, "<f4").reshape(G, 64)).view(np.uint8)
        rec = np.concatenate([packed, bb], axis=1)
        self.i32([K, Nc, nb, 7]); self.put(np.ascontiguousarray(rec).tobytes())
        return W16.astype(np.float32)

    def qmat8(self, W, bias):
        """H10 lay 8 (blob v4, p0 only): the SAME int8 codes and scales as lay 2, stored k-pair interleaved for the int16-activation
        path: per 64-column group [q int8 [K/2][64][2] | sw (nb, 64) f32 | bias (64) f32] (record size = lay 2's)."""
        W = np.asarray(W, np.float32); K, Nc = W.shape
        assert K % 16 == 0 and Nc % 64 == 0, (K, Nc)
        nb = (K + BLK - 1) // BLK; Kp = nb * BLK
        Wp = np.zeros((Kp, Nc), np.float32); Wp[:K] = W
        Wb = Wp.reshape(nb, BLK, Nc)
        amax = np.abs(Wb).max(axis=1)
        scale = (amax / 127).astype(np.float32); scale[scale == 0] = 1.0
        q = np.clip(np.rint(Wb / scale[:, None, :]), -127, 127).astype(np.int32).reshape(Kp, Nc)[:K]
        G = Nc // 64
        packed = np.ascontiguousarray(q.astype(np.int8).reshape(K // 2, 2, G, 64).transpose(2, 0, 3, 1)).reshape(G, K * 64).view(np.uint8)
        sw = np.ascontiguousarray(scale.astype("<f4").reshape(nb, G, 64).transpose(1, 0, 2)).reshape(G, nb * 64)
        bb = np.ascontiguousarray(np.asarray(bias, "<f4").reshape(G, 64))
        rec = np.concatenate([packed, sw.view(np.uint8), bb.view(np.uint8)], axis=1)
        self.i32([K, Nc, nb, 8]); self.put(np.ascontiguousarray(rec).tobytes())
        return (q.astype(np.float32) * np.repeat(scale, BLK, axis=0)[:K])


INT4_NAMES = ("ih", "head1", "hh", "proj1")
P0_MODES = ("q8", "a32w8", "a32w16", "a16w8")


def write_blob(ws, eps=1e-5, factored=True, blob="v2", fast_gelu=False, int4=(), int4_blk=64, int4_sdt="f16", twins=None, p0="q8",
               posfeat=None) -> bytes:
    """int4: names among INT4_NAMES stored as int4 asym (blob v3). twins (optional list) receives per member a dict
    {"ih": deq (3H, P) torch-orientation, "head1": deq (Hh, H+P), ...} of the dequantised matrices for the fake-quant twin check.
    p0 (H10): "q8" = int8 activations x int8 weights (v2/v3, today's path); "a32w8" = fp32 activations x the same int8 weights
    (lay 6); "a32w16" = fp32 activations x fp16 weights (lay 7); either sets header flags bit 1 and writes blob v4.
    posfeat (H11): None = automatic (v5 + flags bit 2 + the wpos section iff any member has w["posfeat"]); True forces the v5
    posfeat format (zero wpos for members without the column -- tests); False strips the column (TIMING ONLY: wrong outputs)."""
    int4 = tuple(int4)
    for nm in int4: assert nm in INT4_NAMES, nm
    assert p0 in P0_MODES, p0
    if int4 and blob == "v1": raise ValueError("int4 matrices need blob v3")
    if p0 != "q8" and blob == "v1": raise ValueError("the fp32-activation p0 needs blob v4")
    pf = [bool(w.get("posfeat")) for w in ws]
    if posfeat is None: posfeat = any(pf)
    if posfeat and blob == "v1": raise ValueError("posfeat members need blob v5")
    """blob="v1": the Z52 format, byte-identical to the Z52 builder (dense members, one head width only).
    blob="v2" (default): grouped corr-free matrices (same codes/scales -> bit-identical outputs, ~3% fewer bytes, one sequential
    stream per matrix), per-member head width, factored --hh-rank members. fast_gelu (v2 only): flag bit 0, rational-erf GELU."""
    N = len(ws); w0 = ws[0]; H = w0["hidden"]
    Hm = [int(w["h1_W"].shape[1]) for w in ws]
    Rm = [int(w["hh_lr"]["A"].shape[1]) if (factored and w.get("hh_lr") is not None) else 0 for w in ws]
    ver = 1 if blob == "v1" else (4 if (p0 != "q8" or blob == "v4") else (3 if (int4 or blob == "v3") else 2))
    if posfeat: ver = 5
    if ver == 1 and (any(Rm) or len(set(Hm)) > 1 or fast_gelu):
        raise ValueError("v1 blob cannot hold factored members, mixed head widths or the fast GELU flag")
    for i, w in enumerate(ws):
        assert w["layers"] == 1, "wnnfast: single-layer GRU only"
        assert w["hidden"] == H, "member %d geometry differs" % i
        assert Hm[i] % 64 == 0, "member %d head width %d is not a multiple of 64" % (i, Hm[i])
        assert Rm[i] == 0 or (Rm[i] % 64 == 0 and 64 <= Rm[i] < H), "member %d hh rank %d: need a multiple of 64 below H" % (i, Rm[i])
        assert np.array_equal(w["mu"], w0["mu"]) and np.array_equal(w["sd"], w0["sd"]), "member %d input statistics differ" % i
        assert w["clamp"] == w0["clamp"]
        for k in ("ewma", "fastw", "quad"):
            assert not w.get(k), "member %d uses --%s; wnnfast has no path for it" % (i, k)
        assert not w.get("bcnl"), "member %d uses --bookconv-act; wnnfast has no path for it yet (add_struct.py)" % i
        assert not float(w.get("adanorm", 0.0) or 0.0), "member %d uses adanorm" % i
        assert w["proj_W"].shape[0] == 112
        if pf[i]: assert np.asarray(w["pos_W"]).shape == (w["proj"],), "member %d: pos_W must be (P,)" % i
    F = 112
    inv = (1.0 / w0["sd"]).astype(np.float32); add = (-w0["mu"] * inv).astype(np.float32)
    wr = Writer()
    hd = np.zeros(16, "<i4"); hd[:6] = [MAGIC, ver, N, H, max(Hm), F]
    hd[6] = np.float32(w0["clamp"]).view("<i4"); hd[7] = np.float32(eps).view("<i4")
    if ver >= 2: hd[8] = (1 if fast_gelu else 0) | (2 if p0 != "q8" else 0) | (4 if posfeat else 0)
    wr.i32(hd); wr.f32(inv); wr.f32(add)
    qm = wr.qmat if ver == 1 else wr.qmat2
    q4 = lambda W, b: wr.qmat4(W, b, int4_blk, int4_sdt)
    if ver == 1:
        wr.i32(sum(([w["proj"], 1 + len(w["proj_extra"])] for w in ws), []))
    else:
        wr.i32(sum(([w["proj"], 1 + len(w["proj_extra"]), Rm[i], Hm[i]] for i, w in enumerate(ws)), []))
    qp0 = {"q8": qm, "a32w8": wr.qmat6, "a32w16": wr.qmat7, "a16w8": wr.qmat8}[p0]
    qp0(np.concatenate([w["proj_W"] for w in ws], 1), np.concatenate([w["proj_b"] for w in ws]))
    if posfeat:                       # H11: wpos[Ptot] fp32, member slices in order; a member without the column contributes 0 exactly
        wr.f32(np.concatenate([np.asarray(w["pos_W"], np.float32) if pf[i] else np.zeros(w["proj"], np.float32) for i, w in enumerate(ws)]))
    for i, w in enumerate(ws):
        tw = {}
        wr.f32(w["ln1_g"]); wr.f32(w["ln1_b"])
        for j, e in enumerate(w["proj_extra"]):
            if "proj1" in int4: tw["proj%d" % (3 * (j + 1))] = q4(e["W"], e["b"]).T.copy()
            else: qm(e["W"], e["b"])
            wr.f32(e["g"]); wr.f32(e["lb"])
        g = w["gru"][0]
        if "ih" in int4: tw["ih"] = q4(g["W_ih"].T, g["b_ih"]).T.copy()        # torch weight_ih_l0 is (3H, P)
        else: qm(g["W_ih"].T, g["b_ih"])
        if Rm[i]:                     # torch: gh = (hh_A @ hh_B) h + b_hh = A (B h) + b_hh -> u = h @ B^T (no bias); gh = u @ A^T + b_hh
            assert "hh" not in int4, "int4 W_hh is only defined for the dense recurrence"
            A, Bf = np.asarray(w["hh_lr"]["A"], np.float32), np.asarray(w["hh_lr"]["B"], np.float32)
            assert A.shape == (3 * H, Rm[i]) and Bf.shape == (Rm[i], H)
            qm(Bf.T, np.zeros(Rm[i], np.float32)); qm(A.T, g["b_hh"])
        elif "hh" in int4: tw["hh"] = q4(g["W_hh"].T, g["b_hh"]).T.copy()
        else:
            qm(g["W_hh"].T, g["b_hh"])
        wr.f32(w["ln2_g"]); wr.f32(w["ln2_b"])
        if "head1" in int4: tw["head1"] = q4(w["h1_W"], w["h1_b"]).T.copy()     # torch head.0.weight is (Hh, H+P)
        else: qm(w["h1_W"], w["h1_b"])
        if twins is not None: twins.append(tw)
        wr.f32((w["h2_W"].T / float(N)).astype(np.float32))
    wr.f32((np.sum([w["h2_b"] for w in ws], axis=0) / float(N)).astype(np.float32))
    wr.i32([END, N])
    return bytes(wr.b)


POS_COL = 112      # refgru._inputs: parts = [x (112) | diff? | ewma? | posfeat (1) | nlfeat? | ofi? | bookconv ... ]


class _NoPosfeat:
    """H11: a view of a --posfeat core for refgru_build.weights(): column POS_COL of in_proj.0.weight removed and posfeat=False,
    everything else delegated. weights() then runs its (unchanged, linear-in-x) bookconv fold on the 112 + 2*C*11 remaining
    columns exactly as for a non-posfeat member; the removed column is returned separately."""
    posfeat = False

    def __init__(self, core):
        import torch
        self._c = core
        if getattr(core, "diff", False) or getattr(core, "ewma", ()) or getattr(core, "nlfeat", ()) or getattr(core, "ofi", False) \
                or getattr(core, "split", False) or getattr(core, "quad", False):
            raise SystemExit("posfeat member with --diff/--ewma/--nlfeat/--ofi/--split/--quad: the posfeat column index is not %d" % POS_COL)
        W = core.state_dict()["in_proj.0.weight"]
        self.pos_W = W[:, POS_COL].detach().cpu().numpy().astype(np.float32).copy()          # (P,)
        self._W = torch.cat([W[:, :POS_COL], W[:, POS_COL + 1:]], 1).contiguous()

    def __getattr__(self, k): return getattr(self._c, k)

    def state_dict(self):
        sd = dict(self._c.state_dict()); sd["in_proj.0.weight"] = self._W; return sd


def load_members(ckpts, mults):
    import torch
    from score_ckpt import load
    from refgru_build import weights
    ws, refs = [], []
    for ck in ckpts:
        p = Path(ck).resolve()
        rel = "../" + str(p.parent.relative_to(R)) if str(p).startswith(str(R)) else str(p.parent)
        core, sc, cfg = load(rel, p.name); core = core.cpu().eval()
        if getattr(core, "posfeat", False):                                        # H11
            view = _NoPosfeat(core); w = weights(view, sc, mults)
            w["posfeat"] = True; w["pos_W"] = view.pos_W
            assert w["proj_W"].shape == (112, w["proj"]) and w["pos_W"].shape == (w["proj"],)
        else:
            w = weights(core, sc, mults); w["posfeat"] = False
        ws.append(w); refs.append((core, torch.tensor([sc * mults[0], sc * mults[1]])))
    return ws, refs


def add_posfeat_onnx(m, ows):
    """H11: patch build_fused()'s graph for posfeat members. Adds the state pair pcnt / pcnt_out (1,1,1) (zeroed on reset by the
    shipped Onnx class and by fused_build.run_graph, +1 per row) and rewires the projection pre-activation:
    proj_yN = Reshape(...) + (pcnt / 20000) * pos_wN, pos_wN (1,N,P) = each member's in_proj.0 posfeat column (zeros for a
    non-posfeat member). Requires the single-surface, single-width, single-depth graph (member order = ows order)."""
    import onnx
    from onnx import helper
    from onnx_build import _T, _io
    keys = {(w["proj"], len(w.get("proj_extra", [])), float(w.get("adanorm", 0.0) or 0.0)) for w in ows}
    if len(keys) != 1: raise SystemExit("add_posfeat_onnx: members differ in width/depth/surface; build_fused reorders them -- use --onnx-ckpts or --no-onnx")
    g = m.graph
    rs = [n for n in g.node if n.op_type == "Reshape" and list(n.output) == ["proj_yN"]]
    if len(rs) != 1 or any(n.output[0] == "proj_yN" for n in g.node if n.op_type != "Reshape"):
        raise SystemExit("add_posfeat_onnx: expected exactly one Reshape -> proj_yN (single input surface)")
    rs[0].output[0] = "proj_yN_raw"
    N, P = len(ows), ows[0]["proj"]
    W = np.stack([np.asarray(w["pos_W"], np.float32) if w.get("posfeat") else np.zeros(P, np.float32) for w in ows])[None]   # (1,N,P)
    g.initializer.extend([_T("pos_wN", W), _T("pos_div", np.array(20000.0, np.float32)), _T("pos_one", np.array(1.0, np.float32))])
    new = [helper.make_node("Div", ["pcnt", "pos_div"], ["pos_f"], name="pos_div_node"),
           helper.make_node("Mul", ["pos_f", "pos_wN"], ["pos_term"], name="pos_mul_node"),
           helper.make_node("Add", ["proj_yN_raw", "pos_term"], ["proj_yN"], name="pos_add_node"),
           helper.make_node("Add", ["pcnt", "pos_one"], ["pcnt_out"], name="pos_cnt_node")]
    at = [i for i, n in enumerate(g.node) if n is rs[0]][0] + 1
    nodes = list(g.node); nodes[at:at] = new
    del g.node[:]; g.node.extend(nodes)
    g.input.append(_io("pcnt", [1, 1, 1])); g.output.append(_io("pcnt_out", [1, 1, 1]))
    onnx.checker.check_model(m)
    return m


def twin_refs(refs, twins):
    """Deep copies of the fp32 member cores with the int4 matrices replaced by their dequantised values (= research F1's
    fake-quant harness for those matrices; everything else fp32)."""
    import copy, torch
    out = []
    for (core, k), tw in zip(refs, twins):
        c2 = copy.deepcopy(core)
        with torch.no_grad():
            for nm, W in tw.items():
                if nm == "ih": c2.gru.weight_ih_l0.copy_(torch.from_numpy(np.ascontiguousarray(W)))
                elif nm == "hh": c2.gru.weight_hh_l0.copy_(torch.from_numpy(np.ascontiguousarray(W)))
                elif nm == "head1": c2.head[0].weight.copy_(torch.from_numpy(np.ascontiguousarray(W)))
                elif nm.startswith("proj"): c2.in_proj[int(nm[4:])].weight.copy_(torch.from_numpy(np.ascontiguousarray(W)))
                else: raise ValueError(nm)
        out.append((c2, k))
    return out


def member_bytes(ws, **kw):
    """Blob bytes of each member alone (same options) and of the whole set, for the byte report."""
    return [len(write_blob([w], **kw)) for w in ws], len(write_blob(ws, **kw))


def torch_mean(refs, X, seq=False):
    """Mean of the fp32 torch members. seq=False: every row from a zero state (fused_build's 400-row check);
    seq=True: X is ONE sequence, state carried (the scorer's semantics)."""
    import torch
    N = len(refs); ref = 0
    with torch.no_grad():
        for core, k in refs:
            if seq:
                o, _ = core(torch.from_numpy(X).reshape(len(X), 1, -1), core.init_state(1, torch.device("cpu")))
            else:
                o, _ = core(torch.from_numpy(X).reshape(1, len(X), -1), core.init_state(len(X), torch.device("cpu")))
            o = o.reshape(len(X), -1); o = o if o.shape[1] >= 2 else torch.cat([o, -o], 1)
            ref = ref + (o[:, :2] * k).numpy() / N
    return ref.astype(np.float32)


def write_selfcheck(out: Path, blob: bytes, refs, sol):
    """selfcheck.npz: fixed inputs + the build machine's native outputs (NumPy emulation, which the shipped C matches
    bit-for-bit) + the fp32 torch member mean. solution.py's init check compares against all of it."""
    import hashlib, subprocess, tempfile, shutil as _sh, pyarrow.parquet as pq
    from utils import FEATURE_COLUMNS
    B = sol.Blob(blob)
    xa = sol.selfcheck_rows(B, 512, seed=777)
    t = pq.ParquetFile(str(R/"datasets/valid.parquet")).read_row_group(0, columns=list(FEATURE_COLUMNS))
    xb = np.ascontiguousarray(np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)[:1024])
    xn = xa[:16].copy()
    xn[0, 3] = np.nan; xn[1, :] = np.nan; xn[2, 7] = np.inf; xn[3, 9] = -np.inf
    xn[4, ::3] = np.inf; xn[5, 1::3] = -np.inf; xn[6, 50] = np.nan; xn[7, 100:] = np.nan
    with np.errstate(all="ignore"):
        na = sol.ref_run(B, xa)[0]; nbb = sol.ref_run(B, xb)[0]; nn = sol.ref_run(B, xn)[0]
    ta = torch_mean(refs, xa, seq=True); tb = torch_mean(refs, xb, seq=True)
    corr = lambda a, b: min(float(np.corrcoef(a[:, k], b[:, k])[0, 1]) for k in (0, 1))
    md, cb, ca = float(np.abs(nbb - tb).max()), corr(nbb, tb), corr(na, ta)
    print(f"selfcheck refs: real 1024 rows native vs torch max|d| {md:.3e} corr {cb:.6f}; synthetic 512 corr {ca:.6f}; "
          f"non-finite rows finite: {bool(np.isfinite(nn).all())}", flush=True)
    if not (md <= 0.15 and cb >= 0.999 and ca >= 0.99 and np.isfinite(nn).all()):
        raise SystemExit("this blob would fail its own init self-check -- refusing")
    np.savez_compressed(out/"selfcheck.npz", xa=xa, xb=xb, xn=xn, na=na, nb=nbb, nn=nn, ta=ta, tb=tb,
                        blob_sha256=np.array(hashlib.sha256(blob).hexdigest()))
    # the shipped C must reproduce the stored outputs exactly on this machine
    cc = _sh.which("gcc") or _sh.which("cc") or _sh.which("clang")
    if cc:
        d = tempfile.mkdtemp(prefix="wnnb_"); so = d + "/w.so"
        r = subprocess.run([cc, "-O3", "-shared", "-fPIC", "-ffp-contract=off", "-fno-math-errno", "-fno-stack-protector",
                            "-fvisibility=hidden", "-o", so, str(HERE/"wnnfast.c")], capture_output=True)
        if r.returncode == 0:
            nat = sol.Native(so, blob, -1); worst = 0.0
            for X, ref in ((xa, na), (xb, nbb), (xn, nn)):
                nat.reset(); worst = max(worst, float(np.abs(nat.run(X) - ref).max()))
            print(f"local C build (isa {nat.isa}) vs stored outputs: max|d| {worst:.1e}", flush=True)
            if worst > 1e-6: raise SystemExit("local C build disagrees with the stored outputs -- refusing")
            sol.HERE = out                                  # _selfcheck reads <HERE>/selfcheck.npz
            ok, info = sol._selfcheck(nat, blob, so)
            print(f"full init self-check on this machine: {ok} {info}", flush=True)
            if ok is not True: raise SystemExit("init self-check fails on the build machine -- refusing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--mults", default="0.5,0.5"); ap.add_argument("--prebuilt", default="")
    ap.add_argument("--ref-onnx-zip", default=""); ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--blob", default="v2", choices=["v1", "v2", "v3", "v4"], help="v1 = Z52 format (byte-identical); v2 = RT format (default); v3 = H5 (implied by --int4); v4 = H10 (implied by --p0)")
    ap.add_argument("--p0", default="q8", choices=list(P0_MODES), help="H10: input projection path. q8 = int8 activations (today); a32w8 = fp32 activations x the same int8 weights (lay 6, +0 bytes); a32w16 = fp32 activations x fp16 weights (lay 7); a16w8 = int16 activations x the same int8 weights (lay 8, +0 bytes, VNNI)")
    ap.add_argument("--fast-gelu", action="store_true", help="v2 flag bit 0: rational-erf GELU (opt-in; needs its own parity run)")
    ap.add_argument("--int4", default="", help="H5: comma list of matrices stored int4 asym (F1's quantiser): ih, head1, hh, proj1. E.g. ih,head1")
    ap.add_argument("--int4-block", type=int, default=64, choices=[32, 64, 128], help="H5: rows per int4 block along K (F1 measured 64)")
    ap.add_argument("--int4-scale", default="f16", choices=["f16", "f32"], help="H5: stored dtype of the per-block scale and min (f16 = 0.77x bytes; f32 = F1's exact grid)")
    ap.add_argument("--no-zip", action="store_true", help="stop after the blob + selfcheck.npz + init self-check (research runs)")
    ap.add_argument("--hh-dense", action="store_true", help="materialise W_hh for --hh-rank members (v1 blob) instead of the factored path")
    ap.add_argument("--onnx-members", default="",
                    help="comma list of member indices for the ONNX FALLBACK graph (default: all if N <= 3, else the first 3). "
                         "An N > 3 ONNX graph cannot finish inside 60 min (Z10: ~22 us + 14-17 us/member), so a fallback "
                         "built from every member would be a guaranteed timeout, not a fallback.")
    ap.add_argument("--onnx-ckpts", default="",
                    help="H11: comma list of OTHER checkpoints for the ONNX fallback graph (e.g. 3 non-posfeat members); overrides "
                         "--onnx-members. Default for posfeat members: the first 3 members with the posfeat state patched into the graph.")
    a = ap.parse_args()
    mults = tuple(float(v) for v in a.mults.split(","))
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    ws, refs = load_members(a.ckpts, mults)
    int4 = tuple(v for v in a.int4.split(",") if v)
    kw = dict(factored=not a.hh_dense, blob=a.blob, fast_gelu=a.fast_gelu, int4=int4, int4_blk=a.int4_block, int4_sdt=a.int4_scale, p0=a.p0)
    twins = []
    blob = write_blob(ws, twins=twins, **kw)
    (out/"weights.wnn").write_bytes(blob)
    sol = load_sol(); B = sol.Blob(blob)
    mb, tot = member_bytes(ws, **kw)
    mb8, tot8 = member_bytes(ws, **dict(kw, int4=(), blob="v2" if a.blob in ("v3", "v4") else a.blob, p0="q8"))
    _, totq8 = member_bytes(ws, **dict(kw, p0="q8", blob="v3" if a.blob == "v4" else a.blob))
    print(f"bytes: blob {tot:,} ({tot/1e6:.3f} MB); single-member blobs {mb[0]:,} .. {max(mb):,}; int8-only equivalent {tot8:,} "
          f"-> ratio {tot/tot8:.4f}; int4 {int4 or 'none'} block {a.int4_block} scale {a.int4_scale}; p0 {a.p0} "
          f"(same build with p0 q8: {totq8:,} B -> {tot/totq8:.4f}x)", flush=True)
    # torch parity of the blob (NumPy emulation of the C arithmetic), fused_build's 400-row protocol
    Xr = np.random.default_rng(0).standard_normal((400, 112)).astype(np.float32)
    ref = torch_mean(refs, Xr, seq=True)          # fused_build's protocol: the 400 rows are ONE sequence, state carried
    em = sol.ref_run(B, Xr)[0]
    d = np.abs(em - ref).max()
    print(f"blob: {len(blob)/1e6:.3f} MB, N={B.N}, P={B.P}, depth={B.D}, ver {B.ver} flags {B.flags} p0 lay {B.p0.get('lay', 2)} "
          f"posfeat {[bool(w.get('posfeat')) for w in ws]}; emulation vs torch mean on 400 rows (one sequence): max |diff| {d:.2e}", flush=True)
    if int4:
        tref = torch_mean(twin_refs(refs, twins), Xr, seq=True)
        dt_ = np.abs(em - tref).max()
        corr = lambda p, q: min(float(np.corrcoef(p[:, k], q[:, k])[0, 1]) for k in (0, 1))
        print(f"int4 twin gate: emulation vs fake-quant torch twin max |diff| {dt_:.2e} (corr {corr(em, tref):.6f}) vs fp32 {d:.2e} (corr {corr(em, ref):.6f}); "
              f"twin vs fp32 max |diff| {np.abs(tref - ref).max():.2e}", flush=True)
        if dt_ > d * 1.05 + 1e-3: raise SystemExit("the emulation is not closer to the fake-quant twin than to fp32 -- the int4 codes or grid are wrong; refusing")
        # the 400 random rows are far out of distribution (most features clamp at +-10); there the int4 fake-quant itself moves the
        # mean by up to ~0.1 (twin vs fp32 above), so the fp32 distance is only a sanity bound and the gate is vs the twin at the
        # int8-noise scale (v118 int8 vs fp32 on the same rows: 2.6e-2)
        if dt_ > 0.05 or d > 0.2: raise SystemExit("int4 blob too far from its fake-quant twin / from fp32 -- refusing")
    elif d > 0.1: raise SystemExit("blob does not reproduce the torch member mean -- refusing")
    write_selfcheck(out, blob, refs, sol)
    if a.no_zip:
        print("--no-zip: blob and selfcheck written to %s; no ONNX graph, no zip" % out, flush=True); raise SystemExit(0)
    if not a.no_onnx:
        import onnx
        from fused_build import build_fused, run_graph
        if a.onnx_ckpts:                                  # H11: the fallback graph from OTHER members (their own torch mean is the reference)
            ows, orefs = load_members([v for v in a.onnx_ckpts.split(",") if v], mults)
            oref = torch_mean(orefs, Xr, seq=True)
            print("ONNX fallback graph: %d SEPARATE members %s (posfeat %s)" % (len(ows), a.onnx_ckpts, [w.get("posfeat") for w in ows]), flush=True)
        else:
            oi = [int(v) for v in a.onnx_members.split(",")] if a.onnx_members else list(range(min(3, len(ws))))
            ows = [ws[i] for i in oi]
            oref = ref if len(oi) == len(ws) else torch_mean([refs[i] for i in oi], Xr, seq=True)
            print("ONNX fallback graph: members %s of %d (posfeat %s)" % (oi, len(ws), [w.get("posfeat") for w in ows]), flush=True)
        for mode in ("fp32", "nbits"):
            m = build_fused(ows, mode=mode, bits=8, block=128)
            if any(w.get("posfeat") for w in ows): m = add_posfeat_onnx(m, ows)       # H11: pcnt state + exact fp32 posfeat term
            pth = out/("model_fp32.onnx" if mode == "fp32" else "model.onnx"); onnx.save(m, str(pth))
            got, _ = run_graph(pth, Xr, ows[0]["hidden"], ows[0]["layers"], len(ows))
            dd = np.abs(got - oref).max()
            print(f"onnx {mode}: {len(m.graph.node)} nodes, vs torch mean max |diff| {dd:.2e}", flush=True)
            if mode == "fp32" and dd > 1e-4: raise SystemExit("fused fp32 graph does not match the member mean")
        (out/"model_fp32.onnx").unlink()
        if a.ref_onnx_zip:
            refb = zipfile.ZipFile(a.ref_onnx_zip).read("model.onnx")
            same = refb == (out/"model.onnx").read_bytes()
            print("model.onnx byte-identical to %s: %s" % (a.ref_onnx_zip, same), flush=True)
            if not same: raise SystemExit("model.onnx differs from the reference zip's -- refusing")
    shutil.copyfile(HERE/"wnnfast.c", out/"wnnfast.c")
    shutil.copyfile(HERE/"solution_native.py", out/"solution.py")
    files = ["solution.py", "wnnfast.c", "weights.wnn", "selfcheck.npz"] + ([] if a.no_onnx else ["model.onnx"])
    if a.prebuilt:
        shutil.copyfile(a.prebuilt, out/"wnnfast_x86_64.so"); files.append("wnnfast_x86_64.so")
    zp = out.parent/("submission_%s.zip" % out.name)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files: z.write(out/f, f)
    print("packaged %s (%.2f MB): %s" % (zp, zp.stat().st_size / 1e6, files))
    for f in files: print("  %-20s sha256 %s" % (f, hashlib.sha256((out/f).read_bytes()).hexdigest()))
    print("  zip sha256 %s" % hashlib.sha256(zp.read_bytes()).hexdigest())
