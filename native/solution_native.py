"""[research H11 dev copy: blob v5 = v4 + exact --posfeat support (header flags bit 2; one fp32 section wpos[sum P] after p0;
 the kernel keeps a per-sequence row counter, reset with the state, and adds (rows_seen / 20000) * wpos to the projection
 pre-activation in fp32 -- refgru.py's `row index / 20000` input column, exactly);
 research H10 dev copy: blob v4 = v3 + an exact fp32-activation path for the input projection p0 (header flags bit 1;
 p0 layout 6 = int8 weights k-major with lay-2 scales, 7 = fp16 weights, 8 = int16 activations x lay-2 int8 weights);
 research H5: blob v3 = v2 + per-matrix int4 (asym
 min/max, block 32/64/128, fp32 or fp16 scale+min) matrices; research RT: blob v2 = per-member head width + factored W_hh]
Z52 native submission: fused N-member refgru int8 step in C (wnnfast.c), with a fallback chain.

  (a) gcc -O3 -march=native the shipped wnnfast.c into a fresh temp dir (60 s timeout; retried without -march)
  (b) the PREBUILT wnnfast_x86_64.so shipped in the zip (generic x86-64, runtime AVX-512 VNNI / AVX2 / scalar dispatch),
      loaded from a temp copy, then in place (either location may be mounted noexec)
  (c) the fused ONNX graph (model.onnx, exactly the IOBinding loop of src/infer/solution_iob.py)

Each native library is first self-checked in a CHILD process (python solution.py --wnn-probe LIB): a crash there
(illegal instruction, segfault) rejects the library without touching the scorer process.

A native library is only used after an end-to-end init self-check (_selfcheck): the selected ISA must match a NumPy
emulation of the kernel's arithmetic (int8, int4 and the H10 fp32-activation p0), the library's own scalar path, the build machine's outputs stored in selfcheck.npz
(bit-exact by design), and the fp32 torch member mean stored there (int8-noise tolerance, corr >= 0.999 on real rows).
Any failure -> next link in the chain.
Test hook (default off): env WNN_FORCE = onnx | prebuilt | scalar | nocc | badso | badcheck | crash.
"""
from __future__ import annotations
import os, sys, ctypes, subprocess, tempfile, shutil
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
_ZERO2 = np.zeros(2, dtype=np.float32)
_FORCE = os.environ.get("WNN_FORCE", "")
MAGIC, END, BLK = 0x464E4E57, 0x444E4557, 128
F32 = np.float32


# ----------------------------------------------------------------------------------------- blob (mirrors wnnfast.c)
class Blob:
    """Parse the weight blob exactly as wnn_parse() does (64-byte aligned sections, same order)."""

    def __init__(self, buf: bytes):
        self.buf = buf; self.off = 0
        hd = self._i32(16)
        if hd[0] != MAGIC or hd[1] not in (1, 2, 3, 4, 5): raise ValueError("bad blob header")
        self.ver = int(hd[1])
        self.N, self.H, self.Hh, self.F = (int(v) for v in hd[2:6])
        self.clamp = F32(hd[6:7].view(np.float32)[0]); self.eps = F32(hd[7:8].view(np.float32)[0])
        self.flags = int(hd[8]) if self.ver >= 2 else 0                                  # bit 0: rational-erf GELU; bit 1 (v4): fp32-activation p0; bit 2 (v5): posfeat
        if self.flags & ~(7 if self.ver >= 5 else (3 if self.ver >= 4 else 1)): raise ValueError("unknown blob flags")
        self.inv = self._f32(self.F); self.add = self._f32(self.F)
        MH = 4 if self.ver >= 2 else 2
        mh = self._i32(MH * self.N)
        self.P = [int(mh[MH * i]) for i in range(self.N)]; self.D = [int(mh[MH * i + 1]) for i in range(self.N)]
        self.R = [int(mh[MH * i + 2]) if MH == 4 else 0 for i in range(self.N)]       # research RT: factored W_hh rank
        self.Hm = [int(mh[MH * i + 3]) if MH == 4 else self.Hh for i in range(self.N)]  # research RT: per-member head width
        self.p0 = self._qm(self.F, sum(self.P), p0=True)
        if bool(self.flags & 2) != (self.p0.get("lay", 2) >= 6): raise ValueError("p0 layout / flag mismatch")   # H10
        self.posfeat = bool(self.flags & 4)                                                                     # H11
        self.wpos = self._f32(sum(self.P)) if self.posfeat else None
        self.m = []
        for i in range(self.N):
            P, H, Hh = self.P[i], self.H, self.Hm[i]; mm = {"ln": [(self._f32(P), self._f32(P))], "px": [None]}
            for d in range(1, self.D[i]):
                mm["px"].append(self._qm(P, P)); mm["ln"].append((self._f32(P), self._f32(P)))
            mm["ih"] = self._qm(P, 3 * H)
            if self.R[i]:
                mm["hb"] = self._qm(H, self.R[i]); mm["ha"] = self._qm(self.R[i], 3 * H)
            else:
                mm["hh"] = self._qm(H, 3 * H)
            mm["ln2"] = (self._f32(H + P), self._f32(H + P)); mm["h1"] = self._qm(H + P, Hh)
            mm["w2"] = self._f32(2 * Hh).reshape(2, Hh)
            self.m.append(mm)
        self.b2 = self._f32(2)
        tail = self._i32(2)
        if tail[0] != END or tail[1] != self.N: raise ValueError("bad blob tail")

    def _take(self, nbytes):
        self.off = (self.off + 63) & ~63
        if self.off + nbytes > len(self.buf): raise ValueError("blob truncated")
        b = self.buf[self.off:self.off + nbytes]; self.off += nbytes; return b

    def _i32(self, n): return np.frombuffer(self._take(4 * n), dtype="<i4").astype(np.int64)
    def _f32(self, n): return np.frombuffer(self._take(4 * n), dtype="<f4").copy()

    def _qm(self, K, Nc, p0=False):
        hd = self._i32(4)
        if hd[0] != K or hd[1] != Nc: raise ValueError("matrix shape mismatch")
        nb = (K + BLK - 1) // BLK
        if self.ver >= 4 and p0 and hd[3] in (6, 7, 8):   # H10 p0: lay 6 [q int8 [K][64] | sw (nb,64) | bias], lay 7 [w f16 [K][64] | bias], lay 8 [q int8 [K/2][64][2] | sw | bias]
            if hd[2] != nb: raise ValueError("bad p0 block count")
            G = Nc // 64; lay = int(hd[3]); gs = K * 128 + 256 if lay == 7 else K * 64 + nb * 256 + 256
            raw = np.frombuffer(self._take(gs * G), dtype=np.uint8).reshape(G, gs)
            if lay == 8:
                q = np.ascontiguousarray(raw[:, :K * 64]).view(np.int8).reshape(G, K // 2, 64, 2).transpose(1, 3, 0, 2).reshape(K, Nc).astype(np.int32)
                sw = np.ascontiguousarray(raw[:, K * 64:K * 64 + nb * 256]).view("<f4").reshape(G, nb, 64).transpose(1, 0, 2).reshape(nb, Nc).copy()
                bias = np.ascontiguousarray(raw[:, K * 64 + nb * 256:]).view("<f4").reshape(Nc).copy()
                return {"K": K, "Nc": Nc, "nb": nb, "lay": 8, "q": q, "sw": sw, "b": bias}
            if lay == 6:
                wf = np.ascontiguousarray(raw[:, :K * 64]).view(np.int8).reshape(G, K, 64).transpose(1, 0, 2).reshape(K, Nc).astype(np.float32)
                sw = np.ascontiguousarray(raw[:, K * 64:K * 64 + nb * 256]).view("<f4").reshape(G, nb, 64).transpose(1, 0, 2).reshape(nb, Nc).copy()
                bias = np.ascontiguousarray(raw[:, K * 64 + nb * 256:]).view("<f4").reshape(Nc).copy()
                return {"K": K, "Nc": Nc, "nb": nb, "lay": 6, "wf": wf, "sw": sw, "b": bias}
            wf = np.ascontiguousarray(raw[:, :K * 128]).view("<f2").reshape(G, K, 64).transpose(1, 0, 2).reshape(K, Nc).astype(np.float32)   # exact
            bias = np.ascontiguousarray(raw[:, K * 128:]).view("<f4").reshape(Nc).copy()
            return {"K": K, "Nc": Nc, "nb": nb, "lay": 7, "wf": wf, "b": bias}
        if self.ver >= 3 and hd[3] in (4, 5):  # H5 int4 asym: per group [nibbles K*32 | s (nb4,64) | min (nb4,64) | bias (64)]
            nb4 = int(hd[2]); ssz = 4 if hd[3] == 4 else 2; dt = "<f4" if ssz == 4 else "<f2"
            if nb4 <= 0 or K % nb4 or (K // nb4) not in (32, 64, 128): raise ValueError("bad int4 block")
            G = Nc // 64; gs = K * 32 + nb4 * 128 * ssz + 256
            raw = np.frombuffer(self._take(gs * G), dtype=np.uint8).reshape(G, gs)
            nib = raw[:, :K * 32].reshape(G, K // 4, 2, 16, 4)
            t = np.empty((G, K // 4, 4, 16, 4), np.int32)
            t[:, :, 0::2] = nib & 15; t[:, :, 1::2] = nib >> 4                    # tile 2v = low nibbles, 2v+1 = high
            q = t.transpose(1, 4, 0, 2, 3).reshape(K, Nc)
            o = K * 32
            s = np.ascontiguousarray(raw[:, o:o + nb4 * 64 * ssz]).view(dt).reshape(G, nb4, 64).transpose(1, 0, 2).reshape(nb4, Nc).astype(np.float32)
            o += nb4 * 64 * ssz
            mn = np.ascontiguousarray(raw[:, o:o + nb4 * 64 * ssz]).view(dt).reshape(G, nb4, 64).transpose(1, 0, 2).reshape(nb4, Nc).astype(np.float32)
            o += nb4 * 64 * ssz
            bias = np.ascontiguousarray(raw[:, o:o + 256]).view("<f4").reshape(Nc).copy()
            return {"K": K, "Nc": Nc, "nb": nb, "nb4": nb4, "blk4": K // nb4, "q": q, "s": s, "mn": mn, "b": bias}
        if self.ver >= 2:                      # lay 2: per 64-col group [u8 weights (q+128) | sw (nb,64) | bias (64)], one stream
            if hd[3] != 2: raise ValueError("bad matrix layout")
            G = Nc // 64; gs = K * 64 + nb * 256 + 256
            raw = np.frombuffer(self._take(gs * G), dtype=np.uint8).reshape(G, gs)
            q = (raw[:, :K * 64].astype(np.int32) - 128).reshape(G, K // 4, 4, 16, 4).transpose(1, 4, 0, 2, 3).reshape(K, Nc)
            sw = np.ascontiguousarray(raw[:, K * 64:K * 64 + nb * 256]).view("<f4").reshape(G, nb, 64).transpose(1, 0, 2).reshape(nb, Nc).copy()
            bias = np.ascontiguousarray(raw[:, K * 64 + nb * 256:]).view("<f4").reshape(Nc).copy()
            return {"K": K, "Nc": Nc, "nb": nb, "q": q, "sw": sw, "b": bias}
        Wp = np.frombuffer(self._take(K * Nc), dtype=np.int8)
        q = Wp.reshape(Nc // 64, K // 4, 4, 16, 4).transpose(1, 4, 0, 2, 3).reshape(K, Nc)   # unpack -> (K, Nc)
        sw = self._f32(nb * Nc).reshape(nb, Nc); _ = self._i32(nb * Nc); bias = self._f32(Nc)
        return {"K": K, "Nc": Nc, "nb": nb, "q": q.astype(np.int32), "sw": sw, "b": bias}


# ----------------------------------------------------------------------------------------- NumPy emulation (float32, same op order)
def _wexp(x):
    x = np.minimum(np.maximum(x, F32(-87.0)), F32(88.0)).astype(F32)
    t = x * F32(1.44269504088896341)
    n = (t + F32(12582912.0)) - F32(12582912.0)
    r = x - n * F32(0.693359375); r = r - n * F32(-2.12194440e-4)
    p = np.full_like(r, F32(1.9875691500e-4))
    for cf in (1.3981999507e-3, 8.3334519073e-3, 4.1665795894e-2, 1.6666665459e-1, 5.0000001201e-1):
        p = p * r + F32(cf)
    y = p * (r * r); y = y + r; y = y + F32(1.0)
    s = ((n.astype(np.int32) + 127) << 23).astype(np.int32).view(F32)
    return (y * s).astype(F32)


def _gelu_erf(x):
    zz = x * F32(-0.70710678118654752); z = np.abs(zz)
    t = F32(1.0) / (F32(1.0) + F32(0.5) * z)
    p = np.full_like(t, F32(0.17087277))
    for cf in (-0.82215223, 1.48851587, -1.13520398, 0.27886807, -0.18628806, 0.09678418, 0.37409196, 1.00002368, -1.26551223):
        p = F32(cf) + t * p
    e = t * _wexp((-z * z) + p)
    erfc = np.where(zz >= 0, e, F32(2.0) - e).astype(F32)
    return (x * (F32(0.5) * erfc)).astype(F32)


_ERF_A = (-2.72614225801306e-10, 2.77068142495902e-08, -2.10102402082508e-06, -5.69250639462346e-05, -7.34990630326855e-04,
          -2.95459980854025e-03, -1.60960333262415e-02)
_ERF_B = (-1.45660718464996e-05, -2.13374055278905e-04, -1.68282697438203e-03, -7.37332916720468e-03, -1.42647390514189e-02)


def _gelu_r(x):
    """opt-in GELU (blob flag bit 0): rational erf, mirrors vgelu_r in wnnfast.c op for op."""
    s = (x * F32(0.70710678118654752)).astype(F32)
    s = np.minimum(np.maximum(s, F32(-4.0)), F32(4.0)).astype(F32); s2 = s * s
    p = np.full_like(s, F32(_ERF_A[0]))
    for cf in _ERF_A[1:]: p = p * s2 + F32(cf)
    p = p * s
    q = np.full_like(s, F32(_ERF_B[0]))
    for cf in _ERF_B[1:]: q = q * s2 + F32(cf)
    return (x * (F32(0.5) + F32(0.5) * (p / q))).astype(F32)


def _sigm(x): return (F32(1.0) / (F32(1.0) + _wexp(-x))).astype(F32)


def _tanh(x):
    x = np.minimum(np.maximum(x, F32(-9.0)), F32(9.0)).astype(F32)
    return (F32(1.0) - F32(2.0) / (_wexp(F32(2.0) * x) + F32(1.0))).astype(F32)


def _sum16(x):
    s = np.zeros(16, F32)
    for row in x.reshape(-1, 16): s = s + row
    w = 8
    while w >= 1:
        s[:w] = s[:w] + s[w:2 * w]; w >>= 1
    return s[0]


def _ln(x, g, b, eps):
    n = F32(x.size); mean = _sum16(x) / n
    d = x - mean; var = _sum16(d * d) / n
    rs = F32(1.0) / np.sqrt(var + eps, dtype=F32)
    return (((x - mean) * rs) * g + b).astype(F32)


def _quant(a):
    K = a.size; nb = (K + BLK - 1) // BLK; q = np.zeros(K, np.int32); sa = np.zeros(nb, F32)
    for b in range(nb):
        s = a[b * BLK:(b + 1) * BLK]; amax = F32(np.abs(s).max())
        inv = F32(127.0) / amax if amax > 0 else F32(0.0)
        sa[b] = amax / F32(127.0)
        v = (s * inv + F32(12582912.0)) - F32(12582912.0)
        q[b * BLK:(b + 1) * BLK] = np.minimum(np.maximum(v, F32(-127.0)), F32(127.0)).astype(np.int32)
    return q, sa


def _qmv(M, a):
    q, sa = _quant(a); y = M["b"].copy()
    if "s" in M:                                                                  # H5 int4 asym (mirrors qmv4 op for op)
        blk = M["blk4"]
        for b in range(M["nb4"]):
            qa = q[b * blk:(b + 1) * blk]
            acc = qa @ M["q"][b * blk:(b + 1) * blk]                              # exact int64, q in 0..15
            y = y + acc.astype(F32) * (sa[(b * blk) // BLK] * M["s"][b])
            y = y + F32(int(qa.sum())) * (sa[(b * blk) // BLK] * M["mn"][b])
        return y.astype(F32)
    for b in range(M["nb"]):
        acc = q[b * BLK:(b + 1) * BLK] @ M["q"][b * BLK:(b + 1) * BLK]          # exact int64
        y = y + acc.astype(F32) * (sa[b] * M["sw"][b])
    return y.astype(F32)


def _quant16(a):
    """H10 lay 8: symmetric int16 per 128-block (the int8 quantiser with 32767 in place of 127; mirrors quant16 op for op)."""
    K = a.size; nb = (K + BLK - 1) // BLK; q = np.zeros(K, np.int64); sa = np.zeros(nb, F32)
    for b in range(nb):
        s = a[b * BLK:(b + 1) * BLK]; amax = F32(np.abs(s).max())
        inv = F32(32767.0) / amax if amax > 0 else F32(0.0)
        sa[b] = amax / F32(32767.0)
        v = (s * inv + F32(12582912.0)) - F32(12582912.0)
        q[b * BLK:(b + 1) * BLK] = np.minimum(np.maximum(v, F32(-32767.0)), F32(32767.0)).astype(np.int64)
    return q, sa


def _qmv16(M, xs):
    """H10 lay 8 (p0): int16 activations x int8 weights, exact integer dots, lay 2's epilogue (mirrors qmv16 op for op)."""
    q, sa = _quant16(np.asarray(xs, F32)); y = M["b"].copy()
    for b in range(M["nb"]):
        acc = q[b * BLK:(b + 1) * BLK] @ M["q"][b * BLK:(b + 1) * BLK].astype(np.int64)      # exact int64
        y = y + acc.astype(F32) * (sa[b] * M["sw"][b])
    return y.astype(F32)


def _qmvf(M, xs):
    """H10 lay 6/7 (p0): fp32 activations, no quantisation; mirrors qmvf op for op (lane = column, k sequential, mul then add):
    lay 6: per 128-block acc = ((0 + xs[k0]*q[k0]) + xs[k0+1]*q[k0+1]) + ...; y = y + acc * sw[b]  (y starts at the bias);
    lay 7: y = ((b + xs[0]*w[0]) + xs[1]*w[1]) + ... with the fp16 weights converted exactly."""
    xs = np.asarray(xs, F32); y = M["b"].copy(); wf = M["wf"]; K = M["K"]
    if M["lay"] == 6:
        for b in range(M["nb"]):
            k0, k1 = b * BLK, min((b + 1) * BLK, K)
            acc = np.zeros(M["Nc"], F32)
            for k in range(k0, k1): acc = acc + xs[k] * wf[k]
            y = y + acc * M["sw"][b]
        return y.astype(F32)
    for k in range(K): y = y + xs[k] * wf[k]
    return y.astype(F32)


def ref_run(B: Blob, X, h=None):
    """Emulate wnn_step over rows X (n, F), state carried. Returns (n, 2) float32 and the final state.
    State = list of the N member hidden vectors; a posfeat blob (H11) carries one more entry, a 1-element float32 array
    holding the rows stepped since the reset (exactly torch's state slot `t0`), 0 at a reset."""
    H = B.H; h = [np.zeros(H, F32) for _ in range(B.N)] + ([np.zeros(1, F32)] if getattr(B, "posfeat", False) else []) if h is None else h
    _gelu = _gelu_r if getattr(B, "flags", 0) & 1 else _gelu_erf
    out = np.zeros((len(X), 2), F32)
    for r, x in enumerate(np.asarray(X, F32)):
        xs = x * B.inv + B.add
        xs = np.where(np.isnan(xs), F32(0.0), xs)
        xs = np.minimum(np.maximum(xs, -B.clamp), B.clamp).astype(F32)
        lp = B.p0.get("lay", 2)
        y0 = _qmv16(B.p0, xs) if lp == 8 else (_qmvf(B.p0, xs) if lp >= 6 else _qmv(B.p0, xs)); off = 0; acc = [F32(0.0), F32(0.0)]
        if getattr(B, "posfeat", False):                                       # H11: y0 + (rows_seen / 20000) * wpos, one mul + one add
            pf = F32(h[B.N][0]) / F32(20000.0)
            y0 = (y0 + pf * B.wpos).astype(F32)
            h[B.N][0] = F32(h[B.N][0] + F32(1.0))
        for i, mm in enumerate(B.m):
            P = B.P[i]; z = _ln(_gelu(y0[off:off + P]), *mm["ln"][0], B.eps); off += P
            for d in range(1, B.D[i]):
                z = _ln(_gelu(_qmv(mm["px"][d], z)), *mm["ln"][d], B.eps)
            gi = _qmv(mm["ih"], z)
            gh = _qmv(mm["ha"], _qmv(mm["hb"], h[i])) if "hb" in mm else _qmv(mm["hh"], h[i])
            rr = _sigm(gi[:H] + gh[:H]); u = _sigm(gi[H:2 * H] + gh[H:2 * H])
            n = _tanh(gi[2 * H:] + rr * gh[2 * H:])
            h[i] = (n + u * (h[i] - n)).astype(F32)
            c = _ln(np.concatenate([h[i], z]), *mm["ln2"], B.eps)
            hd = _gelu(_qmv(mm["h1"], c))
            acc[0] = F32(acc[0] + _sum16(hd * mm["w2"][0])); acc[1] = F32(acc[1] + _sum16(hd * mm["w2"][1]))
        out[r] = [B.b2[0] + acc[0], B.b2[1] + acc[1]]
    return out, h


def selfcheck_rows(B: Blob, n=64, seed=12345):
    """Synthetic rows in RAW feature units: standardised N(0, 1.5) plus a few far outliers (exercise the clamp)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, B.F)).astype(F32) * F32(1.5)
    z[::7, ::5] *= F32(20.0)
    inv = np.where(B.inv == 0, F32(1.0), B.inv)
    return ((z - B.add) / inv).astype(F32)


# ----------------------------------------------------------------------------------------- native backend
class Native:
    def __init__(self, so_path: str, blob: bytes, isa: int = -1):
        lib = ctypes.CDLL(so_path)
        if lib.wnn_abi() != 1: raise RuntimeError("wnnfast ABI mismatch")
        vp, sz, ci = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        lib.wnn_ws_size.argtypes = [vp, sz]; lib.wnn_ws_size.restype = sz
        lib.wnn_init.argtypes = [vp, sz, vp, sz, ci]; lib.wnn_init.restype = ci
        lib.wnn_step.argtypes = [vp, vp, vp]; lib.wnn_step.restype = ci
        lib.wnn_run.argtypes = [vp, vp, vp, ci]; lib.wnn_run.restype = ci
        lib.wnn_reset.argtypes = [vp]; lib.wnn_reset.restype = None
        lib.wnn_cpu.restype = ci
        try:                                                           # H11 kernels export the posfeat counter (tests only)
            lib.wnn_pos.argtypes = [vp]; lib.wnn_pos.restype = ci; self._pos = lib.wnn_pos
        except AttributeError:
            self._pos = None
        self.lib = lib
        raw = np.zeros(len(blob) + 128, np.uint8)                     # 64-byte aligned copy, kept alive
        a = (-raw.ctypes.data) % 64
        self.blob = raw[a:a + len(blob)]; self.blob[:] = np.frombuffer(blob, np.uint8)
        need = lib.wnn_ws_size(self.blob.ctypes.data, len(blob))
        if need == 0: raise RuntimeError("wnnfast rejected the blob")
        self.ws = np.zeros(need + 128, np.uint8)
        got = lib.wnn_init(self.ws.ctypes.data, self.ws.nbytes, self.blob.ctypes.data, len(blob), isa)
        if got < 0: raise RuntimeError("wnn_init failed (%d)" % got)
        self.isa = got
        self.F = Blob(blob).F
        self.x = np.zeros(self.F, F32); self.o = np.zeros(2, F32)
        self._ws = ctypes.c_void_p(self.ws.ctypes.data)
        self._xp = ctypes.c_void_p(self.x.ctypes.data); self._op = ctypes.c_void_p(self.o.ctypes.data)
        self._step = lib.wnn_step; self._reset = lib.wnn_reset

    def reset(self): self._reset(self._ws)

    def pos(self): return int(self._pos(self._ws)) if self._pos is not None else -1   # H11: rows stepped since the reset

    def run(self, X):
        X = np.ascontiguousarray(X, F32); out = np.zeros((len(X), 2), F32)
        self.lib.wnn_run(self._ws, X.ctypes.data, out.ctypes.data, len(X)); return out


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else 0.0


def _selfcheck(nat: Native, blob: bytes, so_path: str):
    """End-to-end init check; the native path is used only if EVERY item holds.
      a) 64 synthetic rows: selected ISA == NumPy emulation of the int8 arithmetic (<= 1e-4) and == the library's scalar path (<= 1e-5)
      b) selfcheck.npz (written by the builder for THIS blob, sha256-tied): synthetic 512 / real 1024 / non-finite 16 rows,
         each from a reset: native == the build machine's outputs (<= 1e-4; the C is bit-exact across ISAs, compilers, machines)
      c) native vs the build machine's fp32 TORCH member mean: real rows max|d| <= 0.15 and corr >= 0.999 per target,
         synthetic rows corr >= 0.99 -> the blob really is the trained ensemble, not merely self-consistent
      d) reset reproducibility, finite outputs, pred sd > 1e-3 on the real rows
    Returns (ok, info)."""
    import hashlib
    B = Blob(blob); X = selfcheck_rows(B)
    ref, _ = ref_run(B, X)
    nat.reset(); got = nat.run(X); nat.reset()
    sc = Native(so_path, blob, isa=0); got0 = sc.run(X)
    if _FORCE == "badcheck": ref = ref + F32(0.5)
    d_ref, d_sc = float(np.abs(got - ref).max()), float(np.abs(got - got0).max())
    ok = bool(np.isfinite(got).all() and got.std(0).min() > 1e-4 and d_ref <= 1e-4 and d_sc <= 1e-5)
    info = {"emu": d_ref, "scalar": d_sc}
    S = np.load(str(HERE / "selfcheck.npz"))
    ok &= str(S["blob_sha256"]) == hashlib.sha256(blob).hexdigest()
    outs = {}
    for k in ("a", "b", "n"):
        nat.reset(); outs[k] = nat.run(S["x" + k]); nat.reset()
        info["stored_" + k] = float(np.abs(outs[k] - S["n" + k]).max())
        ok &= bool(np.isfinite(outs[k]).all()) and info["stored_" + k] <= 1e-4
    ob, ta, tb = outs["b"], S["ta"], S["tb"]
    info["torch_maxd"] = float(np.abs(ob - tb).max())
    info["torch_corr"] = min(_corr(ob[:, 0], tb[:, 0]), _corr(ob[:, 1], tb[:, 1]))
    info["torch_corr_synth"] = min(_corr(outs["a"][:, 0], ta[:, 0]), _corr(outs["a"][:, 1], ta[:, 1]))
    ok &= info["torch_maxd"] <= 0.15 and info["torch_corr"] >= 0.999 and info["torch_corr_synth"] >= 0.99
    nat.reset(); r1 = nat.run(S["xb"][:64]); nat.reset(); r2 = nat.run(S["xb"][:64]); nat.reset()
    ok &= bool(np.array_equal(r1, r2)) and float(ob.std(0).min()) > 1e-3
    return bool(ok), info


def _probe(so_path: str):
    """Run the self-check in a CHILD process first, so an illegal instruction or a segfault in native code can only
    kill the child, never the scorer. True = passed, False = failed or crashed, None = the child could not be launched."""
    try:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--wnn-probe", so_path],
                           capture_output=True, timeout=60, cwd=str(HERE))
    except subprocess.TimeoutExpired:                           # a hung check is a failed check
        return False
    except Exception:                                           # could not launch a child at all
        return None
    return r.returncode == 0 and b"WNN_PROBE_OK" in r.stdout


def _compile(src: Path):
    """gcc the shipped source into a fresh temp dir; returns the .so path or None."""
    cc = shutil.which("gcc") or shutil.which("cc")
    if _FORCE == "nocc": cc = None
    if not cc: return None
    d = tempfile.mkdtemp(prefix="wnn_")
    base = ["-O3", "-shared", "-fPIC", "-ffp-contract=off", "-fno-math-errno", "-fno-stack-protector",
            "-fvisibility=hidden", "-I" + str(src.parent)]
    for extra in (["-march=native"], []):
        so = os.path.join(d, "wnnfast_%s.so" % ("native" if extra else "base"))
        try:
            r = subprocess.run([cc] + base + extra + ["-o", so, str(src)], capture_output=True, timeout=60)
            if r.returncode == 0 and os.path.exists(so): return so
        except Exception:
            pass
    return None


# ----------------------------------------------------------------------------------------- ONNX fallback (= solution_iob.py)
class Onnx:
    def __init__(self):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.s = ort.InferenceSession(str(HERE / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
        ins, outs = self.s.get_inputs(), self.s.get_outputs()
        shp = lambda t: [d if isinstance(d, int) else 1 for d in t.shape]
        self.x = np.zeros(shp(next(i for i in ins if i.name == "features")), np.float32)
        self.xrow = self.x.reshape(-1)
        self.sin = [i.name for i in ins if i.name != "features"]
        self.sout = [o.name for o in outs if o.name != "prediction"]
        st_shapes = [shp(next(i for i in ins if i.name == n)) for n in self.sin]
        self.p = np.zeros(shp(next(o for o in outs if o.name == "prediction")), np.float32)
        self.pflat = self.p.reshape(-1)
        self.bank = [[np.zeros(s, np.float32) for s in st_shapes] for _ in range(2)]
        self.bind = []
        for k in range(2):
            b = self.s.io_binding()
            b.bind_input("features", "cpu", 0, np.float32, self.x.shape, self.x.ctypes.data)
            for n, arr in zip(self.sin, self.bank[k]):
                b.bind_input(n, "cpu", 0, np.float32, arr.shape, arr.ctypes.data)
            b.bind_output("prediction", "cpu", 0, np.float32, self.p.shape, self.p.ctypes.data)
            for n, arr in zip(self.sout, self.bank[1 - k]):
                b.bind_output(n, "cpu", 0, np.float32, arr.shape, arr.ctypes.data)
            self.bind.append(b)
        self.k = 0

    def reset(self):
        for bank in self.bank:
            for v in bank: v[:] = 0.0


# ----------------------------------------------------------------------------------------- scorer interface
class PredictionModel:
    def __init__(self):
        self.seq = None
        self.backend = None
        self.log = []
        blob = (HERE / "weights.wnn").read_bytes() if (HERE / "weights.wnn").exists() else None
        isa = 0 if _FORCE == "scalar" else -1
        cands = []
        if blob is not None and _FORCE not in ("onnx", "prebuilt", "badso"):
            cands.append(("compiled", lambda: _compile(HERE / "wnnfast.c")))
        if blob is not None and _FORCE != "onnx":
            def _pre():                                        # a temp copy first (the zip dir may be mounted noexec)
                so = str(HERE / "wnnfast_x86_64.so")
                if _FORCE == "badso":
                    d = tempfile.mkdtemp(prefix="wnn_"); so2 = os.path.join(d, "bad.so")
                    open(so2, "wb").write(b"\x7fELF-not-a-library"); return so2
                if not os.path.exists(so): return None
                d = tempfile.mkdtemp(prefix="wnn_"); so2 = os.path.join(d, "wnnfast_x86_64.so")
                shutil.copyfile(so, so2); return so2
            def _pre_inplace():                                # ... and in place (the temp dir may be noexec instead)
                so = str(HERE / "wnnfast_x86_64.so")
                return so if os.path.exists(so) and _FORCE != "badso" else None
            cands.append(("prebuilt", _pre))
            cands.append(("prebuilt-inplace", _pre_inplace))
        for name, make in cands:
            try:
                so = make()
                if not so: self.log.append(name + ": unavailable"); continue
                pr = _probe(so)
                if pr is False: self.log.append(name + ": child-process probe failed or crashed"); continue
                if pr is None: self.log.append(name + ": probe could not launch, in-process check only")
                nat = Native(so, blob, isa)
                ok, info = _selfcheck(nat, blob, so)
                self.log.append("%s: isa %d, selfcheck %s (%s)" % (name, nat.isa, ok, ", ".join("%s %.1e" % kv for kv in info.items())))
                if ok:
                    self.backend = "native-" + name; self.nat = nat
                    break
            except Exception as e:                              # any failure -> next link
                self.log.append("%s: %s" % (name, repr(e)[:200]))
        if self.backend is None:
            self.onnx = Onnx(); self.backend = "onnx"
        if os.environ.get("WNN_VERBOSE"):
            print("[wnn] backend=%s; %s" % (self.backend, " | ".join(self.log)), file=sys.stderr)
        if self.backend.startswith("native"):
            n = self.nat
            self._step, self._ws, self._xp, self._op = n._step, n._ws, n._xp, n._op
            self._x, self._o = n.x, n.o
            self.predict = self._predict_native
        else:
            self.predict = self._predict_onnx

    def predict(self, data_point):                              # replaced per instance by the selected backend
        return (self._predict_native if self.backend.startswith("native") else self._predict_onnx)(data_point)

    def _predict_native(self, data_point):
        if data_point.seq_ix != self.seq:
            self.seq = data_point.seq_ix
            self.nat.reset()
        self._x[:] = data_point.state
        bad = self._step(self._ws, self._xp, self._op)
        if not data_point.need_prediction:
            return None
        if bad:                                                 # non-finite output (never seen): same policy as solution_iob
            self.nat.reset()
            return _ZERO2
        return self._o.copy()

    def _predict_onnx(self, data_point):
        o = self.onnx
        if data_point.seq_ix != self.seq:
            self.seq = data_point.seq_ix
            o.reset()
        o.xrow[:] = data_point.state
        o.s.run_with_iobinding(o.bind[o.k])
        o.k ^= 1
        if not data_point.need_prediction:
            return None
        p = o.pflat[:2].copy()
        if not np.isfinite(p).all():
            o.reset()
            return _ZERO2
        return p


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == "--wnn-probe":      # child-process self-check
    _blob = (HERE / "weights.wnn").read_bytes()
    if _FORCE == "crash":                                   # TEST HOOK: simulate a native crash inside the child
        ctypes.string_at(0)
    _ok = _selfcheck(Native(sys.argv[2], _blob, 0 if _FORCE == "scalar" else -1), _blob, sys.argv[2])[0]
    print("WNN_PROBE_OK" if _ok else "WNN_PROBE_FAIL")
    sys.exit(0 if _ok else 3)
