/* wnnfast.c -- research H11 (12 Oct 2026): the H10 kernel (offline/research_H10/wnnfast.c = H5 + the fp32-activation p0 paths)
 * plus EXACT support for --posfeat members (blob v5, header flags bit 2): refgru.py feeds `row index / 20000` of the current
 * sequence as one extra input column of in_proj.0 (column 112, right after the 112 raw features and before the bookconv read;
 * the counter rides in a state slot, starts at 0 on every state reset and counts EVERY row stepped, warm-up rows included).
 * v1..v4 blobs parse and compute exactly as before (bit-identical outputs to the H5 / H10 kernels on every ISA).
 *
 * H11 (blob v5; header flags bit 2 = "posfeat"; one extra fp32 section wpos[Ptot] right after p0):
 *   - the 112-column p0 matvec is untouched (same int8/int16/fp32 path as the blob's lay word says, same bits); the posfeat
 *     column is kept EXACT in fp32 as a rank-1 term: y0[n] = y0[n] + pf * wpos[n], pf = (float)rows_seen / 20000.0f, one IEEE
 *     multiply and one IEEE add per column, lane = column, so scalar / AVX2 / AVX-512 / the NumPy emulation agree bit for bit;
 *     wpos[n] = in_proj.0.weight[n, 112] of the owning member (0 for a non-posfeat member in a mixed blob: y0 + 0 is exact);
 *   - the row counter lives in the workspace context (int, reset by wnn_init / wnn_reset, advanced once per wnn_step after
 *     the row has been fed, saturating far beyond any sequence length); it is NOT a float state slot, but it holds exactly
 *     the integers torch's float32 slot holds (< 2^24);
 *   - a v5 blob with flags bit 2 clear is a v4 blob; the bit on a v1..v4 header is rejected.
 *
 * H10 (blob v4; header flags bit 1 = "p0 takes fp32 activations"; p0's layout word 6 or 7, every other matrix as in v3):
 *   - research H9 measured that ~90% of the int8 kernel's deviation from fp32 is the per-128-block int8 quantisation of the
 *     112 standardised input features feeding the bookconv-folded in_proj.0 (p0 = 112 x sum_m P_m, ~4% of the MACs). Here the
 *     input block is NOT quantised: y0 = bias + sum_k xs[k] * W[k][n] in fp32, lane = output column, k strictly sequential,
 *     every product one IEEE multiply and every accumulation one IEEE add (no FMA), so the scalar, AVX2 and AVX-512 bodies
 *     and the NumPy emulation produce the same bits (the lane width does not change any per-column order of operations).
 *   - lay 6: the weights stay the int8 codes + per-(128-row block, column) fp32 scales of lay 2 (same bytes, k-major layout);
 *     per block acc = ((0 + xs[k0] q[k0][n]) + xs[k0+1] q[k0+1][n]) + ..., then y = y + acc * sw[b][n] (y starts at the bias).
 *   - lay 7: fp16 weights (exact -> fp32), y = ((bias + xs[0] w[0][n]) + xs[1] w[1][n]) + ...; 2x p0's bytes, no scales.
 *   - lay 8 (the cheap alternative, measured next to 6/7): the inputs are quantised to INT16 per 128-block (scale amax/32767,
 *     256x finer than today's int8), the weights stay lay 2's int8 codes + scales (k-pair interleaved for vpdpwssd / vpmaddwd),
 *     the dot products are exact int32 (|dot| <= 112*32767*127 < 2^31) and the epilogue is lay 2's two fp32 ops; same bytes.
 *   - a v4 blob with flags bit 1 clear, or a lay 6/7/8 word on any matrix but p0, is rejected; the old paths are untouched.
 *
 * H5 int4 (blob v3; matrix layout word 4 = fp32 scale/min, 5 = fp16 scale/min; only the matrices the builder marks, in
 * practice W_ih and head1, the rest stay int8 lay 2):
 *   - asymmetric min/max per (BLK4-row K block, column), BLK4 in {32, 64, 128}: q = rint((w - min) / s), s = (max - min) / 15,
 *     q in 0..15 -- exactly research F1's fq_asym quantiser (the one whose quality was measured), with the scale and min
 *     stored per block and column as fp32 (lay 4) or fp16 (lay 5; fp16 -> fp32 is exact, so the ISAs still agree);
 *   - nibbles: per 4-k step, 64 columns = 4 tiles of 16; vector v (64 bytes) holds tile 2v in the low nibbles and tile 2v+1 in
 *     the high nibbles, byte = col*4 + k (the same tile/col/k order as lay 2), so one 64-byte load + and/shift feeds two
 *     VNNI dot products; q (0..15) IS the unsigned operand (no offset), the activations are the signed one;
 *   - zero point: sum_k a_k (q s + min) = s * dot(a, q) + min * sum_k a_k, so per block y += float(dot) * (sa * s);
 *     y += float(sum_k aq) * (sa * min), both exact integers first; the same two fp32 ops in the same order on every ISA.
 *
 * One row, N members, each with its own GRU state:
 *   x -> (x*inv + add) -> clamp +-C  (NaN -> 0)                         shared
 *     -> proj0 (one collapsed matvec 112 -> sum P_m: int8 x int8, or H10 fp32 x int8/fp16) -> GELU -> LN  per member slice
 *     -> [extra proj blocks: int8 matvec P->P -> GELU -> LN]
 *     -> GRU cell (int8 W_ih; int8 W_hh dense, OR factored: h -> int8 B (H->R) -> int8 A (R->3H) + b_hh; fp32 gates)
 *     -> LN([h | z]) -> int8 head1 -> GELU -> fp32 head2 (prescaled by scale*mults/N)
 *   prediction = b2 + sum_m head2_m      (= the member mean, as in the fused ONNX graph)
 *
 * RT changes vs Z52 (outputs for a v1 blob are BIT-IDENTICAL to Z52's kernel; every float op is the same op on the
 * same operands, only fused into fewer passes and written with explicit SIMD):
 *   - elementwise work (GELU, LN, gates, quant, input prep, head2) uses explicit 16-/8-lane SIMD through one generic
 *     body (V* macros below); the scalar build of the same body is the reference path;
 *   - GELU fused with the LN mean pass; LN variance without a temp store; LN output fused with the quant amax;
 *     no memcpy into the head's concat (LN reads h and z in place); GELU fused into the head2 dot products;
 *   - blob v2: per-member head width Hh_m and an optional factored recurrence (hh rank R, 0 = dense).
 *     v1 blobs load unchanged.
 *
 * int8 scheme (same as MatMulNBits accuracy_level 4): weights symmetric int8 per (128-row K block, column);
 * activations symmetric int8 per 128-element K block, scale amax/127. Integer dot products are exact,
 * so the scalar, AVX2 and AVX-512 VNNI paths produce BIT-IDENTICAL results: every float operation is
 * done in the same order with no FMA contraction (compile with -ffp-contract=off; the pragma below
 * covers clang). Deterministic: no threads, fixed summation order.
 *
 * No libc/libm calls except memset/memcpy (exp/erf/sqrt are inline; compile with -fno-math-errno).
 * No allocation: the caller passes a workspace of wnn_ws_size() bytes and keeps the blob alive.
 *
 * The file includes ITSELF once per ISA (compile it by ABSOLUTE path, or with -I<its dir>).
 */
#ifndef WNN_BODY_PASS
#pragma STDC FP_CONTRACT OFF
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#define WNN_X86 1
#else
#define WNN_X86 0
#endif

#define WNN_MAGIC 0x464E4E57u      /* "WNNF" */
#define WNN_END 0x444E4557u        /* "WEND" */
#define WNN_VERSION_MAX 5u
#define WNN_MAXM 16
#define WNN_MAXD 4
#define WNN_BLK 128
#define WNN_MAXK 1024
#define WNN_MAXNC 8192

typedef struct {
    int K, Nc, nb, lay;     /* lay 1: Z52 (separate sw/corr/bias arrays); lay 2: grouped stream, u8 weights, no corr;
                               lay 4/5 (v3): int4 asym nibbles, fp32/fp16 scale+min per (blk4-row block, column);
                               lay 6/7 (v4, p0 only): fp32 activations; 6 = int8 k-major weights + lay-2 scales, 7 = fp16 weights;
                               lay 8 (v4, p0 only): int16 activations, int8 weights [K/2][64][2] + lay-2 scales */
    int nb4, blk4, ssz;     /* lay 4/5: number of int4 blocks, rows per block, bytes per stored scale/min (4 or 2) */
    size_t gstride;         /* lay 2: bytes per 64-column group = K*64 + nb*256 + 256; lay 4/5: K*32 + nb4*128*ssz + 256;
                               lay 6: K*64 + nb*256 + 256 (weights [K][64] int8, then sw [nb][64], bias [64]); lay 7: K*128 + 256;
                               lay 8: K*64 + nb*256 + 256 (weights [K/2][64][2] int8, then sw, bias) */
    const int8_t *W;        /* lay 1: packed [Nc/64][K/4][4 tiles][16 cols][4 k] int8 q.
                               lay 2: per group g: [K/4][4][16][4] uint8 (q ^ 0x80 = q + 128), then sw [nb][64] f32, bias [64] f32 */
    const float *sw;        /* [nb][Nc] weight scales */
    const int32_t *corr;    /* [nb][Nc] 128 * sum_{k in block} q[k][n]  (VNNI u8 offset) */
    const float *bias;      /* [Nc] */
} wnn_qmat;

typedef struct {
    int P, depth, off;      /* off: this member's slice of the collapsed proj0 output */
    int R, Hh;              /* R: recurrence rank (0 = dense W_hh); Hh: head hidden width */
    const float *ln_g[WNN_MAXD], *ln_b[WNN_MAXD];
    wnn_qmat px[WNN_MAXD];  /* px[d] for d >= 1 */
    wnn_qmat ih, hh, h1;    /* hh: dense W_hh (R == 0) */
    wnn_qmat hb, ha;        /* factored (R > 0): hb = B^T (H -> R, zero bias), ha = A^T (R -> 3H, bias b_hh) */
    const float *ln2_g, *ln2_b;
    const float *w2;        /* [2][Hh] (scale*mults/N folded in) */
} wnn_member;

typedef struct {
    int N, H, Hh, F, Ptot, Pmax, isa, ver, flags;   /* flags (v2): bit 0 = rational-erf GELU (opt-in); (v4) bit 1 = fp32-activation p0;
                                                       (v5, H11) bit 2 = posfeat: y0 += (rows_seen / 20000) * wpos */
    float clamp, eps;
    const float *inv, *add, *b2;
    const float *wpos;      /* H11: [Ptot] fp32 posfeat column of every member's in_proj.0 (flags bit 2), else 0 */
    int pos;                /* H11: rows stepped since the last reset (the posfeat counter) */
    wnn_qmat p0;
    wnn_member m[WNN_MAXM];
    float *h;               /* [N][H] recurrent state */
    float *xs, *y0, *z, *u, *gi, *gh, *cat, *hdn, *sa;
    int32_t *as;            /* per-block sum of the int8 activations (lay-2 VNNI offset correction) */
    int8_t *aq;
    int16_t *aq16;          /* H10 lay 8: int16 codes of the input block */
} wnn_ctx;

#define WNN_ALIGN(x) (((x) + 63u) & ~(size_t)63u)

/* H5: exact IEEE binary16 -> binary32 (normals, subnormals, zeros, inf, NaN); used by the scalar body and the emulation-side
 * reasoning: the SIMD bodies use vcvtph2ps, which is the same exact conversion. */
static inline float wnn_h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { int sh = 0; while (!(m & 0x400u)) { m <<= 1; sh++; } bits = s | ((uint32_t)(113 - sh) << 23) | ((m & 0x3FFu) << 13); }
    } else if (e == 31) bits = s | 0x7F800000u | (m << 13);
    else bits = s | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

/* ---------------------------------------------------------------- per-ISA bodies */
#define WNN_BODY_PASS 1
#if WNN_X86
#define ISA 2
#define SFX(n) n##_avx512
#define TGT __attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2,fma,popcnt")))
#include __FILE__
#undef ISA
#undef SFX
#undef TGT
#define ISA 1
#define SFX(n) n##_avx2
#define TGT __attribute__((target("avx2,fma,f16c,popcnt")))
#include __FILE__
#undef ISA
#undef SFX
#undef TGT
#endif
#define ISA 0
#define SFX(n) n##_scalar
#define TGT
#include __FILE__
#undef ISA
#undef SFX
#undef TGT

/* ---------------------------------------------------------------- CPU detection (no libgcc) */
static int wnn_cpu_best(void) {
#if WNN_X86
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    int osxsave = (c >> 27) & 1, avx = (c >> 28) & 1, f16c = (c >> 29) & 1;
    if (!osxsave || !avx || !f16c) return 0;           /* H5: the SIMD bodies convert fp16 scales (F16C; every AVX2/AVX-512 CPU has it) */
    unsigned xlo, xhi;
    __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    unsigned long long xcr0 = ((unsigned long long)xhi << 32) | xlo;
    if ((xcr0 & 0x6) != 0x6) return 0;
    if (__get_cpuid_max(0, 0) < 7) return 0;
    __cpuid_count(7, 0, a, b, c, d);
    int avx2 = (b >> 5) & 1;
    int f512 = (b >> 16) & 1, dq = (b >> 17) & 1, bw = (b >> 30) & 1, vl = (b >> 31) & 1;
    int vnni = (c >> 11) & 1;
    if (f512 && dq && bw && vl && vnni && avx2 && (xcr0 & 0xE6) == 0xE6) return 2;
    if (avx2) return 1;
#endif
    return 0;
}

/* ---------------------------------------------------------------- blob parsing */
typedef struct { const uint8_t *base; size_t size, off; int bad; } wnn_rd;

static const void *wnn_take(wnn_rd *r, size_t bytes) {
    r->off = WNN_ALIGN(r->off);
    if (r->bad || r->off > r->size || bytes > r->size - r->off) { r->bad = 1; return 0; }
    const void *p = r->base + r->off; r->off += bytes; return p;
}
static int wnn_i32(wnn_rd *r, int *out, int n) {
    const int32_t *p = (const int32_t *)wnn_take(r, (size_t)n * 4);
    if (!p) return -1;
    for (int i = 0; i < n; i++) { int32_t v; memcpy(&v, p + i, 4); out[i] = v; }
    return 0;
}
static int wnn_qm(wnn_rd *r, wnn_qmat *q, int K, int Nc, int ver, int isp0) {
    int hd[4];
    if (wnn_i32(r, hd, 4)) return -1;
    if (hd[0] != K || hd[1] != Nc) return -2;
    if (K <= 0 || K > WNN_MAXK || (K % 8) || Nc <= 0 || Nc > WNN_MAXNC || (Nc % 64)) return -3;
    q->K = K; q->Nc = Nc; q->nb = (K + WNN_BLK - 1) / WNN_BLK;
    q->nb4 = 0; q->blk4 = 0; q->ssz = 0;
    if (ver >= 4 && isp0 && (hd[3] == 6 || hd[3] == 7 || hd[3] == 8)) {   /* H10 p0 paths; hd[2] = nb as in lay 2 */
        if (hd[2] != q->nb) return -4;
        q->lay = hd[3];
        q->gstride = hd[3] == 7 ? (size_t)K * 128 + 256 : (size_t)K * 64 + (size_t)q->nb * 256 + 256;
        q->W = (const int8_t *)wnn_take(r, q->gstride * (size_t)(Nc / 64));
        q->sw = 0; q->corr = 0; q->bias = 0;
        return r->bad ? -5 : 0;
    }
    if (ver >= 3 && (hd[3] == 4 || hd[3] == 5)) {              /* H5: int4 asym, hd[2] = number of blk4-row blocks */
        int nb4 = hd[2];
        if (nb4 <= 0 || (K % nb4)) return -4;
        int blk = K / nb4;
        if (!(blk == 32 || blk == 64 || blk == 128)) return -4;
        q->lay = hd[3]; q->nb4 = nb4; q->blk4 = blk; q->ssz = hd[3] == 4 ? 4 : 2;
        q->gstride = (size_t)K * 32 + (size_t)nb4 * 128 * (size_t)q->ssz + 256;
        q->W = (const int8_t *)wnn_take(r, q->gstride * (size_t)(Nc / 64));
        q->sw = 0; q->corr = 0; q->bias = 0;
        return r->bad ? -5 : 0;
    }
    if (hd[2] != q->nb) return -4;
    if (ver >= 2) {
        if (hd[3] != 2) return -4;
        q->lay = 2; q->gstride = (size_t)K * 64 + (size_t)q->nb * 256 + 256;
        q->W = (const int8_t *)wnn_take(r, q->gstride * (size_t)(Nc / 64));
        q->sw = 0; q->corr = 0; q->bias = 0;
        return r->bad ? -5 : 0;
    }
    q->lay = 1; q->gstride = 0;
    q->W = (const int8_t *)wnn_take(r, (size_t)K * Nc);
    q->sw = (const float *)wnn_take(r, (size_t)q->nb * Nc * 4);
    q->corr = (const int32_t *)wnn_take(r, (size_t)q->nb * Nc * 4);
    q->bias = (const float *)wnn_take(r, (size_t)Nc * 4);
    return r->bad ? -5 : 0;
}
static const float *wnn_f32(wnn_rd *r, int n) { return (const float *)wnn_take(r, (size_t)n * 4); }

/* workspace layout; returns bytes needed (computed from the header only) */
static size_t wnn_ws_layout(wnn_ctx *c, uint8_t *ws) {
    size_t off = 0;
#define WS(ptr, type, n) do { off = WNN_ALIGN(off); if (ws) c->ptr = (type *)(ws + off); off += (size_t)(n) * sizeof(type); } while (0)
    int KM = c->H + c->Pmax; if (KM < c->F) KM = c->F;
    WS(h, float, c->N * c->H);
    WS(xs, float, WNN_ALIGN(c->F));
    WS(y0, float, c->Ptot);
    WS(z, float, c->Pmax + 64);
    WS(u, float, c->H + 64);
    WS(gi, float, 3 * c->H);
    WS(gh, float, 3 * c->H);
    WS(cat, float, KM + 64);
    WS(hdn, float, c->Hh + 64);
    WS(sa, float, 64);
    WS(as, int32_t, 64);
    WS(aq, int8_t, WNN_ALIGN(KM) + 64);
    WS(aq16, int16_t, WNN_ALIGN(c->F) + 64);
#undef WS
    return off + 64;
}

/* Parse header + all weights into *c (pointers into blob). Returns 0 or a negative error.
 * v1: member header (P, depth); one global head width Hh; dense W_hh.
 * v2: member header (P, depth, R, Hh_m); header Hh = max Hh_m; R > 0 -> W_hh stored as two matrices hb (H x R, zero
 *     bias) and ha (R x 3H, bias b_hh) in place of hh; header word 8 = flags; every matrix in the lay-2 grouped
 *     layout (one sequential stream per matrix, no corr array).
 * v3: any matrix may be int4 (lay 4/5).  v4 (H10): flags bit 1 set and p0 in lay 6 or 7 (fp32 activations) or 8 (int16). */
static int wnn_parse(wnn_ctx *c, const uint8_t *blob, size_t size) {
    memset(c, 0, sizeof(*c));
    wnn_rd r = {blob, size, 0, 0};
    int hd[16];
    if (wnn_i32(&r, hd, 16)) return -10;
    if ((uint32_t)hd[0] != WNN_MAGIC || (uint32_t)hd[1] < 1u || (uint32_t)hd[1] > WNN_VERSION_MAX) return -11;
    c->ver = hd[1];
    c->N = hd[2]; c->H = hd[3]; c->Hh = hd[4]; c->F = hd[5];
    memcpy(&c->clamp, &hd[6], 4); memcpy(&c->eps, &hd[7], 4);
    c->flags = c->ver >= 2 ? hd[8] : 0;
    if (c->flags & ~(c->ver >= 5 ? 7 : (c->ver >= 4 ? 3 : 1))) return -13;
    if (c->N < 1 || c->N > WNN_MAXM || c->H < 16 || c->H > 2048 || (c->H % 64) || c->Hh < 16 || (c->Hh % 64)
        || c->Hh > 4096 || c->F < 8 || c->F > WNN_MAXK || (c->F % 8)) return -12;
    if (!(c->clamp > 0.0f) || !(c->eps > 0.0f)) return -13;
    c->inv = wnn_f32(&r, c->F); c->add = wnn_f32(&r, c->F);
    const int MH = c->ver >= 2 ? 4 : 2;
    int mh[WNN_MAXM * 4];
    if (wnn_i32(&r, mh, MH * c->N)) return -14;
    c->Ptot = 0; c->Pmax = 0;
    for (int i = 0; i < c->N; i++) {
        int P = mh[MH * i], D = mh[MH * i + 1];
        int Rk = MH == 4 ? mh[MH * i + 2] : 0, Hh = MH == 4 ? mh[MH * i + 3] : c->Hh;
        if (P < 64 || P > 2048 || (P % 64) || D < 1 || D > WNN_MAXD) return -15;
        if (Rk < 0 || (Rk && (Rk < 64 || Rk >= c->H || (Rk % 64)))) return -15;
        if (Hh < 64 || Hh > c->Hh || (Hh % 64)) return -15;
        c->m[i].P = P; c->m[i].depth = D; c->m[i].off = c->Ptot; c->Ptot += P;
        c->m[i].R = Rk; c->m[i].Hh = Hh;
        if (P > c->Pmax) c->Pmax = P;
    }
    if (wnn_qm(&r, &c->p0, c->F, c->Ptot, c->ver, 1)) return -16;
    if (((c->flags >> 1) & 1) != (c->p0.lay >= 6)) return -24;      /* H10: flag bit 1 <=> p0 lay 6/7/8 */
    c->wpos = 0; c->pos = 0;
    if (c->flags & 4) {                                              /* H11: the posfeat column, fp32 [Ptot], right after p0 */
        c->wpos = wnn_f32(&r, c->Ptot);
        if (!c->wpos) return -25;
    }
    for (int i = 0; i < c->N; i++) {
        wnn_member *m = &c->m[i]; int P = m->P, H = c->H;
        m->ln_g[0] = wnn_f32(&r, P); m->ln_b[0] = wnn_f32(&r, P);
        for (int d = 1; d < m->depth; d++) {
            if (wnn_qm(&r, &m->px[d], P, P, c->ver, 0)) return -17;
            m->ln_g[d] = wnn_f32(&r, P); m->ln_b[d] = wnn_f32(&r, P);
        }
        if (wnn_qm(&r, &m->ih, P, 3 * H, c->ver, 0)) return -18;
        if (m->R) {
            if (wnn_qm(&r, &m->hb, H, m->R, c->ver, 0)) return -19;
            if (wnn_qm(&r, &m->ha, m->R, 3 * H, c->ver, 0)) return -19;
        } else {
            if (wnn_qm(&r, &m->hh, H, 3 * H, c->ver, 0)) return -19;
        }
        m->ln2_g = wnn_f32(&r, H + P); m->ln2_b = wnn_f32(&r, H + P);
        if (wnn_qm(&r, &m->h1, H + P, m->Hh, c->ver, 0)) return -20;
        m->w2 = wnn_f32(&r, 2 * m->Hh);
    }
    c->b2 = wnn_f32(&r, 2);
    int tail[2];
    if (wnn_i32(&r, tail, 2)) return -21;
    if ((uint32_t)tail[0] != WNN_END || tail[1] != c->N) return -22;
    if (r.bad) return -23;
    return 0;
}

/* ---------------------------------------------------------------- exported API */
#if defined(__GNUC__)
#define WNN_API __attribute__((visibility("default")))
#else
#define WNN_API
#endif

WNN_API int wnn_abi(void) { return 1; }
WNN_API int wnn_blob_version_max(void) { return (int)WNN_VERSION_MAX; }
WNN_API int wnn_cpu(void) { return wnn_cpu_best(); }

/* bytes of workspace needed for this blob (0 if the blob is invalid) */
WNN_API size_t wnn_ws_size(const uint8_t *blob, size_t size) {
    wnn_ctx c;
    if (wnn_parse(&c, blob, size)) return 0;
    return WNN_ALIGN(sizeof(wnn_ctx)) + 64 + wnn_ws_layout(&c, 0);
}

static wnn_ctx *wnn_ctx_of(void *ws) { return (wnn_ctx *)(((uintptr_t)ws + 63u) & ~(uintptr_t)63u); }

/* isa: -1 = best available, 0 scalar, 1 avx2, 2 avx512-vnni. Returns the ISA used (>= 0) or < 0. */
WNN_API int wnn_init(void *ws, size_t ws_size, const uint8_t *blob, size_t size, int isa) {
    if (!ws || !blob) return -1;
    wnn_ctx tmp;
    int e = wnn_parse(&tmp, blob, size);
    if (e) return e;
    size_t need = WNN_ALIGN(sizeof(wnn_ctx)) + 64 + wnn_ws_layout(&tmp, 0);
    if (ws_size < need) return -2;
    wnn_ctx *c = wnn_ctx_of(ws);
    memcpy(c, &tmp, sizeof(tmp));
    uint8_t *rest = (uint8_t *)c + WNN_ALIGN(sizeof(wnn_ctx));
    wnn_ws_layout(c, rest);
    int best = wnn_cpu_best();
    if (isa < 0) isa = best;
    if (isa > best) return -3;
    c->isa = isa;
    memset(c->h, 0, (size_t)c->N * c->H * sizeof(float));
    c->pos = 0;
    return isa;
}

WNN_API void wnn_reset(void *ws) {
    wnn_ctx *c = wnn_ctx_of(ws);
    memset(c->h, 0, (size_t)c->N * c->H * sizeof(float));
    c->pos = 0;                                                      /* H11: the posfeat counter restarts with the sequence */
}

/* H11: rows stepped since the last reset (the posfeat counter; tests) */
WNN_API int wnn_pos(void *ws) { return wnn_ctx_of(ws)->pos; }

WNN_API int wnn_isa(void *ws) { return wnn_ctx_of(ws)->isa; }

/* one row: x[F] -> out[2]; state advances. Returns 0, or 1 if the output is non-finite. */
WNN_API int wnn_step(void *ws, const float *x, float *out) {
    wnn_ctx *c = wnn_ctx_of(ws);
#if WNN_X86
    if (c->isa == 2) return wnn_step_avx512(c, x, out);
    if (c->isa == 1) return wnn_step_avx2(c, x, out);
#endif
    return wnn_step_scalar(c, x, out);
}

/* n rows at once (benchmarks / tests): x[n][F] -> out[n][2] */
WNN_API int wnn_run(void *ws, const float *x, float *out, int n) {
    for (int i = 0; i < n; i++) wnn_step(ws, x + (size_t)i * wnn_ctx_of(ws)->F, out + 2 * (size_t)i);
    return 0;
}

#else /* ================================================================ WNN_BODY_PASS: per-ISA kernels */

/* ---- generic SIMD layer: V = VW float lanes. Every macro is ONE IEEE op (or an exact select/bit op), so the three
 * builds compute the same bits. Selects mirror the scalar ternaries exactly, including NaN behaviour
 * (ordered compares are false on NaN, like C's <, >, >=). */
#undef VW
#undef V
#undef VM
#if ISA == 2
#define VW 16
#define V __m512
#define VM __mmask16
#define VLD(p) _mm512_loadu_ps(p)
#define VST(p, v) _mm512_storeu_ps((p), (v))
#define VSET(x) _mm512_set1_ps(x)
#define VADD(a, b) _mm512_add_ps(a, b)
#define VSUB(a, b) _mm512_sub_ps(a, b)
#define VMUL(a, b) _mm512_mul_ps(a, b)
#define VDIV(a, b) _mm512_div_ps(a, b)
#define VNEG(a) _mm512_xor_ps(a, _mm512_set1_ps(-0.0f))
#define VABS(a) _mm512_andnot_ps(_mm512_set1_ps(-0.0f), a)
#define VLT(a, b) _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)
#define VGT(a, b) _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)
#define VGE(a, b) _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ)
#define VISNAN(a) _mm512_cmp_ps_mask(a, a, _CMP_UNORD_Q)
#define VSEL(m, x, y) _mm512_mask_blend_ps((m), (y), (x))          /* m ? x : y */
#define VPOW2N(n) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvttps_epi32(n), _mm512_set1_epi32(127)), 23))
#define VSTQ8(p, q) _mm_storeu_si128((__m128i *)(p), _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(q)))   /* truncating, like (int8_t)(int32_t) */
#define VSTQ16(p, q) _mm256_storeu_si256((__m256i *)(p), _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(q)))   /* H10 lay 8: 16 int16 codes */
#elif ISA == 1
#define VW 8
#define V __m256
#define VM __m256
#define VLD(p) _mm256_loadu_ps(p)
#define VST(p, v) _mm256_storeu_ps((p), (v))
#define VSET(x) _mm256_set1_ps(x)
#define VADD(a, b) _mm256_add_ps(a, b)
#define VSUB(a, b) _mm256_sub_ps(a, b)
#define VMUL(a, b) _mm256_mul_ps(a, b)
#define VDIV(a, b) _mm256_div_ps(a, b)
#define VNEG(a) _mm256_xor_ps(a, _mm256_set1_ps(-0.0f))
#define VABS(a) _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a)
#define VLT(a, b) _mm256_cmp_ps(a, b, _CMP_LT_OQ)
#define VGT(a, b) _mm256_cmp_ps(a, b, _CMP_GT_OQ)
#define VGE(a, b) _mm256_cmp_ps(a, b, _CMP_GE_OQ)
#define VISNAN(a) _mm256_cmp_ps(a, a, _CMP_UNORD_Q)
#define VSEL(m, x, y) _mm256_blendv_ps((y), (x), (m))
#define VPOW2N(n) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127)), 23))
static inline TGT void wnn_stq8_avx2(int8_t *p, __m256 q) {     /* low byte of each int32 (truncation, like the scalar cast) */
    __m256i i = _mm256_cvttps_epi32(q);
    const __m256i sh = _mm256_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i s = _mm256_shuffle_epi8(i, sh);
    __m128i lo = _mm256_castsi256_si128(s), hi = _mm256_extracti128_si256(s, 1);
    _mm_storel_epi64((__m128i *)p, _mm_unpacklo_epi32(lo, hi));
}
#define VSTQ8(p, q) wnn_stq8_avx2((p), (q))
static inline TGT void wnn_stq16_avx2(int16_t *p, __m256 q) {    /* 8 int16 codes; values are within +-32767 so the pack never saturates */
    __m256i i = _mm256_cvttps_epi32(q);
    _mm_storeu_si128((__m128i *)p, _mm_packs_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1)));
}
#define VSTQ16(p, q) wnn_stq16_avx2((p), (q))
#else
#define VW 1
#define V float
#define VM int
#define VLD(p) (*(p))
#define VST(p, v) (*(p) = (v))
#define VSET(x) ((float)(x))
#define VADD(a, b) ((a) + (b))
#define VSUB(a, b) ((a) - (b))
#define VMUL(a, b) ((a) * (b))
#define VDIV(a, b) ((a) / (b))
#define VNEG(a) (-(a))
#define VABS(a) ((a) < 0.0f ? -(a) : (a))
#define VLT(a, b) ((a) < (b))
#define VGT(a, b) ((a) > (b))
#define VGE(a, b) ((a) >= (b))
#define VISNAN(a) ((a) != (a))
#define VSEL(m, x, y) ((m) ? (x) : (y))
static inline float wnn_pow2n_scalar(float n) { int32_t bits = ((int32_t)n + 127) << 23; float s; memcpy(&s, &bits, 4); return s; }
#define VPOW2N(n) wnn_pow2n_scalar(n)
#define VSTQ8(p, q) (*(p) = (int8_t)(int32_t)(q))
#define VSTQ16(p, q) (*(p) = (int16_t)(int32_t)(q))
#endif
#define NACC (16 / VW)      /* vector accumulators that make up the 16 interleaved partial sums of sum16 */

/* exp: Cephes expf polynomial, Cody-Waite reduction, branch-free (selects only). */
static inline TGT V SFX(vexp)(V x) {
    x = VSEL(VLT(x, VSET(-87.0f)), VSET(-87.0f), x);
    x = VSEL(VGT(x, VSET(88.0f)), VSET(88.0f), x);
    V t = VMUL(x, VSET(1.44269504088896341f));
    V n = VSUB(VADD(t, VSET(12582912.0f)), VSET(12582912.0f));          /* round to nearest */
    V r = VSUB(x, VMUL(n, VSET(0.693359375f)));
    r = VSUB(r, VMUL(n, VSET(-2.12194440e-4f)));
    V p = VSET(1.9875691500e-4f);
    p = VADD(VMUL(p, r), VSET(1.3981999507e-3f));
    p = VADD(VMUL(p, r), VSET(8.3334519073e-3f));
    p = VADD(VMUL(p, r), VSET(4.1665795894e-2f));
    p = VADD(VMUL(p, r), VSET(1.6666665459e-1f));
    p = VADD(VMUL(p, r), VSET(5.0000001201e-1f));
    V y = VMUL(p, VMUL(r, r));
    y = VADD(y, r);
    y = VADD(y, VSET(1.0f));
    return VMUL(y, VPOW2N(n));
}

/* GELU(x) = x * Phi(x), Phi(x) = 0.5 * erfc(-x/sqrt2); erfc from Numerical Recipes (frac. err < 1.2e-7) */
static inline TGT V SFX(vgelu)(V x) {
    V zz = VMUL(x, VSET(-0.70710678118654752f));                       /* erfc argument */
    V z = VSEL(VLT(zz, VSET(0.0f)), VNEG(zz), zz);
    V t = VDIV(VSET(1.0f), VADD(VSET(1.0f), VMUL(VSET(0.5f), z)));
    V p = VSET(0.17087277f);
    p = VADD(VSET(-0.82215223f), VMUL(t, p));
    p = VADD(VSET(1.48851587f), VMUL(t, p));
    p = VADD(VSET(-1.13520398f), VMUL(t, p));
    p = VADD(VSET(0.27886807f), VMUL(t, p));
    p = VADD(VSET(-0.18628806f), VMUL(t, p));
    p = VADD(VSET(0.09678418f), VMUL(t, p));
    p = VADD(VSET(0.37409196f), VMUL(t, p));
    p = VADD(VSET(1.00002368f), VMUL(t, p));
    p = VADD(VSET(-1.26551223f), VMUL(t, p));
    V e = VMUL(t, SFX(vexp)(VADD(VMUL(VNEG(z), z), p)));
    V erfc_v = VSEL(VGE(zz, VSET(0.0f)), e, VSUB(VSET(2.0f), e));
    return VMUL(x, VMUL(VSET(0.5f), erfc_v));
}
/* opt-in (blob v2 flag bit 0): GELU(x) = x * (0.5 + 0.5 * erf(x / sqrt2)) with erf as a clamped rational function
 * (odd degree-13 / even degree-8 minimax form used by Eigen/XLA; max |err| 3.7e-7 on erf, 1.05e-6 on GELU). No exp, one div,
 * two short independent Horner chains. NOT bit-identical to vgelu: blobs that set the flag carry their own selfcheck refs. */
static inline TGT V SFX(vgelu_r)(V x) {
    V s = VMUL(x, VSET(0.70710678118654752f));
    s = VSEL(VLT(s, VSET(-4.0f)), VSET(-4.0f), s);
    s = VSEL(VGT(s, VSET(4.0f)), VSET(4.0f), s);
    V s2 = VMUL(s, s);
    V p = VSET(-2.72614225801306e-10f);
    p = VADD(VMUL(p, s2), VSET(2.77068142495902e-08f));
    p = VADD(VMUL(p, s2), VSET(-2.10102402082508e-06f));
    p = VADD(VMUL(p, s2), VSET(-5.69250639462346e-05f));
    p = VADD(VMUL(p, s2), VSET(-7.34990630326855e-04f));
    p = VADD(VMUL(p, s2), VSET(-2.95459980854025e-03f));
    p = VADD(VMUL(p, s2), VSET(-1.60960333262415e-02f));
    p = VMUL(p, s);
    V q = VSET(-1.45660718464996e-05f);
    q = VADD(VMUL(q, s2), VSET(-2.13374055278905e-04f));
    q = VADD(VMUL(q, s2), VSET(-1.68282697438203e-03f));
    q = VADD(VMUL(q, s2), VSET(-7.37332916720468e-03f));
    q = VADD(VMUL(q, s2), VSET(-1.42647390514189e-02f));
    return VMUL(x, VADD(VSET(0.5f), VMUL(VSET(0.5f), VDIV(p, q))));
}
#define VGELU(fg, x) ((fg) ? SFX(vgelu_r)(x) : SFX(vgelu)(x))

static inline TGT V SFX(vsigm)(V x) { return VDIV(VSET(1.0f), VADD(VSET(1.0f), SFX(vexp)(VNEG(x)))); }
static inline TGT V SFX(vtanh)(V x) {
    x = VSEL(VLT(x, VSET(-9.0f)), VSET(-9.0f), VSEL(VGT(x, VSET(9.0f)), VSET(9.0f), x));
    return VSUB(VSET(1.0f), VDIV(VSET(2.0f), VADD(SFX(vexp)(VMUL(VSET(2.0f), x)), VSET(1.0f))));
}

/* the fixed tree that ends sum16: s[j] += s[j + w] for w = 8, 4, 2, 1 */
static inline TGT float SFX(red16)(const V *acc) {
#if ISA == 2                                            /* the same tree with shuffles: w=8 hi256+lo256, w=4, w=2, w=1 */
    __m256 t8 = _mm256_add_ps(_mm512_castps512_ps256(acc[0]), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(acc[0]), 1)));
    __m128 t4 = _mm_add_ps(_mm256_castps256_ps128(t8), _mm256_extractf128_ps(t8, 1));
    __m128 t2 = _mm_add_ps(t4, _mm_movehl_ps(t4, t4));
    __m128 t1 = _mm_add_ss(t2, _mm_shuffle_ps(t2, t2, 1));
    return _mm_cvtss_f32(t1);
#elif ISA == 1                                          /* acc[0] = s[0..7], acc[1] = s[8..15] */
    __m256 t8 = _mm256_add_ps(acc[0], acc[1]);
    __m128 t4 = _mm_add_ps(_mm256_castps256_ps128(t8), _mm256_extractf128_ps(t8, 1));
    __m128 t2 = _mm_add_ps(t4, _mm_movehl_ps(t4, t4));
    __m128 t1 = _mm_add_ss(t2, _mm_shuffle_ps(t2, t2, 1));
    return _mm_cvtss_f32(t1);
#else
    float s[16];
    for (int a = 0; a < NACC; a++) VST(s + a * VW, acc[a]);
    for (int w = 8; w >= 1; w >>= 1)
        for (int j = 0; j < w; j++) s[j] = s[j] + s[j + w];
    return s[0];
#endif
}
/* lane maxima -> one. Lanes are >= +0 and never NaN (they only take v when v > m), so max is order-free and a max
 * tree equals the sequential scan bit for bit. */
static inline TGT float SFX(vhmax)(V m) {
#if ISA == 2
    return _mm512_reduce_max_ps(m);
#elif ISA == 1
    __m128 t = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    t = _mm_max_ps(t, _mm_movehl_ps(t, t));
    t = _mm_max_ss(t, _mm_shuffle_ps(t, t, 1));
    return _mm_cvtss_f32(t);
#else
    return m > 0.0f ? m : 0.0f;
#endif
}

/* quant rounding of one 128-block (or the tail) given its amax: aq[k] = round(a[k]*127/amax) */
static inline TGT void SFX(qround)(const float *a, int n, float amax, int8_t *aq, float *sa, int32_t *as) {
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    *sa = amax / 127.0f;
    const V vi = VSET(inv);
    int k = 0;
    for (; k + VW <= n; k += VW) {
        V q = VSUB(VADD(VMUL(VLD(a + k), vi), VSET(12582912.0f)), VSET(12582912.0f));
        q = VSEL(VGT(q, VSET(127.0f)), VSET(127.0f), VSEL(VLT(q, VSET(-127.0f)), VSET(-127.0f), q));
        VSTQ8(aq + k, q);
    }
    for (; k < n; k++) {
        float q = (a[k] * inv + 12582912.0f) - 12582912.0f;
        q = q > 127.0f ? 127.0f : (q < -127.0f ? -127.0f : q);
        aq[k] = (int8_t)(int32_t)q;
    }
    /* exact integer sum of the stored int8 codes (the lay-2 VNNI offset: sum (q+128) a = sum q a + 128 sum a) */
    int32_t t = 0; k = 0;
#if ISA == 2
    __m512i iv = _mm512_setzero_si512();
    for (; k + 16 <= n; k += 16) iv = _mm512_add_epi32(iv, _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(aq + k))));
    t = _mm512_reduce_add_epi32(iv);
#elif ISA == 1
    __m256i iv = _mm256_setzero_si256();
    for (; k + 8 <= n; k += 8) iv = _mm256_add_epi32(iv, _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(aq + k))));
    int32_t tt[8]; _mm256_storeu_si256((__m256i *)tt, iv);
    for (int j = 0; j < 8; j++) t += tt[j];
#endif
    for (; k < n; k++) t += aq[k];
    *as = t;
}

/* symmetric int8 per 128-block: aq[k] = round(a[k] * 127/amax_b), sa[b] = amax_b/127 */
static TGT void SFX(quant)(const float *a, int K, int8_t *aq, float *sa, int32_t *as) {
    for (int k0 = 0, b = 0; k0 < K; k0 += WNN_BLK, b++) {
        int k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
        V m = VSET(0.0f);
        int k = k0;
        for (; k + VW <= k1; k += VW) { V v = VABS(VLD(a + k)); m = VSEL(VGT(v, m), v, m); }
        float amax = SFX(vhmax)(m);
        for (; k < k1; k++) { float v = a[k] < 0.0f ? -a[k] : a[k]; amax = v > amax ? v : amax; }
        SFX(qround)(a + k0, k1 - k0, amax, aq + k0, sa + b, as + b);
    }
}

/* LayerNorm (biased variance, like torch) over the concatenation [x1 (n1) | x2 (n2)] (n1 % 16 == 0, n2 % 16 == 0),
 * written to dst, then quantised into aq/sa (if aq). Same float ops as Z52's ln_v + quant; x1/x2 are not modified
 * unless dst aliases them. */
static TGT void SFX(ln_q)(const float *x1, int n1, const float *x2, int n2, const float *g, const float *b, float eps,
                          float *dst, int8_t *aq, float *sa, int32_t *as) {
    const int n = n1 + n2;
    V acc[NACC];
    for (int a = 0; a < NACC; a++) acc[a] = VSET(0.0f);
    for (int i = 0; i < n1; i += 16) for (int a = 0; a < NACC; a++) acc[a] = VADD(acc[a], VLD(x1 + i + a * VW));
    for (int i = 0; i < n2; i += 16) for (int a = 0; a < NACC; a++) acc[a] = VADD(acc[a], VLD(x2 + i + a * VW));
    const float mean = SFX(red16)(acc) / (float)n;
    const V vm = VSET(mean);
    for (int a = 0; a < NACC; a++) acc[a] = VSET(0.0f);
    for (int i = 0; i < n1; i += 16) for (int a = 0; a < NACC; a++) { V d = VSUB(VLD(x1 + i + a * VW), vm); acc[a] = VADD(acc[a], VMUL(d, d)); }
    for (int i = 0; i < n2; i += 16) for (int a = 0; a < NACC; a++) { V d = VSUB(VLD(x2 + i + a * VW), vm); acc[a] = VADD(acc[a], VMUL(d, d)); }
    const float var = SFX(red16)(acc) / (float)n;
    const float rs = 1.0f / __builtin_sqrtf(var + eps);
    const V vr = VSET(rs);
    V mx = VSET(0.0f);
    for (int i = 0; i < n; i += VW) {
        const float *src = i < n1 ? x1 + i : x2 + (i - n1);
        V y = VADD(VMUL(VMUL(VSUB(VLD(src), vm), vr), VLD(g + i)), VLD(b + i));
        VST(dst + i, y);
        if (aq) {
            V v = VABS(y); mx = VSEL(VGT(v, mx), v, mx);
            if (((i + VW) % WNN_BLK) == 0 || i + VW == n) {
                int k0 = (i / WNN_BLK) * WNN_BLK;
                SFX(qround)(dst + k0, i + VW - k0, SFX(vhmax)(mx), aq + k0, sa + i / WNN_BLK, as + i / WNN_BLK);
                mx = VSET(0.0f);
            }
        }
    }
}

/* z = LN(GELU(src)) over n (n % 16 == 0), then quantised into aq/sa. src may equal z. */
static TGT void SFX(gelu_ln_q)(const float *src, int n, const float *g, const float *b, float eps, float *z, int8_t *aq, float *sa,
                               int32_t *as, int fg) {
    V acc[NACC];
    for (int a = 0; a < NACC; a++) acc[a] = VSET(0.0f);
    for (int i = 0; i < n; i += 16)
        for (int a = 0; a < NACC; a++) { V y = VGELU(fg, VLD(src + i + a * VW)); VST(z + i + a * VW, y); acc[a] = VADD(acc[a], y); }
    const float mean = SFX(red16)(acc) / (float)n;
    const V vm = VSET(mean);
    for (int a = 0; a < NACC; a++) acc[a] = VSET(0.0f);
    for (int i = 0; i < n; i += 16) for (int a = 0; a < NACC; a++) { V d = VSUB(VLD(z + i + a * VW), vm); acc[a] = VADD(acc[a], VMUL(d, d)); }
    const float var = SFX(red16)(acc) / (float)n;
    const float rs = 1.0f / __builtin_sqrtf(var + eps);
    const V vr = VSET(rs);
    V mx = VSET(0.0f);
    for (int i = 0; i < n; i += VW) {
        V y = VADD(VMUL(VMUL(VSUB(VLD(z + i), vm), vr), VLD(g + i)), VLD(b + i));
        VST(z + i, y);
        V v = VABS(y); mx = VSEL(VGT(v, mx), v, mx);
        if (((i + VW) % WNN_BLK) == 0 || i + VW == n) {
            int k0 = (i / WNN_BLK) * WNN_BLK;
            SFX(qround)(z + k0, i + VW - k0, SFX(vhmax)(mx), aq + k0, sa + i / WNN_BLK, as + i / WNN_BLK);
            mx = VSET(0.0f);
        }
    }
}

/* lay 2: same arithmetic, one sequential stream per matrix; u8 weights (q + 128) so the VNNI offset correction is the
 * per-block activation sum (no per-column corr array) */
static TGT void SFX(qmv2)(const wnn_qmat *M, const int8_t *aq, const float *sa, const int32_t *as, float *y) {
    const int K = M->K, G = M->Nc / 64, nb = M->nb;
    for (int g = 0; g < G; g++) {
        const uint8_t *Wg = (const uint8_t *)M->W + (size_t)g * M->gstride;
        const float *swg = (const float *)(Wg + (size_t)K * 64), *bg = swg + (size_t)nb * 64;
        float *yg = y + g * 64;
#if ISA == 2
        __m512 y0 = _mm512_loadu_ps(bg), y1 = _mm512_loadu_ps(bg + 16), y2 = _mm512_loadu_ps(bg + 32), y3 = _mm512_loadu_ps(bg + 48);
        const __m512i zero = _mm512_setzero_si512();
        for (int b = 0; b < nb; b++) {
            int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
            __m512i e0 = _mm512_set1_epi32(-128 * as[b]), e1 = e0, e2 = e0, e3 = e0;
            __m512i o0 = zero, o1 = zero, o2 = zero, o3 = zero;
            int k4 = k0 / 4;
            for (; k4 + 2 <= k1 / 4; k4 += 2) {
                int32_t a0, a1; memcpy(&a0, aq + 4 * k4, 4); memcpy(&a1, aq + 4 * k4 + 4, 4);
                const __m512i va = _mm512_set1_epi32(a0), vb = _mm512_set1_epi32(a1);
                const uint8_t *wp = Wg + (size_t)k4 * 256;
                e0 = _mm512_dpbusd_epi32(e0, _mm512_loadu_si512(wp), va);
                e1 = _mm512_dpbusd_epi32(e1, _mm512_loadu_si512(wp + 64), va);
                e2 = _mm512_dpbusd_epi32(e2, _mm512_loadu_si512(wp + 128), va);
                e3 = _mm512_dpbusd_epi32(e3, _mm512_loadu_si512(wp + 192), va);
                o0 = _mm512_dpbusd_epi32(o0, _mm512_loadu_si512(wp + 256), vb);
                o1 = _mm512_dpbusd_epi32(o1, _mm512_loadu_si512(wp + 320), vb);
                o2 = _mm512_dpbusd_epi32(o2, _mm512_loadu_si512(wp + 384), vb);
                o3 = _mm512_dpbusd_epi32(o3, _mm512_loadu_si512(wp + 448), vb);
            }
            if (k4 < k1 / 4) {
                int32_t a0; memcpy(&a0, aq + 4 * k4, 4);
                const __m512i va = _mm512_set1_epi32(a0);
                const uint8_t *wp = Wg + (size_t)k4 * 256;
                e0 = _mm512_dpbusd_epi32(e0, _mm512_loadu_si512(wp), va);
                e1 = _mm512_dpbusd_epi32(e1, _mm512_loadu_si512(wp + 64), va);
                e2 = _mm512_dpbusd_epi32(e2, _mm512_loadu_si512(wp + 128), va);
                e3 = _mm512_dpbusd_epi32(e3, _mm512_loadu_si512(wp + 192), va);
            }
            const float *sw = swg + (size_t)b * 64;
            const __m512 sab = _mm512_set1_ps(sa[b]);
            y0 = _mm512_add_ps(y0, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e0, o0)), _mm512_mul_ps(sab, _mm512_loadu_ps(sw))));
            y1 = _mm512_add_ps(y1, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e1, o1)), _mm512_mul_ps(sab, _mm512_loadu_ps(sw + 16))));
            y2 = _mm512_add_ps(y2, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e2, o2)), _mm512_mul_ps(sab, _mm512_loadu_ps(sw + 32))));
            y3 = _mm512_add_ps(y3, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e3, o3)), _mm512_mul_ps(sab, _mm512_loadu_ps(sw + 48))));
        }
        _mm512_storeu_ps(yg, y0); _mm512_storeu_ps(yg + 16, y1); _mm512_storeu_ps(yg + 32, y2); _mm512_storeu_ps(yg + 48, y3);
#elif ISA == 1
        (void)as;
        __m256 yv[8];
        for (int j = 0; j < 8; j++) yv[j] = _mm256_loadu_ps(bg + 8 * j);
        const __m256i ones = _mm256_set1_epi16(1), flip = _mm256_set1_epi8((char)0x80);
        for (int b = 0; b < nb; b++) {
            int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
            __m256i acc[8];
            for (int j = 0; j < 8; j++) acc[j] = _mm256_setzero_si256();
            for (int k4 = k0 / 4; k4 < k1 / 4; k4++) {
                int32_t a0; memcpy(&a0, aq + 4 * k4, 4);
                __m256i av = _mm256_set1_epi32(a0), aa = _mm256_abs_epi8(av);
                const uint8_t *wp = Wg + (size_t)k4 * 256;
                for (int j = 0; j < 8; j++) {
                    __m256i w = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(wp + 32 * j)), flip);   /* back to int8 q */
                    __m256i p = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w, av));
                    acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(p, ones));
                }
            }
            const float *sw = swg + (size_t)b * 64;
            const __m256 sab = _mm256_set1_ps(sa[b]);
            for (int j = 0; j < 8; j++)
                yv[j] = _mm256_add_ps(yv[j], _mm256_mul_ps(_mm256_cvtepi32_ps(acc[j]), _mm256_mul_ps(sab, _mm256_loadu_ps(sw + 8 * j))));
        }
        for (int j = 0; j < 8; j++) _mm256_storeu_ps(yg + 8 * j, yv[j]);
#else
        (void)as;
        for (int n = 0; n < 64; n++) yg[n] = bg[n];
        for (int b = 0; b < nb; b++) {
            int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
            int32_t acc[64];
            for (int n = 0; n < 64; n++) acc[n] = 0;
            for (int k4 = k0 / 4; k4 < k1 / 4; k4++) {
                const uint8_t *wp = Wg + (size_t)k4 * 256;
                const int32_t a0 = aq[4 * k4], a1 = aq[4 * k4 + 1], a2 = aq[4 * k4 + 2], a3 = aq[4 * k4 + 3];
                for (int n = 0; n < 64; n++) {
                    const uint8_t *w = wp + 4 * n;
                    acc[n] += a0 * ((int32_t)w[0] - 128) + a1 * ((int32_t)w[1] - 128) + a2 * ((int32_t)w[2] - 128) + a3 * ((int32_t)w[3] - 128);
                }
            }
            const float *sw = swg + (size_t)b * 64;
            for (int n = 0; n < 64; n++) yg[n] = yg[n] + (float)acc[n] * (sa[b] * sw[n]);
        }
#endif
    }
}

/* H5 lay 4/5: int4 asymmetric weights. Per blk4-row block b and column n:
 *   y[n] += float(sum_k aq[k] q[k][n]) * (sa[b128] * s[b][n]);  y[n] += float(sum_k aq[k]) * (sa[b128] * min[b][n])
 * (b128 = the 128-element activation block the weight block lies in; both integer sums are exact; the fp32 ops are the same
 * two multiplies and one add per term on every ISA, in this order, no FMA). */
#if ISA == 2
static inline TGT __m512 SFX(ld16)(const uint8_t *p, int ssz) {       /* 16 stored scales/mins -> fp32 (fp16 -> fp32 is exact) */
    return ssz == 4 ? _mm512_loadu_ps((const float *)p) : _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
}
#elif ISA == 1
static inline TGT __m256 SFX(ld8)(const uint8_t *p, int ssz) {
    return ssz == 4 ? _mm256_loadu_ps((const float *)p) : _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}
#else
static inline float SFX(ld1)(const uint8_t *p, int ssz) {
    if (ssz == 4) { float f; memcpy(&f, p, 4); return f; }
    uint16_t h; memcpy(&h, p, 2); return wnn_h2f(h);
}
#endif
static TGT void SFX(qmv4)(const wnn_qmat *M, const int8_t *aq, const float *sa, float *y) {
    const int K = M->K, G = M->Nc / 64, nb4 = M->nb4, blk = M->blk4, ssz = M->ssz;
    int32_t as4[WNN_MAXK / 32];                                        /* exact per-block activation sums (order-free) */
    for (int b = 0; b < nb4; b++) {
        int32_t t = 0;
        for (int k = b * blk; k < (b + 1) * blk; k++) t += aq[k];
        as4[b] = t;
    }
    for (int g = 0; g < G; g++) {
        const uint8_t *Wg = (const uint8_t *)M->W + (size_t)g * M->gstride;
        const uint8_t *scp = Wg + (size_t)K * 32, *mnp = scp + (size_t)nb4 * 64 * (size_t)ssz;
        const float *bg = (const float *)(mnp + (size_t)nb4 * 64 * (size_t)ssz);
        float *yg = y + g * 64;
#if ISA == 2
        __m512 y0 = _mm512_loadu_ps(bg), y1 = _mm512_loadu_ps(bg + 16), y2 = _mm512_loadu_ps(bg + 32), y3 = _mm512_loadu_ps(bg + 48);
        const __m512i zero = _mm512_setzero_si512(), m4 = _mm512_set1_epi8(0x0F);
        for (int b = 0; b < nb4; b++) {
            __m512i e0 = zero, e1 = zero, e2 = zero, e3 = zero, o0 = zero, o1 = zero, o2 = zero, o3 = zero;
            int k4 = b * blk / 4; const int k4e = (b + 1) * blk / 4;
            for (; k4 + 2 <= k4e; k4 += 2) {
                int32_t a0, a1; memcpy(&a0, aq + 4 * k4, 4); memcpy(&a1, aq + 4 * k4 + 4, 4);
                const __m512i va = _mm512_set1_epi32(a0), vb = _mm512_set1_epi32(a1);
                const uint8_t *wp = Wg + (size_t)k4 * 128;
                const __m512i p0 = _mm512_loadu_si512(wp), p1 = _mm512_loadu_si512(wp + 64);
                const __m512i q0 = _mm512_loadu_si512(wp + 128), q1 = _mm512_loadu_si512(wp + 192);
                e0 = _mm512_dpbusd_epi32(e0, _mm512_and_si512(p0, m4), va);
                e1 = _mm512_dpbusd_epi32(e1, _mm512_and_si512(_mm512_srli_epi16(p0, 4), m4), va);
                e2 = _mm512_dpbusd_epi32(e2, _mm512_and_si512(p1, m4), va);
                e3 = _mm512_dpbusd_epi32(e3, _mm512_and_si512(_mm512_srli_epi16(p1, 4), m4), va);
                o0 = _mm512_dpbusd_epi32(o0, _mm512_and_si512(q0, m4), vb);
                o1 = _mm512_dpbusd_epi32(o1, _mm512_and_si512(_mm512_srli_epi16(q0, 4), m4), vb);
                o2 = _mm512_dpbusd_epi32(o2, _mm512_and_si512(q1, m4), vb);
                o3 = _mm512_dpbusd_epi32(o3, _mm512_and_si512(_mm512_srli_epi16(q1, 4), m4), vb);
            }
            if (k4 < k4e) {                                                 /* odd tail (never for blk % 8 == 0) */
                int32_t a0; memcpy(&a0, aq + 4 * k4, 4);
                const __m512i va = _mm512_set1_epi32(a0);
                const uint8_t *wp = Wg + (size_t)k4 * 128;
                const __m512i p0 = _mm512_loadu_si512(wp), p1 = _mm512_loadu_si512(wp + 64);
                e0 = _mm512_dpbusd_epi32(e0, _mm512_and_si512(p0, m4), va);
                e1 = _mm512_dpbusd_epi32(e1, _mm512_and_si512(_mm512_srli_epi16(p0, 4), m4), va);
                e2 = _mm512_dpbusd_epi32(e2, _mm512_and_si512(p1, m4), va);
                e3 = _mm512_dpbusd_epi32(e3, _mm512_and_si512(_mm512_srli_epi16(p1, 4), m4), va);
            }
            const __m512 sab = _mm512_set1_ps(sa[(b * blk) / WNN_BLK]), asf = _mm512_set1_ps((float)as4[b]);
            const uint8_t *sp = scp + (size_t)b * 64 * (size_t)ssz, *mp = mnp + (size_t)b * 64 * (size_t)ssz;
            y0 = _mm512_add_ps(y0, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e0, o0)), _mm512_mul_ps(sab, SFX(ld16)(sp, ssz))));
            y1 = _mm512_add_ps(y1, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e1, o1)), _mm512_mul_ps(sab, SFX(ld16)(sp + 16 * ssz, ssz))));
            y2 = _mm512_add_ps(y2, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e2, o2)), _mm512_mul_ps(sab, SFX(ld16)(sp + 32 * ssz, ssz))));
            y3 = _mm512_add_ps(y3, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e3, o3)), _mm512_mul_ps(sab, SFX(ld16)(sp + 48 * ssz, ssz))));
            y0 = _mm512_add_ps(y0, _mm512_mul_ps(asf, _mm512_mul_ps(sab, SFX(ld16)(mp, ssz))));
            y1 = _mm512_add_ps(y1, _mm512_mul_ps(asf, _mm512_mul_ps(sab, SFX(ld16)(mp + 16 * ssz, ssz))));
            y2 = _mm512_add_ps(y2, _mm512_mul_ps(asf, _mm512_mul_ps(sab, SFX(ld16)(mp + 32 * ssz, ssz))));
            y3 = _mm512_add_ps(y3, _mm512_mul_ps(asf, _mm512_mul_ps(sab, SFX(ld16)(mp + 48 * ssz, ssz))));
        }
        _mm512_storeu_ps(yg, y0); _mm512_storeu_ps(yg + 16, y1); _mm512_storeu_ps(yg + 32, y2); _mm512_storeu_ps(yg + 48, y3);
#elif ISA == 1
        __m256 yv[8];
        for (int j = 0; j < 8; j++) yv[j] = _mm256_loadu_ps(bg + 8 * j);
        const __m256i ones = _mm256_set1_epi16(1), m4 = _mm256_set1_epi8(0x0F);
        for (int b = 0; b < nb4; b++) {
            __m256i acc[8];
            for (int j = 0; j < 8; j++) acc[j] = _mm256_setzero_si256();
            for (int k4 = b * blk / 4; k4 < (b + 1) * blk / 4; k4++) {
                int32_t a0; memcpy(&a0, aq + 4 * k4, 4);
                const __m256i av = _mm256_set1_epi32(a0);
                const uint8_t *wp = Wg + (size_t)k4 * 128;
                for (int v = 0; v < 2; v++)
                    for (int h = 0; h < 2; h++) {                        /* 32 bytes = cols 8h..8h+7 of tiles 2v (lo) and 2v+1 (hi) */
                        __m256i cch = _mm256_loadu_si256((const __m256i *)(wp + 64 * v + 32 * h));
                        __m256i lo = _mm256_and_si256(cch, m4), hi = _mm256_and_si256(_mm256_srli_epi16(cch, 4), m4);
                        acc[4 * v + h] = _mm256_add_epi32(acc[4 * v + h], _mm256_madd_epi16(_mm256_maddubs_epi16(lo, av), ones));
                        acc[4 * v + 2 + h] = _mm256_add_epi32(acc[4 * v + 2 + h], _mm256_madd_epi16(_mm256_maddubs_epi16(hi, av), ones));
                    }
            }
            const __m256 sab = _mm256_set1_ps(sa[(b * blk) / WNN_BLK]), asf = _mm256_set1_ps((float)as4[b]);
            const uint8_t *sp = scp + (size_t)b * 64 * (size_t)ssz, *mp = mnp + (size_t)b * 64 * (size_t)ssz;
            for (int j = 0; j < 8; j++) {
                yv[j] = _mm256_add_ps(yv[j], _mm256_mul_ps(_mm256_cvtepi32_ps(acc[j]), _mm256_mul_ps(sab, SFX(ld8)(sp + 8 * j * ssz, ssz))));
                yv[j] = _mm256_add_ps(yv[j], _mm256_mul_ps(asf, _mm256_mul_ps(sab, SFX(ld8)(mp + 8 * j * ssz, ssz))));
            }
        }
        for (int j = 0; j < 8; j++) _mm256_storeu_ps(yg + 8 * j, yv[j]);
#else
        for (int n = 0; n < 64; n++) yg[n] = bg[n];
        for (int b = 0; b < nb4; b++) {
            int32_t acc[64];
            for (int n = 0; n < 64; n++) acc[n] = 0;
            for (int k4 = b * blk / 4; k4 < (b + 1) * blk / 4; k4++) {
                const uint8_t *wp = Wg + (size_t)k4 * 128;
                const int32_t a0 = aq[4 * k4], a1 = aq[4 * k4 + 1], a2 = aq[4 * k4 + 2], a3 = aq[4 * k4 + 3];
                for (int n = 0; n < 64; n++) {
                    const int t = n >> 4, cc = n & 15;                     /* tile t: vector t/2, low nibbles if t even */
                    const uint8_t *w = wp + 64 * (t >> 1) + 4 * cc;
                    const int sh = (t & 1) * 4;
                    acc[n] += a0 * ((w[0] >> sh) & 15) + a1 * ((w[1] >> sh) & 15) + a2 * ((w[2] >> sh) & 15) + a3 * ((w[3] >> sh) & 15);
                }
            }
            const float sab = sa[(b * blk) / WNN_BLK], asf = (float)as4[b];
            const uint8_t *sp = scp + (size_t)b * 64 * (size_t)ssz, *mp = mnp + (size_t)b * 64 * (size_t)ssz;
            for (int n = 0; n < 64; n++) {
                yg[n] = yg[n] + (float)acc[n] * (sab * SFX(ld1)(sp + (size_t)n * ssz, ssz));
                yg[n] = yg[n] + asf * (sab * SFX(ld1)(mp + (size_t)n * ssz, ssz));
            }
        }
#endif
    }
}



/* H10 lay 8: symmetric int16 per 128-block, aq[k] = round(a[k] * 32767/amax_b), sa[b] = amax_b/32767 (the int8 quant's code
 * with 32767 in place of 127: same amax scan, same magic-number rounding, same clamp; no code sum is needed, the int16 x int8
 * products are signed x signed). */
static TGT void SFX(quant16)(const float *a, int K, int16_t *aq, float *sa) {
    for (int k0 = 0, b = 0; k0 < K; k0 += WNN_BLK, b++) {
        const int k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
        V m = VSET(0.0f);
        int k = k0;
        for (; k + VW <= k1; k += VW) { V v = VABS(VLD(a + k)); m = VSEL(VGT(v, m), v, m); }
        float amax = SFX(vhmax)(m);
        for (; k < k1; k++) { float v = a[k] < 0.0f ? -a[k] : a[k]; amax = v > amax ? v : amax; }
        const float inv = amax > 0.0f ? 32767.0f / amax : 0.0f;
        sa[b] = amax / 32767.0f;
        const V vi = VSET(inv);
        for (k = k0; k + VW <= k1; k += VW) {
            V q = VSUB(VADD(VMUL(VLD(a + k), vi), VSET(12582912.0f)), VSET(12582912.0f));
            q = VSEL(VGT(q, VSET(32767.0f)), VSET(32767.0f), VSEL(VLT(q, VSET(-32767.0f)), VSET(-32767.0f), q));
            VSTQ16(aq + k, q);
        }
        for (; k < k1; k++) {
            float q = (a[k] * inv + 12582912.0f) - 12582912.0f;
            q = q > 32767.0f ? 32767.0f : (q < -32767.0f ? -32767.0f : q);
            aq[k] = (int16_t)(int32_t)q;
        }
    }
}

/* H10 lay 8: y[n] = bias[n] + sum_b float(sum_k aq16[k] q[k][n]) * (sa[b] * sw[b][n]) -- lay 2's epilogue on exact int32 dots
 * (|dot| <= K * 32767 * 127 < 2^31 for K <= 512). Weights per 64-column group: [K/2][64 cols][2 k] int8 (one 32-byte load
 * sign-extends to 16 columns x 2 k for vpdpwssd / vpmaddwd), then sw [nb][64] f32, bias [64] f32. */
static TGT void SFX(qmv16)(const wnn_qmat *M, const int16_t *aq, const float *sa, float *y) {
    const int K = M->K, G = M->Nc / 64, nb = M->nb;
    for (int g = 0; g < G; g++) {
        const uint8_t *Wg = (const uint8_t *)M->W + (size_t)g * M->gstride;
        const float *swg = (const float *)(Wg + (size_t)K * 64), *bg = swg + (size_t)nb * 64;
        float *yg = y + g * 64;
#if ISA == 2
        __m512 y0 = _mm512_loadu_ps(bg), y1 = _mm512_loadu_ps(bg + 16), y2 = _mm512_loadu_ps(bg + 32), y3 = _mm512_loadu_ps(bg + 48);
        const __m512i zero = _mm512_setzero_si512();
        for (int b = 0; b < nb; b++) {
            const int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
            __m512i e0 = zero, e1 = zero, e2 = zero, e3 = zero;
            for (int k2 = k0 / 2; k2 < k1 / 2; k2++) {
                int32_t a; memcpy(&a, aq + 2 * k2, 4);
                const __m512i va = _mm512_set1_epi32(a);
                const __m256i *wp = (const __m256i *)(Wg + (size_t)k2 * 128);
                e0 = _mm512_dpwssd_epi32(e0, _mm512_cvtepi8_epi16(_mm256_loadu_si256(wp)), va);
                e1 = _mm512_dpwssd_epi32(e1, _mm512_cvtepi8_epi16(_mm256_loadu_si256(wp + 1)), va);
                e2 = _mm512_dpwssd_epi32(e2, _mm512_cvtepi8_epi16(_mm256_loadu_si256(wp + 2)), va);
                e3 = _mm512_dpwssd_epi32(e3, _mm512_cvtepi8_epi16(_mm256_loadu_si256(wp + 3)), va);
            }
            const float *sw = swg + (size_t)b * 64;
            const __m512 sab = _mm512_set1_ps(sa[b]);
            y0 = _mm512_add_ps(y0, _mm512_mul_ps(_mm512_cvtepi32_ps(e0), _mm512_mul_ps(sab, _mm512_loadu_ps(sw))));
            y1 = _mm512_add_ps(y1, _mm512_mul_ps(_mm512_cvtepi32_ps(e1), _mm512_mul_ps(sab, _mm512_loadu_ps(sw + 16))));
            y2 = _mm512_add_ps(y2, _mm512_mul_ps(_mm512_cvtepi32_ps(e2), _mm512_mul_ps(sab, _mm512_loadu_ps(sw + 32))));
            y3 = _mm512_add_ps(y3, _mm512_mul_ps(_mm512_cvtepi32_ps(e3), _mm512_mul_ps(sab, _mm512_loadu_ps(sw + 48))));
        }
        _mm512_storeu_ps(yg, y0); _mm512_storeu_ps(yg + 16, y1); _mm512_storeu_ps(yg + 32, y2); _mm512_storeu_ps(yg + 48, y3);
#elif ISA == 1
        __m256 yv[8];
        for (int j = 0; j < 8; j++) yv[j] = _mm256_loadu_ps(bg + 8 * j);
        for (int b = 0; b < nb; b++) {
            const int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
            __m256i acc[8];
            for (int j = 0; j < 8; j++) acc[j] = _mm256_setzero_si256();
            for (int k2 = k0 / 2; k2 < k1 / 2; k2++) {
                int32_t a; memcpy(&a, aq + 2 * k2, 4);
                const __m256i va = _mm256_set1_epi32(a);
                const uint8_t *wp = Wg + (size_t)k2 * 128;
                for (int j = 0; j < 8; j++)
                    acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(wp + 16 * j))), va));
            }
            const float *sw = swg + (size_t)b * 64;
            const __m256 sab = _mm256_set1_ps(sa[b]);
            for (int j = 0; j < 8; j++)
                yv[j] = _mm256_add_ps(yv[j], _mm256_mul_ps(_mm256_cvtepi32_ps(acc[j]), _mm256_mul_ps(sab, _mm256_loadu_ps(sw + 8 * j))));
        }
        for (int j = 0; j < 8; j++) _mm256_storeu_ps(yg + 8 * j, yv[j]);
#else
        for (int n = 0; n < 64; n++) yg[n] = bg[n];
        for (int b = 0; b < nb; b++) {
            const int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
            int32_t acc[64];
            for (int n = 0; n < 64; n++) acc[n] = 0;
            for (int k2 = k0 / 2; k2 < k1 / 2; k2++) {
                const int8_t *wp = (const int8_t *)Wg + (size_t)k2 * 128;
                const int32_t a0 = aq[2 * k2], a1 = aq[2 * k2 + 1];
                for (int n = 0; n < 64; n++) acc[n] += a0 * wp[2 * n] + a1 * wp[2 * n + 1];
            }
            const float *sw = swg + (size_t)b * 64;
            for (int n = 0; n < 64; n++) yg[n] = yg[n] + (float)acc[n] * (sa[b] * sw[n]);
        }
#endif
    }
}
/* H10 lay 6/7 (blob v4, p0 only): fp32 activations, NO activation quantisation. Lane = output column, k strictly sequential:
 *   lay 6 (int8 codes q with lay 2's per-(128-row block, column) scale sw): per block b
 *       acc[n] = ((0 + xs[k0] * q[k0][n]) + xs[k0+1] * q[k0+1][n]) + ... ;   y[n] = y[n] + acc[n] * sw[b][n]     (y starts at bias)
 *   lay 7 (fp16 weights w, converted exactly): y[n] = ((bias[n] + xs[0] * w[0][n]) + xs[1] * w[1][n]) + ...
 * Every product is ONE IEEE fp32 multiply and every accumulation ONE IEEE add (no FMA), and the per-column order is the same
 * for 1, 8 or 16 lanes, so the three bodies and the NumPy emulation are bit-identical. Per 64-column group the weights are
 * k-major: lay 6 [K][64] int8, then sw [nb][64] f32, bias [64] f32 (record size = lay 2's); lay 7 [K][64] f16, then bias. */
static TGT void SFX(qmvf)(const wnn_qmat *M, const float *xs, float *y) {
    const int K = M->K, G = M->Nc / 64, nb = M->nb, lay = M->lay;
    for (int g = 0; g < G; g++) {
        const uint8_t *Wg = (const uint8_t *)M->W + (size_t)g * M->gstride;
        const float *swg = (const float *)(Wg + (size_t)K * 64);                              /* lay 6 only */
        const float *bg = lay == 6 ? swg + (size_t)nb * 64 : (const float *)(Wg + (size_t)K * 128);
        float *yg = y + g * 64;
#if ISA == 2
        __m512 y0 = _mm512_loadu_ps(bg), y1 = _mm512_loadu_ps(bg + 16), y2 = _mm512_loadu_ps(bg + 32), y3 = _mm512_loadu_ps(bg + 48);
        if (lay == 6) {
            for (int b = 0; b < nb; b++) {
                const int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
                __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                for (int k = k0; k < k1; k++) {
                    const __m512 xk = _mm512_set1_ps(xs[k]);
                    const __m128i *wp = (const __m128i *)(Wg + (size_t)k * 64);
                    a0 = _mm512_add_ps(a0, _mm512_mul_ps(xk, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp)))));
                    a1 = _mm512_add_ps(a1, _mm512_mul_ps(xk, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp + 1)))));
                    a2 = _mm512_add_ps(a2, _mm512_mul_ps(xk, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp + 2)))));
                    a3 = _mm512_add_ps(a3, _mm512_mul_ps(xk, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(wp + 3)))));
                }
                const float *sw = swg + (size_t)b * 64;
                y0 = _mm512_add_ps(y0, _mm512_mul_ps(a0, _mm512_loadu_ps(sw)));
                y1 = _mm512_add_ps(y1, _mm512_mul_ps(a1, _mm512_loadu_ps(sw + 16)));
                y2 = _mm512_add_ps(y2, _mm512_mul_ps(a2, _mm512_loadu_ps(sw + 32)));
                y3 = _mm512_add_ps(y3, _mm512_mul_ps(a3, _mm512_loadu_ps(sw + 48)));
            }
        } else {
            for (int k = 0; k < K; k++) {
                const __m512 xk = _mm512_set1_ps(xs[k]);
                const __m256i *wp = (const __m256i *)(Wg + (size_t)k * 128);
                y0 = _mm512_add_ps(y0, _mm512_mul_ps(xk, _mm512_cvtph_ps(_mm256_loadu_si256(wp))));
                y1 = _mm512_add_ps(y1, _mm512_mul_ps(xk, _mm512_cvtph_ps(_mm256_loadu_si256(wp + 1))));
                y2 = _mm512_add_ps(y2, _mm512_mul_ps(xk, _mm512_cvtph_ps(_mm256_loadu_si256(wp + 2))));
                y3 = _mm512_add_ps(y3, _mm512_mul_ps(xk, _mm512_cvtph_ps(_mm256_loadu_si256(wp + 3))));
            }
        }
        _mm512_storeu_ps(yg, y0); _mm512_storeu_ps(yg + 16, y1); _mm512_storeu_ps(yg + 32, y2); _mm512_storeu_ps(yg + 48, y3);
#elif ISA == 1
        __m256 yv[8];
        for (int j = 0; j < 8; j++) yv[j] = _mm256_loadu_ps(bg + 8 * j);
        if (lay == 6) {
            for (int b = 0; b < nb; b++) {
                const int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
                __m256 av[8];
                for (int j = 0; j < 8; j++) av[j] = _mm256_setzero_ps();
                for (int k = k0; k < k1; k++) {
                    const __m256 xk = _mm256_set1_ps(xs[k]);
                    const uint8_t *wp = Wg + (size_t)k * 64;
                    for (int j = 0; j < 8; j++)
                        av[j] = _mm256_add_ps(av[j], _mm256_mul_ps(xk, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(wp + 8 * j))))));
                }
                const float *sw = swg + (size_t)b * 64;
                for (int j = 0; j < 8; j++) yv[j] = _mm256_add_ps(yv[j], _mm256_mul_ps(av[j], _mm256_loadu_ps(sw + 8 * j)));
            }
        } else {
            for (int k = 0; k < K; k++) {
                const __m256 xk = _mm256_set1_ps(xs[k]);
                const uint8_t *wp = Wg + (size_t)k * 128;
                for (int j = 0; j < 8; j++)
                    yv[j] = _mm256_add_ps(yv[j], _mm256_mul_ps(xk, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wp + 16 * j)))));
            }
        }
        for (int j = 0; j < 8; j++) _mm256_storeu_ps(yg + 8 * j, yv[j]);
#else
        for (int n = 0; n < 64; n++) yg[n] = bg[n];
        if (lay == 6) {
            for (int b = 0; b < nb; b++) {
                const int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
                float acc[64];
                for (int n = 0; n < 64; n++) acc[n] = 0.0f;
                for (int k = k0; k < k1; k++) {
                    const float xk = xs[k];
                    const int8_t *wp = (const int8_t *)Wg + (size_t)k * 64;
                    for (int n = 0; n < 64; n++) acc[n] = acc[n] + xk * (float)wp[n];
                }
                const float *sw = swg + (size_t)b * 64;
                for (int n = 0; n < 64; n++) yg[n] = yg[n] + acc[n] * sw[n];
            }
        } else {
            for (int k = 0; k < K; k++) {
                const float xk = xs[k];
                const uint8_t *wp = Wg + (size_t)k * 128;
                for (int n = 0; n < 64; n++) { uint16_t h; memcpy(&h, wp + 2 * n, 2); yg[n] = yg[n] + xk * wnn_h2f(h); }
            }
        }
#endif
    }
}

/* y[Nc] = bias + sum_b float(int_dot_b) * (sa[b] * sw[b][n]) */
static TGT void SFX(qmv)(const wnn_qmat *M, const int8_t *aq, const float *sa, const int32_t *as, float *y) {
    if (M->lay == 4 || M->lay == 5) { SFX(qmv4)(M, aq, sa, y); return; }
    if (M->lay == 2) { SFX(qmv2)(M, aq, sa, as, y); return; }
    const int K = M->K, Nc = M->Nc, G = Nc / 64;
    for (int g = 0; g < G; g++) {
        const int8_t *Wg = M->W + (size_t)g * K * 64;
        float *yg = y + g * 64;
#if ISA == 2
        __m512 y0 = _mm512_loadu_ps(M->bias + g * 64), y1 = _mm512_loadu_ps(M->bias + g * 64 + 16),
               y2 = _mm512_loadu_ps(M->bias + g * 64 + 32), y3 = _mm512_loadu_ps(M->bias + g * 64 + 48);
        const __m512i flip = _mm512_set1_epi32((int)0x80808080u), zero = _mm512_setzero_si512();
        for (int b = 0; b < M->nb; b++) {
            int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
            const int32_t *cr = M->corr + (size_t)b * Nc + g * 64;
            __m512i e0 = _mm512_sub_epi32(zero, _mm512_loadu_si512(cr)), e1 = _mm512_sub_epi32(zero, _mm512_loadu_si512(cr + 16)),
                    e2 = _mm512_sub_epi32(zero, _mm512_loadu_si512(cr + 32)), e3 = _mm512_sub_epi32(zero, _mm512_loadu_si512(cr + 48));
            __m512i o0 = zero, o1 = zero, o2 = zero, o3 = zero;
            int k4 = k0 / 4;
            for (; k4 + 2 <= k1 / 4; k4 += 2) {
                int32_t a0, a1; memcpy(&a0, aq + 4 * k4, 4); memcpy(&a1, aq + 4 * k4 + 4, 4);
                __m512i va = _mm512_xor_si512(_mm512_set1_epi32(a0), flip);
                __m512i vb = _mm512_xor_si512(_mm512_set1_epi32(a1), flip);
                const int8_t *wp = Wg + (size_t)k4 * 256;
                e0 = _mm512_dpbusd_epi32(e0, va, _mm512_loadu_si512(wp));
                e1 = _mm512_dpbusd_epi32(e1, va, _mm512_loadu_si512(wp + 64));
                e2 = _mm512_dpbusd_epi32(e2, va, _mm512_loadu_si512(wp + 128));
                e3 = _mm512_dpbusd_epi32(e3, va, _mm512_loadu_si512(wp + 192));
                o0 = _mm512_dpbusd_epi32(o0, vb, _mm512_loadu_si512(wp + 256));
                o1 = _mm512_dpbusd_epi32(o1, vb, _mm512_loadu_si512(wp + 320));
                o2 = _mm512_dpbusd_epi32(o2, vb, _mm512_loadu_si512(wp + 384));
                o3 = _mm512_dpbusd_epi32(o3, vb, _mm512_loadu_si512(wp + 448));
            }
            if (k4 < k1 / 4) {                                   /* odd K/4 tail (K % 8 == 4 never occurs: K % 8 == 0) */
                int32_t a0; memcpy(&a0, aq + 4 * k4, 4);
                __m512i va = _mm512_xor_si512(_mm512_set1_epi32(a0), flip);
                const int8_t *wp = Wg + (size_t)k4 * 256;
                e0 = _mm512_dpbusd_epi32(e0, va, _mm512_loadu_si512(wp));
                e1 = _mm512_dpbusd_epi32(e1, va, _mm512_loadu_si512(wp + 64));
                e2 = _mm512_dpbusd_epi32(e2, va, _mm512_loadu_si512(wp + 128));
                e3 = _mm512_dpbusd_epi32(e3, va, _mm512_loadu_si512(wp + 192));
            }
            const float *sw = M->sw + (size_t)b * Nc + g * 64;
            const __m512 sab = _mm512_set1_ps(sa[b]);
            y0 = _mm512_add_ps(y0, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e0, o0)), _mm512_mul_ps(sab, _mm512_loadu_ps(sw))));
            y1 = _mm512_add_ps(y1, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e1, o1)), _mm512_mul_ps(sab, _mm512_loadu_ps(sw + 16))));
            y2 = _mm512_add_ps(y2, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e2, o2)), _mm512_mul_ps(sab, _mm512_loadu_ps(sw + 32))));
            y3 = _mm512_add_ps(y3, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(e3, o3)), _mm512_mul_ps(sab, _mm512_loadu_ps(sw + 48))));
        }
        _mm512_storeu_ps(yg, y0); _mm512_storeu_ps(yg + 16, y1); _mm512_storeu_ps(yg + 32, y2); _mm512_storeu_ps(yg + 48, y3);
#elif ISA == 1
        __m256 yv[8];
        for (int j = 0; j < 8; j++) yv[j] = _mm256_loadu_ps(M->bias + g * 64 + 8 * j);
        const __m256i ones = _mm256_set1_epi16(1);
        for (int b = 0; b < M->nb; b++) {
            int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
            __m256i acc[8];
            for (int j = 0; j < 8; j++) acc[j] = _mm256_setzero_si256();
            for (int k4 = k0 / 4; k4 < k1 / 4; k4++) {
                int32_t a0; memcpy(&a0, aq + 4 * k4, 4);
                __m256i av = _mm256_set1_epi32(a0), aa = _mm256_abs_epi8(av);
                const int8_t *wp = Wg + (size_t)k4 * 256;
                for (int j = 0; j < 8; j++) {
                    __m256i w = _mm256_loadu_si256((const __m256i *)(wp + 32 * j));
                    __m256i p = _mm256_maddubs_epi16(aa, _mm256_sign_epi8(w, av));
                    acc[j] = _mm256_add_epi32(acc[j], _mm256_madd_epi16(p, ones));
                }
            }
            const float *sw = M->sw + (size_t)b * Nc + g * 64;
            const __m256 sab = _mm256_set1_ps(sa[b]);
            for (int j = 0; j < 8; j++)
                yv[j] = _mm256_add_ps(yv[j], _mm256_mul_ps(_mm256_cvtepi32_ps(acc[j]), _mm256_mul_ps(sab, _mm256_loadu_ps(sw + 8 * j))));
        }
        for (int j = 0; j < 8; j++) _mm256_storeu_ps(yg + 8 * j, yv[j]);
#else
        for (int n = 0; n < 64; n++) yg[n] = M->bias[g * 64 + n];
        for (int b = 0; b < M->nb; b++) {
            int k0 = b * WNN_BLK, k1 = k0 + WNN_BLK < K ? k0 + WNN_BLK : K;
            int32_t acc[64];
            for (int n = 0; n < 64; n++) acc[n] = 0;
            for (int k4 = k0 / 4; k4 < k1 / 4; k4++) {
                const int8_t *wp = Wg + (size_t)k4 * 256;
                const int32_t a0 = aq[4 * k4], a1 = aq[4 * k4 + 1], a2 = aq[4 * k4 + 2], a3 = aq[4 * k4 + 3];
                for (int n = 0; n < 64; n++) {
                    const int8_t *w = wp + 4 * n;          /* tile n/16, col n%16 -> offset (n/16)*64 + (n%16)*4 = 4n */
                    acc[n] += a0 * w[0] + a1 * w[1] + a2 * w[2] + a3 * w[3];
                }
            }
            const float *sw = M->sw + (size_t)b * Nc + g * 64;
            for (int n = 0; n < 64; n++) yg[n] = yg[n] + (float)acc[n] * (sa[b] * sw[n]);
        }
#endif
    }
}

static TGT int SFX(wnn_step)(wnn_ctx *c, const float *x, float *out) {
    const int F = c->F, H = c->H;
    float *xs = c->xs;
    {
        int i = 0;
        for (; i + VW <= F; i += VW) {
            V v = VADD(VMUL(VLD(x + i), VLD(c->inv + i)), VLD(c->add + i));
            v = VSEL(VISNAN(v), VSET(0.0f), v);                           /* NaN feature -> the training mean */
            v = VSEL(VGT(v, VSET(c->clamp)), VSET(c->clamp), VSEL(VLT(v, VSET(-c->clamp)), VSET(-c->clamp), v));
            VST(xs + i, v);
        }
        for (; i < F; i++) {
            float v = x[i] * c->inv[i] + c->add[i];
            v = v != v ? 0.0f : v;
            v = v > c->clamp ? c->clamp : (v < -c->clamp ? -c->clamp : v);
            xs[i] = v;
        }
    }
    if (c->p0.lay == 8) {                                                   /* H10: int16 activations into p0 */
        SFX(quant16)(xs, F, c->aq16, c->sa);
        SFX(qmv16)(&c->p0, c->aq16, c->sa, c->y0);
    } else if (c->p0.lay >= 6) {                                            /* H10: fp32 activations into p0, no quantisation */
        SFX(qmvf)(&c->p0, xs, c->y0);
    } else {
        SFX(quant)(xs, F, c->aq, c->sa, c->as);
        SFX(qmv)(&c->p0, c->aq, c->sa, c->as, c->y0);
    }
    if (c->flags & 4) {                                                     /* H11 posfeat: exact fp32 rank-1 term, counter from 0 */
        const float pf = (float)c->pos / 20000.0f;
        const V pv = VSET(pf);
        float *y0 = c->y0; const float *wp = c->wpos;
        for (int n = 0; n < c->Ptot; n += VW) VST(y0 + n, VADD(VLD(y0 + n), VMUL(pv, VLD(wp + n))));
        if (c->pos < 0x7fffffff) c->pos++;
    }
    float acc0 = 0.0f, acc1 = 0.0f;
    const int fg = c->flags & 1;
    for (int mi = 0; mi < c->N; mi++) {
        const wnn_member *m = &c->m[mi];
        const int P = m->P, Hh = m->Hh;
        float *z = c->z;
        SFX(gelu_ln_q)(c->y0 + m->off, P, m->ln_g[0], m->ln_b[0], c->eps, z, c->aq, c->sa, c->as, fg);
        for (int d = 1; d < m->depth; d++) {
            SFX(qmv)(&m->px[d], c->aq, c->sa, c->as, z);
            SFX(gelu_ln_q)(z, P, m->ln_g[d], m->ln_b[d], c->eps, z, c->aq, c->sa, c->as, fg);
        }
        /* GRU (torch gate order r | z | n); aq/sa hold quant(z) here */
        float *h = c->h + (size_t)mi * H;
        SFX(qmv)(&m->ih, c->aq, c->sa, c->as, c->gi);
        SFX(quant)(h, H, c->aq, c->sa, c->as);
        if (m->R) {                                                         /* gh = A (B h) + b_hh, two int8 matvecs */
            SFX(qmv)(&m->hb, c->aq, c->sa, c->as, c->u);
            SFX(quant)(c->u, m->R, c->aq, c->sa, c->as);
            SFX(qmv)(&m->ha, c->aq, c->sa, c->as, c->gh);
        } else {
            SFX(qmv)(&m->hh, c->aq, c->sa, c->as, c->gh);
        }
        const float *gi = c->gi, *gh = c->gh;
        for (int j = 0; j < H; j += VW) {
            V r = SFX(vsigm)(VADD(VLD(gi + j), VLD(gh + j)));
            V u = SFX(vsigm)(VADD(VLD(gi + H + j), VLD(gh + H + j)));
            V n = SFX(vtanh)(VADD(VLD(gi + 2 * H + j), VMUL(r, VLD(gh + 2 * H + j))));
            VST(h + j, VADD(n, VMUL(u, VSUB(VLD(h + j), n))));
        }
        /* head: LN([h | z]) -> head1 -> GELU -> head2 (GELU fused into the two dot products) */
        SFX(ln_q)(h, H, z, P, m->ln2_g, m->ln2_b, c->eps, c->cat, c->aq, c->sa, c->as);
        SFX(qmv)(&m->h1, c->aq, c->sa, c->as, c->hdn);
        V a0[NACC], a1[NACC];
        for (int a = 0; a < NACC; a++) { a0[a] = VSET(0.0f); a1[a] = VSET(0.0f); }
        const float *w20 = m->w2, *w21 = m->w2 + Hh;
        for (int i = 0; i < Hh; i += 16)
            for (int a = 0; a < NACC; a++) {
                const int k = i + a * VW;
                V gv = VGELU(fg, VLD(c->hdn + k));
                a0[a] = VADD(a0[a], VMUL(gv, VLD(w20 + k)));
                a1[a] = VADD(a1[a], VMUL(gv, VLD(w21 + k)));
            }
        acc0 = acc0 + SFX(red16)(a0);
        acc1 = acc1 + SFX(red16)(a1);
    }
    out[0] = c->b2[0] + acc0;
    out[1] = c->b2[1] + acc1;
    return (out[0] - out[0] == 0.0f && out[1] - out[1] == 0.0f) ? 0 : 1;   /* 1 = non-finite output */
}

#undef VW
#undef V
#undef VM
#undef VLD
#undef VST
#undef VSET
#undef VADD
#undef VSUB
#undef VMUL
#undef VDIV
#undef VNEG
#undef VABS
#undef VLT
#undef VGT
#undef VGE
#undef VISNAN
#undef VSEL
#undef VPOW2N
#undef VSTQ8
#undef VSTQ16
#undef NACC

#endif /* WNN_BODY_PASS */
