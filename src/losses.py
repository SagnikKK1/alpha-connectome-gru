"""Sequence-anchored training objectives for the weighted-Pearson metric.

Why this module exists (see docs/research/objective.md for the evidence):

  * The scoreboard computes ONE correlation per 20,000-row sequence. Target
    autocorrelation decays over ~500 rows, so with 512-row training windows only
    ~31% of the |y|-weighted target variance is *within* a window; the other 69%
    is the level difference *between* windows. `metric.wp_loss` centres every
    window separately and is therefore blind to 69% of what is scored.
  * For a stateful model that blindness is catastrophic: the GRU is free to put
    slow, state-driven drift into its output (71% of the output variance in the
    live per-window-WP run) because no window ever sees it. Removing that drift
    after the fact takes the same checkpoint from 0.19 to 0.51 WP.

Everything here keeps sequence-level statistics in the loss instead:

  seq_pooled_wp_loss  - the metric itself, with covariance/variance POOLED over K
                        windows drawn from the same sequence (needs the grouped
                        sampler below). Best objective in the linear study.
  zmse_loss           - regression to the per-sequence standardised, clipped
                        single signal. Same population optimum as the metric,
                        lowest gradient variance, no ratio, no drift freedom.
  soft_clip           - 2*tanh(u/2) head: the metric clips predictions at +-2,
                        and an exported std of ~1.5 scores ~+0.005 over std 0.5.
  best_export_scale   - the paired grid search that picks that scale.
  sample_grouped      - G sequences x K windows batch sampler for SequenceCache.
  seq_target_stats    - per-sequence mean/std of the clipped signal for zmse.
  fit_propensity/     - P(is_scored | current row) from labelled valid sequences,
  propensity            for optional loss weighting (train.parquet has no mask).

All losses take signal (B, T) and y (B, T, 2) like `metric.wp_loss`, and reduce
to it when G == 1 and K == 1 (checked in the module self-test).
"""
from __future__ import annotations
import os
import numpy as np
import torch

from metric import wp_loss_common, wp_torch, wp_loss, METRIC_CLIP, EPS

WARMUP = 99
SEQUENCE_LENGTH = 20_000


# --------------------------------------------------------------------------
# Target parameterisation
# --------------------------------------------------------------------------
def clipped_parts(y: torch.Tensor):
    """y (..., 2) -> (s_c, c_c, w): clipped antisymmetric signal, clipped symmetric
    'cost' term, and the metric's row weight averaged over the two sides."""
    t0 = y[..., 0].clamp(-METRIC_CLIP, METRIC_CLIP)
    t1 = y[..., 1].clamp(-METRIC_CLIP, METRIC_CLIP)
    return (t0 - t1) / 2, (t0 + t1) / 2, 0.5 * (t0.abs() + t1.abs())


def soft_clip(u: torch.Tensor) -> torch.Tensor:
    """Bounded head: 2*tanh(u/2). Never clipped by the metric, keeps gradients."""
    return METRIC_CLIP * torch.tanh(u / METRIC_CLIP)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def seq_pooled_wp_loss(signal: torch.Tensor, y: torch.Tensor, groups: int,
                       mask: torch.Tensor | None = None,
                       weight: torch.Tensor | None = None) -> torch.Tensor:
    """1 - mean weighted Pearson, statistics pooled over the K windows of each sequence.

    signal: (G*K, T)   windows of the SAME sequence must be consecutive in the batch
    y:      (G*K, T, 2)
    groups: G          number of distinct sequences in the batch
    mask:   (G*K, T)   optional row weights (propensity, is_scored)
    weight: (G,)       optional per-sequence weight

    With G == B (K == 1) this is exactly `metric.wp_loss`. Pooling the K windows
    restores the between-window level that per-window centring discards, while
    keeping BPTT length at T.
    """
    B, T = signal.shape
    if B % groups:
        raise ValueError(f"batch {B} is not a multiple of groups {groups}")
    K = B // groups
    s = signal.reshape(groups, K * T)
    yy = y.reshape(groups, K * T, 2)
    m = mask.reshape(groups, K * T) if mask is not None else None
    return wp_loss(s, yy, m, weight)


def seq_pooled_wp_loss_common(sc: torch.Tensor, y: torch.Tensor, groups: int,
                              mask: torch.Tensor | None = None,
                              weight: torch.Tensor | None = None) -> torch.Tensor:
    """`seq_pooled_wp_loss` for the (s, c) parameterisation. sc: (G*K, T, 2)."""
    B, T, _ = sc.shape
    if B % groups:
        raise ValueError(f"batch {B} is not a multiple of groups {groups}")
    K = B // groups
    return wp_loss_common(sc.reshape(groups, K * T, 2),
                          y.reshape(groups, K * T, 2),
                          mask.reshape(groups, K * T) if mask is not None else None,
                          weight)


def seq_pooled_wp_loss_horizon(signal: torch.Tensor, y: torch.Tensor, groups: int,
                               offsets, weight_aux: float,
                               mask: torch.Tensor | None = None) -> torch.Tensor:
    """Main pooled-WP loss plus auxiliary heads predicting the target at FUTURE offsets.

    The target is a decaying-kernel sum of the next ~10-20 i0 price moves (regressing s
    on the next 128 decoded jumps gives R^2 0.54, cumulative 0.29 by 4 moves and 0.50 by
    16). So s_{t+k} for small k is a strongly related but differently-noised label. The
    Label Horizon literature reports 8-22% IC gains from training on a horizon other
    than the evaluation one; here we keep the true horizon as the main loss and add the
    others as auxiliaries.

    signal: (B, T, 1+len(offsets)) -- channel 0 is the deployed prediction, channels
    1.. are auxiliary and DROPPED at export, so inference cost is unchanged.
    """
    main = seq_pooled_wp_loss(signal[..., 0], y, groups, mask)
    if not offsets or weight_aux <= 0:
        return main
    aux = 0.0
    for j, k in enumerate(offsets):
        ys = torch.roll(y, shifts=-k, dims=1)
        m = mask
        if m is not None:
            m = m.clone(); m[:, -k:] = 0.0          # rolled-in rows are not real labels
        aux = aux + seq_pooled_wp_loss(signal[..., 1+j], ys, groups, m)
    return main + weight_aux * aux / len(offsets)


def zmse_loss(signal: torch.Tensor, y: torch.Tensor, seq_mu: torch.Tensor,
              seq_sd: torch.Tensor, row_weight: torch.Tensor | None = None) -> torch.Tensor:
    """Weighted MSE to the per-sequence standardised clipped signal z = (s_c - mu_b) / sd_b.

    signal:     (B, T)
    y:          (B, T, 2)
    seq_mu/sd:  (B,)   per-sequence |y|-weighted mean / std of s_c (seq_target_stats)
    row_weight: (B, T) optional (propensity). Plain |y| weighting did NOT help in
                the linear study (-0.004); standardisation is what matters.

    Population optimum is E_w[s_c | x] per sequence, i.e. the correlation-optimal
    predictor, with the sequence's scale divided out so every sequence counts
    equally as in the metric. Output lands on a z scale (std ~0.4-0.6): export
    through `best_export_scale`.
    """
    s_c, _, _ = clipped_parts(y)
    z = (s_c - seq_mu[:, None]) / seq_sd[:, None].clamp_min(EPS)
    r = (signal - z) ** 2
    if row_weight is not None:
        return (row_weight * r).sum() / row_weight.sum().clamp_min(EPS)
    return r.mean()


def anchored_loss(signal, y, groups, seq_mu, seq_sd, lam: float = 0.5,
                  mask=None) -> torch.Tensor:
    """zmse + lam * seq_pooled_wp. Regression gradient for stability, the metric for shape."""
    return zmse_loss(signal, y, seq_mu, seq_sd, mask) + lam * seq_pooled_wp_loss(signal, y, groups, mask)


# --------------------------------------------------------------------------
# Data helpers (work on data.SequenceCache; nothing is materialised)
# --------------------------------------------------------------------------
def seq_target_stats(cache, warmup: int = WARMUP):
    """Per-sequence |y|-weighted mean and std of the clipped signal s_c. Streams Y only."""
    n = len(cache)
    mu = np.zeros(n, np.float32); sd = np.zeros(n, np.float32)
    for i in range(n):
        yy = np.asarray(cache.Y[i])[warmup:]
        t0 = np.clip(yy[:, 0], -METRIC_CLIP, METRIC_CLIP); t1 = np.clip(yy[:, 1], -METRIC_CLIP, METRIC_CLIP)
        s = (t0 - t1) / 2; w = 0.5 * (np.abs(t0) + np.abs(t1)); W = max(w.sum(), EPS)
        m = (w * s).sum() / W
        mu[i] = m; sd[i] = max(np.sqrt((w * (s - m) ** 2).sum() / W), 1e-3)
    return mu, sd


def sample_grouped(cache, rng, groups: int, per_seq: int, body: int, burn: int,
                   row_weight: np.ndarray | None = None, seq_p: np.ndarray | None = None):
    """G sequences x K windows. Windows of one sequence are consecutive in the batch.

    Returns X (G*K, burn+body, F), Y (G*K, body, 2), seq index (G*K,),
    W (G*K, body) = row_weight[seq, rows] if given else ones, and the absolute row at
    which each window starts (G*K,). Drop-in for SequenceCache.batch with
    batch = groups * per_seq.

    `starts` is needed by the state-exposure arms: to replay a window's TRUE prefix the
    caller has to know where in the sequence the window sits, which the features alone
    do not say.
    """
    L = burn + body; B = groups * per_seq
    pick = (rng.integers(0, len(cache), groups) if seq_p is None
            else rng.choice(len(cache), size=groups, p=seq_p))
    seqs = np.repeat(pick, per_seq)
    starts = rng.integers(max(0, WARMUP - burn), SEQUENCE_LENGTH - L, B)
    F = cache.X.shape[-1]
    xb = np.empty((B, L, F), np.float32); yb = np.empty((B, body, 2), np.float32)
    wb = np.ones((B, body), np.float32)
    for k, (i, s) in enumerate(zip(seqs, starts)):
        xb[k] = cache.X[i, s:s + L]
        yb[k] = cache.Y[i, s + burn:s + L]
        if row_weight is not None:
            wb[k] = row_weight[i, s + burn:s + L]
    return xb, yb, seqs, wb, starts


# --------------------------------------------------------------------------
# Export calibration
# --------------------------------------------------------------------------
def best_export_scale(seqs, grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0), weighted_pearson=None):
    """Pick the output scale k that maximises the metric once predictions are clipped at +-2.

    seqs: iterable of (y (T,2), signal (T,), mask (T,)) for eligible sequences.
    The signal is normalised to unit std on scored rows, times k, then scored as
    (k*u, -k*u). Returns (best_k, {k: mean WP}). Export scale = best_k / std of the
    raw signal on scored rows (average over sequences), folded into ExportWrapper.
    """
    if weighted_pearson is None:
        from metric import weighted_pearson_np as weighted_pearson
    out = {k: [] for k in grid}; raw_std = []
    for y, p, m in seqs:
        sd = p[m].std()
        if sd <= 0: continue
        raw_std.append(sd); u = p / sd
        for k in grid:
            out[k].append(0.5 * (weighted_pearson(y[m, 0], k * u[m]) + weighted_pearson(y[m, 1], -k * u[m])))
    means = {k: float(np.mean(v)) for k, v in out.items()}
    best = max(means, key=means.get)
    return best / float(np.mean(raw_std)), means


# --------------------------------------------------------------------------
# Propensity P(is_scored | current row): fit on labelled valid sequences, apply anywhere
# --------------------------------------------------------------------------
def fit_propensity(cache_valid, seq_ids, mu, sd, lam: float = 1e-2, n_bins: int = 40):
    """Ridge probe of +-1 on standardised features, then monotone bin calibration.

    Fit on valid sequences that are NOT in cache/eval. Returns a dict usable by
    `propensity`. Out-of-sample AUC ~0.78 on the current row.
    """
    d = len(mu) + 1
    G = np.zeros((d, d)); b = np.zeros(d); n = 0
    for i in seq_ids:
        Z = np.c_[(np.asarray(cache_valid.X[i])[WARMUP:] - mu) / sd, np.ones(SEQUENCE_LENGTH - WARMUP, np.float32)].astype(np.float64)
        m = np.asarray(cache_valid.M[i])[WARMUP:].astype(np.float64) * 2 - 1
        G += Z.T @ Z; b += Z.T @ m; n += len(Z)
    A = G + lam * n * np.eye(d); A[-1, -1] -= lam * n
    w = np.linalg.solve(A, b)
    sc, ms = [], []
    for i in seq_ids:
        Z = np.c_[(np.asarray(cache_valid.X[i])[WARMUP:] - mu) / sd, np.ones(SEQUENCE_LENGTH - WARMUP, np.float32)]
        sc.append(Z @ w); ms.append(np.asarray(cache_valid.M[i])[WARMUP:])
    sc = np.concatenate(sc); ms = np.concatenate(ms)
    edges = np.quantile(sc, np.linspace(0, 1, n_bins + 1)); edges[0] = -np.inf; edges[-1] = np.inf
    bins = np.clip(np.searchsorted(edges, sc, side="right") - 1, 0, n_bins - 1)
    rate = np.maximum.accumulate(np.array([ms[bins == k].mean() for k in range(n_bins)]))
    centers = np.array([sc[bins == k].mean() for k in range(n_bins)])
    return {"w": w, "centers": centers, "rate": rate, "mu": mu, "sd": sd, "base_rate": float(ms.mean())}


def propensity(X, model, power: float = 1.0):
    """P(is_scored | row)^power for X (T, F). power < 1 tempers the weights (ESS trade-off)."""
    Z = np.c_[(X - model["mu"]) / model["sd"], np.ones(len(X), np.float32)]
    p = np.interp(Z @ model["w"], model["centers"], model["rate"])
    return (p / model["base_rate"]) ** power


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # self-test: reductions to metric.wp_loss and to the numpy reference
    torch.manual_seed(0)
    B, T = 8, 64
    y = torch.randn(B, T, 2) * 0.8; s = torch.randn(B, T)
    assert torch.allclose(seq_pooled_wp_loss(s, y, groups=B), wp_loss(s, y)), "G=B must equal wp_loss"
    pooled = seq_pooled_wp_loss(s, y, groups=2)
    ref = wp_loss(s.reshape(2, -1), y.reshape(2, -1, 2))
    assert torch.allclose(pooled, ref)
    from metric import weighted_pearson_np
    # the numpy reference clips predictions; wp_torch only does so with clip_pred=True
    for b in range(B):
        r = weighted_pearson_np(y[b, :, 0].numpy(), s[b].numpy())
        assert abs(wp_torch(y[b, :, 0], s[b], clip_pred=True).item() - r) < 1e-5, (b, r)
    mu = torch.zeros(B); sd = torch.ones(B)
    z = zmse_loss(s, y, mu, sd); assert z.ndim == 0 and torch.isfinite(z)
    assert torch.allclose(soft_clip(torch.tensor([0.0, 100.0])), torch.tensor([0.0, 2.0]))
    print("losses.py self-test ok")


# --------------------------------------------------------------------------
# Running-statistics whole-sequence weighted Pearson
# --------------------------------------------------------------------------
class RunningWP:
    """Weighted-Pearson sufficient statistics accumulated over a whole 20,000-row sequence.

    `seq_pooled_wp_loss` pools K windows of one sequence -- with the usual settings that is
    4 x 512 = 2,048 rows out of 20,000, so the sequence's centring statistics rest on four
    samples and any LEVEL DRIFT across the remaining 90% is unpenalised. The deployed
    metric centres over every scored row of the sequence. A competitor placing above us
    diagnosed exactly this (per-quarter WP unchanged while whole-sequence WP fell) and our
    own EDA reached it independently: the model tracks increments and has no level.

    The fix is to carry sufficient statistics, not rows. Each chunk contributes to
        S_w, S_wy, S_wp, S_wyy, S_wpp, S_wyp     (per target)
    and the loss is evaluated on prefix + current chunk, with the prefix DETACHED. Memory
    is O(1) per sequence and gradients still only span one chunk, so BPTT depth is
    unchanged -- it is the objective that becomes whole-sequence, not the backward pass.
    """

    __slots__ = ("s", "eps")

    def __init__(self, groups, n_tgt, device, dtype=torch.float64, eps=1e-6):
        self.s = torch.zeros(groups, n_tgt, 6, device=device, dtype=dtype)
        self.eps = eps

    def reset(self, idx=None):
        if idx is None: self.s.zero_()
        else: self.s[idx] = 0.0

    @staticmethod
    def _terms(y, p, row_w):
        """y/p: (G,T,n_tgt) with y ALREADY clipped; row_w: (G,T) or None.

        The metric's own weight is |clip(y_j, +-2)|, per target -- so it is computed here
        rather than passed in. `row_w` multiplies it, which is where a soft `is_scored`
        mask enters: the hidden mask selects ~11% of rows, and weighting by the predicted
        probability is how you train the metric you are actually scored on rather than the
        all-rows proxy.
        """
        w = y.abs()
        if row_w is not None:
            w = w * row_w.unsqueeze(-1)
        return torch.stack([w, w*y, w*p, w*y*y, w*p*p, w*y*p], -1)

    def update(self, y, p, row_w=None):
        """Accumulate a chunk, detached. y/p: (G, T, n_tgt), row_w: (G, T) or None."""
        with torch.no_grad():
            self.s += self._terms(y, p, row_w).sum(1).to(self.s.dtype)

    def decay(self, factor: float):
        """Shrink the accumulated statistics, giving a finite effective window."""
        if factor < 1.0:
            self.s.mul_(factor)

    def loss(self, y, p, row_w=None, gamma: float = 0.0, pooled: bool = False, rescale: bool = False):
        """1 - mean WP over (detached prefix + this chunk), gradient through p only.
        gamma > 0: focal weighting, mean((1 - corr) ** (1 + gamma)) -- harder sequences
        (many scored rows, low prefix correlation) weigh more.

        pooled=True sums the sufficient statistics ACROSS LANES first, so ONE correlation
        is formed over every sequence in flight rather than one per lane and averaged.
        That is the aggregation the live metric uses: moments pooled across sequences, a
        single weighted centre, explicitly not averaged by sequence. Per-lane centring is
        what makes level drift cancel, and level is what the metric now scores."""
        cur = self._terms(y, p, row_w).sum(1)                       # (G, n_tgt, 6)
        tot = self.s.to(cur.dtype) + cur
        if pooled:
            tot = tot.sum(0, keepdim=True)                          # (1, n_tgt, 6)
        S, Sy, Sp, Syy, Spp, Syp = tot.unbind(-1)
        S = S.clamp_min(self.eps)
        my, mp = Sy/S, Sp/S
        cov = Syp/S - my*mp
        # eps goes INSIDE the sqrt. Clamping each variance to 1e-12 and taking
        # sqrt(vy*vp) lets the denominator reach 1e-12, so a sequence whose prediction
        # variance is still near zero -- every sequence's FIRST chunk -- produces a
        # gradient ~1e9x too large. That is what collapsed eval to 0.0032 while the
        # training loss stayed flat. Flooring the product instead bounds it at sqrt(eps).
        vy = (Syy/S - my*my).clamp_min(0)
        vp = (Spp/S - mp*mp).clamp_min(0)
        one_minus = 1.0 - cov / torch.sqrt(vy*vp + self.eps)
        if rescale:
            # research M (27 Sep): the gradient of a running correlation w.r.t. the CURRENT chunk's
            # predictions scales with (chunk mass / accumulated mass), so it decays ~1/k over the
            # sequence (14x from chunk 0 to chunk 19) while the per-chunk MSE terms stay flat -- the
            # Pearson term was >95% absent on most rows. Multiply by the detached mass ratio so every
            # chunk's Pearson gradient has the same per-row magnitude as the first chunk's.
            # Divide by the NOMINAL chunk mass (rows x lanes), never by the chunk's own weight mass:
            # a chunk that is mostly warm-up / unscored rows has a tiny mass and would blow the
            # factor up to S/eps (measured: loss 4.6e7). Per-row gradient ~ (1/S) x (S/T) = 1/T.
            _nominal = float(y.shape[1]) * (float(y.shape[0]) if pooled else 1.0)
            one_minus = one_minus * (S / _nominal).detach()
        if gamma > 0:
            return (one_minus.clamp_min(0) ** (1.0 + gamma)).mean()
        return one_minus.mean()


def _direct_row(mm, i):
    """Row i of a 3-D .npy memmap read with O_DIRECT (bypasses the page cache of THIS filesystem); None if unavailable."""
    import os, mmap as _mmap
    try:
        fn = mm.filename if hasattr(mm, "filename") else getattr(mm, "_mmap", None) and mm._mmap.name
        fn = getattr(mm, "filename", None)
        if fn is None: return None
        off0 = int(mm.offset) + i * int(mm.strides[0]); nbytes = int(mm.strides[0])
        A = 4096; lo = (off0 // A) * A; hi = ((off0 + nbytes + A - 1) // A) * A
        fd = os.open(fn, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        try:
            buf = _mmap.mmap(-1, hi - lo)
            got, pos = 0, 0
            while pos < hi - lo:
                n = os.preadv(fd, [memoryview(buf)[pos:]], lo + pos)
                if n <= 0: break
                pos += n
            if pos < off0 - lo + nbytes: return None
            out = np.frombuffer(buf, dtype=mm.dtype, count=nbytes // mm.dtype.itemsize, offset=off0 - lo).reshape(mm.shape[1:]).copy()
            buf.close(); return out
        finally:
            os.close(fd)
    except Exception:
        return None


class SequenceStream:
    """G concurrent readers, each walking ONE sequence from row 0 to the end in chunks.

    Random-window sampling cannot support a whole-sequence objective: the statistics have
    to be accumulated in order, from the start of the sequence, with the recurrent state
    carried alongside them. When a reader finishes its sequence it draws a new one and
    signals a reset so the caller can clear both the state and the statistics for that lane.

    Warm-up rows (< WARMUP) advance the state but are excluded from the loss, exactly as
    deployment does.

    Row factor r(t), applied ONCE on top of the metric's own |clip(y)| weight, is built
    BY SOURCE, because a ConcatCache of an unmasked primary (train_full) and a masked
    extra (valid_retrain) exposes `M = None` and silently drops the extra's real mask:

        train_row_factor   "one"     r = 1 on every post-warm-up row
                           "pred"    r = row_weight[i, t]  (a predicted P(is_scored))
                           "actual"  r = the cache's own is_scored (needs a masked primary)
        extra_row_factor   "one" | "actual"

    A legacy combined row_weight array with one row per COMBINED-cache sequence is still
    accepted (that is how the RM_* runs were driven) and overrides both factors.

    Sampling: with `epochs=None` sequences are drawn uniformly WITH replacement (three
    nominal passes then cover ~95% of sequences at least once). With `epochs` set, a
    fresh permutation is drawn per epoch and consumed without replacement, so every
    sequence appears exactly once per epoch; `new_epoch` is raised on the step at which
    the permutation was refilled. `summary()` reports what was actually drawn.
    """

    def __init__(self, cache, groups, chunk, rng, row_weight=None, epochs=None,
                 train_row_factor="one", extra_row_factor="one", mix_lambda=3.0, fam0_mix=None, fam_mix=None, fam_subset=None,
                 sign_balance=None, big_emph=0.0, seq_subset=None, fam0_repeat=1, subset_all=False):
        self.c, self.G, self.L, self.rng = cache, groups, chunk, rng
        # subset_all: seq_subset indexes the UNION (primary, then extra) instead of the primary only,
        # so cross-fitting folds can hold out validation-derived sequences too (add_wdrop_cf.py).
        self.subset_all = bool(subset_all)
        # BAGGING. Restrict the PRIMARY cache to these indices; the extra cache is always kept
        # whole (the 0x/1x/3x dose-response showed dropping valid_retrain costs -0.0021, so it
        # must never be subsampled). Ensemble gain is bounded by member decorrelation, and until
        # now every member saw identical data -- they differed only by seed and loss weight.
        self.seq_subset = None if seq_subset is None else np.asarray(seq_subset, dtype=np.int64)
        # FAMILY-0 OVERSAMPLING. src/family.py: family 0 (the largest tick, q=0.328) is ~42% of
        # sequences but carries **49% of the total |y| weight mass** under the globally pooled
        # metric, while the model scores 0.4263 there against 0.72-0.80 on families 1-3. Half the
        # metric comes from the segment we are worst at. Drawing those sequences k times per epoch
        # gives them more gradient steps WITHOUT discarding the other families -- unlike the
        # family-0-only specialist GF0, which lost more in data than it gained in focus and is
        # worse than the general model even on family 0 (-0.0261).
        self.fam0_repeat = int(fam0_repeat)
        # big_emph: rows whose clipped |y| (max over targets) >= 1.5 get row factor x(1 + big_emph)
        self.big_emph = float(big_emph)
        # sign_balance: set of families whose rows get an extra factor 0.5 / P_seq(sign of s
        # at this row), so both directions carry equal weight within each sequence
        self.sign_balance = set(sign_balance) if sign_balance else None
        self._sb = {}
        # Only draw sequences of these tick families (router specialists). Family from the
        # target lattice, computed lazily per sequence and cached.
        self.fam_subset = set(fam_subset) if fam_subset else None
        # Family-0 sequences (identified from their own target lattice, a training-side
        # quantity) get r = fam0_mix * mask_factor + (1 - fam0_mix) * 1: the mask arm beat
        # the all-row arm on families 1-3 and LOST to it on family 0 (holdout, 13 Sep).
        self.fam0_mix = fam0_mix
        self._fam = {}
        self.has_extra = hasattr(cache, "a") and hasattr(cache, "b")
        self.n_primary = len(cache.a) if self.has_extra else len(cache)
        self.n_total = len(cache)
        self.mode_train, self.mode_extra = train_row_factor, extra_row_factor
        self.mix_lambda = float(mix_lambda)      # extra_row_factor "mixed": r = 1 + lambda * is_scored (research K #2)
        self.rw, self.legacy_combined = None, False
        if row_weight is not None:
            if len(row_weight) == self.n_total and self.n_total != self.n_primary:
                self.rw, self.legacy_combined = row_weight, True
                self.mode_train = self.mode_extra = "combined-array"
            elif len(row_weight) == self.n_primary:
                self.rw = row_weight
                if train_row_factor != "pred":
                    raise SystemExit("--row-weight given but --train-row-factor is not 'pred'; "
                                     "say which factor you mean")
            else:
                raise SystemExit(
                    f"row_weight has {len(row_weight)} sequences; the primary cache has "
                    f"{self.n_primary} and the combined cache {self.n_total}. It is indexed "
                    f"by cache position, so it must match one of them exactly.")
        elif train_row_factor == "pred":
            raise SystemExit("--train-row-factor pred needs --row-weight")
        pm = cache.a.M if self.has_extra else getattr(cache, "M", None)
        if train_row_factor == "actual" and pm is None:
            raise SystemExit("--train-row-factor actual: the primary cache carries no mask")
        if extra_row_factor in ("actual", "mixed") and (not self.has_extra or cache.b.M is None):
            raise SystemExit(f"--extra-row-factor {extra_row_factor}: no masked --extra-cache")
        self._pm = pm
        self._em = cache.b.M if self.has_extra else None
        # sampling schedule
        self.epochs = epochs
        # Target FAMILY shares for the epoch pool. train.py computed these and then dropped
        # them: seq_p_fam was only ever read by the windowed sampler, and SequenceStream had
        # no parameter for it, so --family-mix was a SILENT NO-OP under --loss seqrun (the
        # only loss in use). FAM_mixHold_s801 scored 0.665463, byte-identical to its baseline,
        # which is what exposed it. Train shares are 0.426/0.319/0.125/0.130 against a holdout
        # 0.383/0.227/0.160/0.230, so family 3 is ~1.8x under-represented in training.
        self.fam_mix = None if fam_mix is None else np.asarray(fam_mix, dtype=np.float64)
        self._queue = []
        self.epoch = 0
        self.new_epoch = False
        self.drawn = np.zeros(self.n_total, np.int64)
        self.rows_forward = 0
        self.rows_selected = 0
        self.weight_mass = 0.0
        self.seq = np.array([self._draw() for _ in range(groups)])
        self.pos = np.zeros(groups, dtype=int)
        self.new_epoch = False          # the initial fill is not an epoch boundary
        self.teacher = None             # (primary (N,T,2), extra (M,T,2)) float16 memmaps, see set_teacher
        self.T_out = None
        self.teacher2 = None            # optional second teacher (add_distill2.py)
        self.T2_out = None
        # research Z6 (add_signals.py): mask emission for --mask-aux and the --curric difficulty curriculum. Both are
        # off unless enabled, and neither touches the rng or the draw order.
        self.emit_mask, self.M_out, self.MV_out = False, None, None
        self.curric_E, self.curric_thr = 0, None
        self.lane_epoch = np.full(groups, self.epoch, np.int64)   # epoch in which each lane's current sequence was drawn
        # research Z12 (add_z12.py): --elr per-row self-target store (n_total, T, 2) float16, NaN = never visited; next() fills
        # E_out (G,L,2) with the stored target (0 where none) and EV_out (G,L) = 1 where one exists; cur = (seq, a, b) of
        # the chunk next() last emitted per lane, so elr_write() can fold the chunk's predictions back in. Off unless set.
        self.elr, self.E_out, self.EV_out = None, None, None
        self.cur = np.zeros((groups, 3), np.int64)
        # research Z15 (add_z15.py): --chunk-jitter. jit_rng draws the first-chunk length of every sequence cycle (lanes in
        # lock-step); jit_n cycles jittered so far, jit_last the last length drawn. Off unless set_chunk_jitter() is called.
        self.jit_rng, self.jit_n, self.jit_last = None, 0, 0

    def set_teacher(self, primary, extra=None):
        """Distillation targets aligned with the combined cache: row i < n_primary reads primary[i],
        otherwise extra[i - n_primary]. next() then fills self.T_out (G,L,2) beside X/Y."""
        if len(primary) != self.n_primary or (self.has_extra and (extra is None or len(extra) != self.n_total - self.n_primary)):
            raise SystemExit(f"teacher arrays ({len(primary)}, {None if extra is None else len(extra)}) do not match the "
                             f"caches ({self.n_primary}, {self.n_total - self.n_primary})")
        self.teacher = (primary, extra)
        self.T_out = np.zeros((self.G, self.L, 2), np.float32)

    def set_teacher2(self, primary, extra=None):
        """Second distillation teacher, same layout as set_teacher (add_distill2.py)."""
        if len(primary) != self.n_primary or (self.has_extra and (extra is None or len(extra) != self.n_total - self.n_primary)):
            raise SystemExit(f"teacher2 arrays ({len(primary)}, {None if extra is None else len(extra)}) do not match the "
                             f"caches ({self.n_primary}, {self.n_total - self.n_primary})")
        self.teacher2 = (primary, extra)
        self.T2_out = np.zeros((self.G, self.L, 2), np.float32)

    def set_elr(self, store):
        """--elr (add_z12.py): per-row temporal-ensemble self-target over the combined cache (n_total, T, 2) float16, NaN
        where no pass has recorded a prediction yet. next() then fills E_out / EV_out beside X/Y."""
        if store.shape[0] != self.n_total or store.shape[1] != SEQUENCE_LENGTH:
            raise SystemExit(f"elr store {store.shape} does not match the combined cache ({self.n_total}, {SEQUENCE_LENGTH}, 2)")
        self.elr = store
        self.E_out = np.zeros((self.G, self.L, 2), np.float32); self.EV_out = np.zeros((self.G, self.L), np.float32)

    def elr_write(self, P, beta):
        """--elr (add_z12.py): fold the chunk's predictions P (G,L,2) float32 -- for the chunk next() last emitted -- into the
        store: S <- beta S + (1 - beta) P where a target exists, S <- P on a row's first recorded visit."""
        for g in range(self.G):
            i, a, b = (int(v) for v in self.cur[g]); n = b - a
            old = np.asarray(self.elr[i, a:b], np.float32)
            new = P[g, :n].astype(np.float32)
            has = np.isfinite(old[:, 0])
            if has.any():
                new[has] = beta * old[has] + (1.0 - beta) * new[has]
            self.elr[i, a:b] = new.astype(np.float16)

    def set_chunk_jitter(self, rng):
        """--chunk-jitter (add_z15.py): the first chunk of every sequence cycle gets a random length 1..L-1 drawn from
        `rng`, so the TBPTT boundaries sit at a fresh offset per sequence per epoch. Requires lock-step lanes."""
        if not (self.pos == 0).all():
            raise SystemExit("--chunk-jitter must be enabled before the first chunk is drawn and with every lane at row 0 (no --stagger)")
        self.jit_rng = rng

    def enable_mask_out(self):
        """--mask-aux (add_signals.py): next() also fills M_out (G,L) = the source cache's is_scored as 0/1 and MV_out
        (G,L) = 1 on post-warm-up rows whose source cache HAS a mask. train_full carries none, so its rows get MV = 0
        and the head is never supervised there."""
        self.emit_mask = True
        self.M_out = np.zeros((self.G, self.L), np.float32); self.MV_out = np.zeros((self.G, self.L), np.float32)
        _p = "masked" if self._pm is not None else "NO mask -> unsupervised"
        _e = ("masked" if self._em is not None else "NO mask -> unsupervised") if self.has_extra else "none"
        return f"  mask-aux targets: primary {_p}, extra {_e}"

    def set_curriculum(self, epochs, n_seq=200, q=0.90):
        """--curric E (add_signals.py): while a lane's sequence was drawn in epoch <= E, rows whose teacher residual
        max_k |clip(t_k) - clip(y_k)| exceeds the family's q-quantile get row factor 0. Thresholds from n_seq evenly
        spaced sequences of the combined cache (post-warm-up rows); key -1 = pooled, used for unlabelled families."""
        if self.teacher is None: raise SystemExit("--curric needs a teacher (--distill)")
        self.curric_E = int(epochs)
        idx = np.unique(np.linspace(0, self.n_total - 1, min(n_seq, self.n_total)).astype(np.int64))
        res, mass_all, mass_drop = {}, {}, {}
        for i in idx:
            r, wy = self._curric_resid(int(i), WARMUP, SEQUENCE_LENGTH)
            res.setdefault(self.family(int(i)), []).append((r, wy))
        allr = np.concatenate([r for v in res.values() for r, _ in v])
        self.curric_thr = {-1: float(np.quantile(allr, q))}
        for f, v in res.items():
            if f >= 0: self.curric_thr[f] = float(np.quantile(np.concatenate([r for r, _ in v]), q))
            thr = self.curric_thr.get(f, self.curric_thr[-1])
            mass_all[f] = float(sum(wy.sum() for _, wy in v)); mass_drop[f] = float(sum(wy[r > thr].sum() for r, wy in v))
        parts = ", ".join(f"f{f}: thr {self.curric_thr.get(f, self.curric_thr[-1]):.3f} drops {mass_drop[f] / max(mass_all[f], 1e-9):.1%} of |y| mass"
                          for f in sorted(res))
        return (f"  curriculum: epochs 1..{self.curric_E}, rows above the {q:.0%} teacher-residual quantile get factor 0 "
                f"({len(idx)} seqs sampled; pooled thr {self.curric_thr[-1]:.3f}; {parts})")

    def _curric_resid(self, i, a, b):
        """(max_k |clip(t_k) - clip(y_k)|, sum_k |clip(y_k)|) for rows a..b of combined-cache sequence i."""
        src = self.teacher[0][i] if i < self.n_primary else self.teacher[1][i - self.n_primary]
        t = np.clip(np.asarray(src[a:b], np.float32), -METRIC_CLIP, METRIC_CLIP)
        y = np.clip(np.asarray(self.c.Y[i, a:b], np.float32), -METRIC_CLIP, METRIC_CLIP)
        return np.abs(t - y).max(1), np.abs(y).sum(1)

    def _curric_keep(self, i, a, b):
        r, _ = self._curric_resid(i, a, b)
        return (r <= self.curric_thr.get(self.family(i), self.curric_thr[-1])).astype(np.float32)

    def _fam_mix(self, pool):
        """Replicate each family in the epoch pool so its share approaches self.fam_mix."""
        if self.fam_mix is None:
            return pool
        if not hasattr(self, "_fam_mix_pool"):
            fams = np.array([self.family(int(i)) for i in pool])
            # base_of() returns -1 when a sequence's tick base is not identifiable. Those
            # cannot be assigned a target share, so they ride along once rather than being
            # dropped (which would shrink the training set) or miscounted.
            unknown = pool[fams < 0]
            keep = fams >= 0
            pool_k, fams = pool[keep], fams[keep]
            cnt = np.bincount(fams, minlength=len(self.fam_mix)).astype(np.float64)
            per = self.fam_mix / np.maximum(cnt, 1.0)
            r = per / per[per > 0].min()                 # cheapest family stays at 1x
            parts = []
            for f in range(len(self.fam_mix)):
                idx = pool_k[fams == f]
                if not len(idx): continue
                whole = int(np.floor(r[f]))
                extra = int(round((r[f] - whole) * len(idx)))
                parts.extend([idx] * whole)
                if extra > 0:
                    parts.append(idx[:extra])            # fixed slice, not a random subset
            if len(unknown): parts.append(unknown)
            self._fam_mix_pool = np.concatenate(parts) if parts else pool
            _gf = np.array([self.family(int(i)) for i in self._fam_mix_pool])
            got = np.bincount(_gf[_gf >= 0], minlength=len(self.fam_mix)).astype(np.float64)
            print("  family-mix: pool %d -> %d, shares %s (target %s, %d unlabelled)"
                  % (len(pool), len(self._fam_mix_pool),
                     np.round(got / got.sum(), 3), np.round(self.fam_mix, 3), len(unknown)), flush=True)
        return self._fam_mix_pool

    def _allowed(self):
        if self.fam_subset is None:
            k = int(getattr(self, "extra_repeat", 1))
            if k > 1 and self.has_extra:
                # oversample the validation-derived extra cache: each of its sequences appears k
                # times per epoch (the test period resembles valid more than train)
                if not hasattr(self, "_allowed_rep"):
                    ex = np.arange(self.n_primary, self.n_total)
                    self._allowed_rep = np.concatenate([np.arange(self.n_primary)] + [ex] * k)
                    print(f"  extra cache oversampled x{k}: epoch pool {len(self._allowed_rep)} draws", flush=True)
                return self._fam0(self._bag(self._fam_mix(self._allowed_rep)))
            return self._fam0(self._bag(self._fam_mix(np.arange(self.n_total))))
        if not hasattr(self, "_allowed_idx"):
            self._allowed_idx = np.array([i for i in range(self.n_total) if self.family(i) in self.fam_subset])
            print(f"  family subset {sorted(self.fam_subset)}: {len(self._allowed_idx)} of {self.n_total} sequences", flush=True)
        return self._fam0(self._bag(self._fam_mix(self._allowed_idx)))

    def _fam0(self, pool):
        """Repeat family-0 indices fam0_repeat times in the draw pool."""
        if self.fam0_repeat <= 1:
            return pool
        if not hasattr(self, "_fam0_pool"):
            f0 = np.array([i for i in pool if self.family(i) == 0], dtype=np.int64)
            self._fam0_pool = np.concatenate([pool] + [f0] * (self.fam0_repeat - 1))
            print(f"  family-0 oversampled x{self.fam0_repeat}: {len(f0)} of {len(pool)} "
                  f"sequences repeated -> epoch pool {len(self._fam0_pool)}", flush=True)
        return self._fam0_pool

    def _bag(self, pool):
        """Keep only primary indices in seq_subset; every extra-cache index survives."""
        if self.seq_subset is None:
            return pool
        if not hasattr(self, "_bag_cache"):
            keep = np.zeros(self.n_total, bool)
            if self.subset_all:
                keep[self.seq_subset] = True                     # union index: extras can be held out
            else:
                keep[self.seq_subset[self.seq_subset < self.n_primary]] = True
                keep[self.n_primary:] = True
            self._bag_keep = keep
            self._bag_cache = {}
        key = (len(pool), int(pool[0]) if len(pool) else -1, int(pool[-1]) if len(pool) else -1)
        if key not in self._bag_cache:
            out = pool[self._bag_keep[pool]]
            self._bag_cache[key] = out
            print(f"  seq-subset ({'union' if self.subset_all else 'bagging'}): "
                  f"{int(self._bag_keep[:self.n_primary].sum())} of {self.n_primary} primary kept, "
                  f"{int(self._bag_keep[self.n_primary:].sum())} of {self.n_total - self.n_primary} "
                  f"extra kept -> pool {len(out)}", flush=True)
        return self._bag_cache[key]

    def _draw(self):
        pool = self._allowed()
        if getattr(self, "z46_rep", None) is not None:      # add_z46.py: per-sequence repeat counts (train.py sets z46_rep)
            pool = self._z46_pool(pool)
        if self.epochs is None:
            i = int(pool[self.rng.integers(0, len(pool))])
        else:
            if not self._queue:
                self._queue = list(pool[self.rng.permutation(len(pool))])
                self.epoch += 1
                if self.epoch > 1:
                    self.new_epoch = True
            i = int(self._queue.pop())
        self.drawn[i] += 1
        return i

    def _z46_pool(self, pool):
        """add_z46.py: index i appears z46_rep[i] times (0 drops it); cached per pool identity like _bag()."""
        key = (len(pool), int(pool[0]) if len(pool) else -1, int(pool[-1]) if len(pool) else -1)
        c = self.__dict__.setdefault("_z46_cache", {})
        if key not in c:
            c[key] = np.repeat(pool, self.z46_rep[pool])
            if not len(c[key]):
                raise SystemExit("--seq-repeat / --extra-fam-repeat left an empty draw pool")
        return c[key]

    def family(self, i):
        if i not in self._fam:
            from family import base_of
            self._fam[i] = int(base_of(np.asarray(self.c.Y[i])))
        return self._fam[i]

    def factor(self, i, a, b):
        """r(t) for rows a..b of combined-cache sequence i, warm-up already zeroed."""
        rows = np.arange(a, b)
        w = (rows >= WARMUP).astype(np.float32)
        if self.legacy_combined:
            return w * np.asarray(self.rw[i, a:b], np.float32)
        if i < self.n_primary:
            if self.mode_train == "pred":
                f = np.asarray(self.rw[i, a:b], np.float32)
            elif self.mode_train == "actual":
                f = np.asarray(self._pm[i, a:b], np.float32)
            else:
                f = None
        else:
            j = i - self.n_primary
            if self.mode_extra == "actual":
                f = np.asarray(self._em[j, a:b], np.float32)
            elif self.mode_extra == "mixed":
                f = 1.0 + self.mix_lambda * np.asarray(self._em[j, a:b], np.float32)
            else:
                f = None
        if self.big_emph > 0:
            Yb = np.abs(np.clip(np.asarray(self.c.Y[i, a:b], np.float32), -METRIC_CLIP, METRIC_CLIP)).max(1)
            w = w * np.where(Yb >= 1.5, 1.0 + self.big_emph, 1.0).astype(np.float32)
        if self.sign_balance is not None and self.family(i) in self.sign_balance:
            if i not in self._sb:
                Y = np.clip(np.asarray(self.c.Y[i], np.float32), -METRIC_CLIP, METRIC_CLIP)
                sg = (Y[:, 0] - Y[:, 1]) / 2; wa = np.abs(sg); pos = sg > 0
                p_pos = float(wa[pos].sum() / max(wa.sum(), 1e-9)); p_pos = min(max(p_pos, 0.05), 0.95)
                self._sb[i] = np.where(pos, 0.5 / p_pos, 0.5 / (1.0 - p_pos)).astype(np.float32)
            w = w * self._sb[i][a:b]
        if f is None:
            return w
        if self.fam0_mix is not None and self.family(i) == 0:
            f = self.fam0_mix * f + (1.0 - self.fam0_mix)
        return w * f

    def describe(self):
        """One line per source saying what the loss will actually weight, with a check
        on one sequence of each source so the realised mask count is asserted, not
        inferred from a flag name."""
        lines = []
        i = 0
        w = self.factor(i, 0, SEQUENCE_LENGTH)
        lines.append(f"  primary ({self.n_primary} seqs): row factor '{self.mode_train}' -> "
                     f"seq 0 selected rows {(w > 0).sum()} / {SEQUENCE_LENGTH - WARMUP}, "
                     f"mass {w.sum():.1f}, warm-up mass {w[:WARMUP].sum():.3f}")
        assert w[:WARMUP].sum() == 0.0
        if self.mode_train == "one" and not self.legacy_combined:
            assert (w > 0).sum() == SEQUENCE_LENGTH - WARMUP
        if self.has_extra:
            j = self.n_primary
            w = self.factor(j, 0, SEQUENCE_LENGTH)
            exp = (np.asarray(self._em[0], bool)[WARMUP:].sum() if self.mode_extra == "actual"
                   else SEQUENCE_LENGTH - WARMUP)
            lines.append(f"  extra ({self.n_total - self.n_primary} seqs): row factor "
                         f"'{self.mode_extra}' -> seq 0 selected rows {(w > 0).sum()} "
                         f"(expected {exp}), mass {w.sum():.1f}")
            if self.mode_extra in ("one", "actual", "mixed"):
                assert (w > 0).sum() == exp, "realised extra-mask count does not match"
        sched = ("uniform with replacement" if self.epochs is None
                 else f"{self.epochs} epoch(s), permutation without replacement")
        lines.append(f"  sampling: {sched}, {self.G} lanes x {self.L}-row chunks")
        if self.fam0_mix is not None:
            k = next(i for i in range(self.n_total) if self.family(i) == 0)
            w = self.factor(k, 0, SEQUENCE_LENGTH)
            lines.append(f"  family-0 row factor = {self.fam0_mix:g} x mask + {1 - self.fam0_mix:g} x 1: "
                         f"seq {k} (family 0) selected rows {(w > 0).sum()}, mass {w.sum():.1f}")
        return "\n".join(lines)

    def summary(self):
        d = self.drawn
        prim, ext = d[:self.n_primary], d[self.n_primary:]
        return {"drawn_primary": int(prim.sum()), "drawn_extra": int(ext.sum()),
                "unique_primary": int((prim > 0).sum()), "unique_extra": int((ext > 0).sum()),
                "coverage_primary": float((prim > 0).mean()) if len(prim) else 0.0,
                "coverage_extra": float((ext > 0).mean()) if len(ext) else 0.0,
                "max_draws_one_seq": int(d.max()), "epochs_started": int(self.epoch),
                "rows_forward": int(self.rows_forward), "rows_selected": int(self.rows_selected),
                "weight_mass": float(self.weight_mass),
                "train_row_factor": self.mode_train, "extra_row_factor": self.mode_extra}

    def next(self):
        """-> X (G,L,F), Yc (G,L,2) clipped, W (G,L) loss weight, reset (G,) bool."""
        F = self.c.X.shape[-1]
        X = np.zeros((self.G, self.L, F), np.float32)
        Y = np.zeros((self.G, self.L, 2), np.float32)
        W = np.zeros((self.G, self.L), np.float32)
        reset = np.zeros(self.G, bool)
        self.new_epoch = False
        _Ls = self.L                                          # add_z15.py: rows every lane consumes this step (= L unless jittered)
        if self.jit_rng is not None:
            if not (self.pos == self.pos[0]).all():
                raise SystemExit("--chunk-jitter needs the lanes in lock-step (every lane at the same row); not with --stagger")
            for _nm in ("T_out", "T2_out", "M_out", "MV_out", "E_out", "EV_out"):   # re-widen buffers narrowed last step
                _bf = getattr(self, _nm)
                if _bf is not None and _bf.shape[1] != self.L:
                    setattr(self, _nm, np.zeros((self.G, self.L) + _bf.shape[2:], np.float32))
            if self.pos[0] == 0:                              # every lane starts a sequence: draw this cycle's first-chunk length
                self.jit_last = int(self.jit_rng.integers(1, self.L)); self.jit_n += 1
                _Ls = self.jit_last
            _n_common = min(_Ls, SEQUENCE_LENGTH - int(self.pos[0]))
        for g in range(self.G):
            if self.pos[g] == 0:
                reset[g] = True
            i, a = int(self.seq[g]), int(self.pos[g])
            b = min(a + _Ls, SEQUENCE_LENGTH)
            n = b - a
            # whole-sequence prefetch (offline/patches/add_seqprefetch.py): identical values, one sequential read per
            # sequence per lane instead of 40 random 224 KB memmap reads -- the difference between 2 and 5 steps/s on
            # an IOPS-bound volume. Entry: lane -> (seq index, X copy, Y copy); dropped when the lane finishes.
            _pf = self.__dict__.setdefault("_pf", {})
            _e = _pf.get(g)
            if _e is None or _e[0] != i:
                _xd = _direct_row(self.c.X, i) if os.environ.get("WNN_DIRECT_READ", "1") != "0" else None
                _yd = _direct_row(self.c.Y, i) if _xd is not None else None
                _e = (i, (_xd if _xd is not None else np.array(self.c.X[i], dtype=np.float32, copy=True)).astype(np.float32, copy=False),
                      (_yd if _yd is not None else np.array(self.c.Y[i], dtype=np.float32, copy=True)).astype(np.float32, copy=False))
                _pf[g] = _e
            X[g, :n] = _e[1][a:b]
            Y[g, :n] = _e[2][a:b]
            W[g, :n] = self.factor(i, a, b)
            if self.curric_thr is not None and self.lane_epoch[g] <= self.curric_E:
                W[g, :n] *= self._curric_keep(i, a, b)         # add_signals.py: hard rows out during epochs 1..E
            if self.emit_mask:
                _ms = self._pm if i < self.n_primary else self._em
                if _ms is not None:
                    self.M_out[g, :n] = np.asarray(_ms[(i if i < self.n_primary else i - self.n_primary), a:b], np.float32)
                    self.MV_out[g, :n] = (np.arange(a, b) >= WARMUP).astype(np.float32)
                else:
                    self.M_out[g, :n] = 0.0; self.MV_out[g, :n] = 0.0
                self.M_out[g, n:] = 0.0; self.MV_out[g, n:] = 0.0
            if self.teacher is not None:
                src = self.teacher[0][i] if i < self.n_primary else self.teacher[1][i - self.n_primary]
                self.T_out[g, :n] = src[a:b]; self.T_out[g, n:] = 0.0
            if self.teacher2 is not None:
                src2 = self.teacher2[0][i] if i < self.n_primary else self.teacher2[1][i - self.n_primary]
                self.T2_out[g, :n] = src2[a:b]; self.T2_out[g, n:] = 0.0
            self.cur[g] = (i, a, b)                           # add_z12.py: what this lane is about to train on
            if self.elr is not None:
                _es = np.asarray(self.elr[i, a:b], np.float32)
                _ev = np.isfinite(_es[:, 0])
                self.E_out[g, :n] = np.where(_ev[:, None], _es, 0.0); self.E_out[g, n:] = 0.0
                self.EV_out[g, :n] = _ev; self.EV_out[g, n:] = 0.0
            self.pos[g] = b
            if b >= SEQUENCE_LENGTH:
                _pf.pop(g, None)
                self.seq[g] = self._draw()
                self.pos[g] = 0
                self.lane_epoch[g] = self.epoch                 # add_signals.py: the epoch this lane's new sequence belongs to
        if self.jit_rng is not None and _n_common < self.L:  # add_z15.py: emit exactly the rows every lane consumed
            X, Y, W = X[:, :_n_common], Y[:, :_n_common], W[:, :_n_common]
            for _nm in ("T_out", "T2_out", "M_out", "MV_out", "E_out", "EV_out"):
                _bf = getattr(self, _nm)
                if _bf is not None: setattr(self, _nm, _bf[:, :_n_common])
        self.rows_forward += int(X.shape[0] * X.shape[1])
        self.rows_selected += int((W > 0).sum())
        self.weight_mass += float(W.sum())
        return X, np.clip(Y, -METRIC_CLIP, METRIC_CLIP), W, reset


def weighted_mse(pred, y, row_w=None, eps: float = 1e-6):
    """|clip(y)|-weighted MSE. Its job is to GIVE THE OUTPUTS A SCALE.

    Weighted Pearson is invariant to output magnitude, so a pure correlation objective
    leaves the prediction scale unconstrained and it drifts -- measured here as the export
    calibration factor falling monotonically 14.4 -> 0.275 over three epochs while the
    training loss stayed flat. The reference implementation carries this term at weight
    0.1 alongside the sequence Pearson at 0.9, for exactly this reason.
    """
    yc = y.clamp(-METRIC_CLIP, METRIC_CLIP)
    w = yc.abs()
    if row_w is not None:
        w = w * row_w.unsqueeze(-1)
    return (w * (pred - yc)**2).sum() / w.sum().clamp_min(eps)


def clip_ste(p, clip: float = METRIC_CLIP, band: float = 2.2):
    """clip(p, +-clip) in the forward pass with a straight-through gradient (research Z5, --anchor-clip).

    The metric clips predictions at +-2, so an anchor on the raw output penalises over-shoots the metric
    never sees. Forward value: the clipped prediction. Gradient: identity where |p| < band, zero outside,
    so a prediction already past the clip is not pulled further and one just inside it still learns.
    Where |p| < clip the forward value is p itself (bit-identical), so the flag changes nothing unless a
    prediction actually crosses the clip.
    """
    pc = p.clamp(-clip, clip)
    g = (p.abs() < band).to(p.dtype)
    return pc.detach() + (p - p.detach()) * g


def hetero_nll(pred, s, y, row_w=None, s_min: float = -6.0, s_max: float = 4.0, eps: float = 1e-6):
    """|clip(y)|-weighted Gaussian NLL with a PREDICTED per-row log-variance s (research Z5, --hetero).

    0.5 * (exp(-s) * (pred - y)^2 + s), per target, s clamped to [s_min, s_max] (the clamp also zeroes the
    gradient outside the range). The minimiser in pred is the same conditional mean the MSE anchor has;
    the difference is that rows the model expects to miss (large s) pull less and rows it expects to hit
    pull more -- a learned, row-wise reweighting of the regression anchor. Weighted and normalised exactly
    like `weighted_mse`, so W x hetero_nll is commensurate with the anchor term. Can be negative (s < 0).
    """
    yc = y.clamp(-METRIC_CLIP, METRIC_CLIP)
    w = yc.abs()
    if row_w is not None:
        w = w * row_w.unsqueeze(-1)
    s = s.clamp(s_min, s_max)
    nll = 0.5 * (torch.exp(-s) * (pred - yc) ** 2 + s)
    return (w * nll).sum() / w.sum().clamp_min(eps)


def _h12_phi(z):
    return torch.exp(-0.5 * z * z) * 0.3989422804014327


def _h12_Phi(z):
    return 0.5 * (1.0 + torch.erf(z * 0.7071067811865476))


def _h12_cumsum(x):
    """inclusive prefix sum of a 1-D tensor by doubling (Hillis-Steele): deterministic on every device (torch.cumsum on a CUDA
    tensor raises under torch.use_deterministic_algorithms(True), which train.py enables), O(n log n), autograd-friendly."""
    n = x.shape[0]; k = 1
    while k < n:
        x = torch.cat([x[:k], x[k:] + x[:-k]])
        k <<= 1
    return x


def _h12_energy(z, w, eps):
    """energy distance between the w-weighted sample z (1-D) and N(0,1): 2 E_w|z - Z| - E_ww|z - z'| - E|Z - Z'| (research H12)."""
    W = w.sum().clamp_min(eps); wn = w / W
    cross = (wn * (2.0 * _h12_phi(z) + z * (2.0 * _h12_Phi(z) - 1.0))).sum()            # E_w |z - Z|, Z ~ N(0,1), closed form
    zs, idx = torch.sort(z); ws = wn[idx]
    C = _h12_cumsum(ws) - ws                                                             # weight strictly below i
    S = _h12_cumsum(ws * zs) - ws * zs                                                   # weighted z strictly below i
    St = (ws * zs).sum()
    row = zs * C - S + (St - S - ws * zs) - zs * (1.0 - C - ws)                          # sum_j w_j |z_i - z_j|
    within = (ws * row).sum()
    return 2.0 * cross - within - 1.1283791670955126                                     # E|Z - Z'| = 2 / sqrt(pi)


def resnorm_terms(pred, logsig, y, row_w=None, beta: float = 1.0, weighting: str = "metric", fam=None, famnorm: bool = False,
                  ed: bool = False, ed_fam: bool = False, ed_grad_p: bool = False, clip_y: bool = True,
                  s_min: float = -6.0, s_max: float = 4.0, eps: float = 1e-6):
    """Research H12 (add_h12_resnorm.py): terms on the standardised residual z = (y - p) / sigma of a per-row, per-target
    log-sigma head. Returns {"nll": tensor, "ed": tensor or None, "stats": {floats}}.

    nll: Gaussian NLL  log sigma + 0.5 z^2  (constants dropped), beta-NLL (Seitzer et al. 2022): each row's term x sg(sigma^(2 beta)).
         beta = 1 -> the gradient w.r.t. p is exactly the row-weighted MSE's; beta = 0 -> the plain heteroscedastic NLL (p's row
         weight x 1 / sigma^2). famnorm: x sg(H_f), H_f = row-weighted harmonic mean of sigma^2 over the lanes of family f (per
         target), so the mean p-gradient weight within every family equals the anchor's. Normalised like weighted_mse.
    ed:  energy distance of the row-weighted z against N(0,1) per target (or per family x target, mass-weighted mean); >= 0 and 0
         iff z ~ N(0,1) in distribution; p detached inside z unless ed_grad_p (the term then calibrates sigma only).
    weighting "metric": |clip y| x row_w (the anchor's weights); "none": row_w alone. pred/logsig/y: (G, T, 2); row_w: (G, T);
    fam: (G,) long, the lane's family (famnorm / ed_fam). The log-sigma is clamped to [s_min, s_max] (zero gradient outside).
    """
    yc = y.clamp(-METRIC_CLIP, METRIC_CLIP) if clip_y else y
    w = yc.abs() if weighting == "metric" else torch.ones_like(yc)
    if row_w is not None:
        w = w * row_w.unsqueeze(-1)
    s = logsig.clamp(s_min, s_max)
    sig2 = torch.exp(2.0 * s)
    r = yc - pred
    nll = s + 0.5 * r * r / sig2
    if beta > 0:
        nll = nll * sig2.detach().pow(float(beta))
    if famnorm:
        if fam is None: raise ValueError("resnorm_terms: famnorm needs the lane families")
        inv = (w / sig2).detach()
        H = torch.ones_like(sig2)
        for f in fam.unique().tolist():
            m = fam == f                                                                  # (G,)
            Hf = w[m].sum((0, 1)) / inv[m].sum((0, 1)).clamp_min(eps)                     # (2,) harmonic mean of sigma^2
            H[m] = Hf.view(1, 1, -1).expand(int(m.sum()), sig2.shape[1], -1)
        nll = nll * H
    nll_term = (w * nll).sum() / w.sum().clamp_min(eps)
    with torch.no_grad():
        z2 = (w * (r * r / sig2)).sum() / w.sum().clamp_min(eps)
        _sv = torch.sort(torch.exp(s).flatten()).values                                  # sort, not kthvalue: deterministic on CUDA
        stats = {"z2_w": float(z2), "logsig_w": float((w * s).sum() / w.sum().clamp_min(eps)),
                 "sig_p10": float(_sv[min(_sv.numel() - 1, int(0.1 * _sv.numel()))]),
                 "sig_p90": float(_sv[min(_sv.numel() - 1, int(0.9 * _sv.numel()))])}
    ed_term = None
    if ed:
        zp = (yc - (pred if ed_grad_p else pred.detach())) / torch.exp(s)
        parts, mass = [], []
        groups = [None] if not ed_fam else fam.unique().tolist()
        for g in groups:
            wg = w if g is None else w * (fam == g).to(w.dtype).view(-1, 1, 1)             # lane mask on the weights: zero-weight rows vanish
            for k in range(zp.shape[-1]):
                zk, wk = zp[..., k].reshape(-1), wg[..., k].reshape(-1)
                if float(wk.sum()) <= 0: continue
                parts.append(_h12_energy(zk, wk, eps)); mass.append(wk.sum())
        if parts:
            mass = torch.stack(mass); ed_term = (torch.stack(parts) * mass).sum() / mass.sum().clamp_min(eps)
        else:
            ed_term = pred.new_zeros(())
        stats["ed"] = float(ed_term.detach())
    return {"nll": nll_term, "ed": ed_term, "stats": stats}


def xlane_wp_loss(pred, y, row_w=None, eps: float = 1e-6):
    """1 - mean over targets of the weighted Pearson POOLED over every lane and row of the chunk (research Z5, --xlane-pw).

    One centre and one scale across lanes, weights |clip(y)| x row_w (warm-up and padded rows have row_w 0
    and drop out), per target, the two targets averaged -- the per-step analogue of the metric's pooled
    aggregation, where the level and scale of one sequence RELATIVE to another count. Distinct from
    RunningWP (one correlation per lane, prefix-accumulated: invariant to each lane's own level/scale) and
    from --global-pool (lane-summed RUNNING statistics, whose detached prefix shrinks the current chunk's
    share of the moments). Predictions are not clipped, the RunningWP convention; eps sits inside the sqrt
    for the reason documented there. pred/y: (G, T, 2); row_w: (G, T) or None.
    """
    yc = y.clamp(-METRIC_CLIP, METRIC_CLIP)
    w = yc.abs()
    if row_w is not None:
        w = w * row_w.unsqueeze(-1)
    k = yc.shape[-1]
    w2, p2, y2 = w.reshape(-1, k), pred.reshape(-1, k), yc.reshape(-1, k)
    S = w2.sum(0).clamp_min(eps)
    mp, my = (w2 * p2).sum(0) / S, (w2 * y2).sum(0) / S
    pc, ycn = p2 - mp, y2 - my
    cov = (w2 * pc * ycn).sum(0) / S
    vp = (w2 * pc * pc).sum(0) / S
    vy = (w2 * ycn * ycn).sum(0) / S
    r = cov / torch.sqrt(vp * vy + eps)
    return 1.0 - r.mean()


def weighted_softrank(p, y, row_w=None, knots: int = 64, tau: float = 0.25,
                      resolved_only: bool = True):
    """|clip(y)|-weighted soft-Spearman on the spread s = (y0 - y1)/2.

    A RANK objective, which nothing here has tried: --rankgauss is an input transform and
    every loss so far (wp/mse/zmse/anchored/dir/huber/l1) is a magnitude or sign objective.
    It earns a slot the way dir and GC did -- not by beating the metric, but by being a
    DIFFERENT objective that lands within the quality floor, so its errors decorrelate.

    Two details that matter more here than in the usual soft-rank setting:

    * TIES DOMINATE. The target sits on a per-family tick lattice, and the clipped spread
      takes about 7 distinct values in family 0 (10/19/32 in families 1/2/3). Breaking ties
      by argsort position would inject pure index noise into most of the weight mass, so the
      true side uses tie-corrected MIDRANKS via unique(); on that target Spearman is close
      to a 7-level ordinal correlation, which is exactly why it is a different objective.
    * COST IS O(n*K), NOT O(n^2). The textbook soft rank sums a sigmoid over every PAIR,
      which for a 64x1000 chunk is 4.1e9 terms and would OOM the backward pass the same way
      the nlfeat decay matrix did. Summing against K quantile KNOTS of the batch instead is
      the same estimator subsampled to K points: 4.1e6 terms, ~16 MB of activations.

    p, y: (G, T, 2) with y already clipped. Returns 1 - corr, so lower is better.
    """
    import torch
    ps = (0.5 * (p[..., 0] - p[..., 1])).reshape(-1)
    ys = (0.5 * (y[..., 0] - y[..., 1])).reshape(-1)
    w = (0.5 * (y[..., 0].abs() + y[..., 1].abs())).reshape(-1)
    if row_w is not None:
        w = w * row_w.reshape(-1)
    if resolved_only:                      # same restriction weighted_dir justifies
        w = w * (torch.maximum(y[..., 0], y[..., 1]) > 0.05).float().reshape(-1)
    if w.sum() <= 1e-6:
        return ps.sum() * 0.0

    # --- soft rank of the PREDICTION (differentiable) ---
    d = ps.detach()
    k = torch.quantile(d, torch.linspace(0, 1, knots, device=d.device, dtype=d.dtype))
    scale = d.std().clamp_min(1e-6) * tau
    rp = torch.sigmoid((ps[:, None] - k[None, :]) / scale).mean(1)

    # --- tie-corrected midranks of the TARGET (no gradient needed) ---
    with torch.no_grad():
        vals, inv, cnt = torch.unique(ys, return_inverse=True, return_counts=True)
        cum = cnt.cumsum(0)
        ry = ((cum - cnt).to(ps.dtype) + 0.5 * cnt.to(ps.dtype))[inv]
        ry = ry / len(ys)

    # --- weighted Pearson between the two rank vectors ---
    sw = w.sum()
    mp = (w * rp).sum() / sw
    my = (w * ry).sum() / sw
    a, b = rp - mp, ry - my
    cov = (w * a * b).sum()
    den = torch.sqrt((w * a * a).sum() * (w * b * b).sum()).clamp_min(1e-9)
    return 1.0 - cov / den


def weighted_dir(p, y, row_w=None, tau: float = 0.5, resolved_only: bool = True):
    """|clip(y)|-weighted BCE on the sign of the spread component s = (y0 - y1)/2.

    The metric is a weighted correlation, so what it rewards is getting the DIRECTION of
    the large-|y| rows right; weighted MSE spends its gradient on their magnitude instead.
    p, y: (G, T, 2) with y already clipped. Returns a scalar.
    """
    import torch
    ps = 0.5 * (p[..., 0] - p[..., 1])
    ys = 0.5 * (y[..., 0] - y[..., 1])
    w = 0.5 * (y[..., 0].abs() + y[..., 1].abs())
    if row_w is not None:
        w = w * row_w
    live = (ys.abs() > 1e-9).float()          # s == 0 carries no direction
    # Restrict to RESOLVED rows. Perfect direction on unresolved rows is worth +0.0009
    # globally against +0.0891 on resolved ones, because a correlation buys weighted target
    # VARIANCE and the unresolved bucket takes 2-3 fixed values -- ~3% of the variance for
    # ~21% of the weight. The losing leg is pinned at -3 ticks, so the winner is the only
    # positive target and this indicator needs neither q nor the family label; it agrees
    # with the lattice at 0.99998 in family 0.
    if resolved_only:
        live = live * (torch.maximum(y[..., 0], y[..., 1]) > 0.05).float()
    w = w * live
    tgt = (ys > 0).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        ps / tau, tgt, reduction="none")
    return (w * bce).sum() / w.sum().clamp_min(1e-6)
