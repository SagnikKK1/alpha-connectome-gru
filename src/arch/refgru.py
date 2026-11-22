"""GRU with an input projection, a skip connection past the recurrence, and an MLP head.

This is the architecture used by a public competitor solution scoring above us, specified
from its source (the file itself could not be reproduced verbatim):

    112 features -> z-score -> clamp +-CLAMP
      -> input projection : Linear -> GELU -> LayerNorm          (width `proj`)
      -> GRU stack
      -> concat(GRU output, projected input)                     <- skip past the RNN
      -> LayerNorm
      -> head : Linear -> GELU -> Dropout -> Linear

Three of these are things our GRUCore has never had, and each is load-bearing for a
different reason:

* the **input clamp** -- our standardisation reports "max amplification 33.3x", so some
  columns reach the recurrence at enormous scale; a +-10 clamp bounds that without
  changing the typical row.
* the **skip connection** -- the head sees the current row directly, so the recurrence
  only has to supply what history adds. Our head sees the hidden state alone.
* the **MLP head** -- a nonlinear readout over [hidden | projected input].

The output stays (s, -s) via ExportWrapper, so the submission contract is unchanged.
"""
from __future__ import annotations
import sys
from pathlib import Path
import math
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GRUCore, N_FEATURES


class _GatedProj(nn.Module):
    """Gated input projection: y = LayerNorm(a * SiLU(b)) where [a, b] = Linear(x).

    The additive stack (Linear -> GELU -> LayerNorm) cannot cheaply represent a product of two input
    features, and an order book is full of them -- imbalances, ratios, spread-normalised quantities.
    A gated block forms those directly for the same parameter budget, which is why it replaced plain
    MLPs in most modern sequence models.
    """

    def __init__(self, n_in, proj, blocks, dropout=0.0):
        super().__init__()
        self.fc = nn.ModuleList()
        self.ln = nn.ModuleList()
        d = n_in
        for _ in range(blocks):
            self.fc.append(nn.Linear(d, 2 * proj))
            self.ln.append(nn.LayerNorm(proj))
            d = proj

    def forward(self, x):
        for fc, ln in zip(self.fc, self.ln):
            a, b = fc(x).chunk(2, dim=-1)
            x = ln(a * torch.nn.functional.silu(b))
        return x


class RefGRUCore(GRUCore):
    """Input variants, all row-by-row computable and carried in the state tensor:

    cell="lstm"   nn.LSTM instead of nn.GRU (state slots: h then c per layer)
    diff=True     also feed x_t - x_{t-1} (standardised); previous row rides in one state slot
    ewma=(8,64,512)  also feed EWMAs of the standardised input at these half-lives; each
                  EWMA rides in its own state slot (needs hidden >= n_features)

    State layout, shape (n_slots, B, hidden): [rnn slots..., x_prev?, ewma_1?, ewma_2?, ...],
    extra slots hold n_features values zero-padded to `hidden`.
    """

    def __init__(self, hidden=192, layers=2, n_out=1, n_features=N_FEATURES,
                 proj=64, head_hidden=0, dropout=0.1, clamp=10.0,
                 cell="gru", diff=False, ewma=(), posfeat=False, split=False,
                 nlfeat=(), quad=False, bookconv=0, ofi=False, adanorm=0.0, proj_layers=1, proj_gated=False, fastw=0, fw_groups=4,
                 bookconv_taps=3, bookconv_unshare=False, tradeconv=0, instshare=0,     # add_s1.py (research S1)
                 inproj_init_mult=1.0, densein=0,                                        # add_s1b.py (research S1 controls)
                 xinst=0, bookconv_act="", hh_rank=0,                                    # add_struct.py (research ST: structures)
                 ple_bins=0, ple_ch=8,                                                   # add_h4_ple.py (research H4: piecewise-linear embedding)
                 bookconv_split=False, bookconv_side=False,                              # add_h7_bcsplit.py (research H7: restricted kernels)
                 fw_hl_min=32.0, fw_hl_max=4096.0, fw_to_gru=False, bidir=False, dualmode=0, hint=0, aux_next=0, mask_aux=0, hetero=0, resnorm=0, resnorm_sd0=1.2, in_drop=0.0, out_drop=0.0, grid=0, learn_h0=False,
                 adanorm_cols="", hlg_bins=0, mh_heads=0):
        super().__init__(hidden=hidden, layers=layers, n_out=n_out,
                         n_features=n_features)
        self.clamp = float(clamp)
        self.cell, self.diff = cell, bool(diff)
        self.bidir = bool(bidir)              # non-causal teacher only (offline/patches/add_bidir.py)
        self.dualmode = int(dualmode)         # train-only backward branch + aux head (add_dualmode.py)
        self._aux_out = None
        self._gru_out = None                  # last GRU output, for hint distillation (add_hint.py)
        self.hint = int(hint)
        if self.hint:
            self.hint_adapter = nn.Linear(hidden, self.hint)   # train-only, never exported
        self.aux_next = int(aux_next)
        self.mask_aux = bool(mask_aux)        # research Z6: train-only P(is_scored) head (add_signals.py); built LAST, see below
        self.in_drop, self.out_drop = float(in_drop), float(out_drop)   # variational dropout (add_vardrop_sched.py)
        self._x_std = None
        self.hetero = 1 if hetero else 0; self._hetero_out = None   # train-only log-variance head (add_losses2.py)
        if self.aux_next:
            self.aux_next_head = nn.Linear(hidden, n_features)  # train-only, never exported (add_aux_next.py)
        if self.dualmode and (split or cell == "lstm" or fastw or self.bidir):
            raise ValueError("dualmode: plain single-tower causal GRU only")
        if self.bidir and (split or cell == "lstm" or fastw):
            raise ValueError("bidir: plain single-tower GRU only (no split towers, LSTM or fast weights)")
        self.ewma = tuple(float(v) for v in ewma) if ewma else ()
        self.posfeat = bool(posfeat)          # row index / 20000 as an input; counter rides in a state slot
        # --- nonlinear causal state -------------------------------------------------------
        # --ewma feeds EWMAs of the RAW columns, which are linear in the input and which a
        # GRU can already approximate. A residual screen against G_pw50_s801 (featscan.py)
        # showed the signal it has NOT captured is nonlinear: realized volatility (an EWMA of
        # SQUARED mid moves, top of the list at 0.024), |trade volume|, and the cross-sectional
        # dispersion of the per-level move. Those have no cheap path through a GRU gate, so we
        # hand them over directly. 4 instantaneous + 4 stats x len(nlfeat) half-lives.
        # --- multiplicative interaction ---------------------------------------------------
        # The input path is Linear -> GELU -> LayerNorm -> GRU, so the only products available to
        # the model are whatever one narrow hidden layer can approximate. The order-flow quantity
        # the LOB literature points at is inherently multiplicative (a volume TIMES the direction
        # the book moved), and every feature test so far has been additive. `quad` hands the GRU
        # z * (z - z_prev): level x change in the learned projection basis, i.e. flow-shaped terms,
        # for one extra state slot and `proj` multiplies per row.
        self.quad = bool(quad)
        # --- fast weights: identify THIS sequence's dynamics online -------------------------
        # Section 37 found a per-sequence affine correction worth +0.064 that no observable proxy
        # reaches, and section 30 found the model scores 0.703 on sequences it trained on against
        # 0.665 on unseen ones. Both say the same thing: each sequence has its own latent dynamics,
        # and a fixed-weight GRU can only average over them. `fastw` gives the model a rank-updated
        # associative memory A that accumulates the map (previous projected row -> current one) FOR
        # THIS SEQUENCE, decayed at a learned rate, and reads it back with the current row. That is
        # a running estimate of the sequence's own transition operator -- system identification at
        # test time, with no targets needed. Cost is 2*D^2 + 3*proj*D multiply-adds per row (about
        # 8k at D=32 against the GRU's 245k) and D^2 + D extra state.
        # --- book-shape convolution ---------------------------------------------------------
        # The 112 columns are two instruments of 11 bid prices, 11 ask prices, 11 bid volumes and 11
        # ask volumes, plus deltas and eight extras. The projection reads all of that as a flat vector,
        # so nothing tells the model that level 3 and level 4 are neighbours on the same ladder. A
        # 1-D convolution ALONG the depth axis, shared across levels and across both instruments,
        # supplies exactly that: it can only form local book-shape features (slope, curvature,
        # imbalance between adjacent levels) and it does so with 4*C*3 weights instead of a dense map.
        # This is a restriction, not extra expressiveness -- which is the point, because section 30
        # says the model overfits 10,607 sequences and bigger models score worse.
        self.bookconv = int(bookconv)
        # --- conditional order-flow imbalance ------------------------------------------------
        # Cont's OFI, not a plain column difference. At each level the volume change is CONDITIONED
        # on whether the price at that level moved:
        #     bid side:  +q_b,t if P_b rose or held, -q_b,t-1 if P_b fell or held
        #     ask side:  mirrored, and subtracted
        # The conditioning is the whole point: it separates queue replenishment at a standing price
        # from the queue vanishing because the price level itself moved. A raw delta conflates them,
        # which is why the literature reports order-flow inputs beating raw books rather than merely
        # supplementing them (Kolm/Turiel/Westray 2023; Cont/Cucuringu/Zhang 2023).
        #
        # 11 levels x 2 instruments = 22 channels, plus a depth-normalised integrated sum per
        # instrument. Needs only rows t and t-1, so it costs one state slot and no matmuls.
        self.ofi = bool(ofi)
        # --- adaptive causal normalisation ----------------------------------------------------
        # Distinct from --ewma, which ADDS a smoothed copy of each column as an extra input. This
        # DIVIDES each column by its own running scale: x_hat = (x - m_t) / sqrt(v_t + eps), with m
        # and v exponentially weighted and updated per row. The fixed (mu, sd) computed once over the
        # training set cannot track a sequence whose volatility regime differs from the average, and
        # the standardisation report shows some columns amplified 33x, so the clamp at +-10 is doing
        # real damage to exactly those. Two state slots; elementwise, so no matmuls and no extra nodes
        # beyond the update itself.
        self.adanorm = float(adanorm)      # half-life in rows; 0 disables
        # add_z48.py (research Z48): --adanorm-cols. Apply the running normalisation to a column SUBSET only (the regime-
        # sensitive volume / clock columns); the others stay the plain standardised input. Persistent buffer, so a tool that
        # does not know the mask fails loudly on a strict load instead of silently normalising every column.
        if not adanorm_cols:
            self.adanorm_mask = None
        else:
            if self.adanorm <= 0:
                raise ValueError("adanorm_cols needs adanorm > 0 (the half-life of the running statistics)")
            _sets = {"vol": list(range(22, 44)) + list(range(48, 52)) + list(range(74, 96)) + list(range(100, 104)),
                     "clk": list(range(104, 112))}
            _cols = set()
            for _tok in str(adanorm_cols).replace("+", ",").split(","):
                _tok = _tok.strip()
                if not _tok:
                    continue
                if _tok in _sets:
                    _cols.update(_sets[_tok])
                elif "-" in _tok:
                    _lo, _hi = (int(v) for v in _tok.split("-")); _cols.update(range(_lo, _hi + 1))
                else:
                    _cols.add(int(_tok))
            if not _cols or min(_cols) < 0 or max(_cols) >= n_features:
                raise ValueError(f"adanorm_cols {adanorm_cols!r}: columns must lie in 0..{n_features - 1}")
            _m = torch.zeros(n_features)
            _m[sorted(_cols)] = 1.0
            self.register_buffer("adanorm_mask", _m)
        self.fastw = int(fastw)
        # Where the memory readout goes. Into the head it can only correct the readout; into
        # the GRU input the recurrence can CONDITION on what this sequence has been doing,
        # which is the difference between a side-channel and a change of dynamics.
        self.fw_to_gru = bool(fw_to_gru)
        self.nlfeat = tuple(float(v) for v in nlfeat) if nlfeat else ()
        self.n_nl = 12 * len(self.nlfeat)
        self.n_in = (n_features * (1 + int(self.diff) + len(self.ewma)) + int(self.posfeat)
                     + (10 + self.n_nl if self.nlfeat else 0))
        if self.ofi:
            self.n_in += 24        # 22 per-level channels + 2 integrated

        # add_s1.py (research S1): bookconv hyper-parameters plus two more shared LINEAR re-reads of the raw columns. All of
        # them are linear in x, so refgru_build.weights() folds them into in_proj.0 (zero inference cost). Defaults reproduce
        # the pre-S1 model bitwise (same book_w shape, same init scale, same tap loop, no extra draws).
        self.bc_taps = int(bookconv_taps) if self.bookconv else 3
        self.bc_unshare = bool(bookconv_unshare) and bool(self.bookconv)
        # add_h7_bcsplit.py (research H7): two SUB-MODELS of the bookconv kernel. --bookconv-split: block-diagonal, one (K, 2) kernel
        # for the price pair -> channels 0..C/2-1 and one for the volume pair -> C/2..C-1 (no price-volume mixing at the index).
        # --bookconv-side: ONE (K, 2) kernel over (p_side, v_side), applied to the bid side -> 0..C/2-1 and to the ask side -> C/2..C-1
        # (shared over 2 instruments x 2 sides x 11 indices). The dense (K, 4, C) kernel is assembled from the parts on every forward
        # (_h7_kernel), so the tap loop, the 2*C*11 features, in_proj.0's shape / init / Muon scale and the export fold are exactly
        # bookconv's; the control's own kernel draw is consumed below so every other init is bit-identical to the same-seed control.
        self.bc_split = bool(bookconv_split) and bool(self.bookconv)
        self.bc_side = bool(bookconv_side) and bool(self.bookconv)
        if (bookconv_split or bookconv_side) and not self.bookconv: raise ValueError("bookconv_split / bookconv_side need bookconv > 0")
        if self.bc_split and self.bc_side: raise ValueError("bookconv_split and bookconv_side are exclusive")
        if (self.bc_split or self.bc_side) and self.bc_unshare: raise ValueError("bookconv_split / bookconv_side with bookconv_unshare: not implemented")
        if (self.bc_split or self.bc_side) and self.bookconv % 2: raise ValueError("bookconv_split / bookconv_side need an even bookconv C")
        self.tradeconv = int(tradeconv)
        self.instshare = int(instshare)
        if self.bookconv:
            if self.bc_taps < 1 or self.bc_taps % 2 == 0:
                raise ValueError("bookconv_taps must be odd and >= 1 (1, 3, 5, ...)")
            self.n_in += 2 * self.bookconv * 11
            # written as three explicit taps rather than nn.Conv1d: Conv1d SEGFAULTS on this
            # box's CPU torch build (2.11.0+cu128), which would break every CPU verification
            # path, and three small matmuls export to ONNX more simply anyway
            if self.bc_split or self.bc_side:    # add_h7_bcsplit.py (research H7): restricted kernel parts instead of the dense book_w
                torch.randn(self.bc_taps, 4, self.bookconv)       # consume the control's kernel draw: in_proj / GRU / head init stay bit-identical
                _g = torch.Generator().manual_seed((int(torch.initial_seed()) % (2 ** 63 - 1)) ^ 0x4837)
                _h = self.bookconv // 2
                _s = (1.0 / (2 * self.bc_taps)) ** 0.5           # fan-in 2 blocks x K taps per part (the dense kernel: 4 x K)
                if self.bc_split:
                    self.book_wp = nn.Parameter(torch.randn(self.bc_taps, 2, _h, generator=_g) * _s)   # (p_bid, p_ask) -> channels 0..C/2-1
                    self.book_wv = nn.Parameter(torch.randn(self.bc_taps, 2, _h, generator=_g) * _s)   # (v_bid, v_ask) -> channels C/2..C-1
                else:
                    self.book_ws = nn.Parameter(torch.randn(self.bc_taps, 2, _h, generator=_g) * _s)   # (p_side, v_side): bid -> 0..C/2-1, ask -> C/2..C-1
                self.book_b = nn.Parameter(torch.zeros(self.bookconv))
            else:
                self.book_w = nn.Parameter(torch.randn(self.bc_taps, 4, self.bookconv) * (1.0 / (4 * self.bc_taps) ** 0.5))
                self.book_b = nn.Parameter(torch.zeros(self.bookconv))
            if self.bc_unshare:                 # i1 gets its own kernel; i0 keeps book_w / book_b
                self.book_w1 = nn.Parameter(torch.randn(self.bc_taps, 4, self.bookconv) * (1.0 / (4 * self.bc_taps) ** 0.5))
                self.book_b1 = nn.Parameter(torch.zeros(self.bookconv))
        if self.tradeconv:
            self.n_in += 2 * self.tradeconv * 4
            self.trade_w = nn.Parameter(torch.randn(3, 2, self.tradeconv) * (1.0 / 6 ** 0.5))
            self.trade_b = nn.Parameter(torch.zeros(self.tradeconv))
        if self.instshare:
            self.n_in += 2 * self.instshare
            # 3-D on purpose: train._muon_split routes 3-D weights to AdamW (decay), exactly as it does book_w
            self.inst_w = nn.Parameter(torch.randn(1, 52, self.instshare) * (1.0 / 52 ** 0.5))
            self.inst_b = nn.Parameter(torch.zeros(self.instshare))
        # add_s1b.py (research S1 controls): a dense unstructured read of the 88 book columns, and the init multiplier
        self.densein = int(densein)
        self.inproj_init_mult = float(inproj_init_mult)
        if self.densein:
            self.n_in += self.densein
            self.dense_w = nn.Parameter(torch.randn(1, 88, self.densein) * (1.0 / 88 ** 0.5))   # 3-D -> AdamW, like book_w
            self.dense_b = nn.Parameter(torch.zeros(self.densein))
        # add_struct.py (research ST): (a) --bookconv-act: a position-wise GELU on the shared bookconv read, per (instrument,
        # ladder index, channel), BEFORE in_proj.0 -- equivariant NONLINEAR local book features (conv -> GELU -> free mix), the
        # function class the folded linear read cannot express. NOT foldable: refgru_build.weights() exports kernel + its
        # in_proj.0 slice as w["bcnl"] for a native-kernel extension. (b) --xinst C: a CROSS-INSTRUMENT ladder read: at index l
        # the 8 values (p_bid, p_ask, v_bid, v_ask of i0 AND i1) over 3 taps (24 inputs) -> C channels, one kernel shared over
        # the 11 indices (features.md E4: i0 and i1 columns are the SAME mixing of their books, so column l of i0 and column l
        # of i1 are the same functional of two books). Linear in x -> folded into in_proj.0 at export. New parameters are drawn
        # from a FORKED generator (seeded from torch.initial_seed()) so no other init moves.
        self.bc_act = str(bookconv_act or "")
        if self.bc_act and self.bc_act != "gelu": raise ValueError("bookconv_act: '' (linear) or 'gelu'")
        if self.bc_act and not self.bookconv: raise ValueError("bookconv_act needs bookconv > 0")
        if self.bc_act and (self.bc_split or self.bc_side): raise ValueError("bookconv_act with bookconv_split / bookconv_side: export not implemented (add_h7_bcsplit.py)")
        self.xinst = int(xinst)
        if self.xinst:
            self.n_in += self.xinst * 11
            _g = torch.Generator().manual_seed((int(torch.initial_seed()) % (2 ** 63 - 1)) ^ 0x5157)
            self.xinst_w = nn.Parameter(torch.randn(3, 8, self.xinst, generator=_g) * (1.0 / 24 ** 0.5))   # 3-D -> AdamW, like book_w
            self.xinst_b = nn.Parameter(torch.zeros(self.xinst))
        # add_h4_ple.py (research H4): --ple-bins B --ple-ch C. Each of the 44 price-code columns (i0 p 0-21, i1 p 52-73: the
        # quasi-continuous book codes, H3 section 3c) is expanded with a PIECEWISE-LINEAR ENCODING (Gorishniy et al. 2022, eq. 3):
        # B bins whose edges are the column's train quantiles (buffer `ple_edges` (44, B+1), set once by train.py like mu / sd and
        # stored in the checkpoint); e_t = 1 above bin t, 0 below it, the linear position inside it (first / last bin extrapolate).
        # One B -> C linear map `ple_w` SHARED across the 44 positions (bookconv's sharing and rank restriction, with a per-column
        # NONLINEARITY in front of it -- the one shape S1 did not probe) gives 44 * C extra inputs to in_proj.0, appended after
        # every other block. A function of row t only (causal). NOT linear in x: refgru_build.weights() refuses to export until a
        # kernel path exists. ple_w is drawn from a FORKED generator so no other init
        # moves; 3-D so train._muon_split routes it to AdamW with decay, exactly like book_w. Default off: no parameter, no buffer.
        self.ple_bins = int(ple_bins or 0)
        self.ple_ch = int(ple_ch) if self.ple_bins else 0
        if self.ple_bins:
            if self.ple_bins < 2: raise ValueError("ple_bins must be >= 2 (0 = off)")
            if self.ple_ch < 1: raise ValueError("ple_ch must be >= 1")
            if n_features != N_FEATURES: raise ValueError("ple: the 112-column layout only (price codes at 0-21 and 52-73)")
            if split: raise ValueError("ple: not implemented for split towers")
            self.n_in += 44 * self.ple_ch
            self.register_buffer("ple_edges", torch.zeros(44, self.ple_bins + 1))          # persistent: set_ple_edges / checkpoint
            _fl = torch.zeros(self.ple_bins); _fl[0] = -math.inf                                # bin 0: no floor (extrapolates below b_0)
            _ce = torch.ones(self.ple_bins); _ce[-1] = math.inf                                 # bin B-1: no ceiling (extrapolates above b_B)
            self.register_buffer("_ple_floor", _fl, persistent=False)
            self.register_buffer("_ple_ceil", _ce, persistent=False)
            _g = torch.Generator().manual_seed((int(torch.initial_seed()) % (2 ** 63 - 1)) ^ 0x504C45)
            self.ple_w = nn.Parameter(torch.randn(1, self.ple_bins, self.ple_ch, generator=_g) * (1.0 / self.ple_bins ** 0.5))   # 3-D -> AdamW, like book_w
            self._ple_ok = False
        self.hh_rank = int(hh_rank)
        self.n_bank = 18 * len(self.nlfeat) + 1 if self.nlfeat else 0
        if (self.diff or self.ewma or self.nlfeat) and hidden < n_features:
            raise ValueError("diff/ewma/nlfeat need hidden >= n_features so a row fits in one state slot")
        self.split = bool(split)
        if self.split:
            # two towers: i0 (cols 0-52) + extras (104-112), i1 (52-104) + extras; each tower a
            # proj/2 projection and a hidden/2 GRU; outputs concatenated to `hidden`, z to `proj`
            if diff or ewma or posfeat or nlfeat or cell != "gru": raise ValueError("split towers: plain GRU inputs only")
            self.i0 = list(range(0, 52)) + list(range(104, 112)); self.i1 = list(range(52, 104)) + list(range(104, 112))
            self.in_proj = nn.ModuleList([nn.Sequential(nn.Linear(60, proj // 2), nn.GELU(), nn.LayerNorm(proj // 2)) for _ in range(2)])
            self.gru = nn.ModuleList([nn.GRU(proj // 2, hidden // 2, layers, batch_first=False, dropout=dropout if layers > 1 else 0.0) for _ in range(2)])
        else:
            # Input capacity is the one axis that pays here: widening this projection 64 -> 192 is
            # worth +0.0010, while the recurrence is at an optimum (256 beats 96/128/192 below it and
            # 384-768 above). Width has now flattened -- 192 and 256 score the same -- so the next
            # thing to try on the same axis is DEPTH: a second Linear/GELU/LayerNorm block before the
            # recurrence, which adds instantaneous nonlinearity without touching the recurrent path.
            # Three independent routes now reach the same input-path ceiling: width saturates at 192,
            # depth saturates at two blocks, and a teacher with 7x the input parameters lands on the
            # same 0.680. That says the ceiling may belong to the FUNCTION CLASS rather than to
            # capacity -- so try a gated block, which is the standard answer when a plain
            # Linear/GELU/LayerNorm stack saturates. Each block emits 2*proj and multiplies one half
            # by SiLU of the other, giving multiplicative interactions the additive stack cannot form
            # at the same parameter count.
            if proj_gated:
                self.in_proj = _GatedProj(self.n_in, proj, max(1, int(proj_layers)), dropout=0.0)
            else:
                _blocks = [nn.Linear(self.n_in, proj), nn.GELU(), nn.LayerNorm(proj)]
                for _ in range(max(0, int(proj_layers) - 1)):
                    _blocks += [nn.Linear(proj, proj), nn.GELU(), nn.LayerNorm(proj)]
                self.in_proj = nn.Sequential(*_blocks)
                if self.inproj_init_mult != 1.0:               # add_s1b.py: shrink the first Linear's init in place (no RNG draw)
                    with torch.no_grad():
                        self.in_proj[0].weight.mul_(self.inproj_init_mult)
            RNN = nn.LSTM if cell == "lstm" else nn.GRU
            _gin = proj * (2 if self.quad else 1) + (self.fastw if self.fw_to_gru else 0)
            self.gru = RNN(_gin, hidden, layers, batch_first=False,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=self.bidir)
        if self.hh_rank:
            # add_struct.py (research ST): a RANK-r RECURRENCE, W_hh = hh_A (3H, r) @ hh_B (r, H): the three gates read the state
            # through ONE shared r-dim projection (LSTMP-style; the recurrent analogue of bookconv's shared restricted read).
            # nn.GRU keeps W_ih and the biases; its weight_hh_l0 is removed and the product is handed to the same cuDNN / CPU
            # kernel through torch._VF.gru each forward. Export: refgru_build.weights() materialises W_hh = A @ B (the shipped
            # graph is unchanged); a factored native path would stream 3H*r + r*H int8 bytes instead of 3H*H.
            if self.split or cell != "gru" or layers != 1 or self.bidir or self.fastw:
                raise ValueError("hh_rank: plain single-layer single-tower causal GRU only")
            if not (16 <= self.hh_rank < hidden): raise ValueError("hh_rank must satisfy 16 <= r < hidden")
            _s = float(3 * hidden * self.hh_rank) ** -0.25         # Var(A @ B) = r s^4 = 1/(3H) = Var(U(-1/sqrt(H), 1/sqrt(H)))
            _g = torch.Generator().manual_seed((int(torch.initial_seed()) % (2 ** 63 - 1)) ^ 0x4848)
            self.hh_A = nn.Parameter(torch.randn(3 * hidden, self.hh_rank, generator=_g) * _s)
            self.hh_B = nn.Parameter(torch.randn(self.hh_rank, hidden, generator=_g) * _s)
            delattr(self.gru, "weight_hh_l0"); self.gru._init_flat_weights()   # state_dict: gru.weight_ih_l0, gru.bias_*, hh_A, hh_B
            self._z14_gate_blocks = {id(self.hh_A): 3, id(self.hh_B): 1}      # train._muon_split: A orthogonalised per gate block
            import warnings as _w
            _w.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk")
        if self.fastw:
            D = self.fastw
            self.fw_q = nn.Linear(proj, D, bias=False)
            self.fw_k = nn.Linear(proj, D, bias=False)
            self.fw_v = nn.Linear(proj, D, bias=False)
            G = max(1, int(fw_groups))
            if D % G: raise ValueError("fastw width must divide by fastw-groups")
            self.fw_groups = G
            # One decay per group, geometrically spread over the half-lives that matter here. A
            # single fast decay is just a local average and identifies nothing; a single slow one
            # cannot track a regime. The sequence is 20,000 rows, so the slowest group has to see a
            # meaningful fraction of it.
            hl = torch.logspace(math.log10(fw_hl_min), math.log10(fw_hl_max), G)
            lam0 = 2.0 ** (-1.0 / hl)
            self.fw_decay = nn.Parameter(torch.log(lam0 / (1.0 - lam0)))       # sigmoid -> lam0
            self.fw_slots = (D * D + hidden - 1) // hidden + 2  # A, its normaliser, and z_prev
        _hy = hidden * (2 if self.bidir else 1)          # bidirectional GRU output width
        self.post_ln = nn.LayerNorm(_hy + proj + (0 if self.fw_to_gru else self.fastw))
        # the head's hidden width is `hidden`, not a separate constant -- that is what
        # reconstructs the reference model's 428,546 parameters exactly
        hh = head_hidden or hidden
        # --grid-head K (research P, round 6): a family-agnostic value grid over [-2, 2]; the head emits
        # K logits per target, the output is the softmax expectation. Composed here so every consumer
        # (loss, eval, dumps) sees (T, B, n_out) as before. NOT exportable by refgru_build yet.
        self.grid = int(grid); self._grid_logits = None
        if self.grid:
            if n_out != 2: raise ValueError("grid head: two-head only (n_out == 2)")
            self.register_buffer("grid_v", torch.linspace(-2.0, 2.0, self.grid))
        self.head = nn.Sequential(nn.Linear(_hy + proj + (0 if self.fw_to_gru else self.fastw), hh), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hh, n_out * self.grid if self.grid else n_out))
        # add_z59.py (research Z59): --hlg = train-only HL-Gauss head on the head's hidden layer; --mh = K sub-heads whose
        # mean is the output (folded into this single head at save time by train.py). Both are initialised from a private
        # generator with the global RNG saved/restored, so every other weight and the dropout stream match the control.
        self.hlg_bins = int(hlg_bins); self._hlg_logits = None
        self.mh_heads = int(mh_heads); self._mh_out = None
        if self.hlg_bins or self.mh_heads:
            if self.grid or split or self.bidir or self.dualmode:
                raise ValueError("--hlg / --mh: plain refgru head only (no grid head, split towers, bidir or dualmode)")
            if self.hlg_bins and self.mh_heads:
                raise ValueError("--hlg with --mh is not implemented")
            if n_out < 2:
                raise ValueError("--hlg / --mh: two-head models only (n_out >= 2)")
            _rs59 = torch.get_rng_state()
            _g59 = torch.Generator().manual_seed(int(torch.initial_seed()) % (2 ** 62) + 59)
            _hin59 = _hy + proj + (0 if self.fw_to_gru else self.fastw)
            if self.hlg_bins:
                self.hlg_head = nn.Linear(hh, 2 * self.hlg_bins)
                _lins59 = [self.hlg_head]
            else:
                if hh % self.mh_heads:
                    raise ValueError(f"--mh {self.mh_heads}: the head width {hh} is not divisible by K")
                _hk59 = hh // self.mh_heads
                _mh59 = nn.ModuleList([nn.Sequential(nn.Linear(_hin59, _hk59), nn.GELU(), nn.Dropout(dropout),
                                                     nn.Linear(_hk59, n_out)) for _ in range(self.mh_heads)])
                _lins59 = [m for _hd in _mh59 for m in (_hd[0], _hd[3])]
                # mh_head takes head's slot in the module order: state-dict, export and optimizer order match the control
                self._modules = type(self._modules)((("mh_head", _mh59) if _n == "head" else (_n, _m))
                                                    for _n, _m in self._modules.items())
            with torch.no_grad():                         # nn.Linear's default distribution U(+-1/sqrt(fan_in)), own stream
                for _l59 in _lins59:
                    _b59 = 1.0 / _l59.in_features ** 0.5
                    _l59.weight.copy_((torch.rand(_l59.weight.shape, generator=_g59) * 2.0 - 1.0) * _b59)
                    _l59.bias.copy_((torch.rand(_l59.bias.shape, generator=_g59) * 2.0 - 1.0) * _b59)
            torch.set_rng_state(_rs59)
        if self.dualmode:
            self.aux_bwd = nn.GRU(_gin, hidden, 1, batch_first=False)
            self.aux_ln = nn.LayerNorm(2 * hidden + proj)
            self.aux_head = nn.Sequential(nn.Linear(2 * hidden + proj, hh), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(hh, n_out))
        self.proj = None          # not the output bottleneck GRUCore knows about
        if self.ewma:
            # alpha = 1 - 2^(-1/half_life): the EWMA of a unit step reaches 1/2 after `half_life` rows
            self.register_buffer("ewma_alpha", torch.tensor([1.0 - 2.0 ** (-1.0 / hl) for hl in self.ewma]))
        if self.nlfeat:
            self.register_buffer("nl_alpha", torch.tensor([1.0 - 2.0 ** (-1.0 / hl) for hl in self.nlfeat]))
            self.register_buffer("nl_rt", torch.tensor([float(hl) ** 0.5 for hl in self.nlfeat]))
        if self.hetero:
            # --hetero (research Z5): per-target log-variance read from the same post-LN head input. Created LAST and
            # without advancing the global RNG, so a paired arm starts from the control's exact init (and, on CPU, the
            # same dropout stream). Keys hetero_head.*: refgru_build/fused_build read named keys and ignore them;
            # score_ckpt.load detects hetero_head.0.weight and rebuilds with hetero=1.
            _rs = torch.get_rng_state()
            self.hetero_head = nn.Sequential(nn.Linear(_hy + proj + (0 if self.fw_to_gru else self.fastw), 64), nn.GELU(),
                                             nn.Linear(64, 2))
            torch.set_rng_state(_rs)
        self.resnorm = 1 if resnorm else 0; self._resnorm_out = None   # train-only per-target log-sigma head (research H12)
        if self.resnorm:
            # add_h12_resnorm.py: two extra 'head outputs' (log sigma per target) as a SEPARATE Linear on the head's hidden layer,
            # so head.3 keeps its exact shape and the exporters (which read head.3 rows 0..1 by name) never see it. Private
            # generator with the global RNG saved/restored: every other weight, the dropout stream and the data order match the
            # control. Weights at a tenth of nn.Linear's default range and bias log(sd0): sigma starts near-constant at sd0.
            if self.grid or self.hlg_bins or self.mh_heads or split or self.bidir or self.dualmode:
                raise ValueError('resnorm: plain refgru head only (no grid head, --hlg, --mh, split towers, bidir or dualmode)')
            if n_out < 2:
                raise ValueError('resnorm: two-head models only (n_out >= 2)')
            _rs12 = torch.get_rng_state()
            _g12 = torch.Generator().manual_seed(int(torch.initial_seed()) % (2 ** 62) + 12)
            self.resnorm_head = nn.Linear(hh, 2)
            with torch.no_grad():
                _b12 = 0.1 / float(hh) ** 0.5
                self.resnorm_head.weight.copy_((torch.rand(self.resnorm_head.weight.shape, generator=_g12) * 2.0 - 1.0) * _b12)
                self.resnorm_head.bias.fill_(float(torch.log(torch.tensor(float(resnorm_sd0)))))
            torch.set_rng_state(_rs12)
        self.learn_h0 = bool(learn_h0)
        if self.learn_h0:                 # research X: learned initial state (add_learn_h0.py)
            if self.split: raise ValueError("--learn-h0 is not implemented for split towers")
            _n0 = self.rnn_slots + int(self.diff) + len(self.ewma) + int(self.posfeat) + (2 if self.nlfeat else 0) + int(self.quad) + int(self.ofi) + (2 if self.adanorm > 0 else 0) + (self.fw_slots if self.fastw else 0)
            self.h0 = nn.Parameter(torch.zeros(_n0, 1, hidden))

        if self.mask_aux:
            # research Z6 (add_signals.py). Built LAST and under a forked RNG, so every other parameter's init and the
            # dropout stream are identical to the same-seed control: the paired screen then sees the loss change alone.
            # (Built before in_proj/gru/head it shifted their init: 0.91 relative weight movement at 20 steps vs 0.01-0.03.)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0x5A6)
                self.mask_head = nn.Linear(hidden, 1)   # train-only, never exported (refgru_build reads explicit keys)

    def _h7_kernel(self):
        """add_h7_bcsplit.py (research H7): the dense (K, 4, C) bookconv kernel assembled from the restricted parts, so the tap loop
        and the export fold are exactly bookconv's. Block rows: 0 p_bid, 1 p_ask, 2 v_bid, 3 v_ask; channels 0..C/2-1 then C/2..C-1."""
        if self.bc_split:
            z = torch.zeros_like(self.book_wp)
            return torch.cat([torch.cat([self.book_wp, z], 1), torch.cat([z, self.book_wv], 1)], 2)
        w = self.book_ws; z = torch.zeros_like(w[:, 0])
        kb = torch.stack([w[:, 0], z, w[:, 1], z], 1)             # bid-side channels read (p_bid, v_bid)
        ka = torch.stack([z, w[:, 0], z, w[:, 1]], 1)             # ask-side channels read (p_ask, v_ask) with the SAME kernel
        return torch.cat([kb, ka], 2)

    def set_ple_edges(self, edges):
        """add_h4_ple.py (research H4): the fixed per-column bin edges (44, ple_bins + 1) of the standardised, clamped price codes
        (train quantiles; train.py computes them once, like mu / sd), strictly increasing per column. Saved in the checkpoint."""
        if not self.ple_bins: raise ValueError("set_ple_edges: this model has no piecewise-linear embedding (ple_bins 0)")
        e = torch.as_tensor(edges, dtype=torch.float32)
        if tuple(e.shape) != (44, self.ple_bins + 1):
            raise ValueError(f"ple_edges must have shape (44, {self.ple_bins + 1}), got {tuple(e.shape)}")
        if not bool(torch.isfinite(e).all()) or not bool((e[:, 1:] > e[:, :-1]).all()):
            raise ValueError("ple_edges must be finite and strictly increasing along each column")
        with torch.no_grad():
            self.ple_edges.copy_(e.to(self.ple_edges.device))
        self._ple_ok = True

    @property
    def rnn_slots(self):
        return self.layers * (2 if self.cell == "lstm" else 1) * (2 if getattr(self, "bidir", False) else 1)

    def init_state(self, batch: int, device=None):
        if self.split:
            return torch.zeros(2 * self.layers, batch, self.hidden, device=device or next(self.parameters()).device)
        n = self.rnn_slots + int(self.diff) + len(self.ewma) + int(self.posfeat) + (2 if self.nlfeat else 0) + int(self.quad) + int(self.ofi) + (2 if self.adanorm > 0 else 0) + (self.fw_slots if self.fastw else 0)
        if getattr(self, "learn_h0", False):
            return self.h0.to(device or self.h0.device).expand(n, batch, self.hidden).contiguous()
        return torch.zeros(n, batch, self.hidden, device=device or next(self.parameters()).device)

    def _inputs(self, x, state):
        """x: (T,B,F) standardised+clamped. Returns (model input (T,B,n_in), updated extra slots)."""
        T, B, F = x.shape
        parts, new = [x], []
        k = self.rnn_slots
        if self.diff:
            prev = state[k, :, :F]                                   # (B,F) previous row
            shifted = torch.cat([prev.unsqueeze(0), x[:-1]], 0)
            parts.append(x - shifted)
            new.append(torch.nn.functional.pad(x[-1], (0, self.hidden - F)))
            k += 1
        for j, hl in enumerate(self.ewma):
            a = self.ewma_alpha[j]
            e = state[k + j, :, :F]                                  # (B,F) running EWMA
            outs = []
            for t in range(T):                                       # sequential scan (T <= chunk)
                e = e + a * (x[t] - e); outs.append(e)
            parts.append(torch.stack(outs, 0))
            new.append(torch.nn.functional.pad(e, (0, self.hidden - F)))
        if self.posfeat:
            k2 = self.rnn_slots + int(self.diff) + len(self.ewma)
            t0 = state[k2, :, 0]                                     # (B,) rows already seen
            pos = t0.unsqueeze(0) + torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1)   # (T,B)
            parts.append((pos / 20000.0).unsqueeze(-1))
            nxt = torch.zeros(B, self.hidden, device=x.device, dtype=x.dtype); nxt[:, 0] = t0 + T
            new.append(nxt)
        if self.nlfeat:
            kn = self.rnn_slots + int(self.diff) + len(self.ewma) + int(self.posfeat)
            H, NS, NL = len(self.nlfeat), 7, 11
            prev = state[kn, :, :F]                                   # (B,F) previous raw row
            bank = state[kn + 1]
            es = bank[:, :NS * H].view(B, NS, H)                      # scalar EWMA bank
            el = bank[:, NS * H:(NS + NL) * H].view(B, H, NL)         # per-LEVEL bid EWMA bank
            # At a sequence start `prev` is zeros, so row 0 would report a spurious move of the
            # full size of mid[0]. At half-life 1024 that fake spike is still 93% intact at the
            # first scored row. `warm` is 0 only on the first row after a state reset.
            warm = bank[:, (NS + NL) * H:(NS + NL) * H + 1].transpose(0, 1)      # (1,B)
            xs = torch.cat([prev.unsqueeze(0), x], 0)                 # (T+1,B,F)
            bp = x[..., 0:11]
            mid = 0.5 * (xs[..., 0:11].mean(-1) + xs[..., 11:22].mean(-1))
            mid1 = 0.5 * (xs[..., 52:63].mean(-1) + xs[..., 63:74].mean(-1))
            imb_a = xs[..., 22:33].mean(-1) - xs[..., 33:44].mean(-1)  # (T+1,B)
            d, d1 = mid[1:] - mid[:-1], mid1[1:] - mid1[:-1]
            d_bid = xs[1:, :, 0:11].mean(-1) - xs[:-1, :, 0:11].mean(-1)
            d_ask = xs[1:, :, 11:22].mean(-1) - xs[:-1, :, 11:22].mean(-1)
            d_imb = imb_a[1:] - imb_a[:-1]
            disp = (xs[1:, :, 0:11] - xs[:-1, :, 0:11]).std(-1)
            gate = torch.ones_like(d); gate[:1] = warm
            d, d1, d_bid, d_ask, d_imb, disp = (v * gate for v in (d, d1, d_bid, d_ask, d_imb, disp))
            midc, imb = mid[1:], imb_a[1:]
            trd_p = x[..., 44:48].mean(-1)
            trd_rel = trd_p - midc                                    # trade price vs book mid
            flow = torch.sign(trd_rel) * x[..., 48:52].abs().mean(-1)  # signed trade flow
            base = torch.stack([d * d, d, d1, imb, midc, d * d * d, imb * imb], -1)  # (T,B,7)
            E, es = self._scan(base, es)
            EL, el = self._scan_lvl(bp, el)
            eps = 1e-6
            rv = E[:, :, 0].clamp_min(0.0).sqrt()                     # (T,B,H)
            a0 = x[..., 104].unsqueeze(-1)                            # (T,B,1)
            imb_sd = (E[:, :, 6] - E[:, :, 3] ** 2).clamp_min(0.0).sqrt()
            feats = torch.cat([
                midc.unsqueeze(-1) - E[:, :, 4],                      # rev: mean reversion
                E[:, :, 2],                                           # cross-instrument momentum
                (bp.unsqueeze(2) - EL).mean(-1),                      # lvl_gap  (group-rolled)
                bp.std(-1).unsqueeze(-1) - EL.std(-1),                # lvl_disp (group-rolled)
                E[:, :, 1] / (rv + eps),                              # vol-normalised momentum
                E[:, :, 3],                                           # EWMA imbalance
                rv,
                flow.unsqueeze(-1) * rv,                              # flow x vol
                a0 * rv,                                              # a0 x vol: -0.028 on t1
                E[:, :, 5] / (rv ** 3 + eps),                         # realized skew (t0 side)
                (imb.unsqueeze(-1) - E[:, :, 3]) / (imb_sd + eps),    # imbalance z
                (midc.unsqueeze(-1) - E[:, :, 4]) / (rv * self.nl_rt + eps)], -1)   # mid z
            parts.append(torch.stack([d, d_bid, d_ask, d_bid - d_ask, d1, d_imb,
                                      disp, trd_rel, trd_p, flow], -1))
            parts.append(feats)
            new.append(torch.nn.functional.pad(x[-1], (0, self.hidden - F)))
            nb = torch.cat([es.reshape(B, NS * H), el.reshape(B, NL * H),
                            torch.ones(B, 1, device=x.device, dtype=x.dtype)], -1)
            new.append(torch.nn.functional.pad(nb, (0, self.hidden - nb.shape[1])))
        if self.ofi:
            ko = (self.rnn_slots + int(self.diff) + len(self.ewma) + int(self.posfeat)
                  + (2 if self.nlfeat else 0) + int(self.quad))
            F_ = x.shape[-1]
            prev = state[ko, :, :F_]
            xs = torch.cat([prev.unsqueeze(0), x], 0)          # (T+1,B,F)
            ofis = []
            for base in (0, 52):
                pb, pa = xs[..., base + 0:base + 11], xs[..., base + 11:base + 22]
                qb, qa = xs[..., base + 22:base + 33], xs[..., base + 33:base + 44]
                dpb, dpa = pb[1:] - pb[:-1], pa[1:] - pa[:-1]
                # bid: add the new queue when the price rose or held, drop the old when it fell or held
                eb = torch.where(dpb > 0, qb[1:], torch.zeros_like(qb[1:])) \
                   + torch.where(dpb == 0, qb[1:] - qb[:-1], torch.zeros_like(qb[1:])) \
                   - torch.where(dpb < 0, qb[:-1], torch.zeros_like(qb[:-1]))
                ea = torch.where(dpa < 0, qa[1:], torch.zeros_like(qa[1:])) \
                   + torch.where(dpa == 0, qa[1:] - qa[:-1], torch.zeros_like(qa[1:])) \
                   - torch.where(dpa > 0, qa[:-1], torch.zeros_like(qa[:-1]))
                e = eb - ea                                     # (T,B,11)
                depth = (qb[1:].abs().mean(-1, keepdim=True) + qa[1:].abs().mean(-1, keepdim=True)) * 0.5
                ofis.append(e / (depth + 1e-3))
                ofis.append((e.sum(-1, keepdim=True)) / (11.0 * depth + 1e-3))
            parts = list(parts) + ofis
            new.append(torch.nn.functional.pad(x[-1], (0, self.hidden - F_)))
        if self.bookconv:
            T_, B_ = x.shape[0], x.shape[1]
            cs = []
            _P = self.bc_taps // 2                                    # add_s1.py: K taps, zero-padded by K//2 (K = 3: pad 1)
            for base in (0, 52):
                if self.bc_split or self.bc_side:                      # add_h7_bcsplit.py: dense (K, 4, C) kernel assembled from the restricted parts
                    _bw, _bb = self._h7_kernel(), self.book_b
                else:
                    _bw, _bb = (self.book_w1, self.book_b1) if (self.bc_unshare and base == 52) else (self.book_w, self.book_b)
                lad = torch.stack([x[..., base + 0:base + 11], x[..., base + 11:base + 22],
                                   x[..., base + 22:base + 33], x[..., base + 33:base + 44]], -2)
                lad = lad.reshape(T_, B_, 4, 11)
                pad = torch.nn.functional.pad(lad, (_P, _P))           # zero outside the ladder
                acc = _bb.view(1, 1, -1, 1).expand(T_, B_, self.bookconv, 11).clone()
                for j in range(self.bc_taps):
                    # (T,B,4,11) x (4,C) -> (T,B,C,11), one tap of the depth kernel
                    acc = acc + torch.einsum("tbcl,co->tbol", pad[..., j:j + 11], _bw[j])
                if self.bc_act:                                        # add_struct.py: position-wise GELU on the shared read
                    acc = torch.nn.functional.gelu(acc)
                cs.append(acc.reshape(T_, B_, self.bookconv * 11))
            parts = list(parts) + cs
        if self.tradeconv:                                             # add_s1.py: 3 taps along the 4 trade slots, (dp, dv) channels, shared across instruments
            T_, B_ = x.shape[0], x.shape[1]
            cs = []
            for base in (44, 96):
                lad = torch.stack([x[..., base + 0:base + 4], x[..., base + 4:base + 8]], -2)   # (T,B,2,4)
                pad = torch.nn.functional.pad(lad, (1, 1))
                acc = self.trade_b.view(1, 1, -1, 1).expand(T_, B_, self.tradeconv, 4).clone()
                for j in range(3):
                    acc = acc + torch.einsum("tbcl,co->tbol", pad[..., j:j + 4], self.trade_w[j])
                cs.append(acc.reshape(T_, B_, self.tradeconv * 4))
            parts = list(parts) + cs
        if self.instshare:                                             # add_s1.py: one dense encoder of the 52 instrument columns, shared by i0 and i1
            cs = [torch.einsum("tbf,fo->tbo", x[..., base:base + 52], self.inst_w[0]) + self.inst_b for base in (0, 52)]
            parts = list(parts) + cs
        if self.densein:                                               # add_s1b.py: dense read of the 88 book columns (no structure)
            xb = torch.cat([x[..., 0:44], x[..., 52:96]], -1)
            parts = list(parts) + [torch.einsum("tbf,fo->tbo", xb, self.dense_w[0]) + self.dense_b]
        if self.xinst:                                                 # add_struct.py: cross-instrument ladder read, shared over the 11 indices
            T_, B_ = x.shape[0], x.shape[1]
            lad = torch.stack([x[..., b + c * 11:b + c * 11 + 11] for b in (0, 52) for c in range(4)], -2)   # (T,B,8,11)
            pad = torch.nn.functional.pad(lad, (1, 1))                 # zero outside the ladder
            acc = self.xinst_b.view(1, 1, -1, 1).expand(T_, B_, self.xinst, 11).clone()
            for j in range(3):
                acc = acc + torch.einsum("tbcl,co->tbol", pad[..., j:j + 11], self.xinst_w[j])
            parts = list(parts) + [acc.reshape(T_, B_, self.xinst * 11)]
        if self.ple_bins:                                              # add_h4_ple.py: piecewise-linear embedding of the 44 price codes, row t only
            if not getattr(self, "_ple_ok", False):                    # once per process: the edges came from set_ple_edges or a checkpoint
                if not bool((self.ple_edges[:, 1:] > self.ple_edges[:, :-1]).all()):
                    raise RuntimeError("ple_edges are not set: train.py computes them from the training quantiles (set_ple_edges); "
                                       "loading a checkpoint restores them")
                self._ple_ok = True
            T_, B_ = x.shape[0], x.shape[1]
            xp = torch.cat([x[..., 0:22], x[..., 52:74]], -1)        # (T,B,44) standardised + clamped price codes of THIS row
            lo = self.ple_edges[:, :-1]; wd = self.ple_edges[:, 1:] - lo                 # (44,B)
            e = (xp.unsqueeze(-1) - lo) / wd                                              # (T,B,44,B) linear position inside each bin
            e = torch.minimum(torch.maximum(e, self._ple_floor), self._ple_ceil)           # PLE: 1 above the bin, 0 below, edges extrapolate
            parts = list(parts) + [torch.einsum("tbpk,kc->tbpc", e, self.ple_w[0]).reshape(T_, B_, 44 * self.ple_ch)]
        return torch.cat(parts, -1), new

    def _nl_scan(self, T, device, dtype):
        """(H,T,T) lower-triangular decay weights and (H,T) carry powers, cached."""
        key = (T, str(device), str(dtype))
        cache = getattr(self, "_nl_cache", None)
        if cache is None:
            cache = self._nl_cache = {}
        if key not in cache:
            a = self.nl_alpha.to(device=device, dtype=dtype).view(-1, 1, 1)   # (H,1,1)
            idx = torch.arange(T, device=device, dtype=dtype)
            lag = idx.view(1, T, 1) - idx.view(1, 1, T)                       # (1,T,T) t-s
            W = torch.where(lag >= 0, a * (1.0 - a) ** lag.clamp_min(0.0),
                            torch.zeros((), device=device, dtype=dtype))
            dpow = (1.0 - a.view(-1, 1)) ** (idx.view(1, T) + 1.0)            # (H,T)
            cache[key] = (W, dpow)
        return cache[key]

    NL_BLOCK = 1024
    FW_BLOCK = 64      # chunked linear attention: T/64 sequential steps instead of T

    def _scan(self, base, e0):
        """base (T,B,C), e0 (B,C,H) -> (T,B,C,H). Blocked so memory is O(B*T*C*H), not O(T^2)."""
        outs, e, T = [], e0, base.shape[0]
        for a in range(0, T, self.NL_BLOCK):
            L = min(self.NL_BLOCK, T - a)
            W, dpow = self._nl_scan(L, base.device, base.dtype)
            E = torch.einsum("hts,sbc->tbch", W, base[a:a + L])
            E = E + dpow.permute(1, 0).view(L, 1, 1, -1) * e.unsqueeze(0)
            outs.append(E); e = E[-1]
        return (outs[0] if len(outs) == 1 else torch.cat(outs, 0)), e

    def _scan_lvl(self, base, e0):
        """base (T,B,NL), e0 (B,H,NL) -> (T,B,H,NL). Blocked, same reason."""
        outs, e, T = [], e0, base.shape[0]
        for a in range(0, T, self.NL_BLOCK):
            L = min(self.NL_BLOCK, T - a)
            W, dpow = self._nl_scan(L, base.device, base.dtype)
            E = torch.einsum("hts,sbc->tbhc", W, base[a:a + L])
            E = E + dpow.permute(1, 0).view(L, 1, -1, 1) * e.unsqueeze(0)
            outs.append(E); e = E[-1]
        return (outs[0] if len(outs) == 1 else torch.cat(outs, 0)), e

    def forward(self, x, h=None, return_states=False):
        """return_states also yields the per-row hidden sequence (T,B,H) for the state
        regularisers; it is nn.GRU's own output, so nothing extra is computed."""
        if h is None: h = self.init_state(x.shape[1], x.device)
        x = ((x - self.mu) / self.sd).clamp(-self.clamp, self.clamp)
        if self.aux_next and self.training: self._x_std = x
        if self.adanorm > 0:
            ka = (self.rnn_slots + int(self.diff) + len(self.ewma) + int(self.posfeat)
                  + (2 if self.nlfeat else 0) + int(self.quad) + int(self.ofi))
            F_ = x.shape[-1]
            al = 1.0 - 2.0 ** (-1.0 / self.adanorm)
            m_ = h[ka, :, :F_]; v_ = h[ka + 1, :, :F_]
            _x48 = x                                             # add_z48.py: pre-adanorm input (--adanorm-cols)
            outs = []
            for t in range(x.shape[0]):
                m_ = m_ + al * (x[t] - m_)
                v_ = v_ + al * ((x[t] - m_) ** 2 - v_)
                outs.append((x[t] - m_) / (v_ + 1.0).sqrt())    # +1 so an all-quiet column is not amplified
            x = torch.stack(outs, 0)
            if getattr(self, "adanorm_mask", None) is not None:   # add_z48.py: only the selected columns are normalised
                x = torch.where(self.adanorm_mask > 0.5, x, _x48)
            ada_state = [torch.nn.functional.pad(m_, (0, self.hidden - F_)),
                         torch.nn.functional.pad(v_, (0, self.hidden - F_))]
        else:
            ada_state = []
        if self.split:
            L, H2 = self.layers, self.hidden // 2
            zs, ys, hs = [], [], []
            for k, cols in enumerate((self.i0, self.i1)):
                z = self.in_proj[k](x[..., cols]); y, hn = self.gru[k](z, h[k*L:(k+1)*L, :, :H2].contiguous())
                zs.append(z); ys.append(y); hs.append(torch.nn.functional.pad(hn, (0, self.hidden - H2)))
            z = torch.cat(zs, -1); y = torch.cat(ys, -1)
            _o = self.head(self.post_ln(torch.cat([y, z], -1)))
            return (_o, torch.cat(hs, 0), y) if return_states else (_o, torch.cat(hs, 0))
        xin, extra = self._inputs(x, h)
        z = self.in_proj(xin)
        r = self.rnn_slots
        if self.quad:
            kq = self.rnn_slots + int(self.diff) + len(self.ewma) + int(self.posfeat) + (2 if self.nlfeat else 0)
            P = z.shape[-1]
            zp = h[kq, :, :P]                                   # previous row's projection
            zprev = torch.cat([zp.unsqueeze(0), z[:-1]], 0)
            zq = z * (z - zprev)                                # level x change: flow-shaped
            gru_in = torch.cat([z, zq], -1)
            extra = list(extra) + [torch.nn.functional.pad(z[-1], (0, self.hidden - P))]
        else:
            gru_in = z
        r_out = None
        if self.fastw:
            r_out, fw_extra = self._fast_weights(z, h)
            extra = list(extra) + fw_extra
            if self.fw_to_gru:
                gru_in = torch.cat([gru_in, r_out], -1)
        if self.cell == "lstm":
            L = self.layers
            y, (hn, cn) = self.gru(gru_in, (h[:L].contiguous(), h[L:r].contiguous()))
            rnn_state = torch.cat([hn, cn], 0)
        else:
            if self.training and self.in_drop > 0:
                # variational: one mask per lane, shared over the chunk's time steps
                _m = (torch.rand(1, gru_in.shape[1], gru_in.shape[2], device=gru_in.device) >= self.in_drop).to(gru_in.dtype) / (1.0 - self.in_drop)
                gru_in = gru_in * _m
            if self.hh_rank:                                           # add_struct.py: rank-r recurrence, the same kernel via torch._VF.gru
                y, rnn_state = torch._VF.gru(gru_in, h[:r].contiguous(), [self.gru.weight_ih_l0, self.hh_A @ self.hh_B,
                                             self.gru.bias_ih_l0, self.gru.bias_hh_l0], True, 1, 0.0, self.training, False, False)
            else:
                y, rnn_state = self.gru(gru_in, h[:r].contiguous())
            if self.training and self.out_drop > 0:
                _m = (torch.rand(1, y.shape[1], y.shape[2], device=y.device) >= self.out_drop).to(y.dtype) / (1.0 - self.out_drop)
                y = y * _m
        self._gru_out = y
        if self.dualmode and self.training:
            # backward GRU over W-row windows, all windows of the chunk in one batched cuDNN call;
            # the backward state restarts at every window boundary (window-W lookahead)
            T, B, F = gru_in.shape; W = self.dualmode; Tp = ((T + W - 1) // W) * W
            xb = gru_in if Tp == T else torch.cat([gru_in, gru_in.new_zeros(Tp - T, B, F)], 0)
            xb = xb.view(Tp // W, W, B, F).permute(1, 0, 2, 3).reshape(W, (Tp // W) * B, F).flip(0)
            yb, _ = self.aux_bwd(xb)
            yb = yb.flip(0).reshape(W, Tp // W, B, -1).permute(1, 0, 2, 3).reshape(Tp, B, -1)[:T]
            self._aux_out = self.aux_head(self.aux_ln(torch.cat([y, yb, z], -1)))
        head_in = torch.cat([y, z], -1) if (r_out is None or self.fw_to_gru) else torch.cat([y, z, r_out], -1)
        extra = list(extra) + ada_state
        state = torch.cat([rnn_state] + [e.unsqueeze(0) for e in extra], 0) if extra else rnn_state
        _hin = self.post_ln(head_in)
        if getattr(self, "_z48_keep", False):                  # add_z48.py: --dann reads the head input (train-only)
            self._z48_hin = _hin
        if getattr(self, "mh_heads", 0):                    # add_z59.py: K sub-heads, output = their mean
            _mho = torch.stack([_hd(_hin) for _hd in self.mh_head], -2)          # (T,B,K,n_out)
            self._mh_out = _mho if self.training else None
            _out = _mho.mean(-2)
        elif getattr(self, "hlg_bins", 0) and self.training:  # add_z59.py: the same ops as self.head(_hin), hidden layer kept
            _h59 = self.head[2](self.head[1](self.head[0](_hin)))
            _out = self.head[3](_h59)
            self._hlg_logits = self.hlg_head(_h59).view(_h59.shape[0], _h59.shape[1], 2, self.hlg_bins)   # (T,B,2,K)
        elif getattr(self, "resnorm", 0) and self.training:   # add_h12_resnorm.py: the same ops as self.head(_hin), hidden layer kept
            _h12 = self.head[2](self.head[1](self.head[0](_hin)))
            _out = self.head[3](_h12)
            self._resnorm_out = self.resnorm_head(_h12)        # (T,B,2) log-sigma per target, train-only (research H12)
        else:
            _out = self.head(_hin)
        if getattr(self, "hetero", 0) and self.training:
            self._hetero_out = self.hetero_head(_hin)        # (T,B,2) log-variance, train-only (add_losses2.py)
        if self.grid:
            _lg = _out.view(_out.shape[0], _out.shape[1], 2, self.grid)
            self._grid_logits = _lg
            _out = torch.softmax(_lg, -1) @ self.grid_v
        return (_out, state, y) if return_states else (_out, state)

    def _fast_weights(self, z, h):
        """Running associative memory of this sequence's own transition operator.

        A_t = lam * A_{t-1} + phi(k_t) v_t^T with k from the PREVIOUS projected row and v from the
        current one, so A accumulates (previous -> current) for this sequence and nothing else. The
        readout queries it with the current row, giving the head a running answer to "in this
        sequence, what usually follows a row like this one". phi = elu + 1 keeps the keys positive so
        the normaliser cannot change sign, which is what makes the linear-attention form stable.

        The scan is sequential in T because A is a matrix; T is at most one BPTT chunk.
        """
        T, B, P = z.shape
        D = self.fastw
        base = (self.rnn_slots + int(self.diff) + len(self.ewma) + int(self.posfeat)
                + (2 if self.nlfeat else 0) + int(self.quad))
        nA = (D * D + self.hidden - 1) // self.hidden
        A = h[base:base + nA].permute(1, 0, 2).reshape(B, -1)[:, :D * D].reshape(B, D, D)
        nrm = h[base + nA, :, :D]
        zp = h[base + nA + 1, :, :P]
        zprev = torch.cat([zp.unsqueeze(0), z[:-1]], 0)
        phi = lambda t: torch.nn.functional.elu(t) + 1.0
        q = phi(self.fw_q(z)).transpose(0, 1)          # (B,T,D)
        k = phi(self.fw_k(zprev)).transpose(0, 1)
        v = self.fw_v(z).transpose(0, 1)
        lam = torch.sigmoid(self.fw_decay)                                 # (G,) one decay per key group
        G = self.fw_groups; Dg = D // G
        lam_row = lam.repeat_interleave(Dg)                                # (D,) decay of each key row of A
        outs = []
        for a in range(0, T, self.FW_BLOCK):
            L = min(self.FW_BLOCK, T - a)
            qb, kb, vb = q[:, a:a + L], k[:, a:a + L], v[:, a:a + L]
            i_ = torch.arange(L, device=z.device, dtype=z.dtype)
            gap = (i_.view(L, 1) - i_.view(1, L)).clamp_min(0.0)
            dec = lam.view(G, 1) ** (i_.view(1, L) + 1.0)                  # (G,L)
            qd = qb * dec.repeat_interleave(Dg, 0).transpose(0, 1).unsqueeze(0)
            num = torch.bmm(qd, A)                                         # (B,L,D) carry of the block-start memory
            den = (qd * nrm.unsqueeze(1)).sum(-1)                          # (B,L)
            for g in range(G):
                sl = slice(g * Dg, (g + 1) * Dg)
                Mi = torch.tril(lam[g] ** gap)                             # (L,L) within-block decay for this group
                Sm = torch.bmm(qb[:, :, sl], kb[:, :, sl].transpose(1, 2)) * Mi.unsqueeze(0)
                num = num + torch.bmm(Sm, vb)
                den = den + Sm.sum(-1)
            outs.append(num / den.clamp_min(1e-4).unsqueeze(-1))
            kd = kb * (lam.view(G, 1) ** (L - 1.0 - i_.view(1, L))).repeat_interleave(Dg, 0).transpose(0, 1).unsqueeze(0)
            A = (lam_row ** L).view(1, D, 1) * A + torch.bmm(kd.transpose(1, 2), vb)
            nrm = (lam_row ** L).view(1, D) * nrm + kd.sum(1)
        r_out = torch.cat(outs, 1).transpose(0, 1)                         # (T,B,D)
        flatA = torch.nn.functional.pad(A.reshape(B, D * D), (0, nA * self.hidden - D * D))
        slots = [flatA.reshape(B, nA, self.hidden)[:, jj] for jj in range(nA)]
        slots.append(torch.nn.functional.pad(nrm, (0, self.hidden - D)))
        slots.append(torch.nn.functional.pad(z[-1], (0, self.hidden - P)))
        return r_out, slots

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        return 16.8 + 18.0 * self.n_params() / 1e6
