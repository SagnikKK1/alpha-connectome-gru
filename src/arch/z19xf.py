"""Z19XF: a lookahead TEACHER of a different model class -- a transformer encoder over the W-row window,
on top of a causal GRU that carries the past (research Z19, 5 Oct 2026). Offline only, never a member,
not exportable.

Why. Lookahead distillation is the one lever that transferred to the board (+0.0025, Sep): a frozen
bidirectional GRU sees a 100-row future window and the causal student is pulled to its predictions.
Every later teacher kept the same class (bidirectional GRU: BT_*, BTMU_*, BT2_*, the w50/w200 dumps),
and a BETTER teacher of that class did not help (MUT: 0.9965 vs 0.99 holdout, student null). What the
student can absorb from a teacher is bounded by E[teacher | past]; what differs between teacher classes
is which part of the target's future-determined noise they explain and how they smear it over the
window. The targets are bracket P&Ls: a stop or exit event at a specific future row decides them, and a
backward GRU integrates the window smoothly, whereas content-based attention can align exactly on the
row where the barrier is hit (research Z16 SS C idea 3). This class gives the student a differently
denoised target over the SAME window; the student and its export are untouched.

Architecture (W = window, default 100; the shipped dump is `teacher_dump.py --window 100`):

    112 raw -> z-score (mu/sd buffers) -> clamp +-10
      -> in_proj : Linear -> GELU -> LayerNorm (x proj_layers, width `proj`)        [= refgru's input stack]
      -> causal GRU (hidden, layers) with the state CARRIED across calls              [= the past]
      -> window tokens u_t = tok_in([p_t ; g_t]) + pos_bias[t - window_start]
      -> xf_layers pre-LN encoder layers: full (non-causal) self-attention WITHIN the window with a
         learned per-head relative-position bias, GELU FFN                            [= the lookahead]
      -> post_ln([g_t ; z_t ; p_t]) -> Linear -> GELU -> Linear (n_out)              [= refgru's head]

Window semantics -- intrinsic, not a flag. Every call splits its input into consecutive W-row windows
starting at row 0 of the call; the GRU state flows across windows, the attention never crosses one. A
500- or 1,000-row training chunk is therefore 5 / 10 windows (all aligned to multiples of W, because
chunks start at multiples of the chunk length), the full-sequence `evaluate()` is 200 windows run as
one batch (the in-training WP of this teacher IS its window-W dump quality -- not inflated by the chunk
length the way a --bidir GRU's is), and `teacher_dump.py --window 100` is one window per call, which
reproduces the training computation exactly. Row k of a window sees rows k+1..W-1 ahead (the same
sawtooth as the GRU teacher's window-100 dump). A call whose length is neither <= W nor a multiple of
W is refused (it would silently be something else): `teacher_dump.py --min-look` is NOT supported by
this class, and a `--window 200` dump of a W=100 checkpoint is two 100-row windows, not a 200-row one --
for another window train the matching preset (z19xf_w50 / z19xf_w200).

Contract with the trainer and the dump (no change to either): `bidir = True` and `layers`, so train.py
and teacher_dump zero the odd rows of the (2*layers, B, hidden) state at every chunk / window (they are
the backward GRU's rows for a --bidir refgru; here they are unused and always zero); `init_state`,
`forward(x, h) -> (out, h)` over (T, B, F); `set_norm`, `n_params`, `est_us_per_row` from GRUCore; the
readout is `head.0` / `head.3` so `score_ckpt.load` reads n_out from `head.3.weight`. Geometry comes from
the preset dict plus the three kwargs train.py forwards to every preset (`hidden`, `layers`, `proj`);
`score_ckpt.load` forwards the same three for a `z19*` preset (offline/patches/add_z19.py), so a dump
rebuilds the trained model. `--proj-layers`, `--head-hidden`, `--dropout`, `--clamp` are refgru-only
flags and are INERT here (the preset dict fixes them).

Determinism: no advanced indexing or scatter in the differentiable path (the relative bias is a matmul
against a fixed one-hot buffer; the position slice is a view), attention is explicit matmul + softmax
(no SDPA kernel selection), the GRU is cuDNN under cudnn.deterministic as today. Parameters named
`*_bias` (pos_bias, rel_bias) fall in the trainer's no-weight-decay / not-a-Muon-matrix group by the
existing name rule.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GRUCore, N_FEATURES


class _WindowAttnLayer(nn.Module):
    """Pre-LN encoder layer: full self-attention within a window (+ additive per-head bias), GELU FFN."""

    def __init__(self, d, heads, ff):
        super().__init__()
        if d % heads: raise ValueError("z19xf: d_model must divide by heads")
        self.heads, self.dh = int(heads), d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, bias):
        # x: (N, W, d) windows batch-first; bias: (1, heads, W, W) additive, broadcast over N
        N, W, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(N, W, 3, self.heads, self.dh).unbind(2)   # (N, W, h, dh) each
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)              # (N, h, W, dh)
        s = torch.matmul(q, k.transpose(-1, -2)) * (self.dh ** -0.5) + bias           # (N, h, W, W)
        a = torch.matmul(torch.softmax(s, dim=-1), v)                                  # (N, h, W, dh)
        x = x + self.out(a.transpose(1, 2).reshape(N, W, d))
        return x + self.ff(self.ln2(x))


class Z19XFCore(GRUCore):
    bidir = True          # a non-causal teacher: train.py / teacher_dump zero the odd state rows per chunk / window

    def __init__(self, hidden=256, layers=1, n_out=1, n_features=N_FEATURES, proj=192, proj_layers=2,
                 window=100, d_model=256, heads=4, ff=512, xf_layers=4, head_hidden=0, clamp=10.0, dropout=0.0):
        super().__init__(hidden=hidden, layers=layers, n_out=n_out, n_features=n_features)
        self.clamp, self.window = float(clamp), int(window)
        self.d_model, self.heads, self.xf_layers = int(d_model), int(heads), int(xf_layers)
        self.n_in, self.proj_width = int(n_features), int(proj)
        if self.window < 2: raise ValueError("z19xf: window must be >= 2")
        blocks = [nn.Linear(n_features, proj), nn.GELU(), nn.LayerNorm(proj)]
        for _ in range(max(0, int(proj_layers) - 1)):
            blocks += [nn.Linear(proj, proj), nn.GELU(), nn.LayerNorm(proj)]
        self.in_proj = nn.Sequential(*blocks)
        self.gru = nn.GRU(proj, hidden, layers, batch_first=False, dropout=dropout if layers > 1 else 0.0)
        self.tok_in = nn.Linear(proj + hidden, d_model)
        self.pos_bias = nn.Parameter(torch.zeros(self.window, d_model))         # position within the window
        self.rel_bias = nn.Parameter(torch.zeros(self.heads, 2 * self.window - 1))   # per-head relative-offset bias
        self.xf = nn.ModuleList([_WindowAttnLayer(d_model, heads, ff) for _ in range(self.xf_layers)])
        self.xf_ln = nn.LayerNorm(d_model)
        self.post_ln = nn.LayerNorm(hidden + d_model + proj)
        hh = head_hidden or hidden
        self.head = nn.Sequential(nn.Linear(hidden + d_model + proj, hh), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hh, n_out))
        self.proj = None          # not GRUCore's output bottleneck
        nn.init.normal_(self.pos_bias, std=0.02)
        # one-hot of the relative offset j - i (+W-1) for every (i, j) pair: the relative bias is then a plain
        # matmul rel_bias @ onehot -> (heads, W, W) -- no gather/scatter in the differentiable path
        idx = torch.arange(self.window)
        rel = (idx[None, :] - idx[:, None] + self.window - 1)                 # (W, W) in 0..2W-2
        self.register_buffer("rel_onehot", torch.nn.functional.one_hot(rel, 2 * self.window - 1)
                             .permute(2, 0, 1).to(torch.float32).contiguous(), persistent=False)   # (2W-1, W, W)

    # -- trainer / dump contract -----------------------------------------------------------------------
    @property
    def rnn_slots(self):
        return 2 * self.layers

    def init_state(self, batch: int, device=None):
        return torch.zeros(2 * self.layers, batch, self.hidden,
                           device=device or next(self.parameters()).device)

    def _rel(self, Wc):
        """(1, heads, Wc, Wc) additive attention bias for a window of Wc <= W rows (prefix of the full table)."""
        oh = self.rel_onehot if Wc == self.window else self.rel_onehot[:, :Wc, :Wc]
        return torch.matmul(self.rel_bias, oh.reshape(oh.shape[0], -1)).view(1, self.heads, Wc, Wc)

    def forward(self, x, h=None, return_states=False):
        T, B, _ = x.shape
        if h is None: h = self.init_state(B, x.device)
        W = self.window
        if T > W and T % W:
            raise ValueError(f"z19xf: a call of {T} rows is neither <= the window ({W}) nor a multiple of it; "
                             f"the window is intrinsic to the checkpoint (teacher_dump --window {W}; --min-look unsupported)")
        x = ((x - self.mu) / self.sd).clamp(-self.clamp, self.clamp)
        p = self.in_proj(x)                                                   # (T, B, proj)
        g, hT = self.gru(p, h[0:2 * self.layers:2].contiguous())             # the past: causal, state carried
        tok = self.tok_in(torch.cat([p, g], -1))                              # (T, B, d)
        Wc = min(W, T); nw = T // Wc
        u = tok.view(nw, Wc, B, self.d_model).permute(0, 2, 1, 3).reshape(nw * B, Wc, self.d_model)
        u = u + self.pos_bias[:Wc].unsqueeze(0)
        bias = self._rel(Wc)
        for layer in self.xf:
            u = layer(u, bias)
        z = self.xf_ln(u).view(nw, B, Wc, self.d_model).permute(0, 2, 1, 3).reshape(T, B, self.d_model)
        out = self.head(self.post_ln(torch.cat([g, z, p], -1)))
        # state rows interleaved like a bidirectional GRU's: even = forward GRU, odd = unused (always zero)
        state = torch.stack([hT, torch.zeros_like(hT)], 1).reshape(2 * self.layers, B, self.hidden)
        return (out, state, g) if return_states else (out, state)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
