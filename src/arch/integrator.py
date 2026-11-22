"""GRU with a parallel bank of LINEAR leaky integrators.

Measured motivation (experiments/eda/FAMILY0_VS_FAMILY1.md, reproduced independently):
family 0's deficit is the LEVEL of the target, not its increments. Stopped states carry
66% of its scored weight; inside a walk-back the model flips sign on each +1 increment
and stays wrong for 150-400 rows. An oracle given the true level `k - k[99]` and nothing
else scores 0.873 where the deployed model scores ~0.48.

A GRU cannot hold that level. Its state is squashed through tanh and its gates are
convex combinations, so representing an unbounded running sum over 20,000 rows means
sitting in the saturated tail where gradients vanish -- and training windows are 768
rows from a zero state, so the level is never even exercised.

This adds a bank of LINEAR recurrences beside it:

    z_t = a * z_{t-1} + W_u x_t ,    a = sigmoid(a_logit) in (0,1), no nonlinearity

With `a` initialised near 1 these are near-perfect accumulators; with `a` small they are
fast-decaying local averages, so the bank can span timescales and the data picks. z is
concatenated onto the GRU output for the head, so the readout sees a level directly.

Cost is small and the export path is unchanged in kind: z is one more recurrent state
of width N, updated by an elementwise multiply-add. The decoded-increment route was
tried first and is a dead end -- increments are perfectly DETECTABLE from features
(1.000 at every magnitude) but not decodable, because the price columns are transformed,
so an integrator fed by decoded jumps drifts (level correlation -0.03).
"""
from __future__ import annotations
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import GRUCore, N_FEATURES


class IntGRUCore(GRUCore):
    """GRU(hidden) + `n_int` linear leaky integrators, both read by one head."""

    def __init__(self, hidden=128, layers=2, n_out=1, n_features=N_FEATURES,
                 n_int=32, dropout=0.0):
        super().__init__(hidden=hidden, layers=layers, n_out=n_out,
                         n_features=n_features, dropout=dropout)
        self.n_int = n_int
        # Driven by the GRU's OUTPUT, not the raw input. The per-row increment is not a
        # linear function of the features -- a least-squares decoder gets 0.117 exact --
        # so a bank accumulating W_u x_t could not track the level either. Accumulating
        # W_u y_t lets the GRU compute the increment nonlinearly and leaves the bank to do
        # only what a GRU cannot: integrate it without saturating.
        self.u = nn.Linear(hidden, n_int, bias=False)
        # Spread the initial timescales geometrically from ~4 rows to ~20,000 so the
        # bank covers the flicker regime and the whole-sequence level at once, rather
        # than starting every channel at the same horizon.
        tau = torch.logspace(0.6, 4.3, n_int)          # ~4 to ~20,000 rows
        self.a_logit = nn.Parameter(torch.logit((1.0 - 1.0/tau).clamp(1e-4, 1 - 1e-4)))
        self.head = nn.Linear(hidden + n_int, n_out)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)

    def init_state(self, batch, device):
        """(layers + 1, B, hidden): the GRU stack plus one row carrying z."""
        h = super().init_state(batch, device)
        return torch.cat([h, torch.zeros(1, batch, h.shape[-1], device=device)], 0)

    def forward(self, x, h=None):
        """x: (T,B,F). State is the GRU stack; z is carried in its own rows.

        z is appended to the GRU state tensor so the existing (T,B,F) -> (out, state)
        contract and the export path's state handling stay unchanged.
        """
        T, B, _ = x.shape
        xn = (x - self.mu) / self.sd
        if h is None:
            h = self.init_state(B, x.device)
        hg, z = h[:self.layers], h[self.layers][:, :self.n_int] if h.shape[0] > self.layers \
            else torch.zeros(B, self.n_int, device=x.device, dtype=x.dtype)
        y, hg = self.gru(xn, hg)

        a = torch.sigmoid(self.a_logit)            # (n_int,)
        # LRU-style input normalisation. An UNNORMALISED accumulator z = a z + u drives
        # z to O(1/(1-a)) -- with a = 0.9999 that is O(10^4) after 20,000 rows, so the
        # head weights must be ~1e-4 and the whole bank is badly conditioned. Scaling the
        # input by sqrt(1-a^2) holds z at unit variance for white-noise u at every
        # timescale, so all n_int channels reach the head on the same scale.
        u = self.u(y) * torch.sqrt(1.0 - a * a)    # (T,B,n_int) <- from the GRU output
        zs = []
        for t in range(T):
            z = a * z + u[t]
            zs.append(z)
        Z = torch.stack(zs)                        # (T,B,n_int)
        out = self.head(torch.cat([y, Z], -1))
        pad = torch.zeros(1, B, hg.shape[-1], device=x.device, dtype=x.dtype)
        pad[0, :, :self.n_int] = z
        return out, torch.cat([hg, pad], 0)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def est_us_per_row(self):
        return 16.8 + 18.0 * self.n_params() / 1e6
