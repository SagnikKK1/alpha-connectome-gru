"""Gate initialisation and recurrent-matrix initialisation for nn.GRU.

Three zero-parameter, zero-inference-cost changes, all with isolated evidence.

**The sign convention is the trap.** PyTorch computes
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
so `z` is the RETENTION gate, and Tallec & Ollivier's "write gate" g is (1 - z). Their
chrono prescription b_g ~ -log(U([1, T_max-1])) therefore becomes a POSITIVE bias on z:
    b_z ~ +log(U([1, T_max - 1]))
Getting this backwards initialises every unit to forget as fast as possible, which looks
like a plausible run that simply never learns long structure.

UGI (Gu et al. 2019) instead draws b_z ~ sigmoid^-1(U[eps, 1-eps]). It permits NEGATIVE
biases, induces a Pareto(2) distribution over decay periods rather than chrono's much
heavier tail, and is insensitive to a T_max guess. On the one real (non-synthetic) task
where both were compared, chrono gave nothing (35.8 -> 35.8 ppl) and UGI helped
(-> 34.9). It also shows the common "forget bias = 1.0" heuristic is worth a decay period
of only 1/(1-sigmoid(1)) = 3.7 steps.

Orthogonal init: PyTorch initialises W_hh from U(-1/sqrt(H), 1/sqrt(H)) like everything
else. Orthogonal INITIALISATION beat both Glorot and identity across tasks (Vorontsov
et al. 2017), while hard orthogonality CONSTRAINTS hurt -- so initialise, never constrain.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

EPS = 1e-3


def _z_slice(bias: torch.Tensor, hidden: int) -> torch.Tensor:
    """PyTorch packs GRU biases as [r, z, n]; z is the middle third."""
    return bias[hidden:2 * hidden]


@torch.no_grad()
def apply_gate_init(gru: nn.GRU, mode: str, t_max: int = 20000) -> str:
    """'ugi' or 'chrono' on the update (retention) gate. Returns a log line."""
    if mode in (None, "", "default"):
        return "gate init: PyTorch default"
    H = gru.hidden_size
    for l in range(gru.num_layers):
        bh = getattr(gru, f"bias_hh_l{l}")
        bi = getattr(gru, f"bias_ih_l{l}")
        if mode == "ugi":
            u = torch.empty(H).uniform_(EPS, 1 - EPS)
            b = torch.log(u / (1 - u))                 # sigmoid^-1
        elif mode == "chrono":
            t = torch.empty(H).uniform_(1, max(t_max - 1, 2))
            b = torch.log(t)                           # POSITIVE: z is retention
        else:
            raise ValueError(mode)
        _z_slice(bh, H).copy_(b)
        _z_slice(bi, H).zero_()                        # the two biases add; use one
    if mode == "ugi":
        lo, hi = torch.sigmoid(torch.tensor([math.log(EPS/(1-EPS)),
                                             math.log((1-EPS)/EPS)]))
        return (f"gate init: UGI on z, retention in [{lo:.3f}, {hi:.3f}] "
                f"-> decay periods 1 to {1/(1-hi):.0f} rows")
    return f"gate init: chrono on z, timescales 1 to {t_max} rows (positive bias)"


@torch.no_grad()
def apply_orth_init(gru: nn.GRU) -> str:
    """Orthogonal init on each of the three W_hh gate blocks, per layer."""
    n = 0
    for l in range(gru.num_layers):
        W = getattr(gru, f"weight_hh_l{l}")            # (3H, H)
        H = gru.hidden_size
        for g in range(3):
            blk = torch.empty(H, H)
            nn.init.orthogonal_(blk)
            W[g*H:(g+1)*H].copy_(blk)
            n += 1
    return f"W_hh init: orthogonal ({n} gate blocks)"


def tar_penalty(h_seq: torch.Tensor, time_dim: int = 0) -> torch.Tensor:
    """Temporal Activation Regularisation: a slowness prior on the hidden trajectory.

    beta * ||h_t - h_{t+1}||^2 along TIME, normalised by the number of differences so the
    scale does not depend on window size.

    `time_dim` is explicit and not optional in spirit: the first version defaulted to axis
    0 and was handed `out` of shape (B, body, n_out), so it differenced the BATCH axis. It
    then rewarded windows for being identical to each OTHER -- exactly the between-window
    variation sequence-pooled WP exists to learn -- and penalised nothing in time.
    Verified inverted on probes: two time-constant windows differing from each other
    scored 1.0 (should be 0), two identical time-ramps scored 0.0 (should be 1).

    AR (penalising ||h||) is deliberately not offered: its rationale is keeping logits
    small under cross-entropy, and weighted Pearson is scale-invariant.
    """
    h = h_seq.transpose(0, time_dim) if time_dim else h_seq
    d = h[1:] - h[:-1]
    return d.pow(2).sum() / d.numel()
