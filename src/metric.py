"""Weighted Pearson: the competition metric, and a differentiable form to train on.

The scoreboard (see METRIC.md, spec v2_wp_block_clip2_v1) computes, per sequence
and per target, a Pearson correlation weighted by w = |clip(y, -2, 2)|, on rows
where need_prediction AND is_scored. Two properties drive everything here:

  * it is a CORRELATION, so within a sequence any offset and any positive
    rescaling of our predictions are free;
  * the weight is the magnitude of the true move, so flat rows are worth
    literally nothing and the ~22% of rows with y == 0 drop out entirely.

Plain MSE optimises neither. `wp_loss` below optimises the thing itself.
"""
from __future__ import annotations
import numpy as np
import torch

METRIC_CLIP = 2.0
EPS = 1e-8


# --------------------------------------------------------------------------
# Reference implementation (numpy, float64) - mirrors utils.weighted_pearson
# --------------------------------------------------------------------------
def weighted_pearson_np(target, prediction) -> float:
    y = np.clip(np.asarray(target, np.float32), -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    p = np.clip(np.asarray(prediction, np.float32), -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    w = np.abs(y)
    total = w.sum()
    if total < EPS:
        return 0.0
    my = (w * y).sum() / total
    mp = (w * p).sum() / total
    yc, pc = y - my, p - mp
    cov = (w * yc * pc).sum() / total
    sy = np.sqrt((w * yc * yc).sum() / total)
    sp = np.sqrt((w * pc * pc).sum() / total)
    if sy <= EPS or sp <= EPS:
        return 0.0
    return float(np.clip(cov / (sy * sp), -1.0, 1.0))


# --------------------------------------------------------------------------
# Differentiable form
# --------------------------------------------------------------------------
def wp_torch(y: torch.Tensor, p: torch.Tensor, mask: torch.Tensor | None = None,
             clip_pred: bool = False) -> torch.Tensor:
    """Weighted Pearson over the last dim. y/p: (..., T). Returns (...,).

    `mask` (..., T) of 0/1 restricts to scored rows; None means all rows.
    Predictions are NOT clipped by default: clipping zeroes the gradient outside
    [-2, 2] and the correlation is scale-invariant anyway, so the model has no
    incentive to run away. Keep the output scale honest at export instead.
    """
    yc_ = torch.clamp(y, -METRIC_CLIP, METRIC_CLIP)
    pc_ = torch.clamp(p, -METRIC_CLIP, METRIC_CLIP) if clip_pred else p
    w = yc_.abs()
    if mask is not None:
        w = w * mask
    total = w.sum(-1)
    safe = total.clamp_min(EPS)
    my = (w * yc_).sum(-1) / safe
    mp = (w * pc_).sum(-1) / safe
    ycen = yc_ - my.unsqueeze(-1)
    pcen = pc_ - mp.unsqueeze(-1)
    cov = (w * ycen * pcen).sum(-1) / safe
    vy = (w * ycen * ycen).sum(-1) / safe
    vp = (w * pcen * pcen).sum(-1) / safe
    rho = cov / torch.sqrt(vy.clamp_min(EPS) * vp.clamp_min(EPS))
    ok = (total > EPS) & (vy > EPS * EPS) & (vp > EPS * EPS)
    return torch.where(ok, rho.clamp(-1.0, 1.0), torch.zeros_like(rho))


def wp_loss(signal: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None,
            weight: torch.Tensor | None = None) -> torch.Tensor:
    """Loss for the single-signal parameterisation: emit s, score it as (s, -s).

    signal: (B, T)      the one scalar the model predicts
    y:      (B, T, 2)   the two targets
    mask:   (B, T)      optional scored-row / propensity weights
    weight: (B,)        optional per-window weight in the final mean

    Returns 1 - mean WP, so 0 is perfect and 1 is uncorrelated.
    """
    m = mask
    r0 = wp_torch(y[..., 0], signal, m)
    r1 = wp_torch(y[..., 1], -signal, m)
    per_window = 0.5 * (r0 + r1)
    if weight is not None:
        per_window = per_window * weight
        return 1.0 - per_window.sum() / weight.sum().clamp_min(EPS)
    return 1.0 - per_window.mean()


def wp_loss_common(sc: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None,
                   weight: torch.Tensor | None = None) -> torch.Tensor:
    """Loss for y0 = s + c, y1 = -s + c.

    sc: (B, T, 2)  channel 0 is the antisymmetric signal s, channel 1 the common
                   component c shared by both targets.

    The single-signal form hard-ties the outputs to exact negatives because t0 and
    t1 correlate -0.91. That correlation does NOT establish that their PREDICTABLE
    parts are exact negatives. With c initialised to zero this loss starts exactly
    at `wp_loss`, so any movement measures predictable signal in (y0 + y1) / 2.
    """
    sig, c = sc[..., 0], sc[..., 1]
    r0 = wp_torch(y[..., 0], sig + c, mask)
    r1 = wp_torch(y[..., 1], -sig + c, mask)
    per_window = 0.5 * (r0 + r1)
    if weight is not None:
        per_window = per_window * weight
        return 1.0 - per_window.sum() / weight.sum().clamp_min(EPS)
    return 1.0 - per_window.mean()


def wp_loss_two_head(pred: torch.Tensor, y: torch.Tensor,
                     mask: torch.Tensor | None = None) -> torch.Tensor:
    """Same, for a model that predicts both targets independently. pred: (B,T,2)."""
    r0 = wp_torch(y[..., 0], pred[..., 0], mask)
    r1 = wp_torch(y[..., 1], pred[..., 1], mask)
    return 1.0 - (0.5 * (r0 + r1)).mean()
