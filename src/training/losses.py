"""Correlation-aligned losses (N2).

The benchmark scores Pearson (PCC) and Spearman (SCC) correlation, but plain MSE optimises neither
directly. These differentiable batch-level losses target the actual metrics:
  * pearson_loss  — 1 - Pearson(pred, target), optimises the linear metric.
  * spearman_loss — Pearson over differentiable *soft ranks* (SoDeep-style), optimises the rank metric.
A small MSE term is kept for calibration (correlation is shift/scale invariant).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = pred - pred.mean()
    t = target - target.mean()
    corr = (p * t).sum() / (p.norm() * t.norm() + eps)
    return 1.0 - corr


def _soft_rank(x: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Differentiable rank approximation: rank_i = sum_j sigmoid((x_i - x_j)/temp)."""
    diff = (x.unsqueeze(1) - x.unsqueeze(0)) / temp
    return torch.sigmoid(diff).sum(1)


def spearman_loss(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    # scale predictions to unit std so a single temperature works across batches
    rp = _soft_rank((pred - pred.mean()) / (pred.std() + 1e-6), temp)
    rt = _soft_rank((target - target.mean()) / (target.std() + 1e-6), temp)
    return pearson_loss(rp, rt)


def correlation_loss(pred: torch.Tensor, target: torch.Tensor,
                     mse_w: float = 0.2, pearson_w: float = 1.0, spearman_w: float = 0.5,
                     temp: float = 0.1) -> torch.Tensor:
    """Weighted combination; needs a batch large enough (>=32) for a stable correlation estimate."""
    loss = mse_w * F.mse_loss(pred, target)
    if pearson_w:
        loss = loss + pearson_w * pearson_loss(pred, target)
    if spearman_w and pred.numel() >= 8:
        loss = loss + spearman_w * spearman_loss(pred, target, temp)
    return loss
