# losses/focal.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gamma: float = 2.0,
    beta: float = 0.9999,
    ignore_index: int = -1,
    eps: float = 1e-6,
    tilde_alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Focal loss for sparse multi-label prediction.

    Expected shapes:
        logits: [..., V]
        labels: [..., V] in {-1, 0, 1}

    The ignore_index entries are excluded from both weight estimation and loss averaging.

    The focal loss uses:
        p_t = p if y=1, and 1-p if y=0
        modulation = (1 - p_t) ** gamma

    Code-level alpha is derived from the same effective-number weighting used
    by the class-balanced BCE baseline:

        alpha_raw_v = (1 - beta) / (1 - beta ** max(n_v, 1))
        alpha_v = alpha_raw_v / mean(alpha_raw)
        tilde_alpha_v = alpha_v / (1 + alpha_v)

    Then:
        alpha_t = tilde_alpha_v for positives,
        alpha_t = 1 - tilde_alpha_v for negatives.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")

    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must satisfy 0 <= beta < 1, got {beta}")

    if logits.shape != labels.shape:
        raise ValueError(
            f"logits and labels must have same shape, "
            f"got {logits.shape} vs {labels.shape}"
        )

    if logits.dim() < 2:
        raise ValueError(
            f"Expected logits/labels shape [..., V], got {logits.shape}"
        )

    V = logits.shape[-1]

    # Compute in float32 for mixed precision safety.
    x = logits.float().reshape(-1, V)
    y_raw = labels.reshape(-1, V)

    valid = y_raw != ignore_index
    valid_f = valid.to(dtype=x.dtype)

    y01 = y_raw.clamp(min=0, max=1).to(dtype=x.dtype)

    # Stable BCEWithLogits equals -log(p_t).
    bce = F.binary_cross_entropy_with_logits(x, y01, reduction="none")

    p = torch.sigmoid(x)
    p_t = p * y01 + (1.0 - p) * (1.0 - y01)
    p_t = p_t.clamp(min=eps, max=1.0 - eps)

    mod = torch.pow(1.0 - p_t, gamma)

    if tilde_alpha is None:
        n_v = (y01 * valid_f).sum(dim=0).clamp_min(1.0)

        beta_t = torch.tensor(beta, device=x.device, dtype=x.dtype)

        alpha_raw = (1.0 - beta_t) / (1.0 - torch.pow(beta_t, n_v))
        alpha = alpha_raw / alpha_raw.mean().clamp_min(eps)

        tilde = alpha / (1.0 + alpha)
        tilde = tilde.clamp(min=eps, max=1.0 - eps).view(1, V)
    else:
        tilde = torch.as_tensor(tilde_alpha, device=x.device, dtype=x.dtype)

        if tilde.dim() == 1:
            tilde = tilde.view(1, V)

        if tilde.shape != (1, V):
            raise ValueError(
                f"tilde_alpha must have shape [V] or [1, V], got {tilde.shape}"
            )

        tilde = tilde.clamp(min=eps, max=1.0 - eps)

    alpha_t = y01 * tilde + (1.0 - y01) * (1.0 - tilde)

    loss = alpha_t * mod * bce
    loss = loss * valid_f

    denom = valid.sum().clamp_min(1)

    return loss.sum() / denom


class FocalLoss(nn.Module):
    """
    nn.Module wrapper for focal_loss_with_logits.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        beta: float = 0.9999,
        ignore_index: int = -1,
        eps: float = 1e-6,
        tilde_alpha: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.gamma = float(gamma)
        self.beta = float(beta)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

        if tilde_alpha is None:
            self.tilde_alpha = None
        else:
            self.register_buffer(
                "tilde_alpha",
                tilde_alpha.detach().clone().float(),
            )

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return focal_loss_with_logits(
            logits,
            labels,
            gamma=self.gamma,
            beta=self.beta,
            ignore_index=self.ignore_index,
            eps=self.eps,
            tilde_alpha=self.tilde_alpha,
        )

