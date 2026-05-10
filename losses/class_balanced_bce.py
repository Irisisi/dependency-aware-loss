# losses/class_balanced_bce.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    beta: float = 0.9999,
    ignore_index: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Class-balanced BCE using effective-number weights.

    Expected shapes:
        logits: [..., V]
        labels: [..., V] in {-1, 0, 1}

    The ignore_index entries are excluded from both weight estimation and loss averaging.

    Code-level effective-number weight:
        alpha_raw_v = (1 - beta) / (1 - beta ** max(n_v, 1))

    Then normalised so that mean(alpha_v) = 1.

    In this implementation, alpha_v is applied to all valid entries for code v,
    including positive and negative entries. This matches the manuscript's
    baseline formulation.
    """
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

    n_v = (y01 * valid_f).sum(dim=0)
    n_eff = n_v.clamp_min(1.0)

    beta_t = torch.tensor(beta, device=x.device, dtype=x.dtype)

    alpha_raw = (1.0 - beta_t) / (1.0 - torch.pow(beta_t, n_eff))
    alpha = alpha_raw / alpha_raw.mean().clamp_min(eps)
    alpha = alpha.view(1, V)

    bce = F.binary_cross_entropy_with_logits(x, y01, reduction="none")

    loss = alpha * bce * valid_f
    denom = valid.sum().clamp_min(1)

    return loss.sum() / denom


class ClassBalancedBCELoss(nn.Module):
    """
    nn.Module wrapper for class_balanced_bce_with_logits.
    """

    def __init__(
        self,
        beta: float = 0.9999,
        ignore_index: int = -1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return class_balanced_bce_with_logits(
            logits,
            labels,
            beta=self.beta,
            ignore_index=self.ignore_index,
            eps=self.eps,
        )

