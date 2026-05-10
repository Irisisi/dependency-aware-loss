# losses/weighted_bce.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Weighted BCE with per-code positive weights estimated on the current mini-batch.

    Expected shapes:
        logits: [..., V]
        labels: [..., V] in {-1, 0, 1}

    The ignore_index entries are excluded from both weight estimation and loss averaging.

    Positive weight for code v:
        pos_weight_v = number_of_valid_negatives_v / max(number_of_valid_positives_v, 1)

    This matches PyTorch's BCEWithLogitsLoss(pos_weight=...), where pos_weight
    is applied only to the positive term.
    """
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

    # Per-code counts over valid entries.
    S_v = valid_f.sum(dim=0)
    n_v = (y01 * valid_f).sum(dim=0)

    pos_weight = (S_v - n_v) / n_v.clamp_min(1.0)

    bce = F.binary_cross_entropy_with_logits(
        x,
        y01,
        pos_weight=pos_weight,
        reduction="none",
    )

    loss = bce * valid_f
    denom = valid.sum().clamp_min(1)

    return loss.sum() / denom


class WeightedBCELoss(nn.Module):
    """
    nn.Module wrapper for weighted_bce_with_logits.
    """

    def __init__(self, ignore_index: int = -1) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return weighted_bce_with_logits(
            logits,
            labels,
            ignore_index=self.ignore_index,
        )

