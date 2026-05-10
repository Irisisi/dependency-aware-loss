# losses/dependency_aware.py
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class DependencyAwareLoss(nn.Module):
    """
    Dependency-aware loss for sparse multi-label prediction.

    Expected shapes:
        logits: [..., V]
            Raw logits for V labels.

        labels: [..., V]
            Binary targets in {-1, 0, 1}, where -1 means ignore_index.

    The loss is averaged over rows with at least one valid label.
    Rows containing only ignore_index values are excluded from the mean.

    Main hyperparameters:
        alpha_neg:
            Negative-weight multiplier used in the class-weighted correctness term.

        lambda_reg:
            Strength of the output-space interaction regulariser.

        loss_scale:
            Optional constant multiplier applied to the final loss.
            Default is 1.0 so that the implementation matches the manuscript equation.
            If a different value was used in experiments, set it explicitly in the
            training config and document it in the manuscript or supplement.
    """

    def __init__(
        self,
        vocab_size: int,
        alpha_neg: float = 25.0,
        lambda_reg: float = 10.0,
        loss_scale: float = 1.0,
        ignore_index: int = -1,
        init: str = "kaiming_uniform",
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.alpha_neg = float(alpha_neg)
        self.lambda_reg = float(lambda_reg)
        self.loss_scale = float(loss_scale)
        self.ignore_index = int(ignore_index)

        # Learnable W in R^{V x V}.
        # Symmetry and zero diagonal are enforced during each forward pass.
        self.W_raw = nn.Parameter(torch.empty(self.vocab_size, self.vocab_size))

        if init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.W_raw, a=math.sqrt(5))
        elif init == "zeros":
            nn.init.zeros_(self.W_raw)
        else:
            raise ValueError(
                f"Unknown init={init!r}. Expected 'kaiming_uniform' or 'zeros'."
            )

        # Rank-weight cache. This is not a model parameter and is recomputed
        # when V, device, or dtype changes.
        self._rank_cache_V = 0
        self._rank_w: Optional[torch.Tensor] = None

    @staticmethod
    def _sym_zero_diag(W: torch.Tensor) -> torch.Tensor:
        """
        Enforce symmetry and zero diagonal.

        Returns:
            Symmetric matrix with exactly zero diagonal.
        """
        W = 0.5 * (W + W.transpose(0, 1))
        return W - torch.diag_embed(torch.diagonal(W, 0))

    def _rank_weights(
        self,
        V: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return rank weights of shape [1, V]:

            w_i = 2i / (V(V+1)), i = 1, ..., V

        These weights sum to 1 over V entries and place larger mass on later
        sorted positions. Since correctness values are sorted descending,
        later positions correspond to lower class-weighted correctness.
        """
        if (
            self._rank_cache_V != V
            or self._rank_w is None
            or self._rank_w.device != device
            or self._rank_w.dtype != dtype
        ):
            i = torch.arange(1, V + 1, device=device, dtype=dtype)
            w = 2.0 * i / (float(V) * float(V + 1))
            self._rank_w = w.view(1, V)
            self._rank_cache_V = V

        return self._rank_w

    def _rank_aggregate(
        self,
        wc: torch.Tensor,
        valid: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Rank-weighted aggregation of class-weighted correctness.

        Args:
            wc:
                Tensor of shape [N, V], class-weighted correctness values.

            valid:
                Boolean tensor of shape [N, V], True for valid entries.

            eps:
                Numerical stability constant.

        Returns:
            Tensor of shape [N], one rank-weighted aggregate per row.
        """
        n_rows, V = wc.shape

        # Invalid values are sent to the tail after descending sort.
        scores = wc.masked_fill(~valid, float("-inf"))
        sorted_wc, _ = torch.sort(scores, dim=1, descending=True)

        # Row-specific valid count.
        n_valid = valid.sum(dim=1).clamp_min(1)

        # Full rank weights for V positions.
        w = self._rank_weights(V, wc.device, wc.dtype).expand(n_rows, V)

        # Keep only the first n_valid sorted positions per row and renormalise.
        idx = torch.arange(V, device=wc.device).view(1, V).expand(n_rows, V)
        keep = idx < n_valid.view(n_rows, 1)

        w = w * keep.to(dtype=w.dtype)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)

        # Replace invalid sorted tail with zeros before multiplying.
        sorted_wc = torch.where(
            torch.isfinite(sorted_wc),
            sorted_wc,
            torch.zeros_like(sorted_wc),
        )

        return (sorted_wc * w).sum(dim=1)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        eps: float = 1e-6,
        lambda_reg: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute dependency-aware loss.

        Args:
            logits:
                Tensor of shape [..., V].

            labels:
                Tensor of shape [..., V] with values in {-1, 0, 1}.

            eps:
                Numerical stability constant.

            lambda_reg:
                Optional override for the interaction regularisation strength.

        Returns:
            Scalar loss.
        """
        if logits.shape != labels.shape:
            raise ValueError(
                f"logits and labels must have the same shape, "
                f"got {logits.shape} vs {labels.shape}"
            )

        if logits.dim() < 2:
            raise ValueError(
                f"Expected logits/labels shape [..., V], got {logits.shape}"
            )

        V = logits.shape[-1]
        if V != self.vocab_size:
            raise ValueError(
                f"Expected final dimension vocab_size={self.vocab_size}, got {V}"
            )

        # Always compute the loss in float32.
        # This avoids overflow or rounding issues under mixed precision.
        x = logits.float().reshape(-1, V)
        y_raw = labels.reshape(-1, V)

        valid = y_raw != self.ignore_index
        row_active = valid.any(dim=1)

        # If there are no valid rows, return a differentiable zero.
        if not row_active.any():
            return logits.float().sum() * 0.0

        valid_f = valid.to(dtype=x.dtype)
        y01 = y_raw.clamp(min=0, max=1).to(dtype=x.dtype)

        # Correctness:
        # y=1 -> p
        # y=0 -> 1-p
        p = torch.sigmoid(x)
        c = 1.0 - torch.abs(p - y01)
        c = c * valid_f

        # Batch-level counts over valid entries.
        P = (y01 * valid_f).sum()
        N = ((1.0 - y01) * valid_f).sum()

        # Class weights.
        w_pos = N / (P + eps)
        w_neg = self.alpha_neg * P / (N + eps)

        wc = torch.where(y01 == 1.0, c * w_pos, c * w_neg)
        wc = wc * valid_f

        # Rank-weighted aggregation.
        C = self._rank_aggregate(wc, valid, eps=eps)

        # Label-space interaction.
        W = self._sym_zero_diag(self.W_raw).to(dtype=x.dtype, device=x.device)

        v = wc * valid_f
        Wv = (W @ v.transpose(0, 1)).transpose(0, 1)
        numer = (v * Wv).sum(dim=1)

        denom = valid.sum(dim=1).to(dtype=x.dtype).clamp_min(1.0)
        raw = numer / denom

        S = torch.sigmoid(raw)
        I = 1.0 - S

        lam = self.lambda_reg if lambda_reg is None else float(lambda_reg)

        L = 1.0 - C + lam * I

        # Average only over rows with at least one valid label.
        out = self.loss_scale * L[row_active].mean()

        return out

