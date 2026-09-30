from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedJointBCELoss(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        channel_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("channel_weight", channel_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        losses = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if self.channel_weight is not None:
            losses = losses * self.channel_weight.view(1, 1, 2)

        mask = (
            torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        ).unsqueeze(-1)

        mask = mask.to(losses.dtype)
        denominator = mask.sum() * logits.shape[-1]
        return (losses * mask).sum() / denominator.clamp_min(1.0)
