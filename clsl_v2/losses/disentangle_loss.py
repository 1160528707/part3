from __future__ import annotations

import torch
import torch.nn.functional as F


def observation_mask_loss(mask_logits: torch.Tensor, effective_mask: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(mask_logits, effective_mask.float())


def view_classification_loss(view_logits: torch.Tensor, view_idx: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(view_logits, view_idx.long())
