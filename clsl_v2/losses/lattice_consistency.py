from __future__ import annotations

import torch
import torch.nn.functional as F


def evidence_lattice_consistency(
    coarse_probs: torch.Tensor,
    fine_probs: torch.Tensor,
    coarse_entropy: torch.Tensor,
    fine_entropy: torch.Tensor,
    entropy_margin: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior consistency and entropy monotonicity between coarse and fine views.

    This is a practical proxy: full/fine posterior is used as a target for same-sample
    consistency, while allowing coarse view to remain more uncertain.
    """
    consistency = F.mse_loss(coarse_probs, fine_probs.detach())
    # Coarse entropy should be >= fine entropy + margin.
    monotonic = F.relu(fine_entropy.detach() + entropy_margin - coarse_entropy).mean()
    return consistency, monotonic
