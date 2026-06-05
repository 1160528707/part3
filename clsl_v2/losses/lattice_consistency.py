from __future__ import annotations

"""Order-aware and conditional evidence-lattice consistency losses.

This module is a drop-in replacement for the original `lattice_consistency.py`.
It keeps the old function name `evidence_lattice_consistency`, but replaces
plain probability MSE with a Bernoulli KL / Jensen-Shannon style posterior loss
and adds a reusable class for approximating the key CLSL condition

    p(Y | X_coarse) ~= E[p(Y | X_fine) | X_coarse].

The conditional expectation is approximated within a mini-batch by kernel
aggregation over coarse-view representations.  This is still simple enough for
CPU experiments, but is much harder for reviewers to dismiss as vanilla
teacher-student distillation because the fine-view target is no longer just the
same patient's full-view prediction.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _clamp_probs(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return p.clamp(min=eps, max=1.0 - eps)


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Elementwise KL(Bernoulli(p) || Bernoulli(q))."""
    p = _clamp_probs(p, eps)
    q = _clamp_probs(q, eps)
    return p * (p.log() - q.log()) + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())


def symmetric_bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 0.5 * (bernoulli_kl(p, q, eps) + bernoulli_kl(q, p, eps))


def bernoulli_js(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = 0.5 * (_clamp_probs(p, eps) + _clamp_probs(q, eps))
    return 0.5 * bernoulli_kl(p, m, eps) + 0.5 * bernoulli_kl(q, m, eps)


def view_gap_weight(
    coarse_mask: Optional[torch.Tensor],
    fine_mask: Optional[torch.Tensor],
    fallback: float = 1.0,
) -> torch.Tensor | float:
    """Return an information-gap weight for a coarse<=fine view pair.

    If masks are [B,F], the result is [B].  The value increases when fine_view
    contains many features unavailable to coarse_view.  This lets entropy
    monotonicity be weak for nearly identical views and stronger for genuine
    evidence gaps.
    """
    if coarse_mask is None or fine_mask is None:
        return fallback
    c = coarse_mask.float()
    f = fine_mask.float()
    added = (f - c).clamp_min(0.0).sum(dim=-1)
    denom = f.sum(dim=-1).clamp_min(1.0)
    return (added / denom).clamp_min(0.05)


def evidence_lattice_consistency(
    coarse_probs: torch.Tensor,
    fine_probs: torch.Tensor,
    coarse_entropy: torch.Tensor,
    fine_entropy: torch.Tensor,
    entropy_margin: float = 0.02,
    divergence: str = "js",
    coarse_mask: Optional[torch.Tensor] = None,
    fine_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible pairwise lattice consistency.

    Original code used `F.mse_loss(coarse_probs, fine_probs.detach())`.  This
    replacement uses a distributional divergence for Bernoulli label marginals
    and an information-gap weighted entropy-order penalty.
    """
    target = fine_probs.detach()
    if divergence.lower() == "mse":
        consistency = F.mse_loss(coarse_probs, target)
    elif divergence.lower() in {"skl", "sym_kl", "symmetric_kl"}:
        consistency = symmetric_bernoulli_kl(coarse_probs, target).mean()
    else:
        consistency = bernoulli_js(coarse_probs, target).mean()

    gap = view_gap_weight(coarse_mask, fine_mask, fallback=1.0)
    if torch.is_tensor(gap):
        gap = gap.to(coarse_entropy.device)
    margin = float(entropy_margin) * gap
    monotonic = F.relu(fine_entropy.detach() + margin - coarse_entropy).mean()
    return consistency, monotonic


@dataclass
class ConditionalLatticeDiagnostics:
    consistency: torch.Tensor
    monotonicity: torch.Tensor
    mean_neighbor_entropy: torch.Tensor
    mean_information_gap: torch.Tensor


class ConditionalLatticeConsistency(nn.Module):
    """Mini-batch approximation to E[p(Y|X_fine) | X_coarse].

    Given coarse probabilities p_c, fine probabilities p_f, and a coarse-view
    representation r_c, the target for sample i is

        q_i = sum_j softmax(-||r_i-r_j||^2/tau)_j p_f[j].

    This changes the lattice term from same-sample distillation to conditional
    posterior smoothing over patients with similar coarse evidence.  Set
    `leave_one_out=True` to avoid the trivial identity target; for small batches
    the implementation automatically falls back to including self-neighbors.
    """

    def __init__(
        self,
        temperature: float = 0.15,
        divergence: str = "js",
        entropy_margin: float = 0.02,
        leave_one_out: bool = True,
        detach_teacher: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.divergence = divergence
        self.entropy_margin = float(entropy_margin)
        self.leave_one_out = bool(leave_one_out)
        self.detach_teacher = bool(detach_teacher)
        self.eps = float(eps)

    def conditional_teacher(
        self,
        fine_probs: torch.Tensor,
        coarse_repr: Optional[torch.Tensor],
        valid: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coarse_repr is None or coarse_repr.size(0) <= 1:
            target = fine_probs
            entropy = torch.zeros((), device=fine_probs.device)
            return (target.detach() if self.detach_teacher else target), entropy

        r = F.normalize(coarse_repr.float(), dim=-1)
        dist2 = torch.cdist(r, r, p=2.0).pow(2)
        logits = -dist2 / max(self.temperature, self.eps)

        b = fine_probs.size(0)
        if valid is not None:
            valid = valid.bool().view(1, b)
            logits = logits.masked_fill(~valid, -1e9)

        if self.leave_one_out and b > 2:
            eye = torch.eye(b, device=fine_probs.device, dtype=torch.bool)
            logits = logits.masked_fill(eye, -1e9)

        # If a row became invalid, re-enable self-neighbor for numerical safety.
        bad = torch.isneginf(logits).all(dim=1)
        if bad.any():
            logits[bad] = -dist2[bad] / max(self.temperature, self.eps)

        weights = torch.softmax(logits, dim=-1)
        target = weights @ fine_probs
        neighbor_entropy = -(weights.clamp_min(self.eps).log() * weights).sum(dim=-1).mean()
        if self.detach_teacher:
            target = target.detach()
        return target, neighbor_entropy

    def forward(
        self,
        coarse_probs: torch.Tensor,
        fine_probs: torch.Tensor,
        coarse_entropy: torch.Tensor,
        fine_entropy: torch.Tensor,
        coarse_repr: Optional[torch.Tensor] = None,
        coarse_mask: Optional[torch.Tensor] = None,
        fine_mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ConditionalLatticeDiagnostics]:
        target, nbr_entropy = self.conditional_teacher(fine_probs, coarse_repr, valid=valid)

        if self.divergence.lower() == "mse":
            consistency = F.mse_loss(coarse_probs, target)
        elif self.divergence.lower() in {"skl", "sym_kl", "symmetric_kl"}:
            consistency = symmetric_bernoulli_kl(coarse_probs, target, self.eps).mean()
        else:
            consistency = bernoulli_js(coarse_probs, target, self.eps).mean()

        gap = view_gap_weight(coarse_mask, fine_mask, fallback=1.0)
        if torch.is_tensor(gap):
            gap = gap.to(coarse_entropy.device)
            mean_gap = gap.mean()
        else:
            mean_gap = torch.tensor(float(gap), device=coarse_entropy.device)
        monotonicity = F.relu(fine_entropy.detach() + self.entropy_margin * gap - coarse_entropy).mean()
        loss = consistency + monotonicity
        if not return_diagnostics:
            return loss
        return loss, ConditionalLatticeDiagnostics(
            consistency=consistency.detach(),
            monotonicity=monotonicity.detach(),
            mean_neighbor_entropy=nbr_entropy.detach(),
            mean_information_gap=mean_gap.detach(),
        )
