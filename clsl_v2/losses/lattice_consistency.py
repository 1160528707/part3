from __future__ import annotations

"""Conditional evidence-lattice consistency losses for CLSL-v2.

The old prototype used same-sample MSE between coarse and fine marginal
probabilities. This file keeps the public function name but upgrades it to
Bernoulli KL/Jensen-Shannon divergence and adds a mini-batch approximation to

    p(Y | X_coarse) ~= E[p(Y | X_fine) | X_coarse].

The conditional expectation is approximated by kernel aggregation over coarse
representations, so the target is not merely the same patient's fine-view output.
"""

from dataclasses import dataclass
from typing import Optional

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
    """Information-gap weight for a coarse <= fine pair.

    Larger when fine_view contains many features unavailable to coarse_view.
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

    Returns (posterior_consistency, entropy_monotonicity).
    """
    target = fine_probs.detach()
    div = divergence.lower()
    if div == "mse":
        consistency = F.mse_loss(coarse_probs, target)
    elif div in {"skl", "sym_kl", "symmetric_kl"}:
        consistency = symmetric_bernoulli_kl(coarse_probs, target).mean()
    elif div in {"kl", "forward_kl"}:
        consistency = bernoulli_kl(target, coarse_probs).mean()
    else:
        consistency = bernoulli_js(coarse_probs, target).mean()

    gap = view_gap_weight(coarse_mask, fine_mask, fallback=1.0)
    if torch.is_tensor(gap):
        gap = gap.to(coarse_entropy.device, coarse_entropy.dtype)
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

    Given coarse probabilities p_c, fine probabilities p_f, and a coarse
    representation r_c, target q_i is

        q_i = sum_j softmax(-||r_i-r_j||^2 / tau)_j p_f[j].

    With leave_one_out=True, the same sample is removed when the batch size is
    large enough, reducing collapse into trivial same-patient distillation.
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
            entropy = torch.zeros((), device=fine_probs.device, dtype=fine_probs.dtype)
            return (target.detach() if self.detach_teacher else target), entropy

        r = F.normalize(coarse_repr.float(), dim=-1)
        dist2 = torch.cdist(r, r, p=2.0).pow(2)
        logits = -dist2 / max(self.temperature, self.eps)
        b = fine_probs.size(0)

        if valid is not None:
            valid_mask = valid.bool().view(1, b)
            logits = logits.masked_fill(~valid_mask, -1e9)

        if self.leave_one_out and b > 2:
            eye = torch.eye(b, device=fine_probs.device, dtype=torch.bool)
            logits = logits.masked_fill(eye, -1e9)
            bad = torch.isneginf(logits).all(dim=1)
            if bad.any():
                logits[bad] = -dist2[bad] / max(self.temperature, self.eps)

        weights = torch.softmax(logits, dim=-1).to(fine_probs.dtype)
        target = weights @ fine_probs
        neighbor_entropy = -(weights.clamp_min(self.eps).log() * weights).sum(dim=-1).mean()
        if self.detach_teacher:
            target = target.detach()
        return target, neighbor_entropy

    def _divergence(self, coarse_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        div = self.divergence.lower()
        if div == "mse":
            return F.mse_loss(coarse_probs, target)
        if div in {"skl", "sym_kl", "symmetric_kl"}:
            return symmetric_bernoulli_kl(coarse_probs, target, self.eps).mean()
        if div in {"kl", "forward_kl"}:
            return bernoulli_kl(target, coarse_probs, self.eps).mean()
        return bernoulli_js(coarse_probs, target, self.eps).mean()

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
        consistency = self._divergence(coarse_probs, target)

        gap = view_gap_weight(coarse_mask, fine_mask, fallback=1.0)
        if torch.is_tensor(gap):
            gap = gap.to(coarse_entropy.device, coarse_entropy.dtype)
            mean_gap = gap.mean()
        else:
            mean_gap = torch.tensor(float(gap), device=coarse_entropy.device, dtype=coarse_entropy.dtype)

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
