from __future__ import annotations

"""Exact and diagnostically rich label-marginalized graph-energy loss.

Drop-in replacement for the original `LabelMarginalizationLoss`.  It keeps the
same public methods (`log_probs`, `marginal_probs`, `entropy`, `forward`) while
adding:

1. numerically safer NaN handling;
2. pairwise marginal computation for hidden-label recovery analysis;
3. an explicit independent/masked-BCE equivalence helper;
4. a configurable guard for K that is too large for exact 2^K enumeration.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MarginalNLLDiagnostics:
    nll: torch.Tensor
    observed_per_sample: torch.Tensor
    marginal_probs: torch.Tensor
    joint_entropy: torch.Tensor


class LabelMarginalizationLoss(nn.Module):
    """Exact marginal NLL for partially observed binary multi-label states.

    For pairwise energy
        E_x(y)=sum_k unary_k y_k + sum_{k<l} pairwise_kl y_k y_l,
    this loss computes
        -log sum_{y_miss} p_x(y_obs, y_miss)
    exactly by enumerating all states.  When `pairwise=0`, this is exactly the
    sample-mean masked BCE over observed labels, which is useful in the paper as
    a formal degeneration result.
    """

    def __init__(self, num_labels: int, max_exact_labels: int = 20, eps: float = 1e-8):
        super().__init__()
        self.num_labels = int(num_labels)
        self.max_exact_labels = int(max_exact_labels)
        self.eps = float(eps)
        if self.num_labels > self.max_exact_labels:
            raise ValueError(
                f"Exact state enumeration requires 2^K states; got K={self.num_labels}. "
                f"Increase max_exact_labels only if memory allows, or implement a variational decoder."
            )
        states = []
        for s in range(2 ** self.num_labels):
            states.append([(s >> i) & 1 for i in range(self.num_labels)])
        self.register_buffer("states", torch.tensor(states, dtype=torch.float32), persistent=False)
        self.register_buffer("upper_mask", torch.triu(torch.ones(self.num_labels, self.num_labels), diagonal=1), persistent=False)

    def _states(self, device: torch.device) -> torch.Tensor:
        return self.states.to(device=device)

    def log_probs(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        states = self._states(unary.device)  # [S,K]
        upper = self.upper_mask.to(device=unary.device, dtype=pairwise.dtype)
        unary_energy = torch.einsum("bk,sk->bs", unary, states)
        yy = states[:, :, None] * states[:, None, :]
        pair_energy = torch.einsum("bij,sij->bs", pairwise * upper.unsqueeze(0), yy)
        energy = unary_energy + pair_energy
        return energy - torch.logsumexp(energy, dim=1, keepdim=True)

    def marginal_probs(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        lp = self.log_probs(unary, pairwise)
        return lp.exp() @ self._states(unary.device)

    def pairwise_marginals(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        """Return E[Y_i Y_j | X] for all label pairs, shape [B,K,K]."""
        states = self._states(unary.device)
        lp = self.log_probs(unary, pairwise)
        probs = lp.exp()
        yy = states[:, :, None] * states[:, None, :]
        return torch.einsum("bs,sij->bij", probs, yy)

    def entropy(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        lp = self.log_probs(unary, pairwise)
        p = lp.exp()
        return -(p * lp).sum(dim=1)

    def masked_bce_equivalent(self, unary: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
        """Sample-mean masked BCE used for the pairwise=0 degeneration check."""
        labels = torch.nan_to_num(labels.float(), nan=0.0)
        mask = label_mask.float()
        loss_per_label = F.binary_cross_entropy_with_logits(unary, labels, reduction="none") * mask
        observed_any = (mask.sum(dim=1) > 0).float()
        if observed_any.sum() < 1:
            return unary.sum() * 0.0
        return (loss_per_label.sum(dim=1) * observed_any).sum() / observed_any.sum().clamp_min(1.0)

    def forward(
        self,
        unary: torch.Tensor,
        pairwise: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MarginalNLLDiagnostics]:
        lp = self.log_probs(unary, pairwise)  # [B,S]
        states = self._states(unary.device)
        labels = torch.nan_to_num(labels.float(), nan=0.0)
        label_mask = label_mask.float()
        diff = (states.unsqueeze(0) - labels.unsqueeze(1)).abs() * label_mask.unsqueeze(1)
        consistent = diff.sum(dim=-1) < 0.5

        masked_lp = lp.masked_fill(~consistent, torch.finfo(lp.dtype).min / 4)
        log_marginal = torch.logsumexp(masked_lp, dim=1)
        nll = -log_marginal
        observed_any = (label_mask.sum(dim=1) > 0).float()
        if observed_any.sum() < 1:
            loss = unary.sum() * 0.0
        else:
            loss = (nll * observed_any).sum() / observed_any.sum().clamp_min(1.0)
        if not return_diagnostics:
            return loss
        return loss, MarginalNLLDiagnostics(
            nll=nll.detach(),
            observed_per_sample=label_mask.sum(dim=1).detach(),
            marginal_probs=self.marginal_probs(unary, pairwise).detach(),
            joint_entropy=self.entropy(unary, pairwise).detach(),
        )


def brier_loss_from_marginals(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    labels = torch.nan_to_num(labels.float(), nan=0.0)
    mask = mask.float()
    loss = ((probs - labels) ** 2) * mask
    return loss.sum() / mask.sum().clamp_min(1.0)
