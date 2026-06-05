from __future__ import annotations

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


class LabelMarginalizationLoss(nn.Module):
    """Exact marginal negative log-likelihood for partially observed binary labels.

    For K=6 diseases, all 2^K states are enumerated. If a label is unobserved,
    it is summed out rather than treated as negative.
    """

    def __init__(self, num_labels: int):
        super().__init__()
        states = []
        for s in range(2 ** num_labels):
            bits = [(s >> i) & 1 for i in range(num_labels)]
            states.append(bits)
        self.register_buffer("states", torch.tensor(states, dtype=torch.float32), persistent=False)
        upper = torch.triu(torch.ones(num_labels, num_labels), diagonal=1)
        self.register_buffer("upper_mask", upper, persistent=False)

    def log_probs(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        states = self.states.to(unary.device)  # [S,K]
        upper = self.upper_mask.to(unary.device)
        unary_energy = torch.einsum("bk,sk->bs", unary, states)
        yy = states[:, :, None] * states[:, None, :]
        pair_energy = torch.einsum("bij,sij->bs", pairwise * upper.unsqueeze(0), yy)
        energy = unary_energy + pair_energy
        return energy - torch.logsumexp(energy, dim=1, keepdim=True)

    def marginal_probs(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        lp = self.log_probs(unary, pairwise)
        probs = lp.exp()
        return probs @ self.states.to(unary.device)

    def entropy(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        lp = self.log_probs(unary, pairwise)
        p = lp.exp()
        return -(p * lp).sum(dim=1)

    def forward(self, unary: torch.Tensor, pairwise: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> torch.Tensor:
        lp = self.log_probs(unary, pairwise)  # [B,S]
        states = self.states.to(unary.device)
        labels = labels.float()
        label_mask = label_mask.float()
        diff = (states.unsqueeze(0) - labels.unsqueeze(1)).abs() * label_mask.unsqueeze(1)
        consistent = diff.sum(dim=-1) < 0.5
        # When no labels observed, every state is consistent and NLL is 0.
        masked_lp = lp.masked_fill(~consistent, -1e9)
        log_marginal = torch.logsumexp(masked_lp, dim=1)
        nll = -log_marginal
        observed_any = (label_mask.sum(dim=1) > 0).float()
        if observed_any.sum() < 1:
            return unary.sum() * 0.0
        return (nll * observed_any).sum() / observed_any.sum().clamp_min(1.0)


def brier_loss_from_marginals(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = ((probs - labels.float()) ** 2) * mask.float()
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom
