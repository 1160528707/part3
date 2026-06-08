from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from clsl_v3.models.components import MLP


@dataclass
class LabelPosteriorDiagnostics:
    nll: torch.Tensor
    mode: str
    observed_per_sample: torch.Tensor
    marginal_probs: torch.Tensor
    joint_entropy: torch.Tensor
    tb_loss: torch.Tensor


def energy_of_labels(unary: torch.Tensor, pairwise: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    upper = torch.triu(torch.ones(pairwise.size(-1), pairwise.size(-1), device=pairwise.device, dtype=pairwise.dtype), diagonal=1)
    unary_e = (unary * y).sum(dim=-1)
    pair_e = torch.einsum("bij,bi,bj->b", pairwise * upper.unsqueeze(0), y, y)
    return unary_e + pair_e


class ExactLabelMarginalizer(nn.Module):
    def __init__(self, num_labels: int, max_exact_labels: int = 20):
        super().__init__()
        self.num_labels = int(num_labels)
        self.max_exact_labels = int(max_exact_labels)
        if self.num_labels > self.max_exact_labels:
            raise ValueError(f"Exact state enumeration requires 2^K states; got K={self.num_labels}")
        states = []
        for s in range(2 ** self.num_labels):
            states.append([(s >> i) & 1 for i in range(self.num_labels)])
        self.register_buffer("states", torch.tensor(states, dtype=torch.float32), persistent=False)
        self.register_buffer("upper_mask", torch.triu(torch.ones(self.num_labels, self.num_labels), diagonal=1), persistent=False)

    def _states(self, device: torch.device) -> torch.Tensor:
        return self.states.to(device=device)

    def log_probs(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        states = self._states(unary.device)
        upper = self.upper_mask.to(unary.device, pairwise.dtype)
        unary_energy = torch.einsum("bk,sk->bs", unary, states)
        yy = states[:, :, None] * states[:, None, :]
        pair_energy = torch.einsum("bij,sij->bs", pairwise * upper.unsqueeze(0), yy)
        energy = unary_energy + pair_energy
        return energy - torch.logsumexp(energy, dim=1, keepdim=True)

    def marginal_probs(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        lp = self.log_probs(unary, pairwise)
        return lp.exp() @ self._states(unary.device)

    def entropy(self, unary: torch.Tensor, pairwise: torch.Tensor) -> torch.Tensor:
        lp = self.log_probs(unary, pairwise)
        p = lp.exp()
        return -(p * lp).sum(dim=1)

    def forward(self, unary: torch.Tensor, pairwise: torch.Tensor, labels: torch.Tensor, label_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lp = self.log_probs(unary, pairwise)
        states = self._states(unary.device)
        labels = torch.nan_to_num(labels.float(), nan=0.0)
        label_mask = label_mask.float()
        diff = (states.unsqueeze(0) - labels.unsqueeze(1)).abs() * label_mask.unsqueeze(1)
        consistent = diff.sum(dim=-1) < 0.5
        masked_lp = lp.masked_fill(~consistent, torch.finfo(lp.dtype).min / 4)
        log_marginal = torch.logsumexp(masked_lp, dim=1)
        nll = -log_marginal
        observed_any = (label_mask.sum(dim=1) > 0).float()
        loss = (nll * observed_any).sum() / observed_any.sum().clamp_min(1.0) if observed_any.sum() > 0 else unary.sum() * 0.0
        return loss, self.marginal_probs(unary, pairwise), self.entropy(unary, pairwise)


class LabelSetGFlowNet(nn.Module):
    """Sequential GFlowNet-style proposal over hidden disease-label sets.

    It samples label configurations consistent with observed labels and assigns
    trajectory log-probabilities. A trajectory-balance loss trains the proposal to
    allocate probability mass proportional to exp(graph_energy(y)). This makes the
    hidden-label posterior scalable beyond the exact 2^K regime.
    """

    def __init__(self, num_labels: int, disease_repr_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.k = int(num_labels)
        self.policy = MLP(disease_repr_dim + 3, hidden_dim, 1, dropout=dropout, layers=3)
        self.log_z = nn.Parameter(torch.tensor(0.0))

    def sample(
        self,
        disease_repr: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        num_samples: int = 16,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k, h = disease_repr.shape
        labels = torch.nan_to_num(labels.float(), nan=0.0)
        label_mask = label_mask.float()
        repr_rep = disease_repr.unsqueeze(1).expand(b, num_samples, k, h).reshape(b * num_samples, k, h)
        lab_rep = labels.unsqueeze(1).expand(b, num_samples, k).reshape(b * num_samples, k)
        mask_rep = label_mask.unsqueeze(1).expand(b, num_samples, k).reshape(b * num_samples, k)
        y = torch.zeros(b * num_samples, k, device=disease_repr.device, dtype=disease_repr.dtype)
        log_q = torch.zeros(b * num_samples, device=disease_repr.device, dtype=disease_repr.dtype)
        for t in range(k):
            observed_flag = mask_rep[:, t : t + 1]
            prev_density = y[:, :t].mean(dim=1, keepdim=True) if t > 0 else y.new_zeros(y.size(0), 1)
            step_frac = y.new_full((y.size(0), 1), float(t) / max(1, k - 1))
            inp = torch.cat([repr_rep[:, t, :], observed_flag, prev_density, step_frac], dim=-1)
            logit = self.policy(inp).squeeze(-1) / max(float(temperature), 1e-4)
            prob = torch.sigmoid(logit).clamp(1e-5, 1 - 1e-5)
            sampled = torch.bernoulli(prob)
            value = torch.where(observed_flag.squeeze(-1) > 0.5, lab_rep[:, t], sampled)
            # Observed labels are constraints, not proposal decisions.
            step_log_q = value * prob.log() + (1.0 - value) * (1.0 - prob).log()
            log_q = log_q + step_log_q * (1.0 - observed_flag.squeeze(-1))
            y[:, t] = value
        return y.view(b, num_samples, k), log_q.view(b, num_samples)

    def trajectory_balance_loss(self, log_q: torch.Tensor, log_reward: torch.Tensor) -> torch.Tensor:
        return ((self.log_z + log_q - log_reward.detach()) ** 2).mean()


class ScalableLabelMarginalizationLoss(nn.Module):
    """Exact-smallK + GFlowNet/importance-estimated largeK marginal NLL.

    For K <= max_exact_labels, this is an exact structured marginal likelihood.
    For K > max_exact_labels, it samples hidden labels from LabelSetGFlowNet and
    estimates log p(y_obs | x) by self-normalized importance over configurations
    consistent with observed labels.
    """

    def __init__(
        self,
        num_labels: int,
        disease_repr_dim: int,
        max_exact_labels: int = 20,
        num_samples: int = 32,
        gfn_hidden_dim: int = 128,
        tb_weight: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.max_exact_labels = int(max_exact_labels)
        self.num_samples = int(num_samples)
        self.tb_weight = float(tb_weight)
        self.exact = ExactLabelMarginalizer(num_labels, max_exact_labels) if num_labels <= max_exact_labels else None
        self.gflownet = LabelSetGFlowNet(num_labels, disease_repr_dim, hidden_dim=gfn_hidden_dim, dropout=dropout)

    def _approximate(
        self,
        unary: torch.Tensor,
        pairwise: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        disease_repr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, k = unary.shape
        y_samples, log_q = self.gflownet.sample(disease_repr, labels, label_mask, num_samples=self.num_samples)
        unary_rep = unary.unsqueeze(1).expand(b, self.num_samples, k).reshape(b * self.num_samples, k)
        pair_rep = pairwise.unsqueeze(1).expand(b, self.num_samples, k, k).reshape(b * self.num_samples, k, k)
        y_flat = y_samples.reshape(b * self.num_samples, k)
        log_reward = energy_of_labels(unary_rep, pair_rep, y_flat).view(b, self.num_samples)
        log_w = log_reward - log_q
        log_marginal = torch.logsumexp(log_w, dim=1) - torch.log(torch.tensor(float(self.num_samples), device=unary.device, dtype=unary.dtype))
        nll = -log_marginal
        observed_any = (label_mask.float().sum(dim=1) > 0).float()
        nll_loss = (nll * observed_any).sum() / observed_any.sum().clamp_min(1.0) if observed_any.sum() > 0 else unary.sum() * 0.0
        tb = self.gflownet.trajectory_balance_loss(log_q, log_reward)
        probs = y_samples.mean(dim=1)
        entropy = -(probs.clamp_min(1e-6).log() * probs + (1 - probs).clamp_min(1e-6).log() * (1 - probs)).sum(dim=1)
        return nll_loss + self.tb_weight * tb, probs, entropy, tb

    def forward(
        self,
        unary: torch.Tensor,
        pairwise: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        disease_repr: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LabelPosteriorDiagnostics]:
        if self.exact is not None:
            loss, probs, entropy = self.exact(unary, pairwise, labels, label_mask)
            tb = unary.sum() * 0.0
            mode = "exact"
        else:
            if disease_repr is None:
                raise ValueError("disease_repr is required when K > max_exact_labels for scalable hidden-label inference")
            loss, probs, entropy, tb = self._approximate(unary, pairwise, labels, label_mask, disease_repr)
            mode = "gflownet_importance"
        if not return_diagnostics:
            return loss
        labels = torch.nan_to_num(labels.float(), nan=0.0)
        return loss, LabelPosteriorDiagnostics(
            nll=loss.detach(),
            mode=mode,
            observed_per_sample=label_mask.float().sum(dim=1).detach(),
            marginal_probs=probs.detach(),
            joint_entropy=entropy.detach(),
            tb_loss=tb.detach(),
        )


def brier_loss_from_marginals(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    labels = torch.nan_to_num(labels.float(), nan=0.0)
    mask = mask.float()
    return (((probs - labels) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
