from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .components import MLP


@dataclass
class PosteriorFlowDiagnostics:
    flow_mse: torch.Tensor
    endpoint_kl: torch.Tensor
    mean_t: torch.Tensor
    mean_gap: torch.Tensor


def _bernoulli_kl_logits(target_logits: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    target = torch.sigmoid(target_logits).clamp(1e-6, 1 - 1e-6)
    pred = torch.sigmoid(pred_logits).clamp(1e-6, 1 - 1e-6)
    return (target * (target.log() - pred.log()) + (1 - target) * ((1 - target).log() - (1 - pred).log())).mean()


class PosteriorRefinementFlow(nn.Module):
    """Evidence-conditioned flow matching from coarse to fine posteriors.

    Rather than forcing low-information predictions to directly imitate full-view
    predictions, this module learns a vector field in posterior-logit space:
        d u_t / dt = v_theta(u_t, t, r_coarse, gap),
    where u_0 is the coarse-view posterior logit and u_1 is the fine-view posterior
    logit. This upgrades lattice consistency from a static distillation penalty to
    an explicit posterior transport mechanism.
    """

    def __init__(
        self,
        num_labels: int,
        repr_dim: int,
        hidden_dim: int = 128,
        time_embed_dim: int = 16,
        dropout: float = 0.1,
        layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.repr_dim = int(repr_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.net = MLP(
            in_dim=self.num_labels + self.repr_dim + self.time_embed_dim + 1,
            hidden_dim=hidden_dim,
            out_dim=self.num_labels,
            dropout=dropout,
            layers=layers,
        )
        self.endpoint_head = MLP(
            in_dim=self.num_labels + self.repr_dim + self.time_embed_dim + 1,
            hidden_dim=hidden_dim,
            out_dim=self.num_labels,
            dropout=dropout,
            layers=layers,
        )

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = max(1, self.time_embed_dim // 2)
        freqs = torch.exp(torch.linspace(0, 4, half, device=t.device, dtype=t.dtype))
        x = t.view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if emb.size(-1) < self.time_embed_dim:
            emb = F.pad(emb, (0, self.time_embed_dim - emb.size(-1)))
        return emb[..., : self.time_embed_dim]

    def forward(self, posterior_logits_t: torch.Tensor, t: torch.Tensor, coarse_repr: torch.Tensor, view_gap: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = posterior_logits_t.size(0)
        if view_gap is None:
            view_gap = posterior_logits_t.new_ones(b, 1)
        if view_gap.dim() == 1:
            view_gap = view_gap.unsqueeze(-1)
        inp = torch.cat([posterior_logits_t, coarse_repr, self.time_embedding(t), view_gap.to(posterior_logits_t.dtype)], dim=-1)
        return self.net(inp)

    def endpoint_predict(self, posterior_logits_0: torch.Tensor, coarse_repr: torch.Tensor, view_gap: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = posterior_logits_0.size(0)
        t1 = posterior_logits_0.new_ones(b)
        if view_gap is None:
            view_gap = posterior_logits_0.new_ones(b, 1)
        if view_gap.dim() == 1:
            view_gap = view_gap.unsqueeze(-1)
        inp = torch.cat([posterior_logits_0, coarse_repr, self.time_embedding(t1), view_gap.to(posterior_logits_0.dtype)], dim=-1)
        return posterior_logits_0 + self.endpoint_head(inp)

    def flow_matching_loss(
        self,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        coarse_repr: torch.Tensor,
        view_gap: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, PosteriorFlowDiagnostics]:
        b = coarse_logits.size(0)
        t = torch.rand(b, device=coarse_logits.device, dtype=coarse_logits.dtype)
        # Linear probability path in logit space. This simple path is stable and
        # keeps the target vector field closed-form.
        u0 = coarse_logits.detach()
        u1 = fine_logits.detach()
        noise = torch.randn_like(u0) * 0.01
        u_t = (1.0 - t.view(b, 1)) * u0 + t.view(b, 1) * u1 + noise
        target_v = u1 - u0
        pred_v = self.forward(u_t, t, coarse_repr, view_gap=view_gap)
        flow_mse = F.mse_loss(pred_v, target_v)
        endpoint = self.endpoint_predict(coarse_logits, coarse_repr, view_gap=view_gap)
        endpoint_kl = _bernoulli_kl_logits(fine_logits.detach(), endpoint)
        loss = flow_mse + 0.25 * endpoint_kl
        if not return_diagnostics:
            return loss
        gap = torch.tensor(1.0, device=coarse_logits.device, dtype=coarse_logits.dtype) if view_gap is None else view_gap.float().mean()
        return loss, PosteriorFlowDiagnostics(
            flow_mse=flow_mse.detach(),
            endpoint_kl=endpoint_kl.detach(),
            mean_t=t.mean().detach(),
            mean_gap=gap.detach(),
        )

    @torch.no_grad()
    def transport(self, coarse_logits: torch.Tensor, coarse_repr: torch.Tensor, view_gap: Optional[torch.Tensor] = None, steps: int = 8) -> torch.Tensor:
        """Numerically transport coarse posterior logits toward a refined posterior."""
        u = coarse_logits.clone()
        b = u.size(0)
        dt = 1.0 / max(1, int(steps))
        for i in range(max(1, int(steps))):
            t = u.new_full((b,), (i + 0.5) * dt)
            u = u + dt * self.forward(u, t, coarse_repr, view_gap=view_gap)
        return u
