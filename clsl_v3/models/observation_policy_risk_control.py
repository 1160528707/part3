from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .components import MLP


@dataclass
class ObservationPolicyDiagnostics:
    mask_bce: torch.Tensor
    propensity_min: torch.Tensor
    propensity_mean: torch.Tensor


class ObservationPropensityModel(nn.Module):
    """Explicit model of selective clinical observation p(mask | latent state).

    It turns the old adversarial observation head into a usable propensity model
    that can reweight losses under selective measurement and analyze which feature
    groups are conditionally observed.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, num_features: int, num_views: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.num_views = int(num_views)
        self.mask_head = MLP(latent_dim + num_views, hidden_dim, num_features, dropout=dropout, layers=3)

    def forward(self, z_global: torch.Tensor, view_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        view_onehot = F.one_hot(view_idx.long().clamp_min(0), num_classes=self.num_views).to(z_global.dtype)
        logits = self.mask_head(torch.cat([z_global, view_onehot], dim=-1))
        propensity = torch.sigmoid(logits).clamp(1e-3, 1 - 1e-3)
        return {"feature_propensity_logits": logits, "feature_propensity": propensity}

    def loss(self, logits: torch.Tensor, x_mask: torch.Tensor, return_diagnostics: bool = False):
        mask_bce = F.binary_cross_entropy_with_logits(logits, x_mask.float())
        if not return_diagnostics:
            return mask_bce
        prop = torch.sigmoid(logits).clamp(1e-3, 1 - 1e-3)
        return mask_bce, ObservationPolicyDiagnostics(mask_bce=mask_bce.detach(), propensity_min=prop.min().detach(), propensity_mean=prop.mean().detach())

    @staticmethod
    def inverse_propensity_weights(propensity: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        return (1.0 / propensity.clamp_min(1e-3)).clamp_max(float(clip))


class ViewConditionalRiskController:
    """Split-conformal style view-conditional thresholding for multi-label risk.

    This is intentionally framework-light: call calibrate(...) on validation logits
    and masks, then apply(...) on test probabilities. It controls a simple missed
    positive label risk proxy within each view group.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = float(alpha)
        self.thresholds: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _scores(probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        labels = torch.nan_to_num(labels.float(), nan=0.0)
        # For positive observed labels, a miss occurs when threshold > probability.
        positive = (labels > 0.5) & (mask > 0.5)
        return (1.0 - probs).masked_fill(~positive, float("nan"))

    def calibrate(self, probs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, view_idx: torch.Tensor) -> Dict[int, torch.Tensor]:
        scores = self._scores(probs.detach(), labels, mask)
        self.thresholds = {}
        for v in torch.unique(view_idx.detach().cpu()).tolist():
            idx = view_idx.detach().cpu() == int(v)
            s = scores[idx.to(scores.device)].reshape(-1)
            s = s[torch.isfinite(s)]
            if s.numel() == 0:
                thr = torch.tensor(0.5, device=probs.device, dtype=probs.dtype)
            else:
                q = torch.quantile(s, min(1.0, max(0.0, 1.0 - self.alpha)))
                thr = 1.0 - q
            self.thresholds[int(v)] = thr.detach()
        return self.thresholds

    def apply(self, probs: torch.Tensor, view_idx: torch.Tensor, default_threshold: float = 0.5) -> torch.Tensor:
        out = torch.zeros_like(probs, dtype=torch.bool)
        for i in range(probs.size(0)):
            v = int(view_idx[i].detach().cpu())
            thr = self.thresholds.get(v, torch.tensor(default_threshold, device=probs.device, dtype=probs.dtype)).to(probs.device, probs.dtype)
            out[i] = probs[i] >= thr
        return out
