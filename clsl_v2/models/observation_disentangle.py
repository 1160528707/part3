from __future__ import annotations

import torch
from torch import nn

from .components import MLP, grad_reverse


class ObservationPolicyHeads(nn.Module):
    """Observation-policy disentanglement heads.

    - obs_repr predicts feature observation masks and view/stage.
    - disease repr passes through gradient reversal before view prediction, encouraging
      disease representation to be less view/system-specific.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, num_features: int, num_views: int, dropout: float = 0.1):
        super().__init__()
        self.mask_decoder = MLP(latent_dim, hidden_dim, num_features, dropout=dropout, layers=2)
        self.obs_view_classifier = MLP(latent_dim, hidden_dim, num_views, dropout=dropout, layers=2)
        self.adv_view_classifier = MLP(latent_dim, hidden_dim, num_views, dropout=dropout, layers=2)

    def forward(self, z_disease: torch.Tensor, z_obs: torch.Tensor, grl_lambda: float = 1.0):
        return {
            "obs_mask_logits": self.mask_decoder(z_obs),
            "obs_view_logits": self.obs_view_classifier(z_obs),
            "adv_view_logits": self.adv_view_classifier(grad_reverse(z_disease, grl_lambda)),
        }
