from __future__ import annotations

import torch


def latent_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def transition_smoothness(z_current: torch.Tensor, z_future: torch.Tensor) -> torch.Tensor:
    return torch.mean((z_future - z_current) ** 2)
