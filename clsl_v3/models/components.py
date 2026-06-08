from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambd)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1, layers: int = 2):
        super().__init__()
        if layers <= 1:
            self.net = nn.Linear(in_dim, out_dim)
            return
        blocks = []
        d = int(in_dim)
        for _ in range(int(layers) - 1):
            blocks += [nn.Linear(d, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)]
            d = hidden_dim
        blocks.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureTokenizer(nn.Module):
    """Tokenizes tabular EHR features with explicit observation semantics."""

    OBSERVED = 0
    RANDOM_MISSING = 1
    NOT_MEASURED = 2
    NOT_YET_AVAILABLE = 3
    STRUCT_UNAVAILABLE = 4

    def __init__(self, num_features: int, num_modalities: int, num_stages: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.feature_emb = nn.Embedding(num_features, hidden_dim)
        self.modality_emb = nn.Embedding(max(1, num_modalities), hidden_dim)
        self.mask_emb = nn.Embedding(2, hidden_dim)
        self.missing_type_emb = nn.Embedding(5, hidden_dim)
        self.stage_emb = nn.Embedding(max(1, num_stages), hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        effective_mask: torch.Tensor,
        modality_idx: torch.Tensor,
        missing_type: torch.Tensor,
        stage_idx: torch.Tensor,
    ) -> torch.Tensor:
        b, f = x.shape
        device = x.device
        feature_ids = torch.arange(f, device=device).unsqueeze(0).expand(b, f)
        stage_ids = stage_idx.view(b, 1).expand(b, f)
        mask_ids = effective_mask.long().clamp(0, 1)
        tok = (
            self.value_proj(x.unsqueeze(-1))
            + self.feature_emb(feature_ids)
            + self.modality_emb(modality_idx.to(device).view(1, f).expand(b, f).long())
            + self.mask_emb(mask_ids)
            + self.missing_type_emb(missing_type.long().clamp(0, 4))
            + self.stage_emb(stage_ids.long().clamp_min(0))
        )
        return self.dropout(self.norm(tok))


class SimpleGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neigh_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.dim() == 2:
            adj = adjacency.unsqueeze(0).expand(h.size(0), -1, -1)
        else:
            adj = adjacency
        denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        neigh = torch.bmm(adj / denom, h)
        out = self.self_proj(h) + self.neigh_proj(neigh)
        return self.norm(h + self.dropout(torch.relu(out)))


class LightweightSetEncoder(nn.Module):
    """Permutation-aware set encoder used as the fast clinical evidence backbone."""

    def __init__(self, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(max(1, int(n_layers)))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.layers])

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = tokens
        w = mask.unsqueeze(-1).float() + 1e-3
        for layer, norm in zip(self.layers, self.norms):
            g = (h * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            h = norm(h + layer(torch.cat([h, g.expand_as(h)], dim=-1)))
        return h
