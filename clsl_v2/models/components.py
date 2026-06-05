from __future__ import annotations

import math
import torch
from torch import nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
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
        d = in_dim
        for _ in range(layers - 1):
            blocks += [nn.Linear(d, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)]
            d = hidden_dim
        blocks.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureTokenizer(nn.Module):
    """Tokenizes fixed tabular features with explicit observation semantics.

    Missing type IDs:
      0 observed
      1 randomly_missing
      2 not_measured
      3 not_yet_available
      4 structurally_unavailable
    """

    OBSERVED = 0
    RANDOM_MISSING = 1
    NOT_MEASURED = 2
    NOT_YET_AVAILABLE = 3
    STRUCT_UNAVAILABLE = 4

    def __init__(self, num_features: int, num_modalities: int, num_stages: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.value_proj = nn.Linear(1, hidden_dim)
        self.feature_emb = nn.Embedding(num_features, hidden_dim)
        self.modality_emb = nn.Embedding(num_modalities, hidden_dim)
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
        v = self.value_proj(x.unsqueeze(-1))
        tok = (
            v
            + self.feature_emb(feature_ids)
            + self.modality_emb(modality_idx.view(1, f).expand(b, f))
            + self.mask_emb(mask_ids)
            + self.missing_type_emb(missing_type.long().clamp(0, 4))
            + self.stage_emb(stage_ids.long())
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
        # h: [B,K,D], adjacency: [K,K] or [B,K,K]
        if adjacency.dim() == 2:
            adj = adjacency.unsqueeze(0).expand(h.size(0), -1, -1)
        else:
            adj = adjacency
        denom = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        neigh = torch.bmm(adj / denom, h)
        out = self.self_proj(h) + self.neigh_proj(neigh)
        return self.norm(h + self.dropout(torch.relu(out)))
