from __future__ import annotations

from typing import Dict, Any
import torch
from torch import nn

from .components import MLP, SimpleGNNLayer
from clsl_v2.data.schema import Schema


def build_prior_adjacency(schema: Schema, prior_edges: Dict[str, Any] | None) -> torch.Tensor:
    k = schema.num_diseases
    adj = torch.eye(k) * 0.05
    disease_to_idx = {d: i for i, d in enumerate(schema.diseases)}
    if prior_edges:
        for src, dsts in prior_edges.items():
            if src not in disease_to_idx:
                continue
            i = disease_to_idx[src]
            if isinstance(dsts, dict):
                items = dsts.items()
            else:
                items = []
            for dst, w in items:
                if dst in disease_to_idx:
                    j = disease_to_idx[dst]
                    adj[i, j] = float(w)
                    adj[j, i] = max(float(w), float(adj[j, i]))
    return adj


class GraphEnergyDecoder(nn.Module):
    """Structured multi-label decoder with exact 2^K energy enumeration.

    Outputs unary logits and pairwise interaction logits. A separate loss module performs
    label marginalization under partial labels.
    """

    def __init__(self, schema: Schema, hidden_dim: int, latent_dim: int, gnn_layers: int, dropout: float, prior_edges: Dict[str, Any] | None):
        super().__init__()
        self.schema = schema
        self.k = schema.num_diseases
        self.disease_emb = nn.Parameter(torch.randn(self.k, hidden_dim) * 0.02)
        self.latent_to_node = nn.Linear(latent_dim, hidden_dim)
        self.node_proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        prior_adj = build_prior_adjacency(schema, prior_edges)
        self.register_buffer("prior_adj", prior_adj, persistent=True)
        self.learned_adj = nn.Parameter(torch.zeros(self.k, self.k))
        self.gnn = nn.ModuleList([SimpleGNNLayer(hidden_dim, dropout=dropout) for _ in range(gnn_layers)])
        self.unary_head = MLP(hidden_dim, hidden_dim, 1, dropout=dropout, layers=2)
        self.pair_q = nn.Linear(hidden_dim, hidden_dim)
        self.pair_k = nn.Linear(hidden_dim, hidden_dim)
        self.pair_bias = nn.Parameter(torch.zeros(self.k, self.k))
        self.reliability_head = MLP(hidden_dim + 2, hidden_dim, 1, dropout=dropout, layers=2)

    def adjacency(self) -> torch.Tensor:
        a = self.prior_adj + torch.sigmoid((self.learned_adj + self.learned_adj.t()) / 2.0) * 0.25
        eye = torch.eye(self.k, device=a.device)
        return a * (1.0 - eye) + eye * 0.1

    def forward(self, disease_evidence: torch.Tensor, z: torch.Tensor, effective_mask: torch.Tensor, attention: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = disease_evidence.size(0)
        z_node = self.latent_to_node(z).unsqueeze(1).expand(b, self.k, -1)
        disease_emb = self.disease_emb.unsqueeze(0).expand(b, self.k, -1)
        h = self.node_proj(torch.cat([disease_evidence, z_node, disease_emb], dim=-1))
        adj = self.adjacency()
        for layer in self.gnn:
            h = layer(h, adj)
        unary = self.unary_head(h).squeeze(-1)
        q = self.pair_q(h)
        kk = self.pair_k(h)
        pairwise = torch.matmul(q, kk.transpose(1, 2)) / (h.size(-1) ** 0.5)
        pairwise = (pairwise + pairwise.transpose(1, 2)) / 2.0
        pairwise = pairwise + (self.pair_bias + self.pair_bias.t()).unsqueeze(0) / 2.0
        eye = torch.eye(self.k, device=h.device).unsqueeze(0)
        pairwise = pairwise * (1.0 - eye)

        # Evidence reliability per disease: high when attention concentrates on observed tokens and coverage is high.
        coverage = effective_mask.mean(dim=1, keepdim=True).expand(b, self.k)
        attn_entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=-1) / max(1.0, float(attention.size(-1)))
        rel_in = torch.cat([h, coverage.unsqueeze(-1), attn_entropy.unsqueeze(-1)], dim=-1)
        reliability = torch.sigmoid(self.reliability_head(rel_in).squeeze(-1))
        return {"node_repr": h, "unary": unary, "pairwise": pairwise, "reliability": reliability}
