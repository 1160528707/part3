from __future__ import annotations

"""Patient-adaptive graph-energy decoder for CLSL.

Drop-in replacement for the original GraphEnergyDecoder.  The class name and
forward signature are kept, but the graph is upgraded from a mostly global
prior+learned adjacency to a patient-adaptive interaction graph:

    A(x) = A_prior + A_global + A_adaptive(z, evidence).

The returned pairwise energy is therefore sample-specific, which strengthens the
methodological claim from "label correlation with a fixed graph" to
"patient-conditioned latent disease-state energy modeling under coarsened
observations".
"""

from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from .components import MLP, SimpleGNNLayer
from clsl_v2.data.schema import Schema


def build_prior_adjacency(schema: Schema, prior_edges: Optional[Dict[str, Any]]) -> torch.Tensor:
    k = schema.num_diseases
    adj = torch.zeros(k, k, dtype=torch.float32)
    disease_to_idx = {d: i for i, d in enumerate(schema.diseases)}
    if prior_edges:
        for src, dsts in prior_edges.items():
            if src not in disease_to_idx or not isinstance(dsts, dict):
                continue
            i = disease_to_idx[src]
            for dst, w in dsts.items():
                if dst in disease_to_idx:
                    j = disease_to_idx[dst]
                    wij = float(w)
                    adj[i, j] = max(adj[i, j].item(), wij)
                    adj[j, i] = max(adj[j, i].item(), wij)
    return adj.clamp_min(0.0)


class GraphEnergyDecoder(nn.Module):
    """Structured decoder with patient-adaptive pairwise disease interactions."""

    def __init__(
        self,
        schema: Schema,
        hidden_dim: int,
        latent_dim: int,
        gnn_layers: int,
        dropout: float,
        prior_edges: Optional[Dict[str, Any]],
        adaptive_rank: int = 8,
        max_global_edge: float = 0.35,
        max_adaptive_edge: float = 0.75,
        init_edge_logit: float = -4.0,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.k = schema.num_diseases
        self.hidden_dim = int(hidden_dim)
        self.adaptive_rank = int(adaptive_rank)
        self.max_global_edge = float(max_global_edge)
        self.max_adaptive_edge = float(max_adaptive_edge)

        self.disease_emb = nn.Parameter(torch.randn(self.k, hidden_dim) * 0.02)
        self.latent_to_node = nn.Linear(latent_dim, hidden_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        prior_adj = build_prior_adjacency(schema, prior_edges)
        self.register_buffer("prior_adj", prior_adj, persistent=True)
        self.edge_logits = nn.Parameter(torch.full((self.k, self.k), float(init_edge_logit)))
        self.edge_context = MLP(latent_dim, hidden_dim, adaptive_rank, dropout=dropout, layers=2)

        self.gnn = nn.ModuleList([SimpleGNNLayer(hidden_dim, dropout=dropout) for _ in range(gnn_layers)])
        self.unary_head = MLP(hidden_dim, hidden_dim, 1, dropout=dropout, layers=2)

        self.pair_q = nn.Linear(hidden_dim, adaptive_rank)
        self.pair_k = nn.Linear(hidden_dim, adaptive_rank)
        self.pair_context_bias = nn.Linear(adaptive_rank, self.k * self.k)
        self.pair_bias = nn.Parameter(torch.zeros(self.k, self.k))
        self.reliability_head = MLP(hidden_dim + 3, hidden_dim, 1, dropout=dropout, layers=2)

    def global_adjacency(self) -> torch.Tensor:
        sym_logits = 0.5 * (self.edge_logits + self.edge_logits.t())
        eye = torch.eye(self.k, device=sym_logits.device, dtype=sym_logits.dtype)
        learned = torch.sigmoid(sym_logits) * self.max_global_edge
        prior = self.prior_adj.to(sym_logits.device, sym_logits.dtype)
        a = (prior + learned) * (1.0 - eye)
        return a + eye * 0.1

    def _adaptive_pairwise(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        q = self.pair_q(h)
        k = self.pair_k(h)
        low_rank = torch.matmul(q, k.transpose(1, 2)) / max(self.adaptive_rank, 1) ** 0.5
        context = self.edge_context(z)
        context_bias = self.pair_context_bias(context).view(-1, self.k, self.k)
        base_bias = 0.5 * (self.pair_bias + self.pair_bias.t()).unsqueeze(0)
        pair = low_rank + context_bias + base_bias
        pair = 0.5 * (pair + pair.transpose(1, 2))
        eye = torch.eye(self.k, device=h.device, dtype=h.dtype).unsqueeze(0)
        return torch.tanh(pair) * self.max_adaptive_edge * (1.0 - eye)

    def forward(
        self,
        disease_evidence: torch.Tensor,
        z: torch.Tensor,
        effective_mask: torch.Tensor,
        attention: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b = disease_evidence.size(0)
        z_node = self.latent_to_node(z).unsqueeze(1).expand(b, self.k, -1)
        disease_emb = self.disease_emb.unsqueeze(0).expand(b, self.k, -1)
        h = self.node_proj(torch.cat([disease_evidence, z_node, disease_emb], dim=-1))

        adj = self.global_adjacency()
        for layer in self.gnn:
            h = layer(h, adj)

        unary = self.unary_head(h).squeeze(-1)
        adaptive_pair = self._adaptive_pairwise(h, z)
        # Prior edges modulate but do not force all pairwise energies to be positive.
        prior_signed = self.prior_adj.to(h.device, h.dtype).unsqueeze(0) * torch.tanh(adaptive_pair)
        pairwise = adaptive_pair + prior_signed

        coverage = effective_mask.mean(dim=1, keepdim=True).expand(b, self.k)
        # Normalize attention entropy by log(F), not by F.
        f = attention.size(-1)
        norm = torch.log(torch.tensor(float(f), device=h.device, dtype=h.dtype)).clamp_min(1.0)
        attn_entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=-1) / norm
        attention_on_observed = (attention * effective_mask.unsqueeze(1)).sum(dim=-1)
        rel_in = torch.cat([h, coverage.unsqueeze(-1), attn_entropy.unsqueeze(-1), attention_on_observed.unsqueeze(-1)], dim=-1)
        reliability = torch.sigmoid(self.reliability_head(rel_in).squeeze(-1))

        return {
            "node_repr": h,
            "unary": unary,
            "pairwise": pairwise,
            "reliability": reliability,
            "global_adjacency": adj,
            "adaptive_pairwise": adaptive_pair,
        }

    def graph_regularization(self) -> Dict[str, torch.Tensor]:
        """Regularizers used by the training patch.

        `sparse` encourages parsimonious disease edges; `prior_alignment` keeps
        learned global edges near clinical priors when priors are supplied.
        """
        adj = self.global_adjacency()
        eye = torch.eye(self.k, device=adj.device, dtype=adj.dtype)
        off = 1.0 - eye
        sparse = (adj * off).abs().mean()
        prior = self.prior_adj.to(adj.device, adj.dtype)
        prior_alignment = ((adj * off - prior * off) ** 2).mean()
        diag = ((adj * eye - 0.1 * eye) ** 2).mean()
        return {"graph_sparse": sparse, "graph_prior_alignment": prior_alignment, "graph_diag": diag}
