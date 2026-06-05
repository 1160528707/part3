from __future__ import annotations

"""Sparse-prior, patient-adaptive graph-energy decoder for CLSL-v2.

The decoder consumes disease-specific evidence and disease-specific latent states
and returns unary/pairwise energies for exact multi-label marginalization.
Compared with the earlier prototype, adjacency is no longer initialized as a
nearly dense sigmoid(0) graph. It is a sparse prior plus a tiny learnable
residual, with patient-adaptive modulation constrained by that support.
"""

from typing import Any, Dict, Optional

import torch
from torch import nn

from .components import MLP, SimpleGNNLayer
from clsl_v2.data.schema import Schema


def build_prior_adjacency(schema: Schema, prior_edges: Optional[Dict[str, Any]]) -> torch.Tensor:
    """Build a symmetric nonnegative [K,K] prior adjacency from config."""
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
    eye = torch.eye(k, dtype=torch.float32)
    return (adj.clamp_min(0.0) * (1.0 - eye)).clamp_max(1.0)


class GraphEnergyDecoder(nn.Module):
    """Structured decoder with sparse-prior + patient-adaptive disease graph."""

    def __init__(
        self,
        schema: Schema,
        hidden_dim: int,
        latent_dim: int,
        gnn_layers: int,
        dropout: float,
        prior_edges: Optional[Dict[str, Any]],
        adaptive_rank: int = 8,
        max_global_edge: float = 0.25,
        max_adaptive_edge: float = 0.50,
        init_edge_logit: float = -6.0,
        self_loop_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.k = schema.num_diseases
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.adaptive_rank = int(max(1, adaptive_rank))
        self.max_global_edge = float(max_global_edge)
        self.max_adaptive_edge = float(max_adaptive_edge)
        self.self_loop_weight = float(self_loop_weight)

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

        self.edge_q = nn.Linear(hidden_dim, self.adaptive_rank)
        self.edge_k = nn.Linear(hidden_dim, self.adaptive_rank)
        self.edge_context = MLP(latent_dim, hidden_dim, self.k * self.k, dropout=dropout, layers=2)

        self.gnn = nn.ModuleList([SimpleGNNLayer(hidden_dim, dropout=dropout) for _ in range(max(0, gnn_layers))])
        self.unary_head = MLP(hidden_dim, hidden_dim, 1, dropout=dropout, layers=2)

        self.pair_q = nn.Linear(hidden_dim, self.adaptive_rank)
        self.pair_k = nn.Linear(hidden_dim, self.adaptive_rank)
        self.pair_context_bias = nn.Linear(latent_dim, self.k * self.k)
        self.pair_bias = nn.Parameter(torch.zeros(self.k, self.k))

        self.reliability_head = MLP(hidden_dim + 3, hidden_dim, 1, dropout=dropout, layers=2)

    def _symmetrize_offdiag(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            eye = torch.eye(self.k, device=x.device, dtype=x.dtype)
            return 0.5 * (x + x.t()) * (1.0 - eye)
        eye = torch.eye(self.k, device=x.device, dtype=x.dtype).unsqueeze(0)
        return 0.5 * (x + x.transpose(1, 2)) * (1.0 - eye)

    def learned_residual_adjacency(self) -> torch.Tensor:
        logits = 0.5 * (self.edge_logits + self.edge_logits.t())
        residual = torch.sigmoid(logits) * self.max_global_edge
        return self._symmetrize_offdiag(residual)

    def global_adjacency(self) -> torch.Tensor:
        """Sparse global graph: clinical prior + tiny learnable residual."""
        residual = self.learned_residual_adjacency()
        prior = self.prior_adj.to(residual.device, residual.dtype)
        eye = torch.eye(self.k, device=residual.device, dtype=residual.dtype)
        adj = self._symmetrize_offdiag(prior + residual)
        return adj + eye * self.self_loop_weight

    # Backward-friendly alias.
    def adjacency(self) -> torch.Tensor:
        return self.global_adjacency()

    def _z_node_and_global(self, z: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if z.dim() == 3:
            if z.size(1) != self.k:
                raise ValueError(f"Expected z shape [B,{self.k},L], got {tuple(z.shape)}")
            z_node = self.latent_to_node(z)
            z_global = z.mean(dim=1)
        elif z.dim() == 2:
            z_node = self.latent_to_node(z).unsqueeze(1).expand(batch_size, self.k, -1)
            z_global = z
        else:
            raise ValueError(f"z must be [B,L] or [B,K,L], got shape {tuple(z.shape)}")
        return z_node, z_global

    def patient_adjacency(self, h0: torch.Tensor, z_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return patient-specific adjacency and adaptive off-diagonal component."""
        b = h0.size(0)
        base = self.global_adjacency().to(h0.device, h0.dtype)
        eye = torch.eye(self.k, device=h0.device, dtype=h0.dtype)
        off = 1.0 - eye

        q = self.edge_q(h0)
        k = self.edge_k(h0)
        low_rank = torch.matmul(q, k.transpose(1, 2)) / (self.adaptive_rank**0.5)
        context = self.edge_context(z_global).view(b, self.k, self.k)
        logits = self._symmetrize_offdiag(low_rank + context)

        # Support is proportional to prior+residual adjacency. With residual
        # initialized near zero, non-prior edges start near absent rather than
        # dense. They can still be learned under graph_sparse regularization.
        support = (base * off).clamp_min(0.0)
        support = support / support.max().clamp_min(1.0)
        adaptive = torch.sigmoid(logits) * self.max_adaptive_edge * support.unsqueeze(0)
        adaptive = self._symmetrize_offdiag(adaptive)
        patient_adj = self._symmetrize_offdiag(base.unsqueeze(0) + adaptive)
        patient_adj = patient_adj + eye.unsqueeze(0) * self.self_loop_weight
        return patient_adj, adaptive

    def _pairwise_energy(self, h: torch.Tensor, z_global: torch.Tensor, patient_adj: torch.Tensor) -> torch.Tensor:
        q = self.pair_q(h)
        k = self.pair_k(h)
        low_rank = torch.matmul(q, k.transpose(1, 2)) / (self.adaptive_rank**0.5)
        context_bias = self.pair_context_bias(z_global).view(-1, self.k, self.k)
        base_bias = 0.5 * (self.pair_bias + self.pair_bias.t()).unsqueeze(0)
        raw = self._symmetrize_offdiag(low_rank + context_bias + base_bias)
        eye = torch.eye(self.k, device=h.device, dtype=h.dtype).unsqueeze(0)
        edge_strength = patient_adj * (1.0 - eye)
        return torch.tanh(raw) * edge_strength

    def forward(
        self,
        disease_evidence: torch.Tensor,
        z: torch.Tensor,
        effective_mask: torch.Tensor,
        attention: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        b = disease_evidence.size(0)
        z_node, z_global = self._z_node_and_global(z, b)
        disease_emb = self.disease_emb.unsqueeze(0).expand(b, self.k, -1)
        h0 = self.node_proj(torch.cat([disease_evidence, z_node, disease_emb], dim=-1))

        patient_adj, adaptive_adj = self.patient_adjacency(h0, z_global)
        h = h0
        for layer in self.gnn:
            h = layer(h, patient_adj)

        unary = self.unary_head(h).squeeze(-1)
        pairwise = self._pairwise_energy(h, z_global, patient_adj)

        coverage = effective_mask.mean(dim=1, keepdim=True).expand(b, self.k)
        f = attention.size(-1)
        norm = torch.log(torch.tensor(float(f), device=h.device, dtype=h.dtype)).clamp_min(1.0)
        attn_entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=-1) / norm
        attention_on_observed = (attention * effective_mask.unsqueeze(1)).sum(dim=-1)
        rel_in = torch.cat(
            [h, coverage.unsqueeze(-1), attn_entropy.unsqueeze(-1), attention_on_observed.unsqueeze(-1)],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_head(rel_in).squeeze(-1))

        return {
            "node_repr": h,
            "unary": unary,
            "pairwise": pairwise,
            "reliability": reliability,
            "global_adjacency": self.global_adjacency(),
            "patient_adjacency": patient_adj,
            "adaptive_adjacency": adaptive_adj,
            "adaptive_pairwise": pairwise,
        }

    def graph_regularization(self) -> Dict[str, torch.Tensor]:
        adj = self.global_adjacency()
        eye = torch.eye(self.k, device=adj.device, dtype=adj.dtype)
        off = 1.0 - eye
        prior = self.prior_adj.to(adj.device, adj.dtype)
        residual = self.learned_residual_adjacency()

        sparse = (adj * off).abs().mean()
        residual_sparse = (residual * off).abs().mean()
        prior_alignment = ((adj * off - prior * off) ** 2).mean()
        diag = ((adj * eye - self.self_loop_weight * eye) ** 2).mean()
        return {
            "graph_sparse": sparse,
            "graph_residual_sparse": residual_sparse,
            "graph_prior_alignment": prior_alignment,
            "graph_diag": diag,
        }
