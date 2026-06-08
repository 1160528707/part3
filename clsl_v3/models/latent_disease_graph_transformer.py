from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .components import MLP, SimpleGNNLayer


def build_prior_adjacency(schema, prior_edges: Optional[Dict[str, Any]]) -> torch.Tensor:
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


def _symmetrize_offdiag(x: torch.Tensor) -> torch.Tensor:
    k = x.size(-1)
    if x.dim() == 2:
        eye = torch.eye(k, device=x.device, dtype=x.dtype)
        return 0.5 * (x + x.t()) * (1.0 - eye)
    eye = torch.eye(k, device=x.device, dtype=x.dtype).unsqueeze(0)
    return 0.5 * (x + x.transpose(1, 2)) * (1.0 - eye)


def _gumbel_sigmoid(logits: torch.Tensor, temperature: float = 0.5, hard: bool = False) -> torch.Tensor:
    u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
    g = torch.log(u) - torch.log1p(-u)
    y = torch.sigmoid((logits + g) / max(float(temperature), 1e-4))
    if hard:
        y_hard = (y > 0.5).to(y.dtype)
        y = y_hard.detach() - y.detach() + y
    return y


@dataclass
class GraphPosterior:
    edge_logits: torch.Tensor
    edge_probs: torch.Tensor
    edge_sample: torch.Tensor
    edge_kl: torch.Tensor
    edge_entropy: torch.Tensor


class LatentDiseaseGraphSampler(nn.Module):
    """Differentiable posterior q(A | patient disease states).

    This replaces a deterministic patient-adaptive adjacency with a latent graph
    posterior. Prior edges seed the graph, but non-prior edges can be discovered
    when supported by disease evidence. The KL term keeps the graph sparse and
    clinically grounded.
    """

    def __init__(
        self,
        num_diseases: int,
        hidden_dim: int,
        latent_dim: int,
        rank: int = 8,
        prior_logit_scale: float = 4.0,
        dropout: float = 0.1,
        temperature: float = 0.5,
        hard_edges: bool = False,
    ) -> None:
        super().__init__()
        self.k = int(num_diseases)
        self.rank = int(max(1, rank))
        self.temperature = float(temperature)
        self.hard_edges = bool(hard_edges)
        self.prior_logit_scale = float(prior_logit_scale)
        self.edge_q = nn.Linear(hidden_dim, self.rank)
        self.edge_k = nn.Linear(hidden_dim, self.rank)
        self.context = MLP(latent_dim, hidden_dim, self.k * self.k, dropout=dropout, layers=2)
        self.global_residual = nn.Parameter(torch.zeros(self.k, self.k))

    def forward(self, h: torch.Tensor, z_global: torch.Tensor, prior_adj: torch.Tensor) -> GraphPosterior:
        b, k, _ = h.shape
        prior_adj = prior_adj.to(h.device, h.dtype)
        q = self.edge_q(h)
        kk = self.edge_k(h)
        low_rank = torch.matmul(q, kk.transpose(1, 2)) / (self.rank ** 0.5)
        context = self.context(z_global).view(b, k, k)
        residual = _symmetrize_offdiag(self.global_residual).unsqueeze(0)
        prior_logits = (prior_adj * 2.0 - 1.0).unsqueeze(0) * self.prior_logit_scale
        logits = _symmetrize_offdiag(low_rank + context + residual + prior_logits)
        probs = torch.sigmoid(logits)
        sample = _gumbel_sigmoid(logits, temperature=self.temperature, hard=self.hard_edges) if self.training else probs
        sample = _symmetrize_offdiag(sample)
        probs = _symmetrize_offdiag(probs)
        eye = torch.eye(k, device=h.device, dtype=h.dtype).unsqueeze(0)
        sample = sample * (1.0 - eye)
        probs = probs * (1.0 - eye)
        prior_prob = (0.05 + 0.90 * prior_adj).clamp(1e-4, 1 - 1e-4).unsqueeze(0)
        qprob = probs.clamp(1e-5, 1 - 1e-5)
        kl = qprob * (qprob.log() - prior_prob.log()) + (1 - qprob) * ((1 - qprob).log() - (1 - prior_prob).log())
        kl = (kl * (1.0 - eye)).mean()
        entropy = -(qprob * qprob.log() + (1 - qprob) * (1 - qprob).log())
        entropy = (entropy * (1.0 - eye)).mean()
        return GraphPosterior(edge_logits=logits, edge_probs=probs, edge_sample=sample, edge_kl=kl, edge_entropy=entropy)


class GraphGPSDiseaseBlock(nn.Module):
    """Local clinical-prior message passing + global disease attention."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.local = SimpleGNNLayer(hidden_dim, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        local_h = self.local(h, adjacency)
        global_h, _ = self.attn(local_h, local_h, local_h, need_weights=False)
        h = self.norm1(local_h + self.dropout(global_h))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class LatentDiseaseGraphTransformerEnergyDecoder(nn.Module):
    """Graph-energy decoder with latent disease graph posterior and GPS blocks.

    Drop-in replacement for GraphEnergyDecoder.forward(...). New outputs include
    edge posterior probabilities, sampled edges, KL-to-clinical-prior and graph
    entropy. Pairwise energies are still compatible with label marginalization.
    """

    def __init__(
        self,
        schema,
        hidden_dim: int,
        latent_dim: int,
        gnn_layers: int,
        dropout: float,
        prior_edges: Optional[Dict[str, Any]],
        adaptive_rank: int = 8,
        num_heads: int = 4,
        edge_temperature: float = 0.5,
        hard_edges: bool = False,
        self_loop_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.k = schema.num_diseases
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.self_loop_weight = float(self_loop_weight)
        self.disease_emb = nn.Parameter(torch.randn(self.k, hidden_dim) * 0.02)
        self.latent_to_node = nn.Linear(latent_dim, hidden_dim)
        self.node_proj = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        prior_adj = build_prior_adjacency(schema, prior_edges)
        self.register_buffer("prior_adj", prior_adj, persistent=True)
        self.graph_sampler = LatentDiseaseGraphSampler(
            num_diseases=self.k,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            rank=adaptive_rank,
            dropout=dropout,
            temperature=edge_temperature,
            hard_edges=hard_edges,
        )
        self.blocks = nn.ModuleList([GraphGPSDiseaseBlock(hidden_dim, num_heads=num_heads, dropout=dropout) for _ in range(max(1, gnn_layers))])
        self.unary_head = MLP(hidden_dim, hidden_dim, 1, dropout=dropout, layers=2)
        self.pair_q = nn.Linear(hidden_dim, adaptive_rank)
        self.pair_k = nn.Linear(hidden_dim, adaptive_rank)
        self.pair_context_bias = nn.Linear(latent_dim, self.k * self.k)
        self.pair_bias = nn.Parameter(torch.zeros(self.k, self.k))
        self.reliability_head = MLP(hidden_dim + 3, hidden_dim, 1, dropout=dropout, layers=2)

    def _z_node_and_global(self, z: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.dim() == 3:
            if z.size(1) != self.k:
                raise ValueError(f"Expected z shape [B,{self.k},L], got {tuple(z.shape)}")
            return self.latent_to_node(z), z.mean(dim=1)
        if z.dim() == 2:
            return self.latent_to_node(z).unsqueeze(1).expand(batch_size, self.k, -1), z
        raise ValueError(f"z must be [B,L] or [B,K,L], got {tuple(z.shape)}")

    def _pairwise_energy(self, h: torch.Tensor, z_global: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        q = self.pair_q(h)
        k = self.pair_k(h)
        raw = torch.matmul(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5)
        raw = raw + self.pair_context_bias(z_global).view(-1, self.k, self.k)
        raw = raw + 0.5 * (self.pair_bias + self.pair_bias.t()).unsqueeze(0)
        raw = _symmetrize_offdiag(raw)
        eye = torch.eye(self.k, device=h.device, dtype=h.dtype).unsqueeze(0)
        return torch.tanh(raw) * adjacency * (1.0 - eye)

    def _decode_with_adjacency(self, h0: torch.Tensor, z_global: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = h0
        for block in self.blocks:
            h = block(h, adjacency)
        unary = self.unary_head(h).squeeze(-1)
        pairwise = self._pairwise_energy(h, z_global, adjacency)
        return h, unary, pairwise

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
        posterior = self.graph_sampler(h0, z_global, self.prior_adj)
        eye = torch.eye(self.k, device=h0.device, dtype=h0.dtype).unsqueeze(0)
        patient_adj = posterior.edge_sample + eye * self.self_loop_weight
        h, unary, pairwise = self._decode_with_adjacency(h0, z_global, patient_adj)

        coverage = effective_mask.mean(dim=1, keepdim=True).expand(b, self.k)
        f = attention.size(-1)
        norm = torch.log(torch.tensor(float(f), device=h.device, dtype=h.dtype)).clamp_min(1.0)
        attn_entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=-1) / norm
        attention_on_observed = (attention * effective_mask.unsqueeze(1)).sum(dim=-1)
        reliability = torch.sigmoid(
            self.reliability_head(torch.cat([h, coverage.unsqueeze(-1), attn_entropy.unsqueeze(-1), attention_on_observed.unsqueeze(-1)], dim=-1)).squeeze(-1)
        )
        global_adj = self.prior_adj.to(h.device, h.dtype) + posterior.edge_probs.mean(dim=0)
        global_adj = _symmetrize_offdiag(global_adj)
        return {
            "node_repr": h,
            "unary": unary,
            "pairwise": pairwise,
            "reliability": reliability,
            "global_adjacency": global_adj,
            "patient_adjacency": patient_adj,
            "latent_edge_probs": posterior.edge_probs,
            "latent_edge_sample": posterior.edge_sample,
            "edge_kl": posterior.edge_kl,
            "edge_entropy": posterior.edge_entropy,
            "adaptive_pairwise": pairwise,
        }

    @torch.no_grad()
    def edge_intervention_effect(
        self,
        disease_evidence: torch.Tensor,
        z: torch.Tensor,
        edge_index: tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Counterfactual effect of removing one disease edge on unary logits.

        This is for analysis/interpretability, not a differentiable training path.
        """
        b = disease_evidence.size(0)
        z_node, z_global = self._z_node_and_global(z, b)
        h0 = self.node_proj(torch.cat([disease_evidence, z_node, self.disease_emb.unsqueeze(0).expand(b, self.k, -1)], dim=-1))
        posterior = self.graph_sampler(h0, z_global, self.prior_adj)
        eye = torch.eye(self.k, device=h0.device, dtype=h0.dtype).unsqueeze(0)
        a_full = posterior.edge_probs + eye * self.self_loop_weight
        a_do = a_full.clone()
        i, j = edge_index
        a_do[:, i, j] = 0.0
        a_do[:, j, i] = 0.0
        _, unary_full, _ = self._decode_with_adjacency(h0, z_global, a_full)
        _, unary_do, _ = self._decode_with_adjacency(h0, z_global, a_do)
        return {"delta_unary": unary_full - unary_do, "full_unary": unary_full, "intervened_unary": unary_do}

    def graph_regularization(self) -> Dict[str, torch.Tensor]:
        prior = self.prior_adj
        eye = torch.eye(self.k, device=prior.device, dtype=prior.dtype)
        return {
            "graph_prior_density": (prior * (1 - eye)).mean(),
            "graph_prior_l2": (prior ** 2).mean(),
        }
