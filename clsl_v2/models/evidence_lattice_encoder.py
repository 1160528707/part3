from __future__ import annotations

import re
from typing import Dict, Any, List
import torch
from torch import nn

from .components import FeatureTokenizer, MLP
from clsl_v2.data.schema import Schema


def _safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", x)


class LightweightSetEncoder(nn.Module):
    """Fast set encoder for tabular clinical evidence tokens.

    It avoids heavy quadratic attention by alternating token-wise MLP and global context
    pooling. This keeps the v2 prototype runnable on CPU while preserving the idea of
    set-based sparse evidence fusion. A full Transformer can be added later if needed.
    """

    def __init__(self, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = tokens
        w = mask.unsqueeze(-1) + 1e-3
        for layer, norm in zip(self.layers, self.norms):
            g = (h * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            g = g.expand_as(h)
            h = norm(h + layer(torch.cat([h, g], dim=-1)))
        return h


class EvidenceLatticeEncoder(nn.Module):
    """Shared encoder for all evidence views in an evidence lattice.

    It explicitly handles feature observation mask, view-based modality availability,
    missing type, stage embedding, modality reliability gates and disease-query attention.
    """

    def __init__(self, schema: Schema, hidden_dim: int, latent_dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.schema = schema
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.view_names = list(schema.views.keys())
        self.view_to_stage = {v: schema.view_stage_index(v) for v in self.view_names}
        self.num_stages = max(self.view_to_stage.values()) + 1 if self.view_to_stage else 1

        modality_idx = torch.tensor(schema.modality_indices, dtype=torch.long)
        self.register_buffer("modality_idx", modality_idx, persistent=False)
        for v in self.view_names:
            mask = torch.tensor(schema.view_feature_mask(v), dtype=torch.float32)
            self.register_buffer(f"viewmask_{_safe_name(v)}", mask, persistent=False)

        self.tokenizer = FeatureTokenizer(
            num_features=schema.num_features,
            num_modalities=schema.num_modalities,
            num_stages=self.num_stages,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        # Fast default: lightweight set encoder.
        # This is intentionally used instead of a heavy TransformerEncoder so the v2
        # prototype can be trained on CPU. The architecture still remains a set-based
        # sparse evidence encoder.
        self.transformer = LightweightSetEncoder(hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

        self.modality_gate = MLP(hidden_dim + 3, hidden_dim, 1, dropout=dropout, layers=2)
        self.disease_queries = nn.Parameter(torch.randn(schema.num_diseases, hidden_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.patient_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.disease_mu = nn.Linear(hidden_dim, latent_dim)
        self.disease_logvar = nn.Linear(hidden_dim, latent_dim)
        self.obs_repr = MLP(hidden_dim + schema.num_features + 1, hidden_dim, latent_dim, dropout=dropout, layers=2)

    def _get_view_mask(self, view_name: str) -> torch.Tensor:
        return getattr(self, f"viewmask_{_safe_name(view_name)}")

    def _build_missing_type(self, x_mask: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        # Effective unavailable due to view = not_yet_available/structural style.
        b, f = x_mask.shape
        vm = view_mask.view(1, f).expand(b, f)
        effective = (x_mask * vm).clamp(0, 1)
        missing_type = torch.full_like(x_mask, FeatureTokenizer.NOT_MEASURED, dtype=torch.long)
        missing_type[effective > 0.5] = FeatureTokenizer.OBSERVED
        missing_type[(vm < 0.5)] = FeatureTokenizer.STRUCT_UNAVAILABLE
        return missing_type

    def _apply_modality_gate(self, tokens: torch.Tensor, effective_mask: torch.Tensor, view_mask: torch.Tensor, stage_idx: torch.Tensor) -> torch.Tensor:
        b, f, d = tokens.shape
        gated = tokens.clone()
        vm = view_mask.view(1, f).expand(b, f)
        for m in range(self.schema.num_modalities):
            idx = (self.modality_idx == m).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            mtok = tokens[:, idx, :]
            mmask = effective_mask[:, idx]
            available = vm[:, idx].max(dim=1).values.unsqueeze(-1)
            coverage = mmask.mean(dim=1, keepdim=True)
            count = torch.full_like(coverage, float(idx.numel())) / max(float(f), 1.0)
            pooled = (mtok * (mmask.unsqueeze(-1) + 1e-3)).sum(dim=1) / (mmask.sum(dim=1, keepdim=True) + 1e-3)
            gate_in = torch.cat([pooled, available, coverage, count], dim=-1)
            gate = torch.sigmoid(self.modality_gate(gate_in)).view(b, 1, 1)
            gated[:, idx, :] = mtok * gate
        return gated

    def _disease_attention(self, tokens: torch.Tensor, effective_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, f, d = tokens.shape
        k = self.schema.num_diseases
        q = self.query_proj(self.disease_queries).unsqueeze(0).expand(b, k, d)
        keys = self.key_proj(tokens)
        vals = self.value_proj(tokens)
        scores = torch.einsum("bkd,bfd->bkf", q, keys) / (d ** 0.5)
        # Missing tokens are not completely removed: they are evidence of absence/unavailability.
        # But observed tokens receive a small positive bias.
        scores = scores + 0.25 * effective_mask.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bkf,bfd->bkd", attn, vals)
        return self.attn_norm(ctx), attn

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, view_name: str) -> Dict[str, torch.Tensor]:
        if view_name not in self.view_names:
            raise KeyError(f"Unknown view '{view_name}'. Available: {self.view_names}")
        b, f = x.shape
        device = x.device
        view_mask = self._get_view_mask(view_name).to(device)
        effective_mask = (x_mask * view_mask.view(1, f)).clamp(0, 1)
        x_eff = x * effective_mask
        stage_id = torch.full((b,), self.view_to_stage[view_name], device=device, dtype=torch.long)
        missing_type = self._build_missing_type(x_mask, view_mask).to(device)

        tokens = self.tokenizer(x_eff, effective_mask, self.modality_idx.to(device), missing_type, stage_id)
        tokens = self.transformer(tokens, effective_mask)
        tokens = self._apply_modality_gate(tokens, effective_mask, view_mask, stage_id)

        disease_evidence, attn = self._disease_attention(tokens, effective_mask)
        mask_weight = effective_mask.unsqueeze(-1) + 1e-3
        pooled = (tokens * mask_weight).sum(dim=1) / mask_weight.sum(dim=1).clamp_min(1e-6)
        patient_repr = self.patient_pool(pooled)
        mu = self.disease_mu(patient_repr)
        logvar = self.disease_logvar(patient_repr).clamp(-8.0, 6.0)
        obs_in = torch.cat([patient_repr, effective_mask, stage_id.float().view(b, 1) / max(1, self.num_stages - 1)], dim=-1)
        obs_repr = self.obs_repr(obs_in)
        return {
            "tokens": tokens,
            "effective_mask": effective_mask,
            "view_mask": view_mask.view(1, f).expand(b, f),
            "missing_type": missing_type,
            "stage_id": stage_id,
            "disease_evidence": disease_evidence,
            "attention": attn,
            "patient_repr": patient_repr,
            "z_mu": mu,
            "z_logvar": logvar,
            "obs_repr": obs_repr,
        }
