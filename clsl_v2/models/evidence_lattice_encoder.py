from __future__ import annotations

import re
from typing import Any, Dict

import torch
from torch import nn

from .components import FeatureTokenizer, MLP
from clsl_v2.data.schema import Schema


def _safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", x)


class LightweightSetEncoder(nn.Module):
    """Fast permutation-aware set encoder for sparse clinical evidence tokens.

    The layer alternates token-wise updates and global-context pooling. It keeps
    CPU smoke tests practical while preserving the core CLSL idea: every view is
    a coarsening of the same patient evidence set.
    """

    def __init__(self, hidden_dim: int, n_layers: int, dropout: float) -> None:
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
            g = g.expand_as(h)
            h = norm(h + layer(torch.cat([h, g], dim=-1)))
        return h


class EvidenceLatticeEncoder(nn.Module):
    """Shared evidence-lattice encoder with disease-specific latent states.

    Major difference from the original prototype:
      * z_mu and z_logvar are now [B, K, latent_dim], not [B, latent_dim].
      * each disease latent is inferred from disease-query evidence + patient
        context + disease identity + evidence-coverage diagnostics.
      * missing types distinguish measured absence, not-yet-available features,
        and structurally unavailable features when the view is coarsened.
    """

    def __init__(
        self,
        schema: Schema,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del n_heads  # kept for config/backward compatibility

        self.schema = schema
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.view_names = list(schema.views.keys())
        self.view_to_stage = {v: schema.view_stage_index(v) for v in self.view_names}
        self.num_stages = max(self.view_to_stage.values()) + 1 if self.view_to_stage else 1

        modality_idx = torch.tensor(schema.modality_indices, dtype=torch.long)
        self.register_buffer("modality_idx", modality_idx, persistent=False)

        for v in self.view_names:
            mask = torch.tensor(schema.view_feature_mask(v), dtype=torch.float32)
            self.register_buffer(f"viewmask_{_safe_name(v)}", mask, persistent=False)

        feature_first_stage = self._build_feature_first_stage(schema)
        self.register_buffer("feature_first_stage", feature_first_stage, persistent=False)

        self.tokenizer = FeatureTokenizer(
            num_features=schema.num_features,
            num_modalities=schema.num_modalities,
            num_stages=self.num_stages,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.transformer = LightweightSetEncoder(hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout)

        self.modality_gate = MLP(hidden_dim + 3, hidden_dim, 1, dropout=dropout, layers=2)

        self.disease_queries = nn.Parameter(torch.randn(schema.num_diseases, hidden_dim) * 0.02)
        self.disease_identity = nn.Parameter(torch.randn(schema.num_diseases, hidden_dim) * 0.02)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.patient_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.patient_to_disease = nn.Linear(hidden_dim, hidden_dim)

        disease_latent_in = hidden_dim * 3 + 4
        self.disease_latent_proj = nn.Sequential(
            nn.Linear(disease_latent_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.disease_mu = nn.Linear(hidden_dim, latent_dim)
        self.disease_logvar = nn.Linear(hidden_dim, latent_dim)

        self.obs_repr = MLP(
            hidden_dim + schema.num_features + 1,
            hidden_dim,
            latent_dim,
            dropout=dropout,
            layers=2,
        )

    def _build_feature_first_stage(self, schema: Schema) -> torch.Tensor:
        modality_first_stage: Dict[str, int] = {}
        for view_name, spec in schema.views.items():
            stage = int(spec.get("stage_index", schema.view_stage_index(view_name)))
            for modality in spec.get("modalities", []):
                if modality not in modality_first_stage:
                    modality_first_stage[modality] = stage
                else:
                    modality_first_stage[modality] = min(modality_first_stage[modality], stage)

        default_stage = max(self.num_stages - 1, 0)
        vals = [modality_first_stage.get(f.modality, default_stage) for f in schema.feature_defs]
        return torch.tensor(vals, dtype=torch.long)

    def _get_view_mask(self, view_name: str) -> torch.Tensor:
        return getattr(self, f"viewmask_{_safe_name(view_name)}")

    def _build_missing_type(
        self,
        x_mask: torch.Tensor,
        view_mask: torch.Tensor,
        stage_idx: torch.Tensor,
    ) -> torch.Tensor:
        b, f = x_mask.shape
        vm = view_mask.view(1, f).expand(b, f)
        observed = (x_mask > 0.5) & (vm > 0.5)

        missing_type = torch.full(
            (b, f),
            FeatureTokenizer.NOT_MEASURED,
            device=x_mask.device,
            dtype=torch.long,
        )
        missing_type[observed] = FeatureTokenizer.OBSERVED

        unavailable = vm < 0.5
        feature_first_stage = self.feature_first_stage.to(x_mask.device).view(1, f).expand(b, f)
        current_stage = stage_idx.view(b, 1).expand(b, f)
        not_yet_available = unavailable & (feature_first_stage > current_stage)
        structurally_unavailable = unavailable & (~not_yet_available)

        missing_type[not_yet_available] = FeatureTokenizer.NOT_YET_AVAILABLE
        missing_type[structurally_unavailable] = FeatureTokenizer.STRUCT_UNAVAILABLE
        return missing_type

    def _apply_modality_gate(
        self,
        tokens: torch.Tensor,
        effective_mask: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, f, _ = tokens.shape
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
            pooled = (mtok * (mmask.unsqueeze(-1) + 1e-3)).sum(dim=1) / (
                mmask.sum(dim=1, keepdim=True) + 1e-3
            )
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
        scores = torch.einsum("bkd,bfd->bkf", q, keys) / (d**0.5)

        # Missing/unavailable tokens are retained because their absence is itself
        # evidence in EHR. Observed evidence receives a modest positive bias.
        scores = scores + 0.25 * effective_mask.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.einsum("bkf,bfd->bkd", attn, vals)
        return self.attn_norm(ctx), attn

    def _disease_latent(
        self,
        disease_evidence: torch.Tensor,
        patient_repr: torch.Tensor,
        attention: torch.Tensor,
        effective_mask: torch.Tensor,
        stage_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, k, _ = disease_evidence.shape
        patient_ctx = self.patient_to_disease(patient_repr).unsqueeze(1).expand(b, k, -1)
        disease_id = self.disease_identity.unsqueeze(0).expand(b, k, -1)

        coverage = effective_mask.mean(dim=1, keepdim=True).expand(b, k)
        attention_on_observed = (attention * effective_mask.unsqueeze(1)).sum(dim=-1)
        norm = torch.log(torch.tensor(float(effective_mask.size(1)), device=effective_mask.device)).clamp_min(1.0)
        attn_entropy = -(attention.clamp_min(1e-8).log() * attention).sum(dim=-1) / norm
        stage_scaled = stage_id.float().view(b, 1).expand(b, k) / max(1, self.num_stages - 1)

        latent_in = torch.cat(
            [
                disease_evidence,
                patient_ctx,
                disease_id,
                coverage.unsqueeze(-1),
                attention_on_observed.unsqueeze(-1),
                attn_entropy.unsqueeze(-1),
                stage_scaled.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = self.disease_latent_proj(latent_in)
        mu = self.disease_mu(hidden)
        logvar = self.disease_logvar(hidden).clamp(-8.0, 6.0)
        return mu, logvar, hidden

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, view_name: str) -> Dict[str, torch.Tensor]:
        if view_name not in self.view_names:
            raise KeyError(f"Unknown view '{view_name}'. Available: {self.view_names}")

        b, f = x.shape
        device = x.device
        view_mask = self._get_view_mask(view_name).to(device)
        effective_mask = (x_mask * view_mask.view(1, f)).clamp(0, 1)
        x_eff = x * effective_mask
        stage_id = torch.full((b,), self.view_to_stage[view_name], device=device, dtype=torch.long)
        missing_type = self._build_missing_type(x_mask, view_mask, stage_id).to(device)

        tokens = self.tokenizer(
            x_eff,
            effective_mask,
            self.modality_idx.to(device),
            missing_type,
            stage_id,
        )
        tokens = self.transformer(tokens, effective_mask)
        tokens = self._apply_modality_gate(tokens, effective_mask, view_mask)

        disease_evidence, attn = self._disease_attention(tokens, effective_mask)

        mask_weight = effective_mask.unsqueeze(-1) + 1e-3
        pooled = (tokens * mask_weight).sum(dim=1) / mask_weight.sum(dim=1).clamp_min(1e-6)
        patient_repr = self.patient_pool(pooled)

        mu, logvar, disease_latent_hidden = self._disease_latent(
            disease_evidence=disease_evidence,
            patient_repr=patient_repr,
            attention=attn,
            effective_mask=effective_mask,
            stage_id=stage_id,
        )

        obs_in = torch.cat(
            [
                patient_repr,
                effective_mask,
                stage_id.float().view(b, 1) / max(1, self.num_stages - 1),
            ],
            dim=-1,
        )
        obs_repr = self.obs_repr(obs_in)

        return {
            "tokens": tokens,
            "effective_mask": effective_mask,
            "view_mask": view_mask.view(1, f).expand(b, f),
            "missing_type": missing_type,
            "stage_id": stage_id,
            "disease_evidence": disease_evidence,
            "disease_latent_hidden": disease_latent_hidden,
            "attention": attn,
            "patient_repr": patient_repr,
            "z_mu": mu,
            "z_logvar": logvar,
            "obs_repr": obs_repr,
        }
