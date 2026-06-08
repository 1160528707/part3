from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence

import torch
from torch import nn

from .components import FeatureTokenizer, LightweightSetEncoder, MLP


def _safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", x)


def _as_splits(latent_dim: int, splits: Sequence[int] | None) -> List[int]:
    if splits is None or len(splits) == 0:
        # Four Matryoshka levels by default, but keep unique positive sizes.
        raw = [max(1, latent_dim // 4), max(1, latent_dim // 2), max(1, (3 * latent_dim) // 4), latent_dim]
    else:
        raw = [int(x) for x in splits]
    out = sorted(set([x for x in raw if 0 < x <= latent_dim]))
    if latent_dim not in out:
        out.append(latent_dim)
    return out


class NestedEvidenceLatticeEncoder(nn.Module):
    """Matryoshka-style evidence lattice encoder for sparse EHRs.

    CLSL-v2 treats each clinical view as a coarsened feature mask and regularizes
    predictions across views. This v3 encoder makes the *latent geometry itself*
    coarsened: disease latent dimensions are organized into nested prefixes.
    Low-information views are encouraged to use a smaller prefix, while refined
    views can use larger prefixes. This turns the evidence lattice from a loss-only
    idea into an explicit representation constraint.

    Public output stays compatible with v2:
      - z_mu, z_logvar: full [B,K,L] latent parameters
      - disease_evidence, attention, effective_mask, obs_repr, etc.

    New outputs:
      - nested_z_mu / nested_z_logvar: dict[str, Tensor] with zero-padded prefixes
      - active_latent_dim: prefix selected by the current view stage
      - latent_splits: tensor of nested dimensions
    """

    def __init__(
        self,
        schema,
        hidden_dim: int,
        latent_dim: int,
        n_layers: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        latent_splits: Sequence[int] | None = None,
        stage_to_split: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        del n_heads  # retained for config compatibility
        self.schema = schema
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.latent_splits = _as_splits(self.latent_dim, latent_splits)
        self.view_names = list(schema.views.keys())
        self.view_to_stage = {v: schema.view_stage_index(v) for v in self.view_names}
        self.num_stages = max(self.view_to_stage.values()) + 1 if self.view_to_stage else 1

        if stage_to_split is None:
            # Map earliest stage to smallest split and latest to largest split.
            if self.num_stages <= 1:
                self.stage_to_split = [self.latent_splits[-1]]
            else:
                self.stage_to_split = []
                for s in range(self.num_stages):
                    idx = round(s * (len(self.latent_splits) - 1) / max(1, self.num_stages - 1))
                    self.stage_to_split.append(self.latent_splits[idx])
        else:
            self.stage_to_split = [int(x) for x in stage_to_split]

        modality_idx = torch.tensor(schema.modality_indices, dtype=torch.long)
        self.register_buffer("modality_idx", modality_idx, persistent=False)
        for v in self.view_names:
            mask = torch.tensor(schema.view_feature_mask(v), dtype=torch.float32)
            self.register_buffer(f"viewmask_{_safe_name(v)}", mask, persistent=False)
        self.register_buffer("feature_first_stage", self._build_feature_first_stage(schema), persistent=False)
        self.register_buffer("latent_splits_tensor", torch.tensor(self.latent_splits, dtype=torch.long), persistent=False)

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

        self.patient_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
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

        self.obs_repr = MLP(hidden_dim + schema.num_features + 1, hidden_dim, latent_dim, dropout=dropout, layers=2)

        # Prefix heads give each nested latent level its own prediction-capable geometry.
        self.prefix_norms = nn.ModuleDict({str(d): nn.LayerNorm(d) for d in self.latent_splits})

    def _build_feature_first_stage(self, schema) -> torch.Tensor:
        modality_first_stage: Dict[str, int] = {}
        for view_name, spec in schema.views.items():
            stage = int(spec.get("stage_index", schema.view_stage_index(view_name)))
            for modality in spec.get("modalities", []):
                modality_first_stage[modality] = min(modality_first_stage.get(modality, stage), stage)
        default_stage = max(self.num_stages - 1, 0)
        return torch.tensor([modality_first_stage.get(f.modality, default_stage) for f in schema.feature_defs], dtype=torch.long)

    def _get_view_mask(self, view_name: str) -> torch.Tensor:
        return getattr(self, f"viewmask_{_safe_name(view_name)}")

    def _build_missing_type(self, x_mask: torch.Tensor, view_mask: torch.Tensor, stage_idx: torch.Tensor) -> torch.Tensor:
        b, f = x_mask.shape
        vm = view_mask.view(1, f).expand(b, f)
        observed = (x_mask > 0.5) & (vm > 0.5)
        missing_type = torch.full((b, f), FeatureTokenizer.NOT_MEASURED, device=x_mask.device, dtype=torch.long)
        missing_type[observed] = FeatureTokenizer.OBSERVED
        unavailable = vm < 0.5
        first_stage = self.feature_first_stage.to(x_mask.device).view(1, f).expand(b, f)
        current_stage = stage_idx.view(b, 1).expand(b, f)
        not_yet = unavailable & (first_stage > current_stage)
        structural = unavailable & (~not_yet)
        missing_type[not_yet] = FeatureTokenizer.NOT_YET_AVAILABLE
        missing_type[structural] = FeatureTokenizer.STRUCT_UNAVAILABLE
        return missing_type

    def _apply_modality_gate(self, tokens: torch.Tensor, effective_mask: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
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
            pooled = (mtok * (mmask.unsqueeze(-1) + 1e-3)).sum(dim=1) / (mmask.sum(dim=1, keepdim=True) + 1e-3)
            gate = torch.sigmoid(self.modality_gate(torch.cat([pooled, available, coverage, count], dim=-1))).view(b, 1, 1)
            gated[:, idx, :] = mtok * gate
        return gated

    def _disease_attention(self, tokens: torch.Tensor, effective_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, f, d = tokens.shape
        k = self.schema.num_diseases
        q = self.query_proj(self.disease_queries).unsqueeze(0).expand(b, k, d)
        keys = self.key_proj(tokens)
        vals = self.value_proj(tokens)
        scores = torch.einsum("bkd,bfd->bkf", q, keys) / (d ** 0.5)
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
        hidden = self.disease_latent_proj(
            torch.cat(
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
        )
        mu = self.disease_mu(hidden)
        logvar = self.disease_logvar(hidden).clamp(-8.0, 6.0)
        return mu, logvar, hidden

    def active_dim_for_stage(self, stage_idx: torch.Tensor) -> torch.Tensor:
        vals = torch.tensor(self.stage_to_split, device=stage_idx.device, dtype=torch.long)
        return vals[stage_idx.long().clamp(0, len(self.stage_to_split) - 1)]

    def zero_pad_prefix(self, z: torch.Tensor, dim: int) -> torch.Tensor:
        out = z.new_zeros(*z.shape[:-1], self.latent_dim)
        out[..., :dim] = self.prefix_norms[str(dim)](z[..., :dim])
        return out

    def nested_prefixes(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {str(d): self.zero_pad_prefix(z, d) for d in self.latent_splits}

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

        tokens = self.tokenizer(x_eff, effective_mask, self.modality_idx.to(device), missing_type, stage_id)
        tokens = self.transformer(tokens, effective_mask)
        tokens = self._apply_modality_gate(tokens, effective_mask, view_mask)
        disease_evidence, attn = self._disease_attention(tokens, effective_mask)

        mask_weight = effective_mask.unsqueeze(-1) + 1e-3
        pooled = (tokens * mask_weight).sum(dim=1) / mask_weight.sum(dim=1).clamp_min(1e-6)
        patient_repr = self.patient_pool(pooled)
        mu, logvar, disease_latent_hidden = self._disease_latent(disease_evidence, patient_repr, attn, effective_mask, stage_id)

        obs_in = torch.cat(
            [patient_repr, effective_mask, stage_id.float().view(b, 1) / max(1, self.num_stages - 1)], dim=-1
        )
        obs_repr = self.obs_repr(obs_in)
        nested_mu = self.nested_prefixes(mu)
        nested_logvar = self.nested_prefixes(logvar)
        return {
            "tokens": tokens,
            "effective_mask": effective_mask,
            "view_mask": view_mask.view(1, f).expand(b, f),
            "missing_type": missing_type,
            "stage_id": stage_id,
            "active_latent_dim": self.active_dim_for_stage(stage_id),
            "latent_splits": self.latent_splits_tensor.to(device),
            "disease_evidence": disease_evidence,
            "disease_latent_hidden": disease_latent_hidden,
            "attention": attn,
            "patient_repr": patient_repr,
            "z_mu": mu,
            "z_logvar": logvar,
            "nested_z_mu": nested_mu,
            "nested_z_logvar": nested_logvar,
            "obs_repr": obs_repr,
        }


class NestedLatticeLoss(nn.Module):
    """Auxiliary Matryoshka latent regularizer.

    Penalizes disagreement between prefix latents and the full latent while respecting
    the fact that smaller prefixes should carry coarser information. This does not
    require labels, so it can be added to the original training objective.
    """

    def __init__(self, latent_splits: Sequence[int], detach_full: bool = True, weight_variance: float = 0.01):
        super().__init__()
        self.latent_splits = list(latent_splits)
        self.detach_full = bool(detach_full)
        self.weight_variance = float(weight_variance)

    def forward(self, nested_z_mu: Dict[str, torch.Tensor], z_mu: torch.Tensor, nested_z_logvar: Dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        target = z_mu.detach() if self.detach_full else z_mu
        losses = []
        for k, z in nested_z_mu.items():
            d = int(k)
            losses.append(torch.mean((z[..., :d] - target[..., :d]) ** 2))
        loss = torch.stack(losses).mean() if losses else z_mu.sum() * 0.0
        if nested_z_logvar is not None and self.weight_variance > 0:
            # Discourage pathological prefix uncertainty explosion.
            var_terms = [torch.relu(v[..., : int(k)] - 4.0).mean() for k, v in nested_z_logvar.items()]
            if var_terms:
                loss = loss + self.weight_variance * torch.stack(var_terms).mean()
        return loss
