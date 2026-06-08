from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from .nested_evidence_lattice_encoder import NestedEvidenceLatticeEncoder
from .latent_disease_graph_transformer import LatentDiseaseGraphTransformerEnergyDecoder, build_prior_adjacency
from .disease_flow_transition import DiseaseFlowTransition, FutureLabelPosteriorEncoder
from .observation_policy_risk_control import ObservationPropensityModel


class CLSLv3(nn.Module):
    """CLSL-v3: Counterfactual Coarsened Latent State Learning.

    Drop-in conceptual successor of CLSLv2. It keeps the batch interface of v2:
        batch = {x, x_mask, delta_t, y_current?, y_current_mask?, y_future?, y_future_mask?}
    and returns current/future unary and pairwise energies compatible with the
    label-marginalization losses.

    Main upgrades:
      1) nested Matryoshka disease latents over evidence views;
      2) latent disease graph posterior with graph-transformer decoding;
      3) flow-matched selective disease transition;
      4) explicit observation propensity output.
    """

    def __init__(self, schema, config: Dict[str, Any]):
        super().__init__()
        self.schema = schema
        mc = config.get("model", config)
        hidden_dim = int(mc.get("hidden_dim", 128))
        latent_dim = int(mc.get("latent_dim", 64))
        dropout = float(mc.get("dropout", 0.1))
        self.use_latent_sampling = bool(mc.get("use_latent_sampling", True))
        self.encoder = NestedEvidenceLatticeEncoder(
            schema=schema,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=int(mc.get("transformer_layers", 2)),
            n_heads=int(mc.get("transformer_heads", 4)),
            dropout=dropout,
            latent_splits=mc.get("latent_splits", [latent_dim // 4, latent_dim // 2, (3 * latent_dim) // 4, latent_dim]),
            stage_to_split=mc.get("stage_to_split"),
        )
        decoder_kwargs = dict(
            schema=schema,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            gnn_layers=int(mc.get("gnn_layers", 2)),
            dropout=dropout,
            prior_edges=mc.get("clinical_prior_edges"),
            adaptive_rank=int(mc.get("adaptive_graph_rank", 8)),
            num_heads=int(mc.get("graph_transformer_heads", 4)),
            edge_temperature=float(mc.get("edge_temperature", 0.5)),
            hard_edges=bool(mc.get("hard_edges", False)),
        )
        self.current_decoder = LatentDiseaseGraphTransformerEnergyDecoder(**decoder_kwargs)
        self.future_decoder = LatentDiseaseGraphTransformerEnergyDecoder(**decoder_kwargs)
        self.transition = DiseaseFlowTransition(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_diseases=schema.num_diseases,
            ode_steps=int(mc.get("ode_steps", 4)),
            time_scale=float(mc.get("transition_time_scale", 180.0)),
            n_time_frequencies=int(mc.get("transition_time_frequencies", 8)),
        )
        self.future_label_encoder = FutureLabelPosteriorEncoder(
            num_diseases=schema.num_diseases,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.obs_policy = ObservationPropensityModel(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_features=schema.num_features,
            num_views=len(schema.views),
            dropout=dropout,
        )
        self.view_to_index = {v: i for i, v in enumerate(schema.views.keys())}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_latent_sampling:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    @staticmethod
    def _globalize_z(z: torch.Tensor) -> torch.Tensor:
        return z.mean(dim=1) if z.dim() == 3 else z

    def forward(self, batch: Dict[str, torch.Tensor], view_name: str) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        x_mask = batch["x_mask"]
        delta_t = batch["delta_t"]
        enc = self.encoder(x, x_mask, view_name=view_name)
        z_current = self.reparameterize(enc["z_mu"], enc["z_logvar"])
        cur = self.current_decoder(enc["disease_evidence"], z_current, enc["effective_mask"], enc["attention"])
        z_future = self.transition(z_current, delta_t, patient_adj=cur["patient_adjacency"].detach())
        fut = self.future_decoder(enc["disease_evidence"], z_future, enc["effective_mask"], enc["attention"])
        z_current_global = self._globalize_z(z_current)
        z_future_global = self._globalize_z(z_future)
        b = x.size(0)
        view_idx = torch.full((b,), self.view_to_index[view_name], device=x.device, dtype=torch.long)
        obs = self.obs_policy(z_current_global, view_idx)

        # Optional weak target for transition-flow loss when future labels exist.
        weak_future_z = None
        if "y_future" in batch and "y_future_mask" in batch:
            weak_future_z = self.future_label_encoder(batch["y_future"], batch["y_future_mask"], base_z=z_future)

        out = {
            **{f"enc_{k}": v for k, v in enc.items()},
            "z_current": z_current,
            "z_future": z_future,
            "weak_future_z": weak_future_z,
            "z_current_global": z_current_global,
            "z_future_global": z_future_global,
            "view_idx": view_idx,
            "current_unary": cur["unary"],
            "current_pairwise": cur["pairwise"],
            "current_reliability": cur["reliability"],
            "current_global_adjacency": cur["global_adjacency"],
            "current_patient_adjacency": cur["patient_adjacency"],
            "current_latent_edge_probs": cur["latent_edge_probs"],
            "current_edge_kl": cur["edge_kl"],
            "current_edge_entropy": cur["edge_entropy"],
            "current_node_repr": cur["node_repr"],
            "future_unary": fut["unary"],
            "future_pairwise": fut["pairwise"],
            "future_reliability": fut["reliability"],
            "future_global_adjacency": fut["global_adjacency"],
            "future_patient_adjacency": fut["patient_adjacency"],
            "future_latent_edge_probs": fut["latent_edge_probs"],
            "future_edge_kl": fut["edge_kl"],
            "future_edge_entropy": fut["edge_entropy"],
            "future_node_repr": fut["node_repr"],
            **obs,
        }
        return out

    def auxiliary_losses(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights: Dict[str, float] | None = None) -> Dict[str, torch.Tensor]:
        """Optional v3-specific losses to add to the main label losses."""
        weights = weights or {}
        losses: Dict[str, torch.Tensor] = {}
        losses["edge_kl"] = weights.get("edge_kl", 1.0) * (out["current_edge_kl"] + out["future_edge_kl"])
        losses["edge_entropy"] = weights.get("edge_entropy", -0.01) * (out["current_edge_entropy"] + out["future_edge_entropy"])
        if "x_mask" in batch:
            mask_loss = self.obs_policy.loss(out["feature_propensity_logits"], batch["x_mask"])
            losses["observation_propensity"] = weights.get("observation_propensity", 0.1) * mask_loss
        if out.get("weak_future_z") is not None:
            tf_loss = self.transition.flow_matching_loss(
                out["z_current"], out["weak_future_z"], batch["delta_t"], patient_adj=out["current_patient_adjacency"].detach()
            )
            losses["transition_flow"] = weights.get("transition_flow", 0.1) * tf_loss
        return losses
