from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from clsl_v2.data.schema import Schema
from .evidence_lattice_encoder import EvidenceLatticeEncoder
from .snapshot_transition import SnapshotToTrajectoryTransition
from .graph_energy_decoder import GraphEnergyDecoder, build_prior_adjacency
from .observation_disentangle import ObservationPolicyHeads


class CLSLv2(nn.Module):
    """Coarsened Latent State Learning v2 with disease-specific latent states."""

    def __init__(self, schema: Schema, config: Dict[str, Any]):
        super().__init__()
        self.schema = schema
        mc = config["model"]
        hidden_dim = int(mc.get("hidden_dim", 128))
        latent_dim = int(mc.get("latent_dim", 64))
        dropout = float(mc.get("dropout", 0.1))
        self.use_latent_sampling = bool(mc.get("use_latent_sampling", True))

        self.encoder = EvidenceLatticeEncoder(
            schema=schema,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=int(mc.get("transformer_layers", 2)),
            n_heads=int(mc.get("transformer_heads", 4)),
            dropout=dropout,
        )

        prior_adj = build_prior_adjacency(schema, mc.get("clinical_prior_edges"))
        self.transition = SnapshotToTrajectoryTransition(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_diseases=schema.num_diseases,
            prior_adjacency=prior_adj,
            ode_steps=int(mc.get("ode_steps", 4)),
            time_scale=float(mc.get("transition_time_scale", 180.0)),
            n_time_frequencies=int(mc.get("transition_time_frequencies", 8)),
            max_residual_edge=float(mc.get("transition_max_residual_edge", 0.25)),
            init_edge_logit=float(mc.get("transition_init_edge_logit", -6.0)),
        )

        decoder_kwargs = dict(
            schema=schema,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            gnn_layers=int(mc.get("gnn_layers", 2)),
            dropout=dropout,
            prior_edges=mc.get("clinical_prior_edges"),
            adaptive_rank=int(mc.get("adaptive_graph_rank", 8)),
            max_global_edge=float(mc.get("max_global_edge", 0.25)),
            max_adaptive_edge=float(mc.get("max_adaptive_edge", 0.50)),
            init_edge_logit=float(mc.get("graph_init_edge_logit", -6.0)),
        )
        self.current_decoder = GraphEnergyDecoder(**decoder_kwargs)
        self.future_decoder = GraphEnergyDecoder(**decoder_kwargs)

        self.obs_heads = ObservationPolicyHeads(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_features=schema.num_features,
            num_views=len(schema.views),
            dropout=dropout,
        )
        self.view_to_index = {v: i for i, v in enumerate(schema.views.keys())}

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_latent_sampling:
            eps = torch.randn_like(mu)
            return mu + eps * torch.exp(0.5 * logvar)
        return mu

    @staticmethod
    def _globalize_z(z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            return z.mean(dim=1)
        return z

    def forward(self, batch: Dict[str, torch.Tensor], view_name: str, grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        x_mask = batch["x_mask"]
        delta_t = batch["delta_t"]

        enc = self.encoder(x, x_mask, view_name=view_name)
        z_current = self.reparameterize(enc["z_mu"], enc["z_logvar"])  # [B,K,L]

        current_state = batch.get("y_current")
        current_state_mask = batch.get("y_current_mask")
        if current_state is not None:
            current_state = torch.nan_to_num(current_state.float(), nan=0.0)
        z_future = self.transition(
            z_current,
            delta_t,
            current_state=current_state,
            current_state_mask=current_state_mask,
        )

        cur = self.current_decoder(enc["disease_evidence"], z_current, enc["effective_mask"], enc["attention"])
        fut = self.future_decoder(enc["disease_evidence"], z_future, enc["effective_mask"], enc["attention"])

        z_current_global = self._globalize_z(z_current)
        z_future_global = self._globalize_z(z_future)
        obs = self.obs_heads(z_current_global, enc["obs_repr"], grl_lambda=grl_lambda)

        b = x.shape[0]
        view_idx = torch.full((b,), self.view_to_index[view_name], device=x.device, dtype=torch.long)

        out = {
            **{f"enc_{k}": v for k, v in enc.items()},
            "z_current": z_current,
            "z_future": z_future,
            "z_current_global": z_current_global,
            "z_future_global": z_future_global,
            "view_idx": view_idx,
            "current_unary": cur["unary"],
            "current_pairwise": cur["pairwise"],
            "current_reliability": cur["reliability"],
            "current_global_adjacency": cur["global_adjacency"],
            "current_patient_adjacency": cur["patient_adjacency"],
            "future_unary": fut["unary"],
            "future_pairwise": fut["pairwise"],
            "future_reliability": fut["reliability"],
            "future_global_adjacency": fut["global_adjacency"],
            "future_patient_adjacency": fut["patient_adjacency"],
            **obs,
        }
        return out
