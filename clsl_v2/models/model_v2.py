from __future__ import annotations

from typing import Dict, Any
import torch
from torch import nn

from clsl_v2.data.schema import Schema
from .evidence_lattice_encoder import EvidenceLatticeEncoder
from .snapshot_transition import SnapshotToTrajectoryTransition
from .graph_energy_decoder import GraphEnergyDecoder
from .observation_disentangle import ObservationPolicyHeads


class CLSLv2(nn.Module):
    """Coarsened Latent State Learning v2."""

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
        self.transition = SnapshotToTrajectoryTransition(latent_dim=latent_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.current_decoder = GraphEnergyDecoder(
            schema=schema,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            gnn_layers=int(mc.get("gnn_layers", 2)),
            dropout=dropout,
            prior_edges=mc.get("clinical_prior_edges"),
        )
        self.future_decoder = GraphEnergyDecoder(
            schema=schema,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            gnn_layers=int(mc.get("gnn_layers", 2)),
            dropout=dropout,
            prior_edges=mc.get("clinical_prior_edges"),
        )
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

    def forward(self, batch: Dict[str, torch.Tensor], view_name: str, grl_lambda: float = 1.0) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        x_mask = batch["x_mask"]
        delta_t = batch["delta_t"]
        enc = self.encoder(x, x_mask, view_name=view_name)
        z_current = self.reparameterize(enc["z_mu"], enc["z_logvar"])
        z_future = self.transition(z_current, delta_t)
        cur = self.current_decoder(enc["disease_evidence"], z_current, enc["effective_mask"], enc["attention"])
        fut = self.future_decoder(enc["disease_evidence"], z_future, enc["effective_mask"], enc["attention"])
        obs = self.obs_heads(z_current, enc["obs_repr"], grl_lambda=grl_lambda)
        b = x.shape[0]
        view_idx = torch.full((b,), self.view_to_index[view_name], device=x.device, dtype=torch.long)
        out = {
            **{f"enc_{k}": v for k, v in enc.items()},
            "z_current": z_current,
            "z_future": z_future,
            "view_idx": view_idx,
            "current_unary": cur["unary"],
            "current_pairwise": cur["pairwise"],
            "current_reliability": cur["reliability"],
            "future_unary": fut["unary"],
            "future_pairwise": fut["pairwise"],
            "future_reliability": fut["reliability"],
            **obs,
        }
        return out
