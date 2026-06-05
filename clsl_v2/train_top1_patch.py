from __future__ import annotations

"""Training-function patch for CLSL-v2.

This file is intentionally not a full replacement for `train.py`.  It gives you
a drop-in `compute_batch_loss_top1` that can replace the existing
`compute_batch_loss` after importing `ConditionalLatticeConsistency`.

Key upgrades:
- conditional lattice target E[p(Y|X_fine)|X_coarse] instead of same-sample MSE;
- information-gap weighted entropy monotonicity;
- optional patient-adaptive graph regularization;
- optional transition semigroup consistency.
"""

from typing import Any, Dict, Tuple

import torch

from clsl_v2.models.model_v2 import CLSLv2
from clsl_v2.losses import LabelMarginalizationLoss, brier_loss_from_marginals, latent_kl_loss, observation_mask_loss, view_classification_loss
from clsl_v2.losses.lattice_consistency import ConditionalLatticeConsistency


def build_loss_and_probs(energy_loss: LabelMarginalizationLoss, out: Dict[str, torch.Tensor], prefix: str, y: torch.Tensor, m: torch.Tensor):
    unary = out[f"{prefix}_unary"]
    pairwise = out[f"{prefix}_pairwise"]
    nll = energy_loss(unary, pairwise, y, m)
    probs = energy_loss.marginal_probs(unary, pairwise)
    entropy = energy_loss.entropy(unary, pairwise)
    return nll, probs, entropy


def _decoder_graph_regularization(model: CLSLv2) -> Dict[str, torch.Tensor]:
    regs: Dict[str, torch.Tensor] = {}
    for name in ["current_decoder", "future_decoder"]:
        dec = getattr(model, name, None)
        if dec is not None and hasattr(dec, "graph_regularization"):
            for k, v in dec.graph_regularization().items():
                regs[f"{name}_{k}"] = v
    return regs


def compute_batch_loss_top1(
    model: CLSLv2,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    energy_loss: LabelMarginalizationLoss,
    device: torch.device,
    lattice_loss_fn: ConditionalLatticeConsistency | None = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    train_cfg = cfg["train"]
    loss_cfg = cfg["loss"]
    primary_view = train_cfg.get("primary_view", "hospital_view")
    lattice_views = list(train_cfg.get("lattice_views", [primary_view]))
    fine_view = train_cfg.get("fine_view", "full")
    if primary_view not in lattice_views:
        lattice_views.append(primary_view)
    if fine_view not in lattice_views and fine_view in model.schema.views:
        lattice_views.append(fine_view)

    outputs = {v: model(batch, v) for v in lattice_views}
    primary = outputs[primary_view]
    future_nll, future_probs, future_entropy = build_loss_and_probs(energy_loss, primary, "future", batch["y_future"], batch["y_future_mask"])
    current_nll, current_probs, current_entropy = build_loss_and_probs(energy_loss, primary, "current", batch["y_current"], batch["y_current_mask"])

    loss = torch.zeros((), device=device)
    stats: Dict[str, float] = {}
    loss = loss + float(loss_cfg.get("future_nll", 1.0)) * future_nll
    loss = loss + float(loss_cfg.get("current_nll", 0.0)) * current_nll
    stats["future_nll"] = float(future_nll.detach().cpu())
    stats["current_nll"] = float(current_nll.detach().cpu())

    if fine_view in outputs:
        fine = outputs[fine_view]
        _, fine_probs, fine_entropy = build_loss_and_probs(energy_loss, fine, "future", batch["y_future"], batch["y_future_mask"])
        if lattice_loss_fn is None:
            lattice_loss_fn = ConditionalLatticeConsistency(
                temperature=float(loss_cfg.get("conditional_lattice_temperature", 0.15)),
                divergence=str(loss_cfg.get("conditional_lattice_divergence", "js")),
                entropy_margin=float(loss_cfg.get("entropy_margin", 0.02)),
                leave_one_out=bool(loss_cfg.get("conditional_lattice_leave_one_out", True)),
            ).to(device)
        lattice_terms = []
        lattice_consistency_vals = []
        lattice_mono_vals = []
        for v, out in outputs.items():
            if v == fine_view:
                continue
            _, coarse_probs, coarse_entropy = build_loss_and_probs(energy_loss, out, "future", batch["y_future"], batch["y_future_mask"])
            term, diag = lattice_loss_fn(
                coarse_probs=coarse_probs,
                fine_probs=fine_probs,
                coarse_entropy=coarse_entropy,
                fine_entropy=fine_entropy,
                coarse_repr=out.get("enc_patient_repr"),
                coarse_mask=out.get("enc_view_mask"),
                fine_mask=fine.get("enc_view_mask"),
                return_diagnostics=True,
            )
            lattice_terms.append(term)
            lattice_consistency_vals.append(diag.consistency)
            lattice_mono_vals.append(diag.monotonicity)
        if lattice_terms:
            lattice = torch.stack(lattice_terms).mean()
            loss = loss + float(loss_cfg.get("conditional_lattice_consistency", loss_cfg.get("lattice_consistency", 0.0))) * lattice
            stats["conditional_lattice"] = float(lattice.detach().cpu())
            stats["lattice_consistency_only"] = float(torch.stack(lattice_consistency_vals).mean().detach().cpu())
            stats["lattice_entropy_mono"] = float(torch.stack(lattice_mono_vals).mean().detach().cpu())

    kl = latent_kl_loss(primary["enc_z_mu"], primary["enc_z_logvar"])
    loss = loss + float(loss_cfg.get("kl", 0.0)) * kl
    stats["kl"] = float(kl.detach().cpu())

    obs_mask = observation_mask_loss(primary["obs_mask_logits"], primary["enc_effective_mask"])
    obs_view = view_classification_loss(primary["obs_view_logits"], primary["view_idx"])
    adv_view = view_classification_loss(primary["adv_view_logits"], primary["view_idx"])
    loss = loss + float(loss_cfg.get("observation_mask", 0.0)) * obs_mask
    loss = loss + float(loss_cfg.get("observation_view", 0.0)) * obs_view
    loss = loss + float(loss_cfg.get("adversarial_view", 0.0)) * adv_view
    stats["obs_mask"] = float(obs_mask.detach().cpu())
    stats["obs_view"] = float(obs_view.detach().cpu())
    stats["adv_view"] = float(adv_view.detach().cpu())

    brier = brier_loss_from_marginals(future_probs, batch["y_future"], batch["y_future_mask"])
    loss = loss + float(loss_cfg.get("brier_calibration", 0.0)) * brier
    stats["brier_loss"] = float(brier.detach().cpu())

    if hasattr(model, "transition") and hasattr(model.transition, "semigroup_loss"):
        semigroup = model.transition.semigroup_loss(primary["z_current"].detach(), batch["delta_t"])
        loss = loss + float(loss_cfg.get("transition_semigroup", 0.0)) * semigroup
        stats["transition_semigroup"] = float(semigroup.detach().cpu())

    graph_regs = _decoder_graph_regularization(model)
    if graph_regs:
        graph_loss = torch.zeros((), device=device)
        for name, value in graph_regs.items():
            # Match by suffix so both current and future decoders share weights.
            if name.endswith("graph_sparse"):
                graph_loss = graph_loss + float(loss_cfg.get("graph_sparse", 0.0)) * value
            elif name.endswith("graph_prior_alignment"):
                graph_loss = graph_loss + float(loss_cfg.get("graph_prior_alignment", 0.0)) * value
            elif name.endswith("graph_diag"):
                graph_loss = graph_loss + float(loss_cfg.get("graph_diag", 0.0)) * value
            stats[name] = float(value.detach().cpu())
        loss = loss + graph_loss

    stats["total"] = float(loss.detach().cpu())
    tensors = {"future_probs": future_probs.detach(), "current_probs": current_probs.detach(), "future_entropy": future_entropy.detach()}
    return loss, stats, tensors
