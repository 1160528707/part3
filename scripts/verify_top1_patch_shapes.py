from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from clsl_v2.data.schema import FeatureDef, Schema
from clsl_v2.models.model_v2 import CLSLv2
from clsl_v2.losses import LabelMarginalizationLoss


def make_schema() -> Schema:
    modalities = {
        "demographic": 0,
        "current_state": 1,
        "diagnosis": 2,
        "lab": 3,
        "medication": 4,
        "procedure": 5,
        "trajectory": 6,
    }
    features = []
    idx = 0
    for modality, midx in modalities.items():
        for j in range(2):
            features.append(FeatureDef(name=f"{modality}_{j}", modality=modality, index=idx, modality_index=midx))
            idx += 1
    diseases = ["af", "renal", "copd", "diabetes", "cad", "anemia"]
    return Schema(
        diseases=diseases,
        feature_defs=features,
        modality_to_index=modalities,
        views={
            "demo": {"stage_index": 0, "modalities": ["demographic"]},
            "diagnosis": {"stage_index": 1, "modalities": ["demographic", "current_state", "diagnosis"]},
            "hospital_view": {
                "stage_index": 2,
                "modalities": ["demographic", "current_state", "diagnosis", "lab"],
            },
            "full": {
                "stage_index": 4,
                "modalities": list(modalities.keys()),
            },
        },
        current_label_cols=[f"{d}_current" for d in diseases],
        future_label_cols=[f"{d}_future" for d in diseases],
        id_column="subject_id",
        time_delta_column="delta_t_days",
        default_delta_days=90.0,
    )


def make_config() -> dict:
    return {
        "model": {
            "hidden_dim": 32,
            "latent_dim": 16,
            "transformer_layers": 1,
            "transformer_heads": 2,
            "dropout": 0.1,
            "gnn_layers": 1,
            "use_latent_sampling": False,
            "adaptive_graph_rank": 4,
            "ode_steps": 2,
            "transition_time_scale": 180.0,
            "clinical_prior_edges": {
                "af": {"cad": 0.35},
                "renal": {"diabetes": 0.45, "anemia": 0.40},
                "diabetes": {"cad": 0.40},
            },
        },
        "train": {"primary_view": "hospital_view"},
        "loss": {},
    }


def main() -> None:
    torch.manual_seed(7)
    schema = make_schema()
    model = CLSLv2(schema, make_config())
    b = 5
    f = schema.num_features
    k = schema.num_diseases
    batch = {
        "x": torch.randn(b, f),
        "x_mask": torch.randint(0, 2, (b, f)).float(),
        "y_current": torch.randint(0, 2, (b, k)).float(),
        "y_current_mask": torch.ones(b, k),
        "y_future": torch.randint(0, 2, (b, k)).float(),
        "y_future_mask": torch.ones(b, k),
        "delta_t": torch.full((b,), 90.0),
    }
    out = model(batch, "hospital_view")
    assert out["z_current"].shape == (b, k, 16), out["z_current"].shape
    assert out["z_future"].shape == (b, k, 16), out["z_future"].shape
    assert out["future_unary"].shape == (b, k), out["future_unary"].shape
    assert out["future_pairwise"].shape == (b, k, k), out["future_pairwise"].shape
    assert out["future_patient_adjacency"].shape == (b, k, k), out["future_patient_adjacency"].shape

    energy = LabelMarginalizationLoss(k)
    loss = energy(out["future_unary"], out["future_pairwise"], batch["y_future"], batch["y_future_mask"])
    assert torch.isfinite(loss), loss
    print("OK: disease-specific CLSL-v2 patch shape test passed")
    print("z_current:", tuple(out["z_current"].shape))
    print("future_pairwise:", tuple(out["future_pairwise"].shape))
    print("loss:", float(loss.detach()))


if __name__ == "__main__":
    main()
