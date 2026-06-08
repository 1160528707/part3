from __future__ import annotations

import torch

from clsl_v3 import CLSLv3
from clsl_v3.data import FeatureDef, Schema
from clsl_v3.losses import ScalableLabelMarginalizationLoss, PosteriorRefinementFlow
from clsl_v3.models.nested_evidence_lattice_encoder import NestedLatticeLoss


def make_schema() -> Schema:
    modalities = {"demo": 0, "diag": 1, "lab": 2}
    fdefs = []
    for i, m in enumerate(["demo", "demo", "diag", "diag", "lab", "lab", "lab", "lab"]):
        fdefs.append(FeatureDef(name=f"f{i}", modality=m, index=i, modality_index=modalities[m]))
    return Schema(
        diseases=["af", "ckd", "copd", "dm", "cad", "anemia"],
        feature_defs=fdefs,
        modality_to_index=modalities,
        views={
            "demo": {"modalities": ["demo"], "stage_index": 0},
            "diagnosis": {"modalities": ["demo", "diag"], "stage_index": 1},
            "full": {"modalities": ["demo", "diag", "lab"], "stage_index": 2},
        },
        current_label_cols=[f"cur_{i}" for i in range(6)],
        future_label_cols=[f"fut_{i}" for i in range(6)],
    )


def test_v3_forward_and_losses():
    torch.manual_seed(7)
    schema = make_schema()
    config = {"model": {"hidden_dim": 32, "latent_dim": 16, "latent_splits": [4, 8, 16], "gnn_layers": 1, "graph_transformer_heads": 4}}
    model = CLSLv3(schema, config)
    b = 3
    batch = {
        "x": torch.randn(b, schema.num_features),
        "x_mask": torch.randint(0, 2, (b, schema.num_features)).float(),
        "delta_t": torch.full((b,), 90.0),
        "y_current": torch.randint(0, 2, (b, schema.num_diseases)).float(),
        "y_current_mask": torch.randint(0, 2, (b, schema.num_diseases)).float(),
        "y_future": torch.randint(0, 2, (b, schema.num_diseases)).float(),
        "y_future_mask": torch.randint(0, 2, (b, schema.num_diseases)).float(),
    }
    out = model(batch, view_name="diagnosis")
    assert out["current_unary"].shape == (b, schema.num_diseases)
    assert out["current_pairwise"].shape == (b, schema.num_diseases, schema.num_diseases)
    assert out["future_unary"].shape == (b, schema.num_diseases)
    assert out["current_latent_edge_probs"].shape == (b, schema.num_diseases, schema.num_diseases)

    label_loss = ScalableLabelMarginalizationLoss(schema.num_diseases, disease_repr_dim=32, max_exact_labels=20)
    loss = label_loss(out["current_unary"], out["current_pairwise"], batch["y_current"], batch["y_current_mask"], out["current_node_repr"])
    assert torch.isfinite(loss)

    nested_loss = NestedLatticeLoss([4, 8, 16])
    nloss = nested_loss(out["enc_nested_z_mu"], out["enc_z_mu"], out["enc_nested_z_logvar"])
    assert torch.isfinite(nloss)

    flow = PosteriorRefinementFlow(num_labels=schema.num_diseases, repr_dim=16, hidden_dim=32)
    fl = flow.flow_matching_loss(out["current_unary"], out["future_unary"], out["z_current_global"])
    assert torch.isfinite(fl)


if __name__ == "__main__":
    test_v3_forward_and_losses()
    print("CLSL-v3 smoke test passed.")
