# CLSL-v3 upgrade overlay

This overlay adds a new `clsl_v3` package without modifying your existing `clsl_v2` code.
It is meant to be copied/unzipped at the root of your repository.

## Main method upgrades

### 1. Nested Evidence Lattice Encoder
File: `clsl_v3/models/nested_evidence_lattice_encoder.py`

Turns evidence coarsening into latent geometry. Disease states are represented by
Matryoshka-style nested latent prefixes, e.g. `[16, 32, 48, 64]`, so lower-information
views are not merely masked inputs; they are constrained to use lower-capacity
subspaces of the same disease state.

### 2. Posterior Refinement Flow
File: `clsl_v3/models/posterior_refinement_flow.py`

Replaces static coarse/fine distillation with flow matching in posterior-logit space:
coarse posterior logits are transported toward fine posterior logits by a learned
vector field conditioned on coarse representations and information gap.

### 3. Latent Disease Graph Transformer
File: `clsl_v3/models/latent_disease_graph_transformer.py`

Replaces deterministic patient-adaptive adjacency with a latent disease graph posterior
`q(A | x)`. It combines clinical-prior local graph propagation with global disease
attention. It also exposes edge posterior probabilities and a counterfactual edge
intervention API.

### 4. Scalable Hidden-Label Inference
File: `clsl_v3/losses/scalable_label_marginalization.py`

Keeps exact marginalization for small K and adds a GFlowNet-style sampler for larger
label spaces. This addresses the main scalability criticism of exact `2^K` enumeration.

### 5. Flow-Matched Disease-State Transition
File: `clsl_v3/models/disease_flow_transition.py`

Upgrades deterministic snapshot transition into selective disease-level vector fields.
If future partial labels are available, `FutureLabelPosteriorEncoder` creates weak
future latent targets for flow-matching supervision.

### 6. Observation Propensity + Risk Control
File: `clsl_v3/models/observation_policy_risk_control.py`

Adds explicit selective-observation propensity modeling and view-conditional
risk-control thresholding for low-information EHR settings.

## Minimal usage

```python
from clsl_v3 import CLSLv3
from clsl_v3.losses import ScalableLabelMarginalizationLoss, PosteriorRefinementFlow
from clsl_v3.models.nested_evidence_lattice_encoder import NestedLatticeLoss

model = CLSLv3(schema, config)
out = model(batch, view_name="diagnosis")

label_loss_fn = ScalableLabelMarginalizationLoss(
    num_labels=schema.num_diseases,
    disease_repr_dim=config["model"]["hidden_dim"],
    max_exact_labels=config["loss"].get("max_exact_labels", 20),
    num_samples=config["loss"].get("label_num_samples", 32),
)

loss_current = label_loss_fn(
    out["current_unary"], out["current_pairwise"],
    batch["y_current"], batch["y_current_mask"],
    disease_repr=out["current_node_repr"],
)
loss_future = label_loss_fn(
    out["future_unary"], out["future_pairwise"],
    batch["y_future"], batch["y_future_mask"],
    disease_repr=out["future_node_repr"],
)

aux = model.auxiliary_losses(out, batch, weights={
    "edge_kl": 0.01,
    "edge_entropy": -0.001,
    "observation_propensity": 0.05,
    "transition_flow": 0.10,
})

nested_loss = NestedLatticeLoss(out["enc_latent_splits"].tolist())(
    out["enc_nested_z_mu"], out["enc_z_mu"], out["enc_nested_z_logvar"]
)

loss = loss_current + loss_future + 0.05 * nested_loss + sum(aux.values())
```

## Smoke test

From the repository root after copying this overlay:

```bash
python tests/test_clsl_v3_smoke.py
```

## How to position it in a CCF-A-style paper

Do not present it as a heart-failure predictor. Present it as:

> Selective-observation-aware structured posterior inference over coarsened evidence.

The three strongest method claims are:

1. Nested evidence-lattice representation rather than view-specific masking.
2. Posterior refinement flow rather than static coarse/fine consistency.
3. Scalable hidden-label graph-energy inference rather than small-K exact enumeration only.
