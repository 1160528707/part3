from __future__ import annotations

"""View-level expected information gain utilities for CLSL.

The model already predicts uncertainty and top evidence.  This module turns that
into a concrete active-acquisition score that can be reported as an additional
contribution: which richer evidence view or feature group would reduce the joint
label uncertainty the most per unit acquisition cost?
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch


@dataclass
class InformationGainResult:
    gain: torch.Tensor
    cost_adjusted_gain: torch.Tensor
    ranking: torch.Tensor


def expected_information_gain(
    base_entropy: torch.Tensor,
    candidate_entropy: Mapping[str, torch.Tensor],
    costs: Optional[Mapping[str, float]] = None,
) -> Dict[str, InformationGainResult]:
    """Compute H(Y|current view)-H(Y|candidate view) for candidate views.

    Entropies are [B].  Scores are returned per sample; positive values mean the
    candidate view is expected to reduce uncertainty.
    """
    out: Dict[str, InformationGainResult] = {}
    for name, ent in candidate_entropy.items():
        gain = (base_entropy - ent).clamp_min(0.0)
        cost = 1.0 if costs is None else float(costs.get(name, 1.0))
        score = gain / max(cost, 1e-6)
        ranking = torch.argsort(score, descending=True)
        out[name] = InformationGainResult(gain=gain, cost_adjusted_gain=score, ranking=ranking)
    return out
