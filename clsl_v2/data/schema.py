from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import json
from pathlib import Path


@dataclass
class FeatureDef:
    name: str
    modality: str
    index: int
    modality_index: int
    mean: float = 0.0
    std: float = 1.0
    is_categorical: bool = False
    categories: Optional[Dict[str, int]] = None


@dataclass
class Schema:
    diseases: List[str]
    feature_defs: List[FeatureDef]
    modality_to_index: Dict[str, int]
    views: Dict[str, Dict[str, Any]]
    current_label_cols: List[str]
    future_label_cols: List[str]
    id_column: Optional[str] = None
    time_delta_column: Optional[str] = None
    default_delta_days: float = 90.0

    @property
    def num_features(self) -> int:
        return len(self.feature_defs)

    @property
    def num_modalities(self) -> int:
        return len(self.modality_to_index)

    @property
    def num_diseases(self) -> int:
        return len(self.diseases)

    @property
    def feature_names(self) -> List[str]:
        return [f.name for f in self.feature_defs]

    @property
    def modality_indices(self) -> List[int]:
        return [f.modality_index for f in self.feature_defs]

    def view_feature_mask(self, view_name: str) -> List[int]:
        if view_name not in self.views:
            raise KeyError(f"Unknown view '{view_name}'. Available views: {list(self.views)}")
        allowed_modalities = set(self.views[view_name]["modalities"])
        return [1 if f.modality in allowed_modalities else 0 for f in self.feature_defs]

    def view_stage_index(self, view_name: str) -> int:
        return int(self.views[view_name].get("stage_index", 0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diseases": self.diseases,
            "feature_defs": [asdict(f) for f in self.feature_defs],
            "modality_to_index": self.modality_to_index,
            "views": self.views,
            "current_label_cols": self.current_label_cols,
            "future_label_cols": self.future_label_cols,
            "id_column": self.id_column,
            "time_delta_column": self.time_delta_column,
            "default_delta_days": self.default_delta_days,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Schema":
        fdefs = [FeatureDef(**x) for x in d["feature_defs"]]
        return cls(
            diseases=d["diseases"],
            feature_defs=fdefs,
            modality_to_index=d["modality_to_index"],
            views=d["views"],
            current_label_cols=d["current_label_cols"],
            future_label_cols=d["future_label_cols"],
            id_column=d.get("id_column"),
            time_delta_column=d.get("time_delta_column"),
            default_delta_days=float(d.get("default_delta_days", 90.0)),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Schema":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
