from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .schema import Schema, FeatureDef


def load_table(path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if suffix in [".parquet"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported data format: {suffix}")


def _is_categorical_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return False
    nunique = s.dropna().astype(str).nunique()
    return nunique <= 50


class TablePreprocessor:
    """Builds a numeric model matrix from a heterogeneous tabular EHR table.

    The class is deliberately conservative:
    - Missing feature values remain tracked in x_mask.
    - NaN labels become label_mask=0, never negative labels.
    - Missing configured feature columns are skipped automatically.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.schema: Schema | None = None
        self.category_maps: Dict[str, Dict[str, int]] = {}

    def build_schema(self, df: pd.DataFrame) -> Schema:
        data_cfg = self.config["data"]
        diseases = list(data_cfg["diseases"])
        current_cols = [f"{d}{data_cfg.get('current_label_suffix', '_current')}" for d in diseases]
        future_cols = [f"{d}{data_cfg.get('future_label_suffix', '_future')}" for d in diseases]

        modality_to_index = {m: i for i, m in enumerate(data_cfg["modalities"].keys())}
        fdefs: List[FeatureDef] = []
        used = set()
        for modality, cols in data_cfg["modalities"].items():
            for col in cols:
                if col in df.columns and col not in used:
                    used.add(col)
                    s = df[col]
                    is_cat = _is_categorical_series(s)
                    fdefs.append(
                        FeatureDef(
                            name=col,
                            modality=modality,
                            index=len(fdefs),
                            modality_index=modality_to_index[modality],
                            is_categorical=is_cat,
                        )
                    )

        if len(fdefs) == 0:
            raise ValueError(
                "No configured feature columns were found in the data. "
                "Edit configs/clsl_v2.yaml -> data.modalities to match your table."
            )

        self.schema = Schema(
            diseases=diseases,
            feature_defs=fdefs,
            modality_to_index=modality_to_index,
            views=data_cfg["views"],
            current_label_cols=current_cols,
            future_label_cols=future_cols,
            id_column=data_cfg.get("id_column"),
            time_delta_column=data_cfg.get("time_delta_column"),
            default_delta_days=float(data_cfg.get("default_delta_days", 90.0)),
        )
        return self.schema

    def fit(self, df: pd.DataFrame) -> "TablePreprocessor":
        if self.schema is None:
            self.build_schema(df)
        assert self.schema is not None
        for f in self.schema.feature_defs:
            s = df[f.name]
            if f.is_categorical:
                values = sorted([str(x) for x in s.dropna().unique()])
                cmap = {v: i + 1 for i, v in enumerate(values)}  # 0 reserved for missing/unknown
                self.category_maps[f.name] = cmap
                f.categories = cmap
                encoded = s.astype(str).map(cmap).replace({"nan": np.nan}).astype(float)
                mean = float(np.nanmean(encoded)) if np.isfinite(np.nanmean(encoded)) else 0.0
                std = float(np.nanstd(encoded)) if np.isfinite(np.nanstd(encoded)) and np.nanstd(encoded) > 1e-6 else 1.0
            else:
                numeric = pd.to_numeric(s, errors="coerce").astype(float)
                mean = float(np.nanmean(numeric)) if np.isfinite(np.nanmean(numeric)) else 0.0
                std = float(np.nanstd(numeric)) if np.isfinite(np.nanstd(numeric)) and np.nanstd(numeric) > 1e-6 else 1.0
            f.mean = mean
            f.std = std
        return self

    def transform_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        assert self.schema is not None
        xs = []
        masks = []
        for f in self.schema.feature_defs:
            s = df[f.name] if f.name in df.columns else pd.Series([np.nan] * len(df))
            if f.is_categorical:
                cmap = self.category_maps.get(f.name, f.categories or {})
                v = s.astype(str).map(cmap).astype(float)
                v[s.isna()] = np.nan
            else:
                v = pd.to_numeric(s, errors="coerce").astype(float)
            mask = (~pd.isna(v)).astype(np.float32).values
            values = ((v.fillna(f.mean).astype(float).values - f.mean) / max(f.std, 1e-6)).astype(np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            xs.append(values)
            masks.append(mask)
        return np.stack(xs, axis=1).astype(np.float32), np.stack(masks, axis=1).astype(np.float32)

    def _labels_and_mask(self, df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        assert self.schema is not None
        labels = []
        masks = []
        suffix = self.config["data"].get("label_mask_suffix", "_mask")
        for col in cols:
            if col in df.columns:
                y = pd.to_numeric(df[col], errors="coerce").astype(float)
                mask = (~pd.isna(y)).astype(np.float32)
                explicit_mask_col = f"{col}{suffix}"
                if explicit_mask_col in df.columns:
                    explicit = pd.to_numeric(df[explicit_mask_col], errors="coerce").fillna(0).astype(float)
                    mask = ((mask > 0) & (explicit > 0)).astype(np.float32)
                y = y.fillna(0).clip(0, 1)
            else:
                y = pd.Series([0.0] * len(df))
                mask = pd.Series([0.0] * len(df))
            labels.append(y.astype(np.float32).values)
            masks.append(mask.astype(np.float32).values)
        return np.stack(labels, axis=1).astype(np.float32), np.stack(masks, axis=1).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        assert self.schema is not None
        x, x_mask = self.transform_features(df)
        y_cur, m_cur = self._labels_and_mask(df, self.schema.current_label_cols)
        y_fut, m_fut = self._labels_and_mask(df, self.schema.future_label_cols)

        if self.schema.time_delta_column and self.schema.time_delta_column in df.columns:
            dt = pd.to_numeric(df[self.schema.time_delta_column], errors="coerce").fillna(self.schema.default_delta_days).astype(float)
        else:
            dt = pd.Series([self.schema.default_delta_days] * len(df)).astype(float)
        delta_t = dt.clip(lower=0).values.astype(np.float32)

        if self.schema.id_column and self.schema.id_column in df.columns:
            ids = df[self.schema.id_column].astype(str).values
        else:
            ids = np.array([str(i) for i in range(len(df))])

        return {
            "x": x,
            "x_mask": x_mask,
            "y_current": y_cur,
            "y_current_mask": m_cur,
            "y_future": y_fut,
            "y_future_mask": m_fut,
            "delta_t": delta_t,
            "ids": ids,
        }

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        self.fit(df)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "TablePreprocessor":
        with Path(path).open("rb") as f:
            return pickle.load(f)


def group_train_val_test_split(
    df: pd.DataFrame,
    id_column: str | None,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if id_column and id_column in df.columns:
        groups = df[id_column].astype(str).values
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        trainval_idx, test_idx = next(gss.split(df, groups=groups))
        trainval = df.iloc[trainval_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        groups_trainval = trainval[id_column].astype(str).values
        val_ratio = val_size / max(1e-8, (1.0 - test_size))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + 1)
        train_idx, val_idx = next(gss2.split(trainval, groups=groups_trainval))
        train = trainval.iloc[train_idx].reset_index(drop=True)
        val = trainval.iloc[val_idx].reset_index(drop=True)
        return train, val, test
    trainval, test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    val_ratio = val_size / max(1e-8, (1.0 - test_size))
    train, val = train_test_split(trainval, test_size=val_ratio, random_state=seed + 1, shuffle=True)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
