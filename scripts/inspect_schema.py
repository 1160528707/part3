from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


import argparse
from pathlib import Path
import yaml

from clsl_v2.data.preprocess import load_table, TablePreprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/clsl_v2.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sheet-name", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    df = load_table(args.data, sheet_name=args.sheet_name)
    prep = TablePreprocessor(cfg)
    schema = prep.build_schema(df)
    print("Data shape:", df.shape)
    print("Diseases:", schema.diseases)
    print("Found features:", schema.num_features)
    for f in schema.feature_defs:
        print(f"  [{f.index:03d}] {f.name:30s} modality={f.modality:18s} categorical={f.is_categorical}")
    print("\nViews:")
    for v in schema.views:
        mask = schema.view_feature_mask(v)
        print(f"  {v:15s} stage={schema.view_stage_index(v)} n_features={sum(mask)}")
    print("\nFuture label columns:", schema.future_label_cols)
    print("Current label columns:", schema.current_label_cols)


if __name__ == "__main__":
    main()
