from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from clsl_v2.data.preprocess import load_table, TablePreprocessor
from clsl_v2.data.dataset import EHRDataset
from clsl_v2.data.schema import Schema
from clsl_v2.models.model_v2 import CLSLv2
from clsl_v2.losses import LabelMarginalizationLoss
from clsl_v2.train import build_loss_and_probs
from clsl_v2.utils.training import move_batch_to_device


def top_evidence(attn: np.ndarray, feature_names: List[str], topk: int = 5) -> List[str]:
    idx = np.argsort(-attn)[:topk]
    return [feature_names[i] for i in idx]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--view", default="hospital_view")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    cfg = ckpt["config"]
    if device.type == "cpu":
        torch.set_num_threads(int(cfg.get("project", {}).get("num_threads", 1)))
    schema = Schema.from_dict(ckpt["schema"])
    prep = TablePreprocessor.load(run_dir / "preprocessor.pkl")
    df = load_table(args.data)
    arrays = prep.transform(df)
    ds = EHRDataset(arrays)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = CLSLv2(schema, cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    energy_loss = LabelMarginalizationLoss(schema.num_diseases).to(device)

    rows = []
    offset = 0
    for batch in loader:
        ids = batch["ids"]
        batch = move_batch_to_device(batch, device)
        out = model(batch, args.view)
        _, probs, entropy = build_loss_and_probs(energy_loss, out, "future", batch["y_future"], batch["y_future_mask"])
        probs_np = probs.cpu().numpy()
        entropy_np = entropy.cpu().numpy()
        reliability_np = out["future_reliability"].cpu().numpy()
        attn_np = out["enc_attention"].cpu().numpy()  # [B,K,F]
        for i in range(probs_np.shape[0]):
            row = {"sample_id": ids[i], "view": args.view, "joint_entropy": float(entropy_np[i])}
            for k, disease in enumerate(schema.diseases):
                row[f"risk_{disease}"] = float(probs_np[i, k])
                row[f"evidence_reliability_{disease}"] = float(reliability_np[i, k])
                row[f"top_evidence_{disease}"] = ";".join(top_evidence(attn_np[i, k], schema.feature_names, topk=5))
            rows.append(row)
            offset += 1
    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
