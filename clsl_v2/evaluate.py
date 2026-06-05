from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from clsl_v2.data.preprocess import load_table, TablePreprocessor
from clsl_v2.data.dataset import EHRDataset
from clsl_v2.data.schema import Schema
from clsl_v2.models.model_v2 import CLSLv2
from clsl_v2.losses import LabelMarginalizationLoss
from clsl_v2.train import build_loss_and_probs
from clsl_v2.utils.training import move_batch_to_device
from clsl_v2.utils.metrics import multilabel_metrics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", default=None, help="If omitted, use <run-dir>/<split>_split.csv")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "external"])
    parser.add_argument("--view", default="hospital_view")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device)
    cfg = ckpt["config"]
    if device.type == "cpu":
        torch.set_num_threads(int(cfg.get("project", {}).get("num_threads", 1)))
    schema = Schema.from_dict(ckpt["schema"])
    prep = TablePreprocessor.load(run_dir / "preprocessor.pkl")
    if args.data is None:
        data_path = run_dir / f"{args.split}_split.csv"
    else:
        data_path = Path(args.data)
    df = load_table(data_path)
    arrays = prep.transform(df)
    ds = EHRDataset(arrays)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = CLSLv2(schema, cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    energy_loss = LabelMarginalizationLoss(schema.num_diseases).to(device)

    ys, ps, ms = [], [], []
    losses = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch, args.view)
        nll, probs, entropy = build_loss_and_probs(energy_loss, out, "future", batch["y_future"], batch["y_future_mask"])
        losses.append(float(nll.cpu()) * batch["x"].size(0))
        ys.append(batch["y_future"].cpu().numpy())
        ps.append(probs.cpu().numpy())
        ms.append(batch["y_future_mask"].cpu().numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    m = np.concatenate(ms, axis=0)
    metrics = multilabel_metrics(y, p, m, schema.diseases, threshold=float(cfg["train"].get("threshold", 0.5)))
    metrics["loss"] = float(np.sum(losses) / max(len(ds), 1))
    metrics["view"] = args.view
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
