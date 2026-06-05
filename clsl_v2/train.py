from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clsl_v2.data.preprocess import load_table, TablePreprocessor, group_train_val_test_split
from clsl_v2.data.dataset import EHRDataset
from clsl_v2.data.schema import Schema
from clsl_v2.models.model_v2 import CLSLv2
from clsl_v2.losses import (
    LabelMarginalizationLoss,
    brier_loss_from_marginals,
    evidence_lattice_consistency,
    latent_kl_loss,
    observation_mask_loss,
    view_classification_loss,
)
from clsl_v2.utils.io import load_yaml, save_yaml, save_json
from clsl_v2.utils.seed import set_seed
from clsl_v2.utils.training import move_batch_to_device, EarlyStopping, save_model
from clsl_v2.utils.metrics import multilabel_metrics


def build_loss_and_probs(energy_loss: LabelMarginalizationLoss, out: Dict[str, torch.Tensor], prefix: str, y: torch.Tensor, m: torch.Tensor):
    unary = out[f"{prefix}_unary"]
    pairwise = out[f"{prefix}_pairwise"]
    nll = energy_loss(unary, pairwise, y, m)
    probs = energy_loss.marginal_probs(unary, pairwise)
    entropy = energy_loss.entropy(unary, pairwise)
    return nll, probs, entropy


def compute_batch_loss(
    model: CLSLv2,
    batch: Dict[str, torch.Tensor],
    cfg: Dict[str, Any],
    energy_loss: LabelMarginalizationLoss,
    device: torch.device,
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

    future_nll, future_probs, future_entropy = build_loss_and_probs(
        energy_loss, primary, "future", batch["y_future"], batch["y_future_mask"]
    )
    current_nll, current_probs, current_entropy = build_loss_and_probs(
        energy_loss, primary, "current", batch["y_current"], batch["y_current_mask"]
    )

    loss = torch.zeros((), device=device)
    stats: Dict[str, float] = {}
    loss = loss + float(loss_cfg.get("future_nll", 1.0)) * future_nll
    loss = loss + float(loss_cfg.get("current_nll", 0.0)) * current_nll
    stats["future_nll"] = float(future_nll.detach().cpu())
    stats["current_nll"] = float(current_nll.detach().cpu())

    # Evidence-lattice consistency: compare each coarse view to fine view.
    if fine_view in outputs:
        fine = outputs[fine_view]
        _, fine_probs, fine_entropy = build_loss_and_probs(
            energy_loss, fine, "future", batch["y_future"], batch["y_future_mask"]
        )
        lattice_losses = []
        mono_losses = []
        for v, out in outputs.items():
            if v == fine_view:
                continue
            _, cp, ce = build_loss_and_probs(energy_loss, out, "future", batch["y_future"], batch["y_future_mask"])
            lc, mono = evidence_lattice_consistency(
                cp, fine_probs, ce, fine_entropy, entropy_margin=float(loss_cfg.get("entropy_margin", 0.02))
            )
            lattice_losses.append(lc)
            mono_losses.append(mono)
        if lattice_losses:
            lattice_loss = torch.stack(lattice_losses).mean()
            monotonic_loss = torch.stack(mono_losses).mean()
            loss = loss + float(loss_cfg.get("lattice_consistency", 0.0)) * lattice_loss
            loss = loss + float(loss_cfg.get("entropy_monotonic", 0.0)) * monotonic_loss
            stats["lattice"] = float(lattice_loss.detach().cpu())
            stats["entropy_mono"] = float(monotonic_loss.detach().cpu())

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
    stats["total"] = float(loss.detach().cpu())
    tensors = {"future_probs": future_probs.detach(), "current_probs": current_probs.detach(), "future_entropy": future_entropy.detach()}
    return loss, stats, tensors


@torch.no_grad()
def evaluate_loader(model: CLSLv2, loader: DataLoader, cfg: Dict[str, Any], energy_loss: LabelMarginalizationLoss, device: torch.device, view: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    ys, ps, ms = [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch, view)
        nll, probs, entropy = build_loss_and_probs(energy_loss, out, "future", batch["y_future"], batch["y_future_mask"])
        total_loss += float(nll.detach().cpu()) * batch["x"].size(0)
        n += batch["x"].size(0)
        ys.append(batch["y_future"].detach().cpu().numpy())
        ps.append(probs.detach().cpu().numpy())
        ms.append(batch["y_future_mask"].detach().cpu().numpy())
    if not ys:
        return {"loss": float("nan")}
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    m = np.concatenate(ms, axis=0)
    metrics = multilabel_metrics(y, p, m, model.schema.diseases, threshold=float(cfg["train"].get("threshold", 0.5)))
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sheet-name", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("project", {}).get("seed", 42)))
    # CPU tabular models can be much faster with fewer intra-op threads.
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        torch.set_num_threads(int(cfg.get("project", {}).get("num_threads", 1)))
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, outdir / "config_used.yaml")
    try:
        shutil.copy(args.config, outdir / Path(args.config).name)
    except Exception:
        pass

    df = load_table(args.data, sheet_name=args.sheet_name)
    train_df, val_df, test_df = group_train_val_test_split(
        df,
        id_column=cfg["data"].get("id_column"),
        test_size=float(cfg["train"].get("test_size", 0.15)),
        val_size=float(cfg["train"].get("val_size", 0.15)),
        seed=int(cfg.get("project", {}).get("seed", 42)),
    )
    train_df.to_csv(outdir / "train_split.csv", index=False)
    val_df.to_csv(outdir / "val_split.csv", index=False)
    test_df.to_csv(outdir / "test_split.csv", index=False)

    prep = TablePreprocessor(cfg)
    prep.fit(train_df)
    train_arrays = prep.transform(train_df)
    val_arrays = prep.transform(val_df)
    test_arrays = prep.transform(test_df)
    prep.save(outdir / "preprocessor.pkl")
    assert prep.schema is not None
    prep.schema.save(outdir / "schema.json")

    train_ds = EHRDataset(train_arrays)
    val_ds = EHRDataset(val_arrays)
    test_ds = EHRDataset(test_arrays)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=int(cfg["train"].get("num_workers", 0)))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = CLSLv2(prep.schema, cfg).to(device)
    energy_loss = LabelMarginalizationLoss(prep.schema.num_diseases).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("learning_rate", 8e-4)), weight_decay=float(cfg["train"].get("weight_decay", 1e-5)))
    stopper = EarlyStopping(patience=int(cfg["train"].get("patience", 8)), mode="min")
    primary_view = cfg["train"].get("primary_view", "hospital_view")

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {}
        count = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, stats, _ = compute_batch_loss(model, batch, cfg, energy_loss, device)
            loss.backward()
            if float(cfg["train"].get("grad_clip", 0)) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 5.0)))
            optimizer.step()
            bs = batch["x"].size(0)
            count += bs
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v * bs
            pbar.set_postfix({"loss": running.get("total", 0.0) / max(count, 1)})

        train_stats = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        val_stats = evaluate_loader(model, val_loader, cfg, energy_loss, device, view=primary_view)
        row = {"epoch": epoch, **train_stats, **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
        print(row)
        val_loss = val_stats.get("loss", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            save_model(outdir / "best_model.pt", model, optimizer, epoch, cfg, prep.schema)
        if stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    # load best for test
    ckpt = torch.load(outdir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    results = {}
    for view in [primary_view, "full", "initial_lab", "diagnosis", "demo"]:
        if view in prep.schema.views:
            results[view] = evaluate_loader(model, test_loader, cfg, energy_loss, device, view=view)
    save_json(results, outdir / "test_metrics.json")
    print("Test metrics:", results)


if __name__ == "__main__":
    main()
