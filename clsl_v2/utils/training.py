from __future__ import annotations

from pathlib import Path
import torch


def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class EarlyStopping:
    def __init__(self, patience: int = 8, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.count = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.count = 0
            return False
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


def save_model(path: str | Path, model, optimizer, epoch: int, config, schema):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "config": config,
        "schema": schema.to_dict(),
    }, path)
