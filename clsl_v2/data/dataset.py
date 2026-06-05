from __future__ import annotations

from typing import Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset


class EHRDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.arrays = arrays
        self.n = arrays["x"].shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x": torch.tensor(self.arrays["x"][idx], dtype=torch.float32),
            "x_mask": torch.tensor(self.arrays["x_mask"][idx], dtype=torch.float32),
            "y_current": torch.tensor(self.arrays["y_current"][idx], dtype=torch.float32),
            "y_current_mask": torch.tensor(self.arrays["y_current_mask"][idx], dtype=torch.float32),
            "y_future": torch.tensor(self.arrays["y_future"][idx], dtype=torch.float32),
            "y_future_mask": torch.tensor(self.arrays["y_future_mask"][idx], dtype=torch.float32),
            "delta_t": torch.tensor(self.arrays["delta_t"][idx], dtype=torch.float32),
            "ids": self.arrays["ids"][idx],
        }
