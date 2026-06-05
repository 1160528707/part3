from __future__ import annotations

from typing import Dict, List
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
except Exception:  # pragma: no cover
    roc_auc_score = average_precision_score = f1_score = brier_score_loss = None


def binary_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if idx.sum() == 0:
            continue
        ece += (idx.mean()) * abs(y_true[idx].mean() - y_prob[idx].mean())
    return float(ece)


def multilabel_metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray, diseases: List[str], threshold: float = 0.5) -> Dict[str, float]:
    out: Dict[str, float] = {}
    y = np.asarray(y)
    p = np.asarray(p)
    mask = np.asarray(mask)
    aucs = []
    aps = []
    briers = []
    eces = []
    f1s = []
    for k, name in enumerate(diseases):
        idx = mask[:, k] > 0.5
        if idx.sum() < 2:
            continue
        yt, pp = y[idx, k], p[idx, k]
        out[f"{name}_n"] = float(idx.sum())
        if len(np.unique(yt)) > 1 and roc_auc_score is not None:
            try:
                auc = float(roc_auc_score(yt, pp))
                out[f"{name}_auc"] = auc
                aucs.append(auc)
            except Exception:
                pass
            try:
                ap = float(average_precision_score(yt, pp))
                out[f"{name}_ap"] = ap
                aps.append(ap)
            except Exception:
                pass
        if brier_score_loss is not None:
            b = float(brier_score_loss(yt, pp))
            out[f"{name}_brier"] = b
            briers.append(b)
        out[f"{name}_ece"] = binary_ece(yt, pp)
        eces.append(out[f"{name}_ece"])
        pred = (pp >= threshold).astype(int)
        if len(np.unique(yt)) > 1 and f1_score is not None:
            f = float(f1_score(yt, pred, zero_division=0))
            out[f"{name}_f1"] = f
            f1s.append(f)
    if aucs:
        out["macro_auc"] = float(np.mean(aucs))
    if aps:
        out["macro_ap"] = float(np.mean(aps))
    if briers:
        out["macro_brier"] = float(np.mean(briers))
    if eces:
        out["macro_ece"] = float(np.mean(eces))
    if f1s:
        out["macro_f1"] = float(np.mean(f1s))
    return out
