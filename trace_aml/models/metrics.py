"""Evaluation metrics at account level (docs/05-FEATURES-MODEL.md section 7)."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def precision_recall_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    """Of the k highest scores: share that are mules, and share of all mules caught."""
    order = np.argsort(-scores, kind="stable")[:k]
    hits = y[order].sum()
    return float(hits / k), float(hits / max(y.sum(), 1))


def evaluate(targets: pd.Series, scores: pd.Series, k_values: list[int], fpr_top_share: float) -> dict:
    """All section 7 metrics for one approach. Accounts with no score count as score 0."""
    s = scores.reindex(targets.index).fillna(scores.min() if len(scores) else 0).to_numpy(dtype=float)
    y = targets.to_numpy()
    result = {
        "n_accounts": int(len(y)),
        "n_mules": int(y.sum()),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "chance_pr_auc": float(y.mean()),
    }
    for k in k_values:
        p, r = precision_recall_at_k(y, s, k)
        result[f"precision@{k}"], result[f"recall@{k}"] = p, r
    top = np.argsort(-s, kind="stable")[: max(1, int(len(s) * fpr_top_share))]
    result["fpr_top_1pct"] = float((y[top] == 0).sum() / max((y == 0).sum(), 1))
    return result
