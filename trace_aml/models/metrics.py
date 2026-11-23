"""Evaluation metrics (docs/05-FEATURES-MODEL.md section 7).

Mules are rare, so accuracy is never used. `target` is 1 for a mule and 0 for a
normal account; `score` is any number where higher means riskier.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def top_k(score: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k highest scores (ties broken by position, so results are repeatable)."""
    return np.argsort(-score, kind="stable")[: min(k, len(score))]


def precision_at_k(target: np.ndarray, score: np.ndarray, k: int) -> float:
    """Of the k highest-risk accounts, the share that are true mules."""
    picked = top_k(score, k)
    return float(target[picked].mean()) if len(picked) else 0.0


def recall_at_k(target: np.ndarray, score: np.ndarray, k: int) -> float:
    """Of all true mules, the share inside the k highest-risk accounts."""
    total = target.sum()
    return float(target[top_k(score, k)].sum() / total) if total else 0.0


def at_cut(target: np.ndarray, score: np.ndarray, cut: float) -> dict[str, float]:
    """Precision, recall and F1 when every account with score >= cut is flagged."""
    flagged = score >= cut
    hits = float((flagged & (target == 1)).sum())
    precision = hits / flagged.sum() if flagged.sum() else 0.0
    recall = hits / target.sum() if target.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "flagged": int(flagged.sum())}


def fpr_at_top_share(target: np.ndarray, score: np.ndarray, share: float) -> float:
    """Share of normal accounts that land inside the top `share` of risk."""
    picked = top_k(score, int(np.ceil(share * len(score))))
    normal = (target == 0).sum()
    return float((target[picked] == 0).sum() / normal) if normal else 0.0


def main_metrics(target: np.ndarray, score: np.ndarray, k_values: list[int], top_share: float) -> dict[str, float]:
    """PR-AUC, ROC-AUC, Precision@K, Recall@K and the false-positive rate at the top share."""
    target, score = np.asarray(target), np.asarray(score, dtype="float64")
    out = {
        "n_accounts": int(len(target)),
        "n_mules": int(target.sum()),
        "pr_auc": float(average_precision_score(target, score)) if target.sum() else 0.0,
        "roc_auc": float(roc_auc_score(target, score)) if 0 < target.sum() < len(target) else 0.0,
        "fpr_at_top_share": fpr_at_top_share(target, score, top_share),
    }
    for k in k_values:
        out[f"precision_at_{k}"] = precision_at_k(target, score, k)
        out[f"recall_at_{k}"] = recall_at_k(target, score, k)
    return out


def mean_and_sd(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """{"metric": {"mean": ..., "sd": ...}} over the seeds."""
    return {name: {"mean": float(np.mean([run[name] for run in per_seed])),
                   "sd": float(np.std([run[name] for run in per_seed]))}
            for name in per_seed[0]}
