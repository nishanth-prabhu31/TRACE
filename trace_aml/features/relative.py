"""Relative-amount features: unusual sizes without any currency (docs/05-FEATURES-MODEL.md 2.4)."""

import numpy as np
import pandas as pd


def _percentile(values: pd.Series, reference_sorted: np.ndarray) -> pd.Series:
    """Percentile (0-100) of each value among the reference amounts."""
    rank = np.searchsorted(reference_sorted, values.to_numpy(), side="right")
    return pd.Series(100 * rank / len(reference_sorted), index=values.index)


def relative_features(events: pd.DataFrame, all_amounts_sorted: np.ndarray,
                      round_divisor: float) -> pd.DataFrame:
    """One row per account. `all_amounts_sorted` = every transaction amount <= cutoff, sorted."""
    out = pd.DataFrame(index=pd.Index(events["account_id"].unique(), name="account_id"))
    out["self_txn_share"] = events.groupby("account_id")["is_self"].mean().reindex(out.index)

    ev = events[~events["is_self"]]
    grouped = ev.groupby("account_id")
    median = grouped["amount"].median()
    max_in = ev[ev["direction"] == "IN"].groupby("account_id")["amount"].max()
    max_out = ev[ev["direction"] == "OUT"].groupby("account_id")["amount"].max()

    out["max_in_vs_own_median"] = (max_in / median).reindex(out.index)
    out["max_out_vs_own_median"] = (max_out / median).reindex(out.index)
    out["max_in_pctl"] = _percentile(max_in, all_amounts_sorted).reindex(out.index)
    out["mean_amount_pctl"] = _percentile(grouped["amount"].mean(), all_amounts_sorted).reindex(out.index)

    is_round = (ev["amount_original"] % round_divisor) == 0
    out["round_amount_share"] = is_round.groupby(ev["account_id"]).mean().reindex(out.index)

    shares = ev.groupby(["account_id", "channel"]).size() / grouped.size()
    entropy = -(shares * np.log2(shares)).groupby(level="account_id").sum()
    out["channel_entropy"] = entropy.reindex(out.index)

    return out.astype("float32")
