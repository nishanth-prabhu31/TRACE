"""Relative-amount features: amount signals that do not depend on the currency
(docs/05-FEATURES-MODEL.md 2.4)."""

import numpy as np
import pandas as pd

from trace_aml.features.prep import History, check_cutoff

PERCENT = 100.0


def percentile_among(sorted_values: np.ndarray, values: pd.Series) -> pd.Series:
    """Share (0-100) of `sorted_values` that are <= each value."""
    position = np.searchsorted(sorted_values, values.to_numpy(dtype="float64"), side="right")
    return pd.Series(PERCENT * position / max(len(sorted_values), 1), index=values.index).where(values.notna())


def channel_entropy(history: History, cutoff: pd.Timestamp) -> pd.Series:
    """How mixed an account's payment channels are, in bits (0 = always the same channel)."""
    check_cutoff(history, cutoff)
    counts = history.legs.groupby(["a", "channel"]).size()
    share = counts / counts.groupby(level="a").transform("sum")
    return (-(share * np.log2(share))).groupby(level="a").sum().abs().reindex(history.active)


def relative_features(history: History, cutoff: pd.Timestamp, settings: dict) -> pd.DataFrame:
    """All relative-amount features for the scored accounts, indexed by account code."""
    check_cutoff(history, cutoff)
    legs = history.legs
    amounts = legs.groupby("a")["amount"]
    own_median = amounts.median().reindex(history.active)
    max_in = legs[legs["is_in"]].groupby("a")["amount"].max().reindex(history.active)
    max_out = legs[~legs["is_in"]].groupby("a")["amount"].max().reindex(history.active)

    every_amount = np.sort(history.flows["amount"].to_numpy())
    out = pd.DataFrame(index=history.active)
    out["max_in_vs_own_median"] = max_in / own_median
    out["max_out_vs_own_median"] = max_out / own_median
    out["max_in_pctl"] = percentile_among(every_amount, max_in)
    out["mean_amount_pctl"] = percentile_among(every_amount, amounts.mean().reindex(history.active))

    divisor = settings["features"]["round_amount_divisor"]
    is_round = np.mod(legs["amount_original"], divisor) == 0
    out["round_amount_share"] = is_round.groupby(legs["a"]).mean().reindex(history.active)
    out["channel_entropy"] = channel_entropy(history, cutoff)
    return out
