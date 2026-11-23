"""Temporal features: how an account's activity behaves over time (docs/05-FEATURES-MODEL.md 2.2)."""

import numpy as np
import pandas as pd

from trace_aml.features.prep import SECONDS_PER_HOUR, History, check_cutoff

HOURS_PER_DAY = 24


def window_counts(history: History, cutoff: pd.Timestamp, hours: int) -> pd.DataFrame:
    """Received and sent transactions in the `hours` before the cut-off."""
    check_cutoff(history, cutoff)
    legs = history.legs
    recent = legs[legs["t"] > history.cutoff_s - hours * SECONDS_PER_HOUR]
    counts = recent.groupby(["a", "is_in"]).size().unstack(fill_value=0)
    out = pd.DataFrame(index=history.active)
    out[f"in_cnt_{hours}h"] = counts[True] if True in counts else 0
    out[f"out_cnt_{hours}h"] = counts[False] if False in counts else 0
    return out.fillna(0)


def burstiness(history: History, cutoff: pd.Timestamp) -> pd.Series:
    """(sd - mean) / (sd + mean) of the gaps between an account's transactions.

    -1 = perfectly regular, +1 = very bursty. Empty if fewer than 3 transactions,
    or if every gap is zero.
    """
    check_cutoff(history, cutoff)
    legs = history.legs
    gap = legs["t"].diff().where(legs["a"].eq(legs["a"].shift())).astype("float64")
    stats = gap.groupby(legs["a"]).agg(["mean", "count"])
    spread = gap.groupby(legs["a"]).std(ddof=0)
    total = spread + stats["mean"]
    value = (spread - stats["mean"]) / total.where(total > 0)
    return value.where(stats["count"] >= 2).reindex(history.active)   # 2 gaps = 3 transactions


def temporal_features(history: History, cutoff: pd.Timestamp, settings: dict) -> pd.DataFrame:
    """All temporal features for the scored accounts, indexed by account code."""
    check_cutoff(history, cutoff)
    legs, features = history.legs, settings["features"]
    out = pd.concat([window_counts(history, cutoff, hours) for hours in features["windows_hours"]], axis=1)

    by_account = legs.groupby("a")
    n_all = by_account.size().reindex(history.active, fill_value=0)
    age_hours = (history.cutoff_s - by_account["t"].min().reindex(history.active)) / SECONDS_PER_HOUR
    out["age_hours"] = age_hours
    per_day = n_all / np.maximum(age_hours / HOURS_PER_DAY, 1.0)
    out["cnt_24h_vs_avg"] = (out["in_cnt_24h"] + out["out_cnt_24h"]) / (per_day + 1.0)
    out["burstiness"] = burstiness(history, cutoff)
    out["active_hours_spread"] = legs.drop_duplicates(["a", "hour"]).groupby("a").size().reindex(history.active)

    first_met = legs.groupby(["a", "o"])["t"].min()
    is_new = first_met > history.cutoff_s - features["new_counterparty_hours"] * SECONDS_PER_HOUR
    new = is_new.groupby(level="a").sum().reindex(history.active, fill_value=0)
    old = (~is_new).groupby(level="a").sum().reindex(history.active, fill_value=0)
    out["new_counterparties_24h_ratio"] = new / (old + 1.0)

    night_start, night_end = features["night_hours"]
    at_night = legs["hour"].between(night_start, night_end - 1)
    out["night_share"] = at_night.groupby(legs["a"]).mean().reindex(history.active)
    return out
