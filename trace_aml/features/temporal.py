"""Temporal features: how active and how bursty an account is (docs/05-FEATURES-MODEL.md 2.2)."""

import numpy as np
import pandas as pd


def temporal_features(events: pd.DataFrame, cutoff: pd.Timestamp, windows_hours: list[int],
                      night_hours: tuple[int, int], new_counterparty_hours: int) -> pd.DataFrame:
    """One row per account. `events` must already be limited to timestamp <= cutoff."""
    ev = events[~events["is_self"]]
    grouped = ev.groupby("account_id")
    out = pd.DataFrame(index=pd.Index(events["account_id"].unique(), name="account_id"))

    for hours in windows_hours:
        recent = ev[ev["timestamp"] > cutoff - pd.Timedelta(hours=hours)]
        counts = recent.groupby(["account_id", "direction"]).size().unstack(fill_value=0)
        out[f"in_cnt_{hours}h"] = counts.get("IN", pd.Series(dtype=float)).reindex(out.index).fillna(0)
        out[f"out_cnt_{hours}h"] = counts.get("OUT", pd.Series(dtype=float)).reindex(out.index).fillna(0)

    first_seen = events.groupby("account_id")["timestamp"].min().reindex(out.index)
    age_hours = (cutoff - first_seen) / pd.Timedelta(hours=1)
    out["age_hours"] = age_hours

    total = grouped.size().reindex(out.index).fillna(0)
    last_24h = ev[ev["timestamp"] > cutoff - pd.Timedelta(hours=24)].groupby("account_id").size()
    days = np.maximum(age_hours / 24, 1)
    out["cnt_24h_vs_avg"] = last_24h.reindex(out.index).fillna(0) / (total / days + 1)

    gaps = ev["timestamp"].diff().dt.total_seconds()
    gaps[ev["account_id"].ne(ev["account_id"].shift())] = np.nan
    gap_stats = gaps.groupby(ev["account_id"]).agg(["mean", "std", "count"])
    burst = (gap_stats["std"] - gap_stats["mean"]) / (gap_stats["std"] + gap_stats["mean"])
    burst[gap_stats["count"] < 2] = np.nan  # needs at least 3 transactions (2 gaps)
    out["burstiness"] = burst.reindex(out.index)

    hours_of_day = ev["timestamp"].dt.hour
    out["active_hours_spread"] = hours_of_day.groupby(ev["account_id"]).nunique().reindex(out.index)
    night = (hours_of_day >= night_hours[0]) & (hours_of_day < night_hours[1])
    out["night_share"] = night.groupby(ev["account_id"]).mean().reindex(out.index)

    pair_first = ev.groupby(["account_id", "counterparty"])["timestamp"].min().reset_index()
    is_new = pair_first["timestamp"] > cutoff - pd.Timedelta(hours=new_counterparty_hours)
    new_cp = is_new.groupby(pair_first["account_id"]).sum()
    old_cp = (~is_new).groupby(pair_first["account_id"]).sum()
    out["new_counterparties_24h_ratio"] = (new_cp / (old_cp + 1)).reindex(out.index)

    return out.astype("float32")
