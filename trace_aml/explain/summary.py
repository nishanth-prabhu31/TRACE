"""Short label and plain-English summary for each account (docs/05-FEATURES-MODEL.md 6.5).

Both come from fixed templates filled with the account's own numbers. No language model.
"""

import pandas as pd

from trace_aml.formatting import format_inr

NORMAL_LABEL, DEFAULT_LABEL = "Normal activity", "Unusual activity"
LABEL_BY_FEATURE = {
    **dict.fromkeys(["pass_through_1h", "pass_through_24h", "holding_median_min", "holding_p10_min",
                     "in_out_similarity", "net_flow_ratio", "rule_pass_through"], "Pass-through"),
    **dict.fromkeys(["fan_in_max_1h", "in_degree", "in_out_degree_ratio", "rule_fan_in"], "Fan-in collector"),
    **dict.fromkeys(["fan_out_max_1h", "out_degree", "rule_fan_out"], "Fan-out distributor"),
    **dict.fromkeys(["chain_count", "cycle_count", "rule_chain", "rule_cycle"], "Layering hop"),
}
MINUTES_PER_HOUR = 60


def short_labels(verdict: pd.Series, reasons: pd.DataFrame) -> pd.Series:
    """Main behaviour in 1-3 words, from the feature that pushed the score up the most."""
    pushing_up = reasons[reasons["direction"] == "UP"].sort_values("position").drop_duplicates("account_id")
    top_feature = verdict.index.to_series().map(pushing_up.set_index("account_id")["feature"])
    label = top_feature.map(LABEL_BY_FEATURE).fillna(DEFAULT_LABEL)
    return label.where(verdict != "NORMAL", NORMAL_LABEL).astype("string")


def held_text(minutes: float) -> str:
    """'25 minutes' or '3.5 hours'."""
    if minutes < MINUTES_PER_HOUR:
        return f"{minutes:.0f} minutes"
    return f"{minutes / MINUTES_PER_HOUR:.1f} hours"


def recent_activity(transactions: pd.DataFrame, accounts: pd.DataFrame, lookback_hours: int) -> pd.DataFrame:
    """Per (account, cutoff): money and counterparties in and out during the look-back window."""
    parts = []
    for cutoff, group in accounts.groupby("cutoff"):
        window = transactions[(transactions["timestamp"] > cutoff - pd.Timedelta(hours=lookback_hours))
                              & (transactions["timestamp"] <= cutoff) & ~transactions["is_self"]]
        wanted = set(group["account_id"])
        got = window[window["receiver_id"].isin(wanted)].groupby("receiver_id").agg(
            in_amount_24h=("amount", "sum"), n_senders=("sender_id", "nunique"))
        sent = window[window["sender_id"].isin(wanted)].groupby("sender_id").agg(
            out_amount_24h=("amount", "sum"), n_receivers=("receiver_id", "nunique"))
        parts.append(group[["account_id"]].join(got, on="account_id").join(sent, on="account_id"))
    return pd.concat(parts).fillna(0.0).set_index("account_id")


def summary_text(verdict: str, activity: pd.Series, features: pd.Series, trail: pd.DataFrame, rate: float) -> str:
    """One account's summary, amounts in the display currency."""
    if verdict == "NORMAL":
        return "No unusual pattern found."
    sentences = [
        f"In the 24 hours before it was scored, it received {format_inr(activity['in_amount_24h'] * rate, short=True)} "
        f"from {int(activity['n_senders'])} account(s) and sent {format_inr(activity['out_amount_24h'] * rate, short=True)} "
        f"to {int(activity['n_receivers'])} account(s)."]
    forwarded, held = features.get("pass_through_24h"), features.get("holding_median_min")
    if pd.notna(forwarded) and pd.notna(held) and forwarded > 0:
        sentences.append(f"Over its history it forwarded {forwarded:.0%} of what it received within a day; "
                         f"money typically stayed {held_text(held)}.")
    if not trail.empty:
        first = trail[trail["trail_id"] == trail["trail_id"].min()].sort_values("hop")
        sentences.append(f"Part of a {len(first)}-hop chain from {first['from_id'].iloc[0]} "
                         f"to {first['to_id'].iloc[-1]}.")
    return " ".join(sentences)
