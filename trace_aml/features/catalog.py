"""Feature catalogue: name, family, human name, meaning and explanation text for every feature.

The table in docs/05-FEATURES-MODEL.md section 2 is the spec; this is the same list in code.
`value_text` is filled with the account's value to show analysts, e.g. "93% forwarded".
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str
    family: str  # STRUCTURAL, TEMPORAL, FLOW, RELATIVE, RULES
    display_name: str
    description: str
    value_text: str  # format string using {v}


def _window_counts() -> list[Feature]:
    items = []
    for hours in (1, 6, 24):
        items.append(Feature(f"in_cnt_{hours}h", "TEMPORAL", f"Received ({hours} h)",
                             f"Transactions received in the last {hours} hours", "{v:.0f} received"))
        items.append(Feature(f"out_cnt_{hours}h", "TEMPORAL", f"Sent ({hours} h)",
                             f"Transactions sent in the last {hours} hours", "{v:.0f} sent"))
    return items


CATALOG: tuple[Feature, ...] = (
    *_window_counts(),
    Feature("age_hours", "TEMPORAL", "Age in data", "Hours since its first transaction in the data",
            "first seen {v:.0f} h ago"),
    Feature("cnt_24h_vs_avg", "TEMPORAL", "Activity spike", "Last 24 h transactions vs its daily average",
            "{v:.1f}x its usual daily activity"),
    Feature("burstiness", "TEMPORAL", "Burstiness", "-1 = regular gaps, +1 = bursts of activity",
            "burstiness {v:.2f}"),
    Feature("active_hours_spread", "TEMPORAL", "Active hours", "Distinct hours of the day it was active",
            "active in {v:.0f} different hours"),
    Feature("night_share", "TEMPORAL", "Night activity", "Share of transactions between midnight and 5 am",
            "{v:.0%} at night"),
    Feature("new_counterparties_24h_ratio", "TEMPORAL", "New counterparties",
            "New counterparties in the last 24 h vs earlier ones", "{v:.1f}x new counterparties"),
    Feature("self_txn_share", "RELATIVE", "Self-transfers", "Share of its transactions sent to itself",
            "{v:.0%} self-transfers"),
    Feature("max_in_vs_own_median", "RELATIVE", "Largest inflow vs usual",
            "Largest received amount divided by its median amount", "largest inflow {v:.0f}x its usual"),
    Feature("max_out_vs_own_median", "RELATIVE", "Largest outflow vs usual",
            "Largest sent amount divided by its median amount", "largest outflow {v:.0f}x its usual"),
    Feature("max_in_pctl", "RELATIVE", "Largest inflow rank",
            "Percentile of its largest inflow among all transactions", "largest inflow in top {v:.0f}%"),
    Feature("mean_amount_pctl", "RELATIVE", "Typical amount rank",
            "Percentile of its mean amount among all transactions", "typical amount at {v:.0f}th percentile"),
    Feature("round_amount_share", "RELATIVE", "Round amounts", "Share of amounts that are round numbers",
            "{v:.0%} round amounts"),
    Feature("channel_entropy", "RELATIVE", "Channel mix", "0 = one payment channel, higher = mixed",
            "channel mix {v:.2f}"),
)

BY_NAME = {f.name: f for f in CATALOG}


def describe(name: str, value: float) -> str:
    """Analyst-facing text for one feature value."""
    feature = BY_NAME.get(name)
    if feature is None or value != value:  # unknown feature or NaN
        return name
    return feature.value_text.format(v=value)
