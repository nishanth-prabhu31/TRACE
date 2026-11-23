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
    Feature("in_degree", "STRUCTURAL", "Distinct senders", "Number of different accounts that sent to it",
            "money from {v:.0f} accounts"),
    Feature("out_degree", "STRUCTURAL", "Distinct receivers", "Number of different accounts it sent to",
            "money to {v:.0f} accounts"),
    Feature("in_out_degree_ratio", "STRUCTURAL", "Senders vs receivers", "Distinct senders / (receivers + 1)",
            "{v:.1f} senders per receiver"),
    Feature("counterparties", "STRUCTURAL", "Counterparties", "Distinct accounts it dealt with",
            "{v:.0f} counterparties"),
    Feature("pagerank", "STRUCTURAL", "Network importance", "Money-weighted PageRank (1 = average)",
            "{v:.1f}x average network importance"),
    Feature("betweenness_local", "STRUCTURAL", "Bridge role", "How often it lies on short paths between others",
            "bridge score {v:.2e}"),
    Feature("clustering", "STRUCTURAL", "Tight circle", "How connected its counterparties are to each other",
            "clustering {v:.2f}"),
    Feature("kcore", "STRUCTURAL", "Core depth", "k-core number: how deep inside the dense core it sits",
            "core depth {v:.0f}"),
    Feature("community_size_log", "STRUCTURAL", "Community size", "log(1 + size of its community)",
            "community of about {v:.1f} (log size)"),
    Feature("community_internal_flow", "STRUCTURAL", "Money stays in community",
            "Share of its community's money that stays inside it", "{v:.0%} of community money stays inside"),
    Feature("nbr_new_share", "STRUCTURAL", "New counterparties' age", "Share of its counterparties first seen in the last 24 h",
            "{v:.0%} of counterparties are brand new"),
    Feature("nbr_mean_degree_log", "STRUCTURAL", "Counterparties' size", "log(1 + mean degree of its counterparties)",
            "counterparties' size {v:.1f} (log)"),
    Feature("pass_through_1h", "FLOW", "Pass-through (1 h)", "Share of received money sent on within 1 hour",
            "forwarded {v:.0%} of inflow within 1 h"),
    Feature("pass_through_24h", "FLOW", "Pass-through (24 h)", "Share of received money sent on within 24 hours",
            "forwarded {v:.0%} of inflow within 24 h"),
    Feature("holding_median_min", "FLOW", "Holding time", "Median minutes money stayed before leaving",
            "money held {v:.0f} min (median)"),
    Feature("holding_p10_min", "FLOW", "Fastest forwards", "10th percentile of holding minutes",
            "fastest 10% held under {v:.0f} min"),
    Feature("net_flow_ratio", "FLOW", "Net flow", "(in - out) / (in + out): 0 = everything that comes in goes out",
            "net flow {v:+.2f}"),
    Feature("in_out_similarity", "FLOW", "In matches out (24 h)", "1 = money in equals money out in the last 24 h",
            "in vs out match {v:.0%}"),
    Feature("fan_in_max_1h", "FLOW", "Fan-in burst", "Most distinct senders within one hour",
            "{v:.0f} senders in one hour"),
    Feature("fan_out_max_1h", "FLOW", "Fan-out burst", "Most distinct receivers within one hour",
            "{v:.0f} receivers in one hour"),
    Feature("chain_count", "FLOW", "Chain hops", "Times it was the middle of a quick multi-hop chain (48 h)",
            "middle of {v:.0f} quick chains"),
    Feature("cycle_count", "FLOW", "Round trips", "Times money left and came back to its sender via it (48 h)",
            "in {v:.0f} money round trips"),
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
