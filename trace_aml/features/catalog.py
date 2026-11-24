"""The feature catalogue: every feature's name, family, and the words used to explain it
(docs/05-FEATURES-MODEL.md section 2).

`value_text` is filled with the feature's value as `v`, e.g. "forwarded {v:.0%} of inflow".
"""

from dataclasses import dataclass

STRUCTURAL, TEMPORAL, FLOW, RELATIVE, RULES = "STRUCTURAL", "TEMPORAL", "FLOW", "RELATIVE", "RULES"


@dataclass(frozen=True)
class Feature:
    """One feature: what it is called in files, and how it is shown to an analyst."""

    name: str
    family: str
    display_name: str
    description: str
    value_text: str


CATALOG: tuple[Feature, ...] = (
    # --- Structural (2.1)
    Feature("in_degree", STRUCTURAL, "Senders", "Distinct accounts that sent to it", "received from {v:.0f} accounts"),
    Feature("out_degree", STRUCTURAL, "Receivers", "Distinct accounts it sent to", "sent to {v:.0f} accounts"),
    Feature("in_out_degree_ratio", STRUCTURAL, "Senders per receiver", "Senders divided by receivers + 1",
            "{v:.1f} senders per receiver"),
    Feature("counterparties", STRUCTURAL, "Counterparties", "Distinct accounts it dealt with", "dealt with {v:.0f} accounts"),
    Feature("pagerank", STRUCTURAL, "Network importance", "Amount-weighted PageRank, 1 = average account",
            "{v:.1f} times the average account"),
    Feature("betweenness_local", STRUCTURAL, "Bridge position", "Share of short paths (up to 3 hops) passing through it",
            "sits on short money paths between other accounts"),
    Feature("clustering", STRUCTURAL, "Neighbours connected", "Share of its counterparties that also deal with each other",
            "{v:.0%} of its counterparties deal with each other"),
    Feature("kcore", STRUCTURAL, "Core depth", "How deep it sits in the dense part of the network", "core level {v:.0f}"),
    Feature("community_size_log", STRUCTURAL, "Group size", "Log of the size of its network group", "group size (log) {v:.1f}"),
    Feature("community_internal_flow", STRUCTURAL, "Money kept in group", "Share of its group's money that stays inside the group",
            "{v:.0%} of its group's money stays inside"),
    Feature("nbr_new_share", STRUCTURAL, "New counterparties", "Share of counterparties first seen in the last 24 h",
            "{v:.0%} of counterparties are new in the last 24 h"),
    Feature("nbr_mean_degree_log", STRUCTURAL, "Counterparty size", "Log of the average connections of its counterparties",
            "counterparty size (log) {v:.1f}"),
    # --- Temporal (2.2)
    Feature("in_cnt_1h", TEMPORAL, "Received (1 h)", "Transactions received in the last hour", "{v:.0f} received in the last hour"),
    Feature("in_cnt_6h", TEMPORAL, "Received (6 h)", "Transactions received in the last 6 hours", "{v:.0f} received in 6 h"),
    Feature("in_cnt_24h", TEMPORAL, "Received (24 h)", "Transactions received in the last 24 hours", "{v:.0f} received in 24 h"),
    Feature("out_cnt_1h", TEMPORAL, "Sent (1 h)", "Transactions sent in the last hour", "{v:.0f} sent in the last hour"),
    Feature("out_cnt_6h", TEMPORAL, "Sent (6 h)", "Transactions sent in the last 6 hours", "{v:.0f} sent in 6 h"),
    Feature("out_cnt_24h", TEMPORAL, "Sent (24 h)", "Transactions sent in the last 24 hours", "{v:.0f} sent in 24 h"),
    Feature("cnt_24h_vs_avg", TEMPORAL, "Activity vs usual", "Transactions in the last 24 h against its daily average",
            "{v:.1f} times its usual daily activity"),
    Feature("burstiness", TEMPORAL, "Burstiness", "-1 = evenly spaced activity, +1 = sudden bursts", "burstiness {v:+.2f}"),
    Feature("age_hours", TEMPORAL, "Age in data", "Hours since its first transaction in the data", "first seen {v:.0f} h ago"),
    Feature("active_hours_spread", TEMPORAL, "Active hours", "Distinct hours of the day it has been active", "active in {v:.0f} hours of the day"),
    Feature("new_counterparties_24h_ratio", TEMPORAL, "New vs old counterparties", "New counterparties in 24 h against earlier ones",
            "{v:.1f} new counterparties per earlier one"),
    Feature("night_share", TEMPORAL, "Night activity", "Share of transactions between midnight and 5 am", "{v:.0%} of activity at night"),
    # --- Fund-flow (2.3)
    Feature("pass_through_1h", FLOW, "Pass-through (1 h)", "Share of received money forwarded within 1 hour",
            "forwarded {v:.0%} of inflow within 1 h"),
    Feature("pass_through_24h", FLOW, "Pass-through (24 h)", "Share of received money forwarded within 24 hours",
            "forwarded {v:.0%} of inflow within 24 h"),
    Feature("holding_median_min", FLOW, "Holding time", "Median minutes between money arriving and leaving",
            "money held for {v:.0f} min (median)"),
    Feature("holding_p10_min", FLOW, "Fastest forwards", "10th percentile of minutes between arriving and leaving",
            "fastest tenth of money left within {v:.0f} min"),
    Feature("net_flow_ratio", FLOW, "Net flow", "(in - out) / (in + out); near 0 = everything is passed on", "net flow {v:+.2f}"),
    Feature("in_out_similarity", FLOW, "In matches out (24 h)", "How close money in and money out were in the last 24 h",
            "in and out match {v:.0%} in 24 h"),
    Feature("fan_in_max_1h", FLOW, "Fan-in burst", "Most distinct senders within any one hour", "{v:.0f} senders within 1 h"),
    Feature("fan_out_max_1h", FLOW, "Fan-out burst", "Most distinct receivers within any one hour", "{v:.0f} receivers within 1 h"),
    Feature("chain_count", FLOW, "Chains", "Time-ordered 3-hop paths through the account", "part of {v:.0f} time-ordered chains"),
    Feature("cycle_count", FLOW, "Cycles", "Times money left and came back within 24 h", "money returned {v:.0f} times within 24 h"),
    Feature("self_txn_share", FLOW, "Self-transfers", "Share of its transactions that are to itself", "{v:.0%} self-transfers"),
    # --- Relative amounts (2.4)
    Feature("max_in_vs_own_median", RELATIVE, "Largest inflow vs usual", "Largest received amount against its own median",
            "largest inflow is {v:.1f} times its usual amount"),
    Feature("max_out_vs_own_median", RELATIVE, "Largest outflow vs usual", "Largest sent amount against its own median",
            "largest outflow is {v:.1f} times its usual amount"),
    Feature("max_in_pctl", RELATIVE, "Largest inflow rank", "Percentile of its largest inflow among all transactions",
            "largest inflow is bigger than {v:.0f}% of all transfers"),
    Feature("mean_amount_pctl", RELATIVE, "Typical amount rank", "Percentile of its mean amount among all transactions",
            "typical amount is bigger than {v:.0f}% of all transfers"),
    Feature("round_amount_share", RELATIVE, "Round amounts", "Share of its amounts that are round numbers", "{v:.0%} round amounts"),
    Feature("channel_entropy", RELATIVE, "Channel mix", "How mixed its payment channels are (0 = one channel)", "channel mix {v:.2f}"),
    # --- Rule flags (2.5)
    Feature("rule_chain", RULES, "Chain rule", "Rule: part of a fast chain of 3 or more hops", "chain rule fired"),
    Feature("rule_fan_in", RULES, "Fan-in rule", "Rule: many senders, then forwarded", "fan-in rule fired"),
    Feature("rule_fan_out", RULES, "Fan-out rule", "Rule: received, then paid out to many", "fan-out rule fired"),
    Feature("rule_cycle", RULES, "Cycle rule", "Rule: money left and returned", "cycle rule fired"),
    Feature("rule_pass_through", RULES, "Pass-through rule", "Rule: forwarded nearly everything within the hour", "pass-through rule fired"),
)

BY_NAME: dict[str, Feature] = {feature.name: feature for feature in CATALOG}

# Per-account counts and amounts only: no counterparty, graph or in-to-out matching information.
TXN_ONLY_NAMES = (
    "in_cnt_1h", "in_cnt_6h", "in_cnt_24h", "out_cnt_1h", "out_cnt_6h", "out_cnt_24h",
    "cnt_24h_vs_avg", "burstiness", "age_hours", "active_hours_spread", "night_share",
    "max_in_vs_own_median", "max_out_vs_own_median", "max_in_pctl", "mean_amount_pctl",
    "round_amount_share", "channel_entropy",
)


def names_in(*families: str) -> list[str]:
    """Feature names belonging to the given families, in catalogue order."""
    return [feature.name for feature in CATALOG if feature.family in families]


FEATURE_SETS: dict[str, list[str]] = {
    "TXN_ONLY": list(TXN_ONLY_NAMES),
    "STRUCTURAL": names_in(STRUCTURAL),
    "TEMPORAL": names_in(TEMPORAL),
    "FLOW": names_in(FLOW),
    "RELATIVE": names_in(RELATIVE),
    "RULES": names_in(RULES),
    "FULL": [feature.name for feature in CATALOG],
}


def value_text(name: str, value: float) -> str:
    """The feature's value in words for the reasons panel."""
    if value != value:                       # NaN
        return "not available"
    return BY_NAME[name].value_text.format(v=value)
