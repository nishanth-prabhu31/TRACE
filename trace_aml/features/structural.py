"""Structural features: where an account sits in the transfer network
(docs/05-FEATURES-MODEL.md 2.1).

The network is every transfer between two different accounts up to the cut-off.
Repeat transfers between the same pair are merged into one arrow weighted by total amount.
"""

import random

import igraph as ig
import numpy as np
import pandas as pd

from trace_aml.features.prep import SECONDS_PER_HOUR, History, check_cutoff


def merged_pairs(history: History, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(directed, undirected) arrows with summed amounts. Undirected rows have lo < hi."""
    check_cutoff(history, cutoff)
    directed = history.flows.groupby(["s", "r"], as_index=False)["amount"].sum()
    lo = np.minimum(directed["s"], directed["r"])
    hi = np.maximum(directed["s"], directed["r"])
    undirected = directed.assign(lo=lo, hi=hi).groupby(["lo", "hi"], as_index=False)["amount"].sum()
    return directed, undirected


def community_features(graph: ig.Graph, undirected: pd.DataFrame, n_accounts: int, seed: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per account: log(1 + size of its Louvain community), and the share of the
    community's money that stays inside it."""
    random.seed(seed)                                   # igraph draws from Python's random
    membership = np.array(graph.community_multilevel(weights="weight").membership)
    size = np.bincount(membership)[membership]

    lo_group, hi_group = membership[undirected["lo"]], membership[undirected["hi"]]
    amount = undirected["amount"].to_numpy()
    inside = lo_group == hi_group
    n_groups = membership.max() + 1
    internal = np.bincount(lo_group[inside], weights=amount[inside], minlength=n_groups)
    crossing = (np.bincount(lo_group[~inside], weights=amount[~inside], minlength=n_groups)
                + np.bincount(hi_group[~inside], weights=amount[~inside], minlength=n_groups))
    total = internal + crossing
    share = np.divide(internal, total, out=np.zeros(n_groups), where=total > 0)
    return np.log1p(size), share[membership]


def structural_features(history: History, cutoff: pd.Timestamp, settings: dict) -> pd.DataFrame:
    """All structural features for the scored accounts, indexed by account code."""
    check_cutoff(history, cutoff)
    n = len(history.names)
    active = history.active
    directed, undirected = merged_pairs(history, cutoff)

    in_degree = np.bincount(directed["r"], minlength=n)
    out_degree = np.bincount(directed["s"], minlength=n)
    counterparties = np.bincount(undirected["lo"], minlength=n) + np.bincount(undirected["hi"], minlength=n)

    out = pd.DataFrame(index=active)
    out["in_degree"] = in_degree[active]
    out["out_degree"] = out_degree[active]
    out["in_out_degree_ratio"] = in_degree[active] / (out_degree[active] + 1.0)
    out["counterparties"] = counterparties[active]

    arrows = ig.Graph(n=n, edges=directed[["s", "r"]].to_numpy().tolist(), directed=True)
    arrows.es["weight"] = directed["amount"].to_numpy()
    links = ig.Graph(n=n, edges=undirected[["lo", "hi"]].to_numpy().tolist(), directed=False)
    links.es["weight"] = undirected["amount"].to_numpy()

    out["pagerank"] = np.array(arrows.pagerank(weights="weight", directed=True))[active] * n
    between = settings["features"]["betweenness"]
    if between["enabled"]:
        pairs = max((n - 1) * (n - 2), 1)
        values = arrows.betweenness(vertices=active.tolist(), directed=True, cutoff=between["cutoff_hops"])
        out["betweenness_local"] = np.array(values) / pairs
    out["clustering"] = links.transitivity_local_undirected(vertices=active.tolist(), mode="zero")
    out["kcore"] = np.array(links.coreness())[active]

    size_log, internal_flow = community_features(links, undirected, n, settings["random_seed"])
    out["community_size_log"] = size_log[active]
    out["community_internal_flow"] = internal_flow[active]

    first_met = history.legs.groupby(["a", "o"])["t"].min()
    is_new = first_met > history.cutoff_s - settings["features"]["new_counterparty_hours"] * SECONDS_PER_HOUR
    out["nbr_new_share"] = is_new.groupby(level="a").mean().reindex(active)

    both_ways = pd.DataFrame({
        "a": np.concatenate([undirected["lo"], undirected["hi"]]),
        "other_degree": np.concatenate([counterparties[undirected["hi"]], counterparties[undirected["lo"]]]),
    })
    out["nbr_mean_degree_log"] = np.log1p(both_ways.groupby("a")["other_degree"].mean().reindex(active))
    return out
