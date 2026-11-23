"""Structural features: where an account sits in the network (docs/05-FEATURES-MODEL.md 2.1).

The graph uses every non-self transaction up to the cut-off; repeat transfers between the
same pair are merged into one edge weighted by total amount.
"""

import random
import time

import igraph as ig
import numpy as np
import pandas as pd


def build_graph(transactions: pd.DataFrame, cutoff: pd.Timestamp) -> ig.Graph:
    """Directed, amount-weighted graph of transactions with timestamp <= cutoff."""
    txns = transactions[(transactions["timestamp"] <= cutoff) & ~transactions["is_self"]]
    edges = (txns.groupby(["sender_id", "receiver_id"], observed=True)["amount"].sum()
             .reset_index().rename(columns={"amount": "weight"}))
    return ig.Graph.DataFrame(edges, directed=True, use_vids=False)


def structural_features(transactions: pd.DataFrame, cutoff: pd.Timestamp, accounts: pd.Index,
                        betweenness_hops: int | None, new_hours: int, seed: int) -> pd.DataFrame:
    """One row per account in `accounts` (accounts with no non-self transfers get NaN)."""
    graph = build_graph(transactions, cutoff)
    names = pd.Index(graph.vs["name"], name="account_id")
    n = graph.vcount()
    out = pd.DataFrame(index=names)

    out["in_degree"] = graph.degree(mode="in")
    out["out_degree"] = graph.degree(mode="out")
    out["in_out_degree_ratio"] = out["in_degree"] / (out["out_degree"] + 1)
    undirected = graph.as_undirected(mode="collapse", combine_edges="sum")
    out["counterparties"] = undirected.degree()
    out["pagerank"] = np.array(graph.pagerank(weights="weight")) * n

    if betweenness_hops:
        start = time.time()
        raw = np.array(graph.betweenness(directed=True, cutoff=betweenness_hops))
        out["betweenness_local"] = raw / max(n - 1, 1)
        print(f"    betweenness ({betweenness_hops} hops) took {time.time() - start:.0f} s")

    out["clustering"] = np.nan_to_num(undirected.transitivity_local_undirected(mode="zero"))
    out["kcore"] = undirected.coreness()

    ig.set_random_number_generator(random.Random(seed))  # same communities every run
    communities = undirected.community_multilevel(weights="weight")
    ig.set_random_number_generator(random)
    membership = np.array(communities.membership)
    sizes = np.bincount(membership)
    out["community_size_log"] = np.log1p(sizes[membership])

    edge_list = np.array(undirected.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    a, b = edge_list[:, 0], edge_list[:, 1]
    src, dst = membership[a], membership[b]
    weight = np.array(undirected.es["weight"])
    touching = np.bincount(src, weight, len(sizes)) + np.bincount(dst, weight, len(sizes))
    inside = np.bincount(src[src == dst], weight[src == dst], len(sizes)) * 2
    out["community_internal_flow"] = (inside / np.clip(touching, 1e-12, None))[membership]

    first_seen = pd.concat([
        transactions.loc[transactions["timestamp"] <= cutoff, ["sender_id", "timestamp"]]
        .rename(columns={"sender_id": "account_id"}),
        transactions.loc[transactions["timestamp"] <= cutoff, ["receiver_id", "timestamp"]]
        .rename(columns={"receiver_id": "account_id"}),
    ]).groupby("account_id")["timestamp"].min()
    is_new = (first_seen.reindex(names) > cutoff - pd.Timedelta(hours=new_hours)).to_numpy()
    degree = np.array(undirected.degree(), dtype=float)
    new_nbr = np.zeros(n)  # sums over each account's neighbours, both edge directions
    nbr_degree = np.zeros(n)
    np.add.at(new_nbr, a, is_new[b])
    np.add.at(new_nbr, b, is_new[a])
    np.add.at(nbr_degree, a, degree[b])
    np.add.at(nbr_degree, b, degree[a])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["nbr_new_share"] = new_nbr / degree
        out["nbr_mean_degree_log"] = np.log1p(nbr_degree / degree)

    return out.reindex(accounts).astype("float32")
