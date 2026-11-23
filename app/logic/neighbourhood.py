"""Build the small network around one account for the graph view (docs/06-DASHBOARD.md 5.2).

`build_network` prepares look-up tables once per run; `build_neighbourhood` then answers
each hop from those tables without scanning every transaction again.
"""

import numpy as np
import pandas as pd

FLAGGED = ("LIKELY_MULE", "WATCH")


def merged_edges(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per (sender, receiver): number of transfers, total amount, first and last time."""
    real = transactions[~transactions["is_self"]]
    return real.groupby(["sender_id", "receiver_id"], as_index=False).agg(
        n_txn=("txn_id", "size"), amount=("amount", "sum"),
        first_time=("timestamp", "min"), last_time=("timestamp", "max"))


def build_network(transactions: pd.DataFrame) -> dict:
    """Look-up tables for one run: merged arrows by sender and by receiver, and transfers by sender.

    Each table is sorted by its key, and the sorted keys are also kept as a plain array
    ("<name>_keys") so look-ups can binary-search them without converting anything.
    """
    edges = merged_edges(transactions)
    real = transactions[~transactions["is_self"]]
    tables = {
        "out": edges.sort_values("sender_id", kind="stable").reset_index(drop=True),
        "in": edges.sort_values("receiver_id", kind="stable").reset_index(drop=True),
        "transfers": real.sort_values("sender_id", kind="stable").reset_index(drop=True),
    }
    key_column = {"out": "sender_id", "in": "receiver_id", "transfers": "sender_id"}
    keys = {f"{name}_keys": tables[name][column].to_numpy(dtype=object) for name, column in key_column.items()}
    return {**tables, **keys}


def rows_for(network: dict, name: str, accounts: list[str]) -> pd.DataFrame:
    """Rows of one look-up table whose key is one of `accounts`, found by binary search."""
    table, keys = network[name], network[f"{name}_keys"]
    wanted = np.array(accounts, dtype=object)
    starts = np.searchsorted(keys, wanted, side="left") if len(wanted) else []
    ends = np.searchsorted(keys, wanted, side="right") if len(wanted) else []
    pieces = [table.iloc[lo:hi] for lo, hi in zip(starts, ends) if hi > lo]
    return pd.concat(pieces, ignore_index=True) if pieces else table.iloc[:0]


def edges_touching(network: dict, accounts: list[str]) -> pd.DataFrame:
    """Every merged arrow that starts or ends at one of `accounts`."""
    both = pd.concat([rows_for(network, "out", accounts), rows_for(network, "in", accounts)], ignore_index=True)
    return both.drop_duplicates(["sender_id", "receiver_id"])


def neighbours_within(network: dict, centre: str, hops: int) -> tuple[set[str], pd.DataFrame]:
    """Accounts reachable from the centre in at most `hops` steps (either direction),
    and the arrows found on the way."""
    reached, frontier, found = {centre}, [centre], []
    for _ in range(hops):
        touching = edges_touching(network, frontier)
        found.append(touching)
        new = (set(touching["sender_id"]) | set(touching["receiver_id"])) - reached
        reached |= new
        frontier = sorted(new)
        if not frontier:
            break
    arrows = pd.concat(found, ignore_index=True).drop_duplicates(["sender_id", "receiver_id"]) if found \
        else network["out"].iloc[:0]
    return reached - {centre}, arrows


def choose_accounts(candidates: set[str], arrows: pd.DataFrame, verdicts: pd.Series,
                    trail_accounts: set[str], max_nodes: int) -> list[str]:
    """Pick which neighbours to draw when there are too many.

    Order: accounts on the centre's money trail, then flagged accounts, then the rest
    by money exchanged on the arrows found. The centre is always kept.
    """
    money = arrows.groupby("sender_id")["amount"].sum().add(arrows.groupby("receiver_id")["amount"].sum(), fill_value=0)

    def priority(account: str) -> tuple:
        return (account not in trail_accounts, verdicts.get(account) not in FLAGGED,
                -float(money.get(account, 0.0)), account)

    return sorted(candidates, key=priority)[: max(max_nodes - 1, 0)]


def build_neighbourhood(network: dict, scores: pd.DataFrame, trails: pd.DataFrame, centre: str,
                        hops: int, max_nodes: int, hover_max_txns: int) -> dict:
    """Nodes and merged arrows around `centre`.

    `scores` must be indexed by account_id. Returns {"nodes": DataFrame, "edges": DataFrame,
    "hidden": int}. Each edge row carries up to `hover_max_txns` of its transfers in "sample".
    """
    candidates, arrows = neighbours_within(network, centre, hops)
    known = scores.reindex(sorted(candidates | {centre}))
    own_trails = trails[trails["account_id"] == centre]
    trail_accounts = (set(own_trails["from_id"]) | set(own_trails["to_id"])) - {centre}

    kept = choose_accounts(candidates, arrows, known["verdict"].dropna(), trail_accounts, max_nodes)
    shown = [centre, *kept]
    between = edges_touching(network, shown)
    shown_edges = between[between["sender_id"].isin(shown) & between["receiver_id"].isin(shown)].reset_index(drop=True)

    detail = rows_for(network, "transfers", shown)
    detail = detail[detail["receiver_id"].isin(shown)].sort_values("timestamp")
    samples = {pair: group[["timestamp", "amount"]].head(hover_max_txns).to_dict("records")
               for pair, group in detail.groupby(["sender_id", "receiver_id"], sort=False)}
    shown_edges["sample"] = [samples.get(pair, []) for pair in zip(shown_edges["sender_id"], shown_edges["receiver_id"])]

    nodes = pd.DataFrame({"account_id": shown})
    nodes["is_centre"] = nodes["account_id"] == centre
    nodes["on_trail"] = nodes["account_id"].isin(trail_accounts)
    for column in ("risk", "verdict", "short_label", "n_txn", "in_amount", "out_amount"):
        nodes[column] = known[column].reindex(shown).to_numpy()
    nodes["has_score"] = nodes["verdict"].notna()
    return {"nodes": nodes, "edges": shown_edges, "hidden": len(candidates) - len(kept)}


def account_transfers(network: dict, account_id: str) -> pd.DataFrame:
    """One account's transfers with other accounts, newest first, with direction and counterparty."""
    sent = rows_for(network, "transfers", [account_id]).assign(direction="out")
    sent = sent.assign(counterparty=sent["receiver_id"])
    senders = rows_for(network, "in", [account_id])["sender_id"].tolist()
    got = rows_for(network, "transfers", senders)
    got = got[got["receiver_id"] == account_id].assign(direction="in")
    got = got.assign(counterparty=got["sender_id"])
    both = pd.concat([sent, got], ignore_index=True)
    return both.sort_values(["timestamp", "txn_id"], ascending=False, ignore_index=True)
