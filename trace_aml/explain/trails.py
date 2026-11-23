"""Money trails: the time-ordered chain of transfers through a flagged account
(docs/05-FEATURES-MODEL.md section 6.4).

This is a search over transfers, not a model prediction. From the account it walks
backwards along money received and forwards along money sent, keeping only the
largest few options at each step, and joins the two halves into trails.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class TransferIndex:
    """Transfers between different accounts, sorted two ways for fast look-ups."""

    names: np.ndarray            # account ID per code
    code_of: dict                # account ID -> code
    by_receiver: dict            # arrays sorted by (receiver, time): off, t, amount, other, txn_id
    by_sender: dict              # arrays sorted by (sender, time): off, t, amount, other, txn_id


def build_index(transactions: pd.DataFrame, cutoff: pd.Timestamp) -> TransferIndex:
    """Index every transfer with timestamp <= cutoff (self-transfers left out)."""
    known = transactions[(transactions["timestamp"] <= cutoff) & ~transactions["is_self"]]
    codes, names = pd.factorize(pd.concat([known["sender_id"], known["receiver_id"]], ignore_index=True))
    table = pd.DataFrame({
        "s": codes[: len(known)], "r": codes[len(known):],
        "t": known["timestamp"].to_numpy(dtype="datetime64[s]").astype("int64"),
        "amount": known["amount"].to_numpy(dtype="float64"), "txn_id": known["txn_id"].to_numpy(),
    })

    def side(own: str, other: str) -> dict:
        ordered = table.sort_values([own, "t", "txn_id"], kind="stable")
        return {"off": np.searchsorted(ordered[own].to_numpy(), np.arange(len(names) + 1)),
                "t": ordered["t"].to_numpy(), "amount": ordered["amount"].to_numpy(),
                "other": ordered[other].to_numpy(), "txn_id": ordered["txn_id"].to_numpy()}

    return TransferIndex(np.asarray(names), {name: code for code, name in enumerate(names)},
                         side("r", "s"), side("s", "r"))


def largest_in_window(side: dict, account: int, t_from: int, t_to: int, branch: int) -> list[int]:
    """Rows of the account's `branch` largest transfers with t_from <= time <= t_to."""
    start, end = side["off"][account], side["off"][account + 1]
    lo = start + np.searchsorted(side["t"][start:end], t_from, side="left")
    hi = start + np.searchsorted(side["t"][start:end], t_to, side="right")
    if hi <= lo:
        return []
    rows = np.arange(lo, hi)
    order = np.lexsort((rows, -side["amount"][lo:hi]))        # largest first; earlier row wins ties
    return rows[order[:branch]].tolist()


def walk(side: dict, account: int, t_edge: int, hops_left: int, gap_s: int, branch: int, forward: bool
         ) -> list[list[int]]:
    """Every path of up to hops_left transfers leading away from `account` in one direction.

    forward = True follows money sent (each hop at or after the previous one, within gap_s);
    forward = False follows money received (each hop at or before the next one, within gap_s).
    Paths are lists of rows, nearest hop first.
    """
    if hops_left == 0:
        return [[]]
    window = (t_edge, t_edge + gap_s) if forward else (t_edge - gap_s, t_edge)
    paths = []
    for row in largest_in_window(side, account, window[0], window[1], branch):
        tails = walk(side, int(side["other"][row]), int(side["t"][row]), hops_left - 1, gap_s, branch, forward)
        paths += [[row, *tail] for tail in tails]
    return paths or [[]]


def trails_for_account(index: TransferIndex, account_id: str, cutoff: pd.Timestamp, settings: dict
                       ) -> list[list[tuple[int, str]]]:
    """Up to max_trails trails through the account: lists of (txn_id, side), earliest hop first.

    A trail needs at least one transfer into the account and one out of it. Trails are
    ranked by the smallest amount along them (the money that clearly went all the way).
    """
    if account_id not in index.code_of:
        return []
    account, cutoff_s = index.code_of[account_id], int(pd.Timestamp(cutoff).value // 10**9)
    gap_s, hops, branch = settings["max_gap_hours"] * SECONDS_PER_HOUR, settings["max_hops"], settings["branch"]
    received, sent = index.by_receiver, index.by_sender

    found: dict[tuple, float] = {}
    for first_in in largest_in_window(received, account, cutoff_s - gap_s, cutoff_s, branch):
        arrived = int(received["t"][first_in])
        earlier = walk(received, int(received["other"][first_in]), arrived, hops - 1, gap_s, branch, forward=False)
        later = [path for path in walk(sent, account, arrived, hops, gap_s, branch, forward=True) if path]
        for before in earlier:
            for after in later:
                before_rows = [first_in, *before][::-1]                    # earliest first
                amounts = [*received["amount"][before_rows], *sent["amount"][after]]
                key = (tuple(int(received["txn_id"][row]) for row in before_rows),
                       tuple(int(sent["txn_id"][row]) for row in after))
                found[key] = float(min(amounts))
    best = sorted(found.items(), key=lambda item: (-item[1], item[0]))[: settings["max_trails"]]
    return [[*((txn, "BEFORE") for txn in before), *((txn, "AFTER") for txn in after)] for (before, after), _ in best]


def build_trails(transactions: pd.DataFrame, accounts: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """trails.parquet rows for every (account_id, cutoff) in `accounts` (contract section 5.3)."""
    by_txn = transactions.set_index("txn_id")[["sender_id", "receiver_id", "timestamp", "amount"]]
    rows = []
    for cutoff, group in accounts.groupby("cutoff"):
        index = build_index(transactions, cutoff)
        for account_id in group["account_id"]:
            for trail_number, trail in enumerate(trails_for_account(index, account_id, cutoff, settings), start=1):
                for hop, (txn_id, side) in enumerate(trail, start=1):
                    rows.append((account_id, trail_number, hop, txn_id, side))
    out = pd.DataFrame(rows, columns=["account_id", "trail_id", "hop", "txn_id", "side"])
    detail = by_txn.reindex(out["txn_id"]).reset_index(drop=True)
    return pd.DataFrame({
        "account_id": out["account_id"].astype("string"),
        "trail_id": out["trail_id"].astype("int8"),
        "hop": out["hop"].astype("int8"),
        "txn_id": out["txn_id"].astype("int64"),
        "from_id": detail["sender_id"].astype("string"),
        "to_id": detail["receiver_id"].astype("string"),
        "timestamp": detail["timestamp"].astype("datetime64[ns]"),
        "amount": detail["amount"].astype("float64"),
        "side": out["side"].astype("string"),
    })
