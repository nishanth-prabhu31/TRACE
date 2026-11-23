"""Shared preparation for all feature families: cut the history at the cut-off, once.

Everything a feature function sees comes from `prepare()`, which keeps only
transactions with timestamp <= cutoff (docs/05-FEATURES-MODEL.md section 1).
Feature code never reads label files.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class History:
    """Transactions up to one cut-off, with account IDs turned into integer codes."""

    cutoff: pd.Timestamp
    cutoff_s: int              # the cut-off in seconds
    names: np.ndarray          # account ID for each code
    flows: pd.DataFrame        # transfers between two different accounts: txn_id, s, r, t, amount, ...
    self_counts: np.ndarray    # number of self-transfers per account code
    active: np.ndarray         # codes of the accounts to score, sorted
    legs: pd.DataFrame         # one row per (scored account, transfer it took part in), see make_legs


def to_seconds(timestamps: pd.Series) -> np.ndarray:
    """Timestamps as whole seconds (int64)."""
    return timestamps.to_numpy(dtype="datetime64[s]").astype("int64")


def make_legs(flows: pd.DataFrame, active: np.ndarray) -> pd.DataFrame:
    """Each transfer seen from each scored account's side.

    Columns: a (the account), o (the other account), is_in (True = money arrived),
    t, amount, amount_original, channel, hour, txn_id. Sorted by account, then time;
    within the same time, arrivals come before departures (timestamps are only
    minute-precise, so money that arrives and leaves in the same minute still matches).
    """
    shared = ["t", "amount", "amount_original", "channel", "hour", "txn_id"]
    incoming = flows[["r", "s", *shared]].rename(columns={"r": "a", "s": "o"}).assign(is_in=True)
    outgoing = flows[["s", "r", *shared]].rename(columns={"s": "a", "r": "o"}).assign(is_in=False)
    legs = pd.concat([incoming, outgoing], ignore_index=True)
    legs = legs[np.isin(legs["a"].to_numpy(), active)]
    return legs.sort_values(["a", "t", "is_in", "txn_id"], ascending=[True, True, False, True],
                            kind="stable", ignore_index=True)


def prepare(transactions: pd.DataFrame, cutoff: pd.Timestamp, lookback_hours: int | None) -> History:
    """Keep only what was known at `cutoff` and choose which accounts to score.

    lookback_hours = N: accounts with a transfer in the N hours before the cut-off.
    lookback_hours = None: every account in the data (uploads and replay windows).
    """
    cutoff = pd.Timestamp(cutoff)
    known = transactions[transactions["timestamp"] <= cutoff]
    codes, names = pd.factorize(pd.concat([known["sender_id"], known["receiver_id"]], ignore_index=True))
    sender, receiver = codes[: len(known)].astype("int32"), codes[len(known):].astype("int32")

    table = pd.DataFrame({
        "txn_id": known["txn_id"].to_numpy(),
        "s": sender,
        "r": receiver,
        "t": to_seconds(known["timestamp"]),
        "amount": known["amount"].to_numpy(dtype="float64"),
        "amount_original": known["amount_original"].to_numpy(dtype="float64"),
        "channel": pd.factorize(known["channel"])[0].astype("int16"),
        "hour": known["timestamp"].dt.hour.to_numpy(dtype="int8"),
    })
    is_self = known["is_self"].to_numpy()
    flows = table[~is_self].reset_index(drop=True)
    self_counts = np.bincount(sender[is_self], minlength=len(names))

    cutoff_s = int(cutoff.value // 10**9)
    if lookback_hours is None:
        recent = flows
    else:
        recent = flows[flows["t"] > cutoff_s - lookback_hours * SECONDS_PER_HOUR]
    active = np.unique(np.concatenate([recent["s"].to_numpy(), recent["r"].to_numpy()]))
    return History(cutoff=cutoff, cutoff_s=cutoff_s, names=np.asarray(names), flows=flows,
                   self_counts=self_counts, active=active, legs=make_legs(flows, active))


def check_cutoff(history: History, cutoff: pd.Timestamp) -> None:
    """Refuse to compute features for a cut-off the history was not prepared for."""
    if pd.Timestamp(cutoff) != history.cutoff:
        raise ValueError(f"History was prepared for {history.cutoff}, not {cutoff}")
    if len(history.flows) and history.flows["t"].max() > history.cutoff_s:
        raise ValueError("History contains transactions after the cut-off")


def account_offsets(sorted_accounts: np.ndarray, accounts: np.ndarray) -> np.ndarray:
    """Start position of each account's rows in an array sorted by account (plus the end)."""
    return np.searchsorted(sorted_accounts, np.append(accounts, accounts[-1] + 1 if len(accounts) else 0))
