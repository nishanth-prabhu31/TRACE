"""Fund-flow features: what an account does with the money it receives
(docs/05-FEATURES-MODEL.md 2.3).

The loops here walk through millions of transfers, so they are compiled with numba.
Each one is an ordinary Python loop; remove the @njit line and it still runs (slowly).
"""

import numpy as np
import pandas as pd
from numba import njit

from trace_aml.features.prep import SECONDS_PER_HOUR, History, account_offsets, check_cutoff

SECONDS_PER_MINUTE = 60
TINY = 1e-9   # amounts smaller than this count as "no money left"


@njit(cache=True)
def fifo_match(offsets, is_in, t, amount, short_s, long_s):
    """First-in-first-out matching of money in to money out, per account.

    Rows are sorted by account, then time (arrivals first within the same time).
    Each sent amount is taken from the oldest received money still unspent.
    Returns per account: total received, amount forwarded within short_s and within
    long_s; and one row per matched piece: (account position, seconds held,
    row of the money arriving, row of the money leaving).
    """
    n_accounts = offsets.size - 1
    total_in = np.zeros(n_accounts)
    forwarded_short = np.zeros(n_accounts)
    forwarded_long = np.zeros(n_accounts)
    piece_account = np.empty(t.size, np.int64)
    piece_held = np.empty(t.size, np.float64)
    piece_in_row = np.empty(t.size, np.int64)
    piece_out_row = np.empty(t.size, np.int64)
    n_pieces = 0
    left = amount.copy()                      # unspent part of each received amount
    for k in range(n_accounts):
        head = offsets[k]                     # oldest row that may still hold unspent money
        for i in range(offsets[k], offsets[k + 1]):
            if is_in[i]:
                total_in[k] += amount[i]
                continue
            need = amount[i]
            while need > TINY and head < i:
                if (not is_in[head]) or left[head] <= TINY:
                    head += 1
                    continue
                take = min(left[head], need)
                left[head] -= take
                need -= take
                held = t[i] - t[head]
                piece_account[n_pieces] = k
                piece_held[n_pieces] = held
                piece_in_row[n_pieces] = head
                piece_out_row[n_pieces] = i
                n_pieces += 1
                if held <= short_s:
                    forwarded_short[k] += take
                if held <= long_s:
                    forwarded_long[k] += take
    return (total_in, forwarded_short, forwarded_long, piece_account[:n_pieces], piece_held[:n_pieces],
            piece_in_row[:n_pieces], piece_out_row[:n_pieces])


@njit(cache=True)
def max_distinct_in_window(offsets, t, other, window_s, n_codes):
    """Per account: the most distinct counterparties seen inside any window of window_s seconds."""
    best = np.zeros(offsets.size - 1)
    seen = np.zeros(n_codes, np.int32)        # how many rows of each counterparty are in the window
    for k in range(offsets.size - 1):
        left, distinct = offsets[k], 0
        for right in range(offsets[k], offsets[k + 1]):
            if seen[other[right]] == 0:
                distinct += 1
            seen[other[right]] += 1
            while t[right] - t[left] > window_s:
                seen[other[left]] -= 1
                if seen[other[left]] == 0:
                    distinct -= 1
                left += 1
            if distinct > best[k]:
                best[k] = distinct
        for j in range(left, offsets[k + 1]):  # empty the window before the next account
            seen[other[j]] = 0
    return best


@njit(cache=True)
def window_of(out_off, out_t, account, t_from, t_to):
    """Positions [lo, hi) of the account's sent transfers with t_from <= time <= t_to."""
    start, end = out_off[account], out_off[account + 1]
    lo = start + np.searchsorted(out_t[start:end], t_from, side="left")
    hi = start + np.searchsorted(out_t[start:end], t_to, side="right")
    return lo, hi


@njit(cache=True)
def count_onward_hops(out_off, out_t, out_amount, out_r, gap_s, keep):
    """For every transfer: how many transfers its receiver then sent on, within gap_s,
    each keeping at least `keep` of the amount."""
    onward = np.zeros(out_t.size)
    for e in range(out_t.size):
        lo, hi = window_of(out_off, out_t, out_r[e], out_t[e], out_t[e] + gap_s)
        need = keep * out_amount[e]
        for j in range(lo, hi):
            if out_amount[j] >= need:
                onward[e] += 1
    return onward


@njit(cache=True)
def count_chains(offsets, accounts, is_in, t, amount, out_off, out_t, out_amount, onward, gap_s, keep):
    """Per account X: number of time-ordered paths A -> X -> C -> D, each hop within
    gap_s of the one before and keeping at least `keep` of its amount."""
    chains = np.zeros(accounts.size)
    for k in range(accounts.size):
        for i in range(offsets[k], offsets[k + 1]):
            if not is_in[i]:
                continue
            lo, hi = window_of(out_off, out_t, accounts[k], t[i], t[i] + gap_s)
            need = keep * amount[i]
            for j in range(lo, hi):
                if out_amount[j] >= need:
                    chains[k] += onward[j]
    return chains


@njit(cache=True)
def count_cycles(accounts, out_off, out_t, out_r, within_s, max_len, max_branch):
    """Per account X: time-ordered cycles X -> ... -> X of 2 to max_len (at most 4) hops,
    completed within within_s of the first hop.

    The search does not continue through an account that sent more than max_branch
    transfers in the time left: such busy accounts are hubs, and following every one
    of their transfers would take too long. Cycles through them are not counted.
    """
    cycles = np.zeros(accounts.size)
    for k in range(accounts.size):
        x = accounts[k]
        for e1 in range(out_off[x], out_off[x + 1]):
            deadline = out_t[e1] + within_s
            lo2, hi2 = window_of(out_off, out_t, out_r[e1], out_t[e1], deadline)
            if hi2 - lo2 > max_branch:
                continue
            for e2 in range(lo2, hi2):
                if out_r[e2] == x:
                    cycles[k] += 1
                    continue
                if max_len < 3:
                    continue
                lo3, hi3 = window_of(out_off, out_t, out_r[e2], out_t[e2], deadline)
                if hi3 - lo3 > max_branch:
                    continue
                for e3 in range(lo3, hi3):
                    if out_r[e3] == x:
                        cycles[k] += 1
                        continue
                    if max_len < 4:
                        continue
                    lo4, hi4 = window_of(out_off, out_t, out_r[e3], out_t[e3], deadline)
                    if hi4 - lo4 > max_branch:
                        continue
                    for e4 in range(lo4, hi4):
                        if out_r[e4] == x:
                            cycles[k] += 1
    return cycles


def sent_transfers(history: History) -> dict[str, np.ndarray]:
    """Every transfer sorted by sender then time, with each sender's start position."""
    flows = history.flows.sort_values(["s", "t", "txn_id"], kind="stable")
    sender = flows["s"].to_numpy()
    return {
        "off": np.searchsorted(sender, np.arange(len(history.names) + 1)),
        "t": flows["t"].to_numpy(dtype="int64"),
        "amount": flows["amount"].to_numpy(dtype="float64"),
        "r": flows["r"].to_numpy(dtype="int64"),
    }


def holding_times(history: History, piece_account: np.ndarray, piece_held: np.ndarray) -> pd.DataFrame:
    """Median and 10th percentile of how long matched money was held, in minutes."""
    minutes = pd.Series(piece_held / SECONDS_PER_MINUTE).groupby(history.active[piece_account])
    return pd.DataFrame({"holding_median_min": minutes.median(), "holding_p10_min": minutes.quantile(0.1)})


def flow_features(history: History, cutoff: pd.Timestamp, settings: dict) -> pd.DataFrame:
    """All fund-flow features for the scored accounts, indexed by account code."""
    check_cutoff(history, cutoff)
    features, legs, active = settings["features"], history.legs, history.active
    out = pd.DataFrame(index=active)
    if len(active) == 0:
        return out
    offsets = account_offsets(legs["a"].to_numpy(), active)
    is_in = legs["is_in"].to_numpy()
    t = legs["t"].to_numpy(dtype="int64")
    amount = legs["amount"].to_numpy(dtype="float64")

    short_h, long_h = features["pass_through_delta_hours"]
    total_in, fwd_short, fwd_long, piece_account, piece_held, _, _ = fifo_match(
        offsets, is_in, t, amount, short_h * SECONDS_PER_HOUR, long_h * SECONDS_PER_HOUR)
    received = np.where(total_in > 0, total_in, np.nan)
    out[f"pass_through_{short_h}h"] = fwd_short / received
    out[f"pass_through_{long_h}h"] = fwd_long / received
    out = out.join(holding_times(history, piece_account, piece_held))

    total_out = np.bincount(np.searchsorted(active, legs["a"].to_numpy()[~is_in]),
                            weights=amount[~is_in], minlength=len(active))
    out["net_flow_ratio"] = (total_in - total_out) / (total_in + total_out)

    recent = legs[legs["t"] > history.cutoff_s - features["windows_hours"][-1] * SECONDS_PER_HOUR]
    recent_in = recent[recent["is_in"]].groupby("a")["amount"].sum().reindex(active, fill_value=0.0)
    recent_out = recent[~recent["is_in"]].groupby("a")["amount"].sum().reindex(active, fill_value=0.0)
    moved = recent_in + recent_out
    out["in_out_similarity"] = 1.0 - (recent_in - recent_out).abs() / moved.where(moved > 0)

    window_s = features["fan_window_hours"] * SECONDS_PER_HOUR
    for name, side in (("fan_in_max_1h", legs[legs["is_in"]]), ("fan_out_max_1h", legs[~legs["is_in"]])):
        out[name] = max_distinct_in_window(
            account_offsets(side["a"].to_numpy(), active), side["t"].to_numpy(dtype="int64"),
            side["o"].to_numpy(dtype="int64"), window_s, len(history.names))

    sent = sent_transfers(history)
    chain, cycle = features["chain"], features["cycle"]
    gap_s = chain["max_gap_hours"] * SECONDS_PER_HOUR
    onward = count_onward_hops(sent["off"], sent["t"], sent["amount"], sent["r"], gap_s, chain["min_keep_share"])
    out["chain_count"] = count_chains(offsets, active.astype("int64"), is_in, t, amount, sent["off"],
                                      sent["t"], sent["amount"], onward, gap_s, chain["min_keep_share"])
    out["cycle_count"] = count_cycles(active.astype("int64"), sent["off"], sent["t"], sent["r"],
                                      cycle["within_hours"] * SECONDS_PER_HOUR, cycle["max_len"],
                                      cycle["max_branch"])

    n_legs = legs.groupby("a").size().reindex(active, fill_value=0)
    n_self = history.self_counts[active]
    out["self_txn_share"] = n_self / (n_self + n_legs)
    return out
