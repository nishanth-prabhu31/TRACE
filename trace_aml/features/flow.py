"""Fund-flow features: what an account does with the money it receives (docs/05-FEATURES-MODEL.md 2.3)."""

from collections import deque

import numpy as np
import pandas as pd


def fifo_flow(events: pd.DataFrame, deltas_hours: list[int]) -> pd.DataFrame:
    """Pass-through shares and holding times from first-in-first-out matching.

    Every received amount joins the back of the account's queue; every sent amount is
    paid from the front (oldest money first). Each matched piece gives a holding time.
    `events` must be sorted by account and time, self-transfers removed.
    """
    accounts = events["account_id"].to_numpy()
    is_in = (events["direction"] == "IN").to_numpy()
    times = (events["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)) / 60.0  # minutes
    amounts = events["amount"].to_numpy()
    deltas_min = [h * 60 for h in deltas_hours]

    rows = {}
    start = 0
    n = len(accounts)
    while start < n:
        end = start
        while end < n and accounts[end] == accounts[start]:
            end += 1
        queue: deque = deque()
        received = 0.0
        forwarded = [0.0] * len(deltas_min)
        holds, hold_weights = [], []
        for i in range(start, end):
            if is_in[i]:
                queue.append([times[i], amounts[i]])
                received += amounts[i]
                continue
            to_pay = amounts[i]
            while to_pay > 1e-9 and queue:
                arrived, left = queue[0]
                piece = min(left, to_pay)
                held = times[i] - arrived
                for k, d in enumerate(deltas_min):
                    if held <= d:
                        forwarded[k] += piece
                holds.append(held)
                hold_weights.append(piece)
                to_pay -= piece
                if piece >= left - 1e-9:
                    queue.popleft()
                else:
                    queue[0][1] = left - piece
        row = {f"pass_through_{h}h": (forwarded[k] / received if received > 0 else np.nan)
               for k, h in enumerate(deltas_hours)}
        if holds:
            holds_arr = np.array(holds)
            row["holding_median_min"] = float(np.median(holds_arr))
            row["holding_p10_min"] = float(np.percentile(holds_arr, 10))
        else:
            row["holding_median_min"] = np.nan
            row["holding_p10_min"] = np.nan
        rows[accounts[start]] = row
        start = end
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "account_id"
    return out


def fan_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Most distinct senders / receivers within one clock hour (an approximation of any 1 h window)."""
    ev = events.assign(hour=events["timestamp"].dt.floor("h"))
    distinct = ev.groupby(["account_id", "direction", "hour"])["counterparty"].nunique()
    best = distinct.groupby(level=["account_id", "direction"]).max().unstack()
    return pd.DataFrame({
        "fan_in_max_1h": best.get("IN", pd.Series(dtype=float)),
        "fan_out_max_1h": best.get("OUT", pd.Series(dtype=float)),
    })


def chains_and_cycles(transactions: pd.DataFrame, cutoff: pd.Timestamp, window_hours: int,
                      max_gap_hours: int, min_keep_share: float) -> pd.DataFrame:
    """Time-ordered chains and short cycles through each account, using the last `window_hours`.

    Hop pair at X: money arrives at X (t1, a1), then X sends on (t2, a2) with
    0 < t2 - t1 <= max_gap and a2 >= min_keep_share * a1.
    chain_count(X): hop pairs at X whose onward transfer is itself followed by another hop.
    cycle_count(X): hop pairs that return to the sender (A->X->A), or A->X->C->A.
    """
    start = cutoff - pd.Timedelta(hours=window_hours)
    w = transactions[(transactions["timestamp"] > start) & (transactions["timestamp"] <= cutoff)
                     & ~transactions["is_self"]][["txn_id", "sender_id", "receiver_id", "timestamp", "amount"]]
    gap = pd.Timedelta(hours=max_gap_hours)

    pairs = w.merge(w, left_on="receiver_id", right_on="sender_id", suffixes=("_1", "_2"))
    dt = pairs["timestamp_2"] - pairs["timestamp_1"]
    pairs = pairs[(dt > pd.Timedelta(0)) & (dt <= gap)
                  & (pairs["amount_2"] >= min_keep_share * pairs["amount_1"])]
    pairs = pairs[["txn_id_1", "txn_id_2", "sender_id_1", "receiver_id_1", "receiver_id_2"]]
    pairs.columns = ["t1", "t2", "a", "x", "c"]

    continues = pairs["t2"].isin(set(pairs["t1"]))
    chain = pairs[continues].groupby("x").size()

    back2 = pairs[pairs["a"] == pairs["c"]].groupby("x").size()
    third = pairs.merge(pairs[["t1", "t2", "c"]].rename(columns={"t1": "t2", "t2": "t3", "c": "d"}), on="t2")
    back3 = third[(third["d"] == third["a"]) & (third["a"] != third["x"])].groupby("x").size()
    cycles = back2.add(back3, fill_value=0)

    return pd.DataFrame({"chain_count": chain, "cycle_count": cycles}).fillna(0)


def flow_features(events: pd.DataFrame, transactions: pd.DataFrame, cutoff: pd.Timestamp,
                  settings: dict) -> pd.DataFrame:
    """All fund-flow features for the accounts in `events` (events limited to <= cutoff)."""
    feats = settings["features"]
    index = pd.Index(events["account_id"].unique(), name="account_id")
    ev = events[~events["is_self"]]

    out = pd.DataFrame(index=index)
    out = out.join(fifo_flow(ev, feats["pass_through_delta_hours"]))

    sums = ev.groupby(["account_id", "direction"])["amount"].sum().unstack(fill_value=0)
    total_in = sums.get("IN", pd.Series(0.0, index=sums.index))
    total_out = sums.get("OUT", pd.Series(0.0, index=sums.index))
    out["net_flow_ratio"] = ((total_in - total_out) / (total_in + total_out)).reindex(index)

    recent = ev[ev["timestamp"] > cutoff - pd.Timedelta(hours=24)]
    rsums = recent.groupby(["account_id", "direction"])["amount"].sum().unstack(fill_value=0)
    r_in = rsums.get("IN", pd.Series(0.0, index=rsums.index))
    r_out = rsums.get("OUT", pd.Series(0.0, index=rsums.index))
    out["in_out_similarity"] = (1 - (r_in - r_out).abs() / (r_in + r_out)).reindex(index)

    out = out.join(fan_counts(ev))
    chain = feats["chain"]
    cc = chains_and_cycles(transactions, cutoff, feats["chain_window_hours"],
                           chain["max_gap_hours"], chain["min_keep_share"])
    out = out.join(cc)
    out[["chain_count", "cycle_count"]] = out[["chain_count", "cycle_count"]].fillna(0)
    return out.astype("float32")
