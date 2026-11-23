"""P5 Rules: five hand-written pattern detectors, each with the transfers that made it fire (F-05).

Every rule looks only at transfers in the look-back window before the cut-off
(docs/05-FEATURES-MODEL.md section 3). Reads transactions only, never labels.
Writes data/features/<dataset>/<cutoff>_rules.parquet and <cutoff>_rules_evidence.parquet.

Usage:
    python -m trace_aml.pipeline.rules --dataset ibm_hi_small --all-snapshots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.features.flow import fifo_match, sent_transfers, window_of
from trace_aml.features.prep import SECONDS_PER_HOUR, History, account_offsets, check_cutoff, prepare
from trace_aml.pipeline.common import file_info, now_utc, write_manifest
from trace_aml.pipeline.features import NEEDED_COLUMNS, cutoff_filename
from trace_aml.pipeline.split import compute_split, snapshot_times

FLAG_COLUMNS = ["rule_chain", "rule_fan_in", "rule_fan_out", "rule_cycle", "rule_pass_through"]
Evidence = list[tuple[int, str, int]]          # (account code, rule name, txn_id)


def window_history(transactions: pd.DataFrame, cutoff: pd.Timestamp, lookback_hours: int) -> History:
    """Only the transfers inside the rule window: cutoff - lookback < timestamp <= cutoff."""
    cutoff = pd.Timestamp(cutoff)
    start = cutoff - pd.Timedelta(hours=lookback_hours)
    inside = transactions[(transactions["timestamp"] > start) & (transactions["timestamp"] <= cutoff)]
    return prepare(inside, cutoff, lookback_hours=None)


# --- PASS_THROUGH --------------------------------------------------------------

def pass_through_rule(history: History, cutoff: pd.Timestamp, rule: dict) -> tuple[np.ndarray, Evidence]:
    """Fires when an account received enough transfers and forwarded nearly all of the money fast."""
    check_cutoff(history, cutoff)
    legs, active = history.legs, history.active
    offsets = account_offsets(legs["a"].to_numpy(), active)
    is_in = legs["is_in"].to_numpy()
    within_s = rule["within_hours"] * SECONDS_PER_HOUR
    total_in, forwarded, _, piece_account, piece_held, piece_in_row, piece_out_row = fifo_match(
        offsets, is_in, legs["t"].to_numpy(dtype="int64"), legs["amount"].to_numpy(dtype="float64"),
        within_s, within_s)
    n_in = np.add.reduceat(is_in.astype("int64"), offsets[:-1]) if len(legs) else np.zeros(len(active))
    share = np.divide(forwarded, total_in, out=np.zeros(len(active)), where=total_in > 0)
    fired = (n_in >= rule["min_in_txns"]) & (share >= rule["forward_share"])

    fast = (piece_held <= within_s) & fired[piece_account]
    txn_id = legs["txn_id"].to_numpy()
    accounts = active[piece_account[fast]]
    evidence = set(zip(accounts.tolist(), txn_id[piece_in_row[fast]].tolist()))
    evidence |= set(zip(accounts.tolist(), txn_id[piece_out_row[fast]].tolist()))
    return fired, [(account, "PASS_THROUGH", txn) for account, txn in sorted(evidence)]


# --- FAN_IN and FAN_OUT --------------------------------------------------------

@njit(cache=True)
def busiest_window(offsets, t, other, window_s, n_codes):
    """Per account: the window of window_s seconds with the most distinct counterparties.

    Returns (count, first row, last row) of the earliest such window.
    """
    n_accounts = offsets.size - 1
    best = np.zeros(n_accounts, np.int64)
    first = np.full(n_accounts, -1, np.int64)
    last = np.full(n_accounts, -1, np.int64)
    seen = np.zeros(n_codes, np.int32)
    for k in range(n_accounts):
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
                best[k], first[k], last[k] = distinct, left, right
        for j in range(left, offsets[k + 1]):
            seen[other[j]] = 0
    return best, first, last


def fan_rule(history: History, cutoff: pd.Timestamp, rule: dict, fan_in: bool) -> tuple[np.ndarray, Evidence]:
    """FAN_IN: many senders in a short window, and most of that money forwarded soon after.
    FAN_OUT: many receivers in a short window, paid from money received shortly before."""
    check_cutoff(history, cutoff)
    name = "FAN_IN" if fan_in else "FAN_OUT"
    legs, active = history.legs, history.active
    burst = legs[legs["is_in"] == fan_in].reset_index(drop=True)       # the many-counterparty side
    other_side = legs[legs["is_in"] != fan_in]
    count, first, last = busiest_window(
        account_offsets(burst["a"].to_numpy(), active), burst["t"].to_numpy(dtype="int64"),
        burst["o"].to_numpy(dtype="int64"), rule["window_hours"] * SECONDS_PER_HOUR, len(history.names))
    needed = rule["min_senders"] if fan_in else rule["min_receivers"]
    share = rule["forward_share"] if fan_in else rule["funded_share"]
    reach_s = (rule["forward_within_hours"] if fan_in else rule["funded_within_hours"]) * SECONDS_PER_HOUR

    fired = np.zeros(len(active), dtype=bool)
    evidence: Evidence = []
    by_account = {code: group for code, group in other_side.groupby("a")}
    for k in np.flatnonzero(count >= needed):
        rows = burst.iloc[first[k]: last[k] + 1]
        opposite = by_account.get(active[k])
        if opposite is None:
            continue
        if fan_in:      # money leaving from the first arrival until reach_s after the last one
            linked = opposite[(opposite["t"] >= rows["t"].iloc[0]) & (opposite["t"] <= rows["t"].iloc[-1] + reach_s)]
        else:           # money arriving from reach_s before the first payout until the last one
            linked = opposite[(opposite["t"] >= rows["t"].iloc[0] - reach_s) & (opposite["t"] <= rows["t"].iloc[-1])]
        if linked["amount"].sum() >= share * rows["amount"].sum():
            fired[k] = True
            evidence += [(int(active[k]), name, int(txn)) for txn in [*rows["txn_id"], *linked["txn_id"]]]
    return fired, evidence


# --- CHAIN -----------------------------------------------------------------------

@njit(cache=True)
def link_hops(out_off, out_t, out_amount, out_r, gap_s, keep):
    """For every transfer: the first transfer that carries the money onward (or -1), and
    one transfer that brought the money in (or -1). "Onward" = sent by the receiver
    within gap_s, keeping at least `keep` of the amount."""
    next_hop = np.full(out_t.size, -1, np.int64)
    previous_hop = np.full(out_t.size, -1, np.int64)
    for e in range(out_t.size):
        lo, hi = window_of(out_off, out_t, out_r[e], out_t[e], out_t[e] + gap_s)
        need = keep * out_amount[e]
        for j in range(lo, hi):
            if out_amount[j] >= need:
                if next_hop[e] == -1:
                    next_hop[e] = j
                if previous_hop[j] == -1:
                    previous_hop[j] = e
    return next_hop, previous_hop


def chain_rule(history: History, cutoff: pd.Timestamp, rule: dict) -> tuple[np.ndarray, Evidence]:
    """Fires for the middle accounts of a time-ordered path of 3 hops: each hop soon after
    the one before and keeping most of the amount."""
    check_cutoff(history, cutoff)
    sent = sent_transfers(history)
    flows = history.flows.sort_values(["s", "t", "txn_id"], kind="stable")
    sender, receiver, txn_id = flows["s"].to_numpy(), flows["r"].to_numpy(), flows["txn_id"].to_numpy()
    next_hop, previous_hop = link_hops(sent["off"], sent["t"], sent["amount"], sent["r"],
                                       rule["hop_within_hours"] * SECONDS_PER_HOUR, rule["min_keep_share"])
    middle = np.flatnonzero((next_hop >= 0) & (previous_hop >= 0))     # hop 2 of a 3-hop path

    fired = np.zeros(len(history.active), dtype=bool)
    fired[np.searchsorted(history.active, np.unique(np.concatenate([sender[middle], receiver[middle]])))] = True
    found = set()
    for e in middle:
        for account in (int(sender[e]), int(receiver[e])):
            found |= {(account, int(txn_id[previous_hop[e]])), (account, int(txn_id[e])),
                      (account, int(txn_id[next_hop[e]]))}
    return fired, [(account, "CHAIN", txn) for account, txn in sorted(found)]


# --- CYCLE -----------------------------------------------------------------------

@njit(cache=True)
def first_cycle(accounts, out_off, out_t, out_r, within_s, max_len, max_branch):
    """Per account: the transfers of the first time-ordered cycle found (rows, -1 padded).

    Same search as the cycle_count feature: 2 to max_len hops (at most 4), completed
    within within_s, not continuing through accounts busier than max_branch.
    """
    found = np.full((accounts.size, 4), -1, np.int64)
    for k in range(accounts.size):
        x = accounts[k]
        done = False
        for e1 in range(out_off[x], out_off[x + 1]):
            if done:
                break
            deadline = out_t[e1] + within_s
            lo2, hi2 = window_of(out_off, out_t, out_r[e1], out_t[e1], deadline)
            if hi2 - lo2 > max_branch:
                continue
            for e2 in range(lo2, hi2):
                if done:
                    break
                if out_r[e2] == x:
                    found[k, 0], found[k, 1] = e1, e2
                    done = True
                    break
                if max_len < 3:
                    continue
                lo3, hi3 = window_of(out_off, out_t, out_r[e2], out_t[e2], deadline)
                if hi3 - lo3 > max_branch:
                    continue
                for e3 in range(lo3, hi3):
                    if done:
                        break
                    if out_r[e3] == x:
                        found[k, 0], found[k, 1], found[k, 2] = e1, e2, e3
                        done = True
                        break
                    if max_len < 4:
                        continue
                    lo4, hi4 = window_of(out_off, out_t, out_r[e3], out_t[e3], deadline)
                    if hi4 - lo4 > max_branch:
                        continue
                    for e4 in range(lo4, hi4):
                        if out_r[e4] == x:
                            found[k, 0], found[k, 1], found[k, 2], found[k, 3] = e1, e2, e3, e4
                            done = True
                            break
    return found


def cycle_rule(history: History, cutoff: pd.Timestamp, rule: dict, max_branch: int) -> tuple[np.ndarray, Evidence]:
    """Fires when money leaves an account and comes back to it through 2 to 4 hops."""
    check_cutoff(history, cutoff)
    sent = sent_transfers(history)
    txn_id = history.flows.sort_values(["s", "t", "txn_id"], kind="stable")["txn_id"].to_numpy()
    found = first_cycle(history.active.astype("int64"), sent["off"], sent["t"], sent["r"],
                        rule["within_hours"] * SECONDS_PER_HOUR, rule["max_len"], max_branch)
    fired = found[:, 0] >= 0
    evidence = [(int(history.active[k]), "CYCLE", int(txn_id[row]))
                for k in np.flatnonzero(fired) for row in found[k] if row >= 0]
    return fired, evidence


# --- All rules together -----------------------------------------------------------

def compute_rules(transactions: pd.DataFrame, cutoff: pd.Timestamp, settings: dict
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the five rules at one cut-off. Returns (flags, evidence) in the contract format.

    `rule_score` here is the plain share of rules that fired. It uses no labels. The
    weighted score of the RULES baseline is worked out later, in training
    (docs/05-FEATURES-MODEL.md section 3.3).
    """
    cutoff = pd.Timestamp(cutoff)
    rules = settings["rules"]
    history = window_history(transactions, cutoff, rules["lookback_hours"])
    results = {
        "rule_chain": chain_rule(history, cutoff, rules["chain"]),
        "rule_fan_in": fan_rule(history, cutoff, rules["fan_in"], fan_in=True),
        "rule_fan_out": fan_rule(history, cutoff, rules["fan_out"], fan_in=False),
        "rule_cycle": cycle_rule(history, cutoff, rules["cycle"], rules["cycle"]["max_branch"]),
        "rule_pass_through": pass_through_rule(history, cutoff, rules["pass_through"]),
    }
    flags = pd.DataFrame({
        "account_id": pd.Series(history.names[history.active], dtype="string"),
        "cutoff": pd.Series(cutoff, index=range(len(history.active))).astype("datetime64[ns]"),
    })
    for column in FLAG_COLUMNS:
        flags[column] = results[column][0].astype("int8")
    flags["rule_score"] = flags[FLAG_COLUMNS].mean(axis=1).astype("float32")

    rows = [row for column in FLAG_COLUMNS for row in results[column][1]]
    evidence = pd.DataFrame(rows, columns=["account", "rule", "txn_id"]).drop_duplicates()
    evidence = pd.DataFrame({
        "account_id": pd.Series(history.names[evidence["account"].to_numpy(dtype="int64")], dtype="string"),
        "rule": evidence["rule"].astype("string").reset_index(drop=True),
        "txn_id": evidence["txn_id"].astype("int64").reset_index(drop=True),
    })
    return flags, evidence.sort_values(["account_id", "rule", "txn_id"], ignore_index=True)


def run_rules(dataset: str, cutoffs: list[pd.Timestamp] | None) -> Path:
    """Run P5 for the given cut-offs (None = every snapshot of the time split)."""
    started = now_utc()
    settings = load_settings()
    source = path("processed") / dataset / c.TRANSACTIONS.filename
    transactions = pd.read_parquet(source, columns=NEEDED_COLUMNS)
    if cutoffs is None:
        split = compute_split(transactions, settings["split"])
        cutoffs = snapshot_times(split, settings["scoring"]["snapshot_every_hours"])

    out = path("features") / dataset
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for cutoff in cutoffs:
        flags, evidence = compute_rules(transactions, cutoff, settings)
        stem = cutoff_filename(cutoff).removesuffix(".parquet")
        flags.to_parquet(out / f"{stem}_rules.parquet", index=False)
        evidence.to_parquet(out / f"{stem}_rules_evidence.parquet", index=False)
        fired = {column: int(flags[column].sum()) for column in FLAG_COLUMNS}
        written[stem] = {"accounts": len(flags), "fired": fired}
        print(f"  {cutoff:%Y-%m-%d %H:%M}: {len(flags):,} accounts, fired {fired}")
    write_manifest(out, "rules", started, dataset=dataset, inputs=[file_info(source)], files=written)
    print(f"Wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 Rules for one cut-off or for every snapshot")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    parser.add_argument("--cutoff", help="one cut-off, e.g. 2022-09-07T00:00")
    parser.add_argument("--all-snapshots", action="store_true", help="every snapshot of the time split")
    args = parser.parse_args()
    if not args.cutoff and not args.all_snapshots:
        parser.error("give --cutoff or --all-snapshots")
    run_rules(args.dataset, None if args.all_snapshots else [pd.Timestamp(args.cutoff)])


if __name__ == "__main__":
    main()
