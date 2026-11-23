"""P5 Rules: five hand-written pattern detectors (F-05, docs/05-FEATURES-MODEL.md section 3).

Each rule looks only at the last `lookback_hours` before the cut-off, outputs a 0/1 flag
per account and the transactions that made it fire. Labels are never read here.

`rule_score` in this file is the plain share of rules that fired (0-1). The weighted
rules-only baseline (each rule weighted by its validation precision) is built at training
time, because weighting needs labels.

Usage:
    python -m trace_aml.pipeline.rules --dataset ibm_hi_small --all
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from trace_aml.config import load_settings, path
from trace_aml.features.events import account_events, active_accounts
from trace_aml.features.flow import fifo_flow
from trace_aml.pipeline.common import now_utc, write_manifest
from trace_aml.pipeline.features import TXN_COLUMNS, cutoff_name, snapshot_cutoffs

RULE_COLUMNS = {"CHAIN": "rule_chain", "FAN_IN": "rule_fan_in", "FAN_OUT": "rule_fan_out",
                "CYCLE": "rule_cycle", "PASS_THROUGH": "rule_pass_through"}


def hop_pairs(window: pd.DataFrame, max_gap_hours: float, min_keep_share: float) -> pd.DataFrame:
    """Money arriving at X (t1) then leaving X (t2) soon after, keeping enough of the amount."""
    pairs = window.merge(window, left_on="receiver_id", right_on="sender_id", suffixes=("_1", "_2"))
    dt = pairs["timestamp_2"] - pairs["timestamp_1"]
    keep = ((dt > pd.Timedelta(0)) & (dt <= pd.Timedelta(hours=max_gap_hours))
            & (pairs["amount_2"] >= min_keep_share * pairs["amount_1"]))
    pairs = pairs.loc[keep, ["txn_id_1", "txn_id_2", "sender_id_1", "receiver_id_1", "receiver_id_2"]]
    pairs.columns = ["t1", "t2", "a", "x", "c"]
    return pairs


def rule_pass_through(window_events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Received >= min_in_txns and forwarded >= forward_share of it within within_hours."""
    ev = window_events[~window_events["is_self"]]
    n_in = ev[ev["direction"] == "IN"].groupby("account_id").size()
    flow = fifo_flow(ev, [cfg["within_hours"]])
    share = flow[f"pass_through_{cfg['within_hours']}h"]
    hit = share[(share >= cfg["forward_share"]) & (n_in.reindex(share.index).fillna(0) >= cfg["min_in_txns"])]
    return ev[ev["account_id"].isin(hit.index)][["account_id", "txn_id"]]


def rule_fan(window_events: pd.DataFrame, cfg: dict, fan_in: bool) -> pd.DataFrame:
    """FAN_IN: many distinct senders in a bucket, mostly forwarded soon after.
    FAN_OUT: many distinct receivers in a bucket, funded by money received just before."""
    ev = window_events[~window_events["is_self"]]
    hours = cfg["window_hours"]
    ev = ev.assign(bucket=ev["timestamp"].dt.floor(f"{hours}h"))
    main, other = ("IN", "OUT") if fan_in else ("OUT", "IN")
    side = ev[ev["direction"] == main]
    distinct = side.groupby(["account_id", "bucket"])["counterparty"].nunique()
    need = cfg["min_senders"] if fan_in else cfg["min_receivers"]
    candidates = distinct[distinct >= need].reset_index()[["account_id", "bucket"]]
    if candidates.empty:
        return pd.DataFrame(columns=["account_id", "txn_id"])

    main_amt = side.groupby(["account_id", "bucket"])["amount"].sum().rename("main")
    gap = pd.Timedelta(hours=cfg["forward_within_hours"] if fan_in else cfg["funded_within_hours"])
    opp = ev[ev["direction"] == other][["account_id", "timestamp", "amount"]]
    joined = candidates.merge(opp, on="account_id")
    if fan_in:  # money forwarded during the bucket or within `gap` after it ends
        inside = (joined["timestamp"] >= joined["bucket"]) & \
                 (joined["timestamp"] < joined["bucket"] + pd.Timedelta(hours=hours) + gap)
        ratio_needed = cfg["forward_share"]
    else:       # money received within `gap` before the bucket, or during it
        inside = (joined["timestamp"] >= joined["bucket"] - gap) & \
                 (joined["timestamp"] < joined["bucket"] + pd.Timedelta(hours=hours))
        ratio_needed = 0.8
    other_amt = joined[inside].groupby(["account_id", "bucket"])["amount"].sum().rename("other")
    check = candidates.set_index(["account_id", "bucket"]).join(main_amt).join(other_amt).fillna(0)
    hits = check[check["other"] >= ratio_needed * check["main"]].reset_index()
    flagged = side.merge(hits[["account_id", "bucket"]], on=["account_id", "bucket"])
    return flagged[["account_id", "txn_id"]]


def rule_chain(window: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Middle of a chain of >= 3 hops, each within hop_within_hours, keeping min_keep_share."""
    pairs = hop_pairs(window, cfg["hop_within_hours"], cfg["min_keep_share"])
    continued = pairs[pairs["t2"].isin(set(pairs["t1"]))]
    evidence = pd.concat([continued[["x", "t1"]].rename(columns={"t1": "txn_id"}),
                          continued[["x", "t2"]].rename(columns={"t2": "txn_id"})])
    return evidence.rename(columns={"x": "account_id"})


def rule_cycle(window: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Money leaves an account and returns to it via 2-3 hops within within_hours."""
    pairs = hop_pairs(window, cfg["within_hours"], 0.5)
    back2 = pairs[pairs["a"] == pairs["c"]]
    nxt = pairs[["t1", "t2", "c"]].rename(columns={"t1": "t2", "t2": "t3", "c": "d"})
    third = pairs.merge(nxt, on="t2")
    back3 = third[(third["d"] == third["a"]) & (third["a"] != third["x"])]
    parts = []
    for df, cols in ((back2, ["t1", "t2"]), (back3, ["t1", "t2", "t3"])):
        for who in ("a", "x"):
            for col in cols:
                parts.append(df[[who, col]].rename(columns={who: "account_id", col: "txn_id"}))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["account_id", "txn_id"])


def compute_rules(transactions: pd.DataFrame, cutoff: pd.Timestamp, settings: dict
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(flags, evidence) for accounts active before `cutoff`, matching the feature files' rows."""
    rules = settings["rules"]
    accounts = active_accounts(transactions, cutoff, settings["scoring"]["active_lookback_hours"])
    start = cutoff - pd.Timedelta(hours=rules["lookback_hours"])
    window = transactions[(transactions["timestamp"] > start) & (transactions["timestamp"] <= cutoff)]
    window_events = account_events(window, cutoff, accounts)
    plain = window[~window["is_self"]][["txn_id", "sender_id", "receiver_id", "timestamp", "amount"]]

    found = {
        "PASS_THROUGH": rule_pass_through(window_events, rules["pass_through"]),
        "FAN_IN": rule_fan(window_events, rules["fan_in"], fan_in=True),
        "FAN_OUT": rule_fan(window_events, rules["fan_out"], fan_in=False),
        "CHAIN": rule_chain(plain, rules["chain"]),
        "CYCLE": rule_cycle(plain, rules["cycle"]),
    }
    evidence = pd.concat([df.assign(rule=name) for name, df in found.items() if not df.empty],
                         ignore_index=True) if any(not d.empty for d in found.values()) else \
        pd.DataFrame(columns=["account_id", "rule", "txn_id"])
    evidence = evidence[evidence["account_id"].isin(accounts)].dropna(subset=["txn_id"])
    evidence = evidence.drop_duplicates(subset=["account_id", "rule", "txn_id"])

    flags = pd.DataFrame(index=accounts)
    for name, col in RULE_COLUMNS.items():
        hit = set(evidence.loc[evidence["rule"] == name, "account_id"])
        flags[col] = flags.index.isin(hit).astype("int8")
    flags["rule_score"] = (flags[list(RULE_COLUMNS.values())].sum(axis=1) / len(RULE_COLUMNS)).astype("float32")
    flags.insert(0, "cutoff", cutoff)
    flags = flags.reset_index()
    flags["account_id"] = flags["account_id"].astype("string")
    flags["cutoff"] = flags["cutoff"].astype("datetime64[ns]")
    flags = flags[["account_id", "cutoff", "rule_chain", "rule_fan_in", "rule_fan_out",
                   "rule_cycle", "rule_pass_through", "rule_score"]]

    evidence = pd.DataFrame({
        "account_id": evidence["account_id"].astype("string"),
        "rule": evidence["rule"].astype("string"),
        "txn_id": evidence["txn_id"].astype("int64"),
    }).reset_index(drop=True)
    return flags, evidence


def run_rules(dataset: str, cutoffs: list[pd.Timestamp] | None = None) -> Path:
    started = now_utc()
    settings = load_settings()
    processed = path("processed") / dataset
    splits = json.loads((processed / "splits.json").read_text(encoding="utf-8"))
    cutoffs = cutoffs or snapshot_cutoffs(splits, settings["scoring"]["snapshot_every_hours"])
    transactions = pd.read_parquet(processed / "transactions.parquet", columns=TXN_COLUMNS)
    out = path("features") / dataset
    out.mkdir(parents=True, exist_ok=True)

    written = {}
    for cutoff in cutoffs:
        flags, evidence = compute_rules(transactions, cutoff, settings)
        name = cutoff_name(cutoff)
        flags.to_parquet(out / f"{name}_rules.parquet", index=False)
        evidence.to_parquet(out / f"{name}_rules_evidence.parquet", index=False)
        fired = {col: int(flags[col].sum()) for col in RULE_COLUMNS.values()}
        written[name] = fired
        print(f"  {cutoff}: {len(flags):,} accounts, fired {fired}")
    write_manifest(out, "rules", started, dataset=dataset, files=written)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 Rules")
    parser.add_argument("--dataset", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cutoff")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    run_rules(args.dataset, None if args.all else [pd.Timestamp(args.cutoff)])


if __name__ == "__main__":
    main()
