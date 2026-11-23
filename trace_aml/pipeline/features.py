"""P4 Features: measure every active account at a cut-off and save one file per cut-off (F-04).

Rule: features at cut-off T use only transactions with timestamp <= T
(docs/05-FEATURES-MODEL.md section 1). Labels are never read here.

Usage:
    python -m trace_aml.pipeline.features --dataset ibm_hi_small --cutoff 2022-09-07T00:00
    python -m trace_aml.pipeline.features --dataset ibm_hi_small --all
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trace_aml.config import load_settings, path
from trace_aml.features.events import account_events, active_accounts
from trace_aml.features.flow import flow_features
from trace_aml.features.relative import relative_features
from trace_aml.features.structural import structural_features
from trace_aml.features.temporal import temporal_features
from trace_aml.pipeline.common import now_utc, write_manifest

TXN_COLUMNS = ["txn_id", "timestamp", "sender_id", "receiver_id", "amount",
               "amount_original", "channel", "is_self"]


def compute_features(transactions: pd.DataFrame, cutoff: pd.Timestamp, settings: dict) -> pd.DataFrame:
    """All features for accounts active in the look-back window before `cutoff`."""
    feats = settings["features"]
    accounts = active_accounts(transactions, cutoff, settings["scoring"]["active_lookback_hours"])
    events = account_events(transactions, cutoff, accounts)
    amounts = np.sort(transactions.loc[transactions["timestamp"] <= cutoff, "amount"].to_numpy())

    families = [
        temporal_features(events, cutoff, feats["windows_hours"], tuple(feats["night_hours"]),
                          feats["new_counterparty_hours"]),
        relative_features(events, amounts, feats["round_amount_divisor"]),
        structural_features(transactions, cutoff, accounts,
                            feats["betweenness"]["cutoff_hops"] if feats["betweenness"]["enabled"] else None,
                            feats["new_counterparty_hours"], settings["random_seed"]),
        flow_features(events, transactions, cutoff, settings),
    ]
    table = pd.concat(families, axis=1).reindex(accounts)
    table.insert(0, "cutoff", cutoff)
    table = table.reset_index()
    table["account_id"] = table["account_id"].astype("string")
    table["cutoff"] = table["cutoff"].astype("datetime64[ns]")
    return table


def snapshot_cutoffs(splits: dict, every_hours: int) -> list[pd.Timestamp]:
    """Every `every_hours` from the first full day to the end of the test period."""
    start, end = pd.Timestamp(splits["data_start"]), pd.Timestamp(splits["test_end"])
    return list(pd.date_range(start + pd.Timedelta(hours=every_hours), end, freq=f"{every_hours}h"))


def cutoff_name(cutoff: pd.Timestamp) -> str:
    return cutoff.strftime("%Y%m%dT%H%M")


def run_features(dataset: str, cutoffs: list[pd.Timestamp] | None = None) -> Path:
    """Run P4 for the given cut-offs (default: every snapshot) and return the output folder."""
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
        table = compute_features(transactions, cutoff, settings)
        name = f"{cutoff_name(cutoff)}.parquet"
        table.to_parquet(out / name, index=False)
        written[name] = {"accounts": len(table), "features": table.shape[1] - 2}
        print(f"  {cutoff}: {len(table):,} accounts x {table.shape[1] - 2} features -> {name}")

    write_manifest(out, "features", started, dataset=dataset, files=written)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 Features")
    parser.add_argument("--dataset", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cutoff", help="one cut-off, e.g. 2022-09-07T00:00")
    group.add_argument("--all", action="store_true", help="every daily snapshot")
    args = parser.parse_args()
    run_features(args.dataset, None if args.all else [pd.Timestamp(args.cutoff)])


if __name__ == "__main__":
    main()
