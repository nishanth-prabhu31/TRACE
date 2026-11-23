"""P4 Features: compute every catalogue feature for the accounts active at a cut-off (F-04).

Reads data/processed/<dataset>/transactions.parquet (never the label files) and writes
data/features/<dataset>/<cutoff>.parquet (docs/04-CONTRACT.md section 4.1).

Usage:
    python -m trace_aml.pipeline.features --dataset ibm_hi_small --cutoff 2022-09-07T00:00
    python -m trace_aml.pipeline.features --dataset ibm_hi_small --all-snapshots
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.features.catalog import FEATURE_SETS
from trace_aml.features.flow import flow_features
from trace_aml.features.prep import prepare
from trace_aml.features.relative import relative_features
from trace_aml.features.structural import structural_features
from trace_aml.features.temporal import temporal_features
from trace_aml.pipeline.common import file_info, now_utc, write_manifest
from trace_aml.pipeline.split import compute_split, snapshot_times

FAMILIES = {"STRUCTURAL": structural_features, "TEMPORAL": temporal_features,
            "FLOW": flow_features, "RELATIVE": relative_features}
NEEDED_COLUMNS = ["txn_id", "timestamp", "sender_id", "receiver_id", "amount", "amount_original",
                  "channel", "is_self"]


def feature_columns(settings: dict) -> list[str]:
    """The non-rule feature columns this run produces, in catalogue order."""
    wanted = [name for name in FEATURE_SETS["FULL"] if name not in FEATURE_SETS["RULES"]]
    if not settings["features"]["betweenness"]["enabled"]:
        wanted.remove("betweenness_local")
    return wanted


def compute_features(transactions: pd.DataFrame, cutoff: pd.Timestamp, settings: dict,
                     score_all: bool = False, timings: dict | None = None) -> pd.DataFrame:
    """Features for one cut-off, using only transactions with timestamp <= cutoff.

    score_all = False: accounts active in the look-back window (split runs).
    score_all = True: every account in the data (uploads, replay windows).
    Returns the contract table: account_id, cutoff, then one float32 column per feature.
    """
    cutoff = pd.Timestamp(cutoff)
    lookback = None if score_all else settings["scoring"]["active_lookback_hours"]
    started = time.perf_counter()
    history = prepare(transactions, cutoff, lookback)
    if timings is not None:
        timings["prepare"] = round(time.perf_counter() - started, 1)

    parts = []
    for family, function in FAMILIES.items():
        started = time.perf_counter()
        parts.append(function(history, cutoff, settings))
        if timings is not None:
            timings[family.lower()] = round(time.perf_counter() - started, 1)
    values = pd.concat(parts, axis=1)[feature_columns(settings)].astype("float32")

    out = pd.DataFrame({
        "account_id": pd.Series(history.names[history.active], dtype="string"),
        "cutoff": pd.Series(cutoff, index=range(len(history.active))).astype("datetime64[ns]"),
    })
    return pd.concat([out, values.reset_index(drop=True)], axis=1)


def cutoff_filename(cutoff: pd.Timestamp) -> str:
    """File name for a cut-off, e.g. 20220907T0000.parquet."""
    return f"{pd.Timestamp(cutoff):%Y%m%dT%H%M}.parquet"


def run_features(dataset: str, cutoffs: list[pd.Timestamp] | None) -> Path:
    """Run P4 for the given cut-offs (None = every snapshot of the time split)."""
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
        timings: dict = {}
        table = compute_features(transactions, cutoff, settings, timings=timings)
        table.to_parquet(out / cutoff_filename(cutoff), index=False)
        written[cutoff_filename(cutoff)] = {"accounts": len(table), "seconds": timings}
        print(f"  {cutoff:%Y-%m-%d %H:%M}: {len(table):,} accounts, seconds {timings}")
    write_manifest(out, "features", started, dataset=dataset, inputs=[file_info(source)], files=written)
    print(f"Wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 Features for one cut-off or for every snapshot")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    parser.add_argument("--cutoff", help="one cut-off, e.g. 2022-09-07T00:00")
    parser.add_argument("--all-snapshots", action="store_true", help="every snapshot of the time split")
    args = parser.parse_args()
    if not args.cutoff and not args.all_snapshots:
        parser.error("give --cutoff or --all-snapshots")
    run_features(args.dataset, None if args.all_snapshots else [pd.Timestamp(args.cutoff)])


if __name__ == "__main__":
    main()
