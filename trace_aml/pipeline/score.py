"""P7 Score and explain: turn a saved model into a run folder the app can read (F-07, F-08).

Scores every account active in a period at each snapshot, keeps each account's
highest-risk snapshot, and writes scores, reasons, trails, the feature rows and a
copy of the transactions to data/runs/<run_id>/ (docs/04-CONTRACT.md section 5).
Reads no label files.

Usage:
    python -m trace_aml.pipeline.score --dataset ibm_hi_small --split TEST
"""

import argparse
import json
import shutil
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.explain.shap_reasons import top_reasons
from trace_aml.explain.summary import recent_activity, short_labels, summary_text
from trace_aml.explain.trails import build_trails
from trace_aml.models.calibrate import to_risk, verdict_for
from trace_aml.models.rows import best_snapshot_per_account
from trace_aml.pipeline.common import now_utc, settings_hash
from trace_aml.pipeline.features import cutoff_filename
from trace_aml.pipeline.rules import FLAG_COLUMNS
from trace_aml.pipeline.split import compute_split, period_of, snapshot_times

FLAGGED = ("LIKELY_MULE", "WATCH")


def load_model(folder: Path) -> dict:
    """Everything saved for one model version (docs/04-CONTRACT.md section 7)."""
    return {
        "booster": lgb.Booster(model_file=str(folder / "model.txt")),
        "calibrator": joblib.load(folder / "calibrator.joblib"),
        "columns": json.loads((folder / "features.json").read_text(encoding="utf-8")),
        "cuts": json.loads((folder / "thresholds.json").read_text(encoding="utf-8")),
    }


def load_feature_rows(features_folder: Path, cutoffs: list[pd.Timestamp]) -> pd.DataFrame:
    """Feature rows plus rule flags for the given snapshots (no labels involved)."""
    parts = []
    for cutoff in cutoffs:
        stem = cutoff_filename(cutoff).removesuffix(".parquet")
        features = pd.read_parquet(features_folder / f"{stem}.parquet")
        flags = pd.read_parquet(features_folder / f"{stem}_rules.parquet", columns=["account_id", *FLAG_COLUMNS])
        rows = features.merge(flags, on="account_id", how="left", validate="one_to_one")
        rows[FLAG_COLUMNS] = rows[FLAG_COLUMNS].fillna(0).astype("float32")
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def account_totals(transactions: pd.DataFrame) -> pd.DataFrame:
    """Per account: total received, total sent, number of transactions and first-seen time."""
    real = transactions[~transactions["is_self"]]
    got = real.groupby("receiver_id").agg(in_amount=("amount", "sum"), n_in=("txn_id", "size"), first_in=("timestamp", "min"))
    sent = transactions.groupby("sender_id").agg(out_amount=("amount", "sum"), n_out=("txn_id", "size"),
                                                 first_out=("timestamp", "min"))
    totals = got.join(sent, how="outer")
    totals["n_txn"] = totals["n_in"].fillna(0) + totals["n_out"].fillna(0)
    totals["first_seen"] = totals[["first_in", "first_out"]].min(axis=1)
    return totals[["in_amount", "out_amount", "n_txn", "first_seen"]].fillna({"in_amount": 0.0, "out_amount": 0.0})


def score_rows(rows: pd.DataFrame, model: dict) -> pd.DataFrame:
    """One row per account: its highest-scoring snapshot, with prob, risk, verdict and rank."""
    raw = pd.Series(model["booster"].predict(rows[model["columns"]]))
    best = best_snapshot_per_account(rows, raw)
    best["prob"], best["risk"] = to_risk(model["calibrator"], best["score"].to_numpy())
    best["verdict"] = verdict_for(best["risk"].to_numpy(), model["cuts"])
    best["rank"] = np.arange(1, len(best) + 1)            # already sorted by raw score, highest first
    return best


def build_run(transactions: pd.DataFrame, rows: pd.DataFrame, model: dict, settings: dict
              ) -> dict[str, pd.DataFrame]:
    """All run tables from feature rows and the transactions the run covers."""
    best = score_rows(rows, model)
    flagged = best[best["verdict"].isin(FLAGGED)].reset_index(drop=True)
    reasons = top_reasons(model["booster"], flagged, model["columns"], settings["explain"]["top_reasons"])
    trails = build_trails(transactions, flagged[["account_id", "cutoff"]], settings["explain"]["trail"])

    activity = recent_activity(transactions, flagged[["account_id", "cutoff"]],
                               settings["scoring"]["active_lookback_hours"])
    rate = settings["currency"]["display_units_per_base"]
    trails_by_account = {account: group for account, group in trails.groupby("account_id")}
    flagged_rows = flagged.set_index("account_id")
    summaries = {account: summary_text(flagged_rows.loc[account, "verdict"], activity.loc[account],
                                       flagged_rows.loc[account], trails_by_account.get(account, trails.iloc[:0]), rate)
                 for account in flagged_rows.index}

    verdict = best.set_index("account_id")["verdict"]
    totals = account_totals(transactions).reindex(best["account_id"])
    scores = pd.DataFrame({
        "account_id": best["account_id"].astype("string"),
        "risk": best["risk"].astype("float32"),
        "prob": best["prob"].astype("float32"),
        "verdict": best["verdict"].astype("string"),
        "rank": best["rank"].astype("int32"),
        "short_label": short_labels(verdict, reasons).to_numpy(),
        "summary": best["account_id"].map(summaries).fillna("No unusual pattern found.").astype("string"),
        "has_trail": best["account_id"].isin(set(trails["account_id"])).to_numpy(),
        "in_amount": totals["in_amount"].to_numpy(dtype="float64"),
        "out_amount": totals["out_amount"].to_numpy(dtype="float64"),
        "n_txn": totals["n_txn"].to_numpy().astype("int32"),
        "first_seen": totals["first_seen"].to_numpy().astype("datetime64[ns]"),
    })
    feature_rows = pd.concat([best[["account_id", "cutoff"]], best["cutoff"].rename("snapshot"),
                              best[model["columns"]].astype("float32")], axis=1)
    view = transactions.sort_values(["sender_id", "timestamp"], kind="stable").reset_index(drop=True)
    return {c.SCORES.filename: scores, c.REASONS.filename: reasons, c.TRAILS.filename: trails,
            c.RUN_FEATURES.filename: feature_rows, c.TRANSACTIONS_VIEW.filename: view}


def write_run(tables: dict[str, pd.DataFrame], manifest: dict, runs_folder: Path) -> Path:
    """Write a run to a temporary folder, then rename it, so a half-written run is never seen."""
    final, temporary = runs_folder / manifest["run_id"], runs_folder / f".{manifest['run_id']}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for filename, table in tables.items():
        table.to_parquet(temporary / filename, index=False)
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    temporary.rename(final)
    return final


def run_score(dataset: str, split_name: str, version: str | None = None) -> Path:
    """Score one period of a dataset with a saved model (default: settings) and write the run folder."""
    settings = load_settings()
    version = version or settings["model"]["version"]
    model = load_model(path("models") / version)
    transactions = pd.read_parquet(path("processed") / dataset / c.TRANSACTIONS.filename)
    split = compute_split(transactions, settings["split"])
    cutoffs = [t for t in snapshot_times(split, settings["scoring"]["snapshot_every_hours"])
               if period_of(t, split) == split_name]
    run_cutoff = max(cutoffs)
    rows = load_feature_rows(path("features") / dataset, cutoffs)
    covered = transactions[transactions["timestamp"] <= run_cutoff]

    tables = build_run(covered, rows, model, settings)
    scores = tables[c.SCORES.filename]
    manifest = {
        "contract_version": c.CONTRACT_VERSION,
        "run_id": f"{dataset}__{split_name.lower()}__{version}",
        "dataset": dataset,
        "kind": "SPLIT",
        "split": split_name,
        "cutoff": run_cutoff.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_version": version,
        "created_at": now_utc(),
        "settings_hash": settings_hash(),
        "n_accounts": int(len(scores)),
        "n_transactions": int(len(covered)),
        "verdict_counts": {verdict: int((scores["verdict"] == verdict).sum()) for verdict in c.VERDICTS},
    }
    folder = write_run(tables, manifest, path("runs"))
    print(f"  accounts {manifest['n_accounts']:,}; verdicts {manifest['verdict_counts']}; "
          f"with trail {int(scores['has_trail'].sum()):,}")
    print(f"Wrote {folder}")
    from trace_aml.pipeline.rings import run_rings      # P10 rings, label-free (F-13)
    print(f"  rings: {run_rings(manifest['run_id'])}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="P7 Score a period and write a run folder")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    parser.add_argument("--split", required=True, choices=c.SPLITS, help="period to score")
    parser.add_argument("--version", help="model version folder under models/ (default: settings)")
    args = parser.parse_args()
    run_score(args.dataset, args.split, args.version)


if __name__ == "__main__":
    main()
