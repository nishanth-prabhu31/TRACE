"""P6 Train: fit the main LightGBM model, calibrate it and choose the verdict cut-offs (F-06, F-07).

Uses the train period to fit and the validation period to calibrate and pick cut-offs.
Never opens the test period (that is evaluate.py's job).
Writes models/<version>/ (docs/04-CONTRACT.md section 7).

Usage:
    python -m trace_aml.pipeline.train --dataset ibm_hi_small
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.features.catalog import FEATURE_SETS
from trace_aml.models import metrics
from trace_aml.models.approaches import Scorer, fit_lightgbm, fit_rule_weights
from trace_aml.models.calibrate import choose_cuts, fit_calibrator, to_risk
from trace_aml.models.rows import (
    NEW_MULE, NORMAL, account_groups, best_snapshot_per_account, load_period_rows, load_split,
    period_bounds, row_labels,
)
from trace_aml.pipeline.common import now_utc, write_manifest


def full_columns(frame: pd.DataFrame) -> list[str]:
    """The FULL feature set, limited to the columns that were actually computed."""
    return [name for name in FEATURE_SETS["FULL"] if name in frame.columns]


def load_training_data(dataset: str, settings: dict) -> dict:
    """Train rows with labels, validation rows with labels, and the validation account groups.

    Train labels use roles as of the end of train. Validation labels use roles as of the
    end of validation, and leave out accounts that were already mules at the end of train.
    """
    split = load_split(dataset)
    every = settings["scoring"]["snapshot_every_hours"]
    account_labels = pd.read_parquet(path("processed") / dataset / c.ACCOUNT_LABELS.filename)

    train = load_period_rows(dataset, "TRAIN", split, every)
    target = row_labels(train, account_labels, split["train_end"])
    used = target.notna().to_numpy()

    validation = load_period_rows(dataset, "VALIDATION", split, every)
    validation_target = row_labels(validation, account_labels, split["validation_end"],
                                   drop_known_before=split["train_end"])
    labelled = validation_target.notna().to_numpy()
    return {
        "split": split,
        "account_labels": account_labels,
        "train": train[used].reset_index(drop=True),
        "target": target[used].astype("int8").reset_index(drop=True),
        "validation": validation,
        "validation_labelled": validation[labelled].reset_index(drop=True),
        "validation_target": validation_target[labelled].astype("int8").reset_index(drop=True),
        "validation_groups": account_groups(account_labels, *period_bounds(split, "VALIDATION")),
    }


def score_period(scorer: Scorer, rows: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """One row per account for a period: its best snapshot, score and evaluation group."""
    best = best_snapshot_per_account(rows, pd.Series(scorer.score(rows)))
    best["group"] = best["account_id"].map(groups).to_numpy()
    return best


def judged(accounts: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(target, score) for the accounts that count in the main metrics: new mules vs normal."""
    counted = accounts[accounts["group"].isin([NEW_MULE, NORMAL])]
    return (counted["group"] == NEW_MULE).to_numpy(dtype="int8"), counted["score"].to_numpy(dtype="float64")


def run_train(dataset: str) -> Path:
    """Fit, calibrate and save the main model; return its folder."""
    started, clock = now_utc(), time.perf_counter()
    settings = load_settings()
    version = settings["model"]["version"]
    data = load_training_data(dataset, settings)
    columns = full_columns(data["train"])
    print(f"Train rows: {len(data['train']):,} ({int(data['target'].sum()):,} mule rows), {len(columns)} features")

    scorer = fit_lightgbm("LIGHTGBM", columns, data["train"], data["target"], data["validation_labelled"],
                          data["validation_target"], settings["model"]["lightgbm"], settings["random_seed"])
    accounts = score_period(scorer, data["validation"], data["validation_groups"])
    target, score = judged(accounts)
    validation_metrics = metrics.main_metrics(target, score, settings["metrics"]["k_values"],
                                              settings["metrics"]["fpr_top_share"])

    calibrator = fit_calibrator(target, score)
    _, risk = to_risk(calibrator, score)
    cuts = choose_cuts(target, risk, settings["verdicts"])
    at_likely = metrics.at_cut(target, risk, cuts["likely_mule_cut"])
    rules = fit_rule_weights(data["validation_labelled"], data["validation_target"])

    folder = path("models") / version
    folder.mkdir(parents=True, exist_ok=True)
    scorer.model.booster_.save_model(str(folder / "model.txt"))
    joblib.dump(calibrator, folder / "calibrator.joblib")
    (folder / "features.json").write_text(json.dumps(columns, indent=2), encoding="utf-8")
    (folder / "thresholds.json").write_text(json.dumps(
        {"likely_mule_cut": cuts["likely_mule_cut"], "watch_cut": cuts["watch_cut"]}, indent=2), encoding="utf-8")
    meta = {
        "version": version,
        "parent_version": None,
        "dataset": dataset,
        "trained_on": f"TRAIN period up to {data['split']['train_end']}, labels as of that time",
        "calibrated_on": f"VALIDATION period up to {data['split']['validation_end']} (new mules vs normal accounts)",
        "train_rows": int(len(data["train"])),
        "train_mule_rows": int(data["target"].sum()),
        "seed": settings["random_seed"],
        "settings": settings["model"]["lightgbm"],
        "trees": int(scorer.model.booster_.num_trees()),
        "validation_metrics": validation_metrics,
        "validation_at_likely_mule_cut": at_likely,
        "precision_target_met": cuts["precision_target_met"],
        "rule_weights": rules.weights,
        "training_seconds": round(time.perf_counter() - clock, 1),
        "created_at": now_utc(),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    write_manifest(folder, "train", started, dataset=dataset, model_version=version)

    shown = {key: round(value, 4) for key, value in validation_metrics.items()}
    print(f"Validation (seed {settings['random_seed']}, synthetic data): {shown}")
    print(f"Cut-offs: {cuts}; at likely_mule_cut on validation: {at_likely}")
    print(f"Rule weights (validation precision): { {k: round(v, 4) for k, v in rules.weights.items()} }")
    print(f"Wrote {folder}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="P6 Train the main model, calibrate, choose cut-offs")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    run_train(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
