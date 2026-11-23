"""P6 Evaluate: compare all approaches on the validation and test periods (F-06).

This is the only step that opens the test period, and it is meant for final numbers.
Every approach is refitted for each seed on the train period, then scored on both
later periods. Writes reports/eval_<version>.md and .json (docs/04-CONTRACT.md section 8).

Usage:
    python -m trace_aml.pipeline.evaluate --dataset ibm_hi_small
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.features.catalog import FEATURE_SETS
from trace_aml.models import metrics
from trace_aml.models.approaches import (
    Scorer, fit_isolation_forest, fit_lightgbm, fit_logreg, fit_random_forest, fit_rule_weights,
)
from trace_aml.models.calibrate import choose_cuts, fit_calibrator, to_risk
from trace_aml.models.rows import (
    EXCLUDED, KNOWN_MULE, NEW_MULE, NORMAL, account_groups, load_period_rows, period_bounds,
)
from trace_aml.pipeline.common import now_utc, write_manifest
from trace_aml.pipeline.train import full_columns, judged, load_training_data, score_period

PERIODS = ("VALIDATION", "TEST")
ABLATIONS = {"FULL - STRUCTURAL": "STRUCTURAL", "FULL - TEMPORAL": "TEMPORAL", "FULL - FLOW": "FLOW",
             "FULL - RULES": "RULES", "FULL - RELATIVE": "RELATIVE"}
NO_PATTERN = "NO_PATTERN"


def pattern_of_accounts(dataset: str) -> pd.Series:
    """account_id -> the laundering pattern type it took part in (first one), for per-pattern recall."""
    folder = path("processed") / dataset
    transactions = pd.read_parquet(folder / c.TRANSACTIONS.filename, columns=["txn_id", "sender_id", "receiver_id"])
    labels = pd.read_parquet(folder / c.LABELS_TXN.filename)
    in_pattern = transactions.merge(labels[labels["pattern_type"].notna()], on="txn_id")
    long = pd.concat([in_pattern[["sender_id", "pattern_type"]].rename(columns={"sender_id": "account_id"}),
                      in_pattern[["receiver_id", "pattern_type"]].rename(columns={"receiver_id": "account_id"})])
    return long.drop_duplicates("account_id").set_index("account_id")["pattern_type"]


def fit_all(data: dict, columns: list[str], settings: dict, seed: int, with_ablations: bool) -> dict[str, Scorer]:
    """Fit every approach (and, if asked, the ablations) for one seed."""
    models, train, target = settings["model"], data["train"], data["target"]

    def boosted(name: str, use: list[str]) -> Scorer:
        return fit_lightgbm(name, use, train, target, data["validation_labelled"], data["validation_target"],
                            models["lightgbm"], seed)

    fitted = {
        "RULES": fit_rule_weights(data["validation_labelled"], data["validation_target"]),
        "TXN_ONLY": boosted("TXN_ONLY", FEATURE_SETS["TXN_ONLY"]),
        "LOGREG": fit_logreg(columns, train, target, models["logreg"], seed),
        "RANDOM_FOREST": fit_random_forest(columns, train, target, models["random_forest"], seed),
        "ISOLATION_FOREST": fit_isolation_forest(columns, train, models["isolation_forest"], seed),
        "LIGHTGBM": boosted("LIGHTGBM", columns),
    }
    if with_ablations:
        for name, family in ABLATIONS.items():
            fitted[name] = boosted(name, [col for col in columns if col not in FEATURE_SETS[family]])
    return fitted


def lightgbm_extras(accounts: dict[str, pd.DataFrame], patterns: pd.Series, settings: dict) -> dict:
    """Numbers reported for the main model only: at the verdict cut-off, per pattern, and for
    the accounts left out of the main metrics (known mules, sources and sinks)."""
    target, score = judged(accounts["VALIDATION"])
    calibrator = fit_calibrator(target, score)
    cuts = choose_cuts(target, to_risk(calibrator, score)[1], settings["verdicts"])

    test = accounts["TEST"].copy()
    test["risk"] = to_risk(calibrator, test["score"].to_numpy())[1]
    test_target, test_score = judged(test)
    counted = test[test["group"].isin([NEW_MULE, NORMAL])]
    out = {f"test_{key}_at_likely_cut": float(value) for key, value in
           metrics.at_cut(test_target, counted["risk"].to_numpy(), cuts["likely_mule_cut"]).items()}
    out["likely_mule_cut"] = cuts["likely_mule_cut"]
    out["watch_cut"] = cuts["watch_cut"]

    k = settings["metrics"]["per_pattern_k"]
    in_top = set(counted.iloc[metrics.top_k(test_score, k)]["account_id"])
    new_mules = counted[counted["group"] == NEW_MULE]
    kind = new_mules["account_id"].map(patterns).fillna(NO_PATTERN)
    for name, members in new_mules.groupby(kind.to_numpy()):
        out[f"test_recall_at_{k}_{name}"] = float(members["account_id"].isin(in_top).mean())
        out[f"test_n_{name}"] = float(len(members))

    known = test[test["group"] == KNOWN_MULE]
    out["test_known_mule_share_at_or_above_watch"] = float((known["risk"] >= cuts["watch_cut"]).mean()) if len(known) else 0.0
    out["test_known_mule_mean_risk"] = float(known["risk"].mean()) if len(known) else 0.0
    out["test_source_sink_mean_risk"] = float(test.loc[test["group"] == EXCLUDED, "risk"].mean())
    out["test_normal_mean_risk"] = float(test.loc[test["group"] == NORMAL, "risk"].mean())
    return out


def report_markdown(report: dict) -> str:
    """The human-readable results table."""
    lines = [f"# Evaluation: {report['dataset']}, model {report['model_version']}", "",
             f"Synthetic data ({report['dataset']}). Time-ordered split; positives are accounts that became "
             f"mules inside the period; mean ± sd over {report['n_seeds']} seeds. "
             "The sd reflects model randomness only, not the small number of positives.", ""]
    shown = ["pr_auc", "precision_at_50", "precision_at_100", "precision_at_500", "recall_at_500", "roc_auc"]
    for period in PERIODS:
        first = next(iter(report["approaches"].values()))[period]
        lines += [f"## {period.title()} period ({int(first['n_mules']['mean'])} new mules among "
                  f"{int(first['n_accounts']['mean']):,} accounts)", "",
                  "| Approach | " + " | ".join(shown) + " |", "|---|" + "---|" * len(shown)]
        for name, result in {**report["approaches"], **report["ablations"]}.items():
            cells = [f"{result[period][m]['mean']:.4f} ± {result[period][m]['sd']:.4f}" for m in shown]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Main model only (test period)", "", "| Measure | Mean | sd |", "|---|---|---|"]
    lines += [f"| {name} | {value['mean']:.4f} | {value['sd']:.4f} |" for name, value in report["lightgbm_extras"].items()]
    return "\n".join(lines) + "\n"


def run_evaluate(dataset: str) -> dict:
    """Fit and score everything for every seed; write the report and return it."""
    started, clock = now_utc(), time.perf_counter()
    settings = load_settings()
    version = settings["model"]["version"]
    data = load_training_data(dataset, settings)
    columns = full_columns(data["train"])
    account_labels, split = data["account_labels"], data["split"]
    rows = {"VALIDATION": data["validation"],
            "TEST": load_period_rows(dataset, "TEST", split, settings["scoring"]["snapshot_every_hours"])}
    groups = {period: account_groups(account_labels, *period_bounds(split, period)) for period in PERIODS}
    patterns = pattern_of_accounts(dataset)
    k_values, top_share = settings["metrics"]["k_values"], settings["metrics"]["fpr_top_share"]

    per_seed: dict[str, dict[str, list]] = {}
    extras: list[dict] = []
    for index in range(settings["n_seeds"]):
        seed = settings["random_seed"] + index
        fitted = fit_all(data, columns, settings, seed, with_ablations=True)
        for name, scorer in fitted.items():
            accounts = {period: score_period(scorer, rows[period], groups[period]) for period in PERIODS}
            for period in PERIODS:
                result = metrics.main_metrics(*judged(accounts[period]), k_values, top_share)
                per_seed.setdefault(name, {}).setdefault(period, []).append(result)
            if name == "LIGHTGBM":
                extras.append(lightgbm_extras(accounts, patterns, settings))
        done = per_seed["LIGHTGBM"]["VALIDATION"][-1]["pr_auc"]
        print(f"  seed {seed} done ({time.perf_counter() - clock:.0f} s); LightGBM validation PR-AUC {done:.4f}", flush=True)

    summary = {name: {period: metrics.mean_and_sd(runs) for period, runs in periods.items()}
               for name, periods in per_seed.items()}
    extra_names = sorted(set().union(*extras))
    report = {
        "dataset": dataset,
        "model_version": version,
        "n_seeds": settings["n_seeds"],
        "split": {key: str(value) for key, value in split.items()},
        "approaches": {name: summary[name] for name in summary if name not in ABLATIONS},
        "ablations": {name: summary[name] for name in ABLATIONS},
        "lightgbm_extras": metrics.mean_and_sd([{name: run.get(name, 0.0) for name in extra_names} for run in extras]),
    }
    reports = path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    stem = f"eval_{version}" if dataset == "ibm_hi_small" else f"eval_{version}_{dataset}"
    (reports / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (reports / f"{stem}.md").write_text(report_markdown(report), encoding="utf-8")
    write_manifest(reports, stem, started, dataset=dataset, model_version=version, n_seeds=settings["n_seeds"])
    print(f"Wrote {reports / stem}.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="P6 Evaluate all approaches on validation and test")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    run_evaluate(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
