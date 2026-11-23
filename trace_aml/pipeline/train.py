"""P6 Train and evaluate: all approaches on the same rows, same split, same metrics (F-06).

Writes models/<version>/ (main LightGBM) and reports/eval_<version>.md + .json.

Usage:
    python -m trace_aml.pipeline.train --dataset ibm_hi_small --version v1
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from trace_aml.config import load_settings, path
from trace_aml.models import approaches as ap
from trace_aml.models.dataset import (evaluation_targets, feature_columns, label_training_rows,
                                      load_rows, period_scores)
from trace_aml.models.metrics import evaluate
from trace_aml.pipeline.common import now_utc, write_manifest

ABLATIONS = ["FULL-STRUCTURAL", "FULL-TEMPORAL", "FULL-FLOW", "FULL-RULES", "FULL-RELATIVE"]


def summarise(runs: list[dict]) -> dict:
    """Mean and standard deviation of each metric over seeds."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
    out = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
           for k in keys}
    out["n_accounts"], out["n_mules"], out["n_seeds"] = runs[0]["n_accounts"], runs[0]["n_mules"], len(runs)
    return out


def run_training(dataset: str, version: str, with_ablations: bool = True) -> dict:
    started = now_utc()
    settings = load_settings()
    cfg_model, k_values = settings["model"], settings["metrics"]["k_values"]
    seeds = [settings["random_seed"] + i for i in range(settings["n_seeds"])]

    print("Loading snapshot rows ...")
    rows, splits = load_rows(dataset)
    numeric = rows.select_dtypes("number").columns
    rows[numeric] = rows[numeric].replace([np.inf, -np.inf], np.nan)
    train = label_training_rows(rows[rows["period"] == "TRAIN"])
    val = label_training_rows(rows[rows["period"] == "VALIDATION"])
    scoring = rows[rows["period"].isin(["VALIDATION", "TEST"])]
    targets = {p: evaluation_targets(rows, p, splits) for p in ("VALIDATION", "TEST")}
    print(f"  train rows {len(train):,} ({int(train['y'].sum())} mule rows), "
          f"validation new mules {int(targets['VALIDATION'].sum())}, test new mules {int(targets['TEST'].sum())}")

    def assess(score_fn) -> dict:
        s = score_fn(scoring)
        return {p: evaluate(targets[p], period_scores(scoring, s, p), k_values,
                            settings["metrics"]["fpr_top_share"]) for p in targets}

    full = feature_columns(rows, "FULL")
    results: dict = {}
    timings: dict = {}

    def record(name: str, runs: list[dict], seconds: float) -> None:
        results[name] = {p: summarise([r[p] for r in runs]) for p in targets}
        timings[name] = round(seconds, 1)
        t = results[name]["TEST"]
        print(f"  {name:18s} test PR-AUC {t['pr_auc']['mean']:.4f} ± {t['pr_auc']['std']:.4f}   "
              f"P@50 {t['precision@50']['mean']:.3f}   R@500 {t['recall@500']['mean']:.3f}   ({seconds:.0f} s)")

    start = time.time()
    record("RULES", [assess(ap.fit_rules(val))], time.time() - start)

    main_models = {}
    for name, cols in (("TXN_ONLY", feature_columns(rows, "TXN_ONLY")), ("LIGHTGBM", full)):
        start, runs = time.time(), []
        for seed in seeds:
            fn = ap.fit_lightgbm(train, val, cols, cfg_model["lightgbm"], seed)
            runs.append(assess(fn))
            main_models.setdefault(name, fn)
        record(name, runs, time.time() - start)

    start = time.time()
    record("LOGREG", [assess(ap.fit_logreg(train, full, seeds[0]))], time.time() - start)
    start = time.time()
    record("RANDOM_FOREST", [assess(ap.fit_random_forest(train, full, cfg_model["random_forest"]["n_estimators"],
                                                         seeds[0]))], time.time() - start)
    start = time.time()
    record("ISOLATION_FOREST", [assess(ap.fit_isolation_forest(train, full, seeds[0]))], time.time() - start)

    if with_ablations:
        for name in ABLATIONS:
            start = time.time()
            cols = feature_columns(rows, name)
            record(name, [assess(ap.fit_lightgbm(train, val, cols, cfg_model["lightgbm"], s)) for s in seeds[:3]],
                   time.time() - start)

    # Save the main model (first seed) for scoring and explanations.
    model_dir = path("models") / version
    model_dir.mkdir(parents=True, exist_ok=True)
    booster = main_models["LIGHTGBM"].model.booster_
    booster.save_model(str(model_dir / "model.txt"))
    (model_dir / "features.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    importance = pd.Series(booster.feature_importance("gain"), index=full).sort_values(ascending=False)
    meta = {
        "version": version, "dataset": dataset, "created_at": now_utc(), "seed": seeds[0],
        "n_trees": int(main_models["LIGHTGBM"].model.booster_.num_trees()),
        "train_rows": len(train), "train_mule_rows": int(train["y"].sum()),
        "validation": results["LIGHTGBM"]["VALIDATION"],
        "top_features_by_gain": importance.head(15).round(1).to_dict(),
        "rule_weights": ap.fit_rules(val).weights,
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    report = {"dataset": dataset, "version": version, "seeds": seeds, "results": results,
              "timings_seconds": timings, "top_features_by_gain": meta["top_features_by_gain"]}
    write_report(report)
    write_manifest(model_dir, "train", started, dataset=dataset)
    return report


def write_report(report: dict) -> None:
    out = path("reports")
    version = report["version"]
    (out / f"eval_{version}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# Evaluation {version}: {report['dataset']}", "",
             "Synthetic data (IBM AMLworld). Account level. Positives = **new** mules in the period; "
             "known mules, sources and sinks excluded. Mean ± std over seeds (LightGBM and TXN_ONLY: "
             f"{len(report['seeds'])} seeds; ablations: 3; others: 1).", ""]
    for period in ("TEST", "VALIDATION"):
        first = next(iter(report["results"].values()))[period]
        lines += [f"## {period.title()} ({first['n_mules']} new mules among {first['n_accounts']:,} accounts; "
                  f"chance PR-AUC {first['chance_pr_auc']['mean']:.4f})", "",
                  "| Approach | PR-AUC | ROC-AUC | P@50 | P@100 | R@100 | R@500 | FPR top 1% |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, res in report["results"].items():
            r = res[period]
            lines.append(f"| {name} | {r['pr_auc']['mean']:.4f} ± {r['pr_auc']['std']:.4f} | "
                         f"{r['roc_auc']['mean']:.3f} | {r['precision@50']['mean']:.3f} | "
                         f"{r['precision@100']['mean']:.3f} | {r['recall@100']['mean']:.3f} | "
                         f"{r['recall@500']['mean']:.3f} | {r['fpr_top_1pct']['mean']:.4f} |")
        lines.append("")
    lines += ["## Top features (LightGBM gain)", ""]
    lines += [f"- {k}: {v}" for k, v in report["top_features_by_gain"].items()]
    (out / f"eval_{version}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="P6 Train and evaluate")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--no-ablations", action="store_true")
    args = parser.parse_args()
    run_training(args.dataset, args.version, not args.no_ablations)


if __name__ == "__main__":
    main()
