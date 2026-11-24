"""F-12 Replay and "caught in time" (docs/05-FEATURES-MODEL.md section 8).

replay:   step through the test period every `replay.step_hours`; at each step T compute
          features and rules from transactions up to T only, score with the frozen model and
          keep every account at or above watch_cut. No labels.
evaluate: for each complete laundering attempt (ring) that ends inside the test period,
          lead = cash-out time - first step at which any member reached likely_mule_cut.

Writes data/replay/<dataset>__<version>/alerts.parquet and reports/replay_<version>.{json,md}.

Usage:
    python -m trace_aml.pipeline.replay --dataset ibm_hi_small            (replay + evaluate)
    python -m trace_aml.pipeline.replay --dataset ibm_hi_small --evaluate-only
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from trace_aml import contract as c
from trace_aml.config import PROJECT_ROOT, load_settings, path
from trace_aml.models.calibrate import to_risk
from trace_aml.pipeline.common import now_utc, write_manifest
from trace_aml.pipeline.features import compute_features
from trace_aml.pipeline.rules import FLAG_COLUMNS, compute_rules
from trace_aml.pipeline.score import load_model


def replay_folder(dataset: str, version: str):
    return path("runs").parent / "replay" / f"{dataset}__{version}"      # data/replay/...


def steps_for(splits: dict, step_hours: int) -> list[pd.Timestamp]:
    start, end = pd.Timestamp(splits["validation_end"]), pd.Timestamp(splits["test_end"])
    return list(pd.date_range(start + pd.Timedelta(hours=step_hours), end, freq=f"{step_hours}h"))


def score_step(transactions: pd.DataFrame, cutoff: pd.Timestamp, model: dict, settings: dict) -> pd.DataFrame:
    """Risk for every account active in the 24 h before `cutoff`, from data up to `cutoff` only."""
    features = compute_features(transactions, cutoff, settings)
    flags, _ = compute_rules(transactions, cutoff, settings)
    rows = features.merge(flags[["account_id", *FLAG_COLUMNS]], on="account_id", how="left")
    rows[FLAG_COLUMNS] = rows[FLAG_COLUMNS].fillna(0).astype("float32")
    raw = model["booster"].predict(rows[model["columns"]])
    _, risk = to_risk(model["calibrator"], raw)
    return pd.DataFrame({"account_id": rows["account_id"], "step": cutoff, "risk": risk.astype("float32")})


def run_replay(dataset: str) -> pd.DataFrame:
    settings = load_settings()
    version = settings["model"]["version"]
    model = load_model(path("models") / version)
    proc = path("processed") / dataset
    transactions = pd.read_parquet(proc / c.TRANSACTIONS.filename)
    splits = json.loads((proc / "splits.json").read_text(encoding="utf-8"))
    steps = steps_for(splits, settings["replay"]["step_hours"])
    started, kept = now_utc(), []
    for number, step in enumerate(steps, start=1):
        clock = time.perf_counter()
        scored = score_step(transactions, step, model, settings)
        kept.append(scored[scored["risk"] >= model["cuts"]["watch_cut"]])
        print(f"  step {number}/{len(steps)} {step:%d %b %H:%M}: {len(scored):,} scored, "
              f"{int((scored['risk'] >= model['cuts']['likely_mule_cut']).sum())} likely mule "
              f"({time.perf_counter() - clock:.0f} s)", flush=True)
    alerts = pd.concat(kept, ignore_index=True)
    out = replay_folder(dataset, version)
    out.mkdir(parents=True, exist_ok=True)
    alerts.to_parquet(out / "alerts.parquet", index=False)
    write_manifest(out, "replay", started, dataset=dataset, model_version=version, steps=len(steps),
                   step_hours=settings["replay"]["step_hours"], cuts=model["cuts"], alert_rows=len(alerts))
    return alerts


def complete_attempts(dataset: str, labels_txn: pd.DataFrame) -> set[str]:
    """Pattern attempts whose every pattern-file row survived cleaning (not cut by the tail)."""
    from trace_aml.pipeline.ingest import read_ibm_patterns
    info = load_settings()["datasets"][dataset]
    in_file = read_ibm_patterns(PROJECT_ROOT / info["raw_dir"] / info["patterns_file"])["pattern_id"].value_counts()
    kept = labels_txn["pattern_id"].dropna().value_counts()
    return {pid for pid, n in in_file.items() if kept.get(pid, 0) == n}


def evaluate(dataset: str) -> dict:
    """Lead times per ring (evaluation only: reads labels)."""
    settings = load_settings()
    version = settings["model"]["version"]
    cuts = load_model(path("models") / version)["cuts"]
    alerts = pd.read_parquet(replay_folder(dataset, version) / "alerts.parquet")
    likely = alerts[alerts["risk"] >= cuts["likely_mule_cut"]]
    first_alert = likely.groupby("account_id")["step"].min()

    proc = path("processed") / dataset
    transactions = pd.read_parquet(proc / c.TRANSACTIONS.filename,
                                   columns=["txn_id", "timestamp", "sender_id", "receiver_id"])
    labels_txn = pd.read_parquet(proc / c.LABELS_TXN.filename)
    roles = pd.read_parquet(proc / c.ACCOUNT_LABELS.filename, columns=["account_id", "role"]).set_index("account_id")["role"]
    splits = json.loads((proc / "splits.json").read_text(encoding="utf-8"))
    start, end = pd.Timestamp(splits["validation_end"]), pd.Timestamp(splits["test_end"])

    complete = complete_attempts(dataset, labels_txn)
    laundering = transactions.merge(labels_txn[labels_txn["is_laundering"] == 1], on="txn_id")
    laundering = laundering[laundering["pattern_id"].isin(complete) & (laundering["sender_id"] != laundering["receiver_id"])]
    rows = []
    for pid, ring in laundering.groupby("pattern_id"):
        if not (start <= ring["timestamp"].max() < end):
            continue                                       # ring's laundering must end inside the test period
        into_sink = ring[ring["receiver_id"].map(roles) == "SINK"]
        cashout = (into_sink if len(into_sink) else ring)["timestamp"].max()
        members = set(ring["sender_id"]) | set(ring["receiver_id"])
        alert = first_alert.reindex(list(members)).min()
        lead = (cashout - alert).total_seconds() / 60 if pd.notna(alert) else None
        rows.append({"pattern_id": pid, "pattern_type": ring["pattern_type"].iloc[0], "members": len(members),
                     "start": ring["timestamp"].min(), "cashout": cashout, "first_alert": alert, "lead_minutes": lead})
    table = pd.DataFrame(rows)
    leads = table["lead_minutes"].dropna()
    before = leads[leads > 0]
    # Strict: the alert came after the ring's first laundering transfer (an alert before it was
    # raised by earlier activity of an account IBM reuses across attempts, not by this ring).
    strict = table[(table["lead_minutes"] > 0) & (table["first_alert"] >= table["start"])]
    result = {
        "dataset": dataset, "model_version": version, "step_hours": settings["replay"]["step_hours"],
        "ring_definition": "PATTERN_FILE (complete attempts only)", "likely_mule_cut": cuts["likely_mule_cut"],
        "rings": int(len(table)),
        "caught_before_cashout": int((table["lead_minutes"] > 0).sum()),
        "caught_after_cashout": int((table["lead_minutes"] <= 0).sum()),
        "never_caught": int(table["lead_minutes"].isna().sum()),
        "share_before_cashout": round(float((table["lead_minutes"] > 0).mean()), 4) if len(table) else None,
        "lead_minutes_median": float(before.median()) if len(before) else None,
        "lead_minutes_p25_p75": [float(before.quantile(0.25)), float(before.quantile(0.75))] if len(before) else None,
        "strict_caught_after_start_before_cashout": int(len(strict)),
        "strict_share": round(len(strict) / len(table), 4) if len(table) else None,
        "strict_lead_minutes_median": float(strict["lead_minutes"].median()) if len(strict) else None,
        "alert_before_ring_start": int(((table["lead_minutes"] > 0) & (table["first_alert"] < table["start"])).sum()),
        "by_pattern": {k: {"rings": int(len(g)), "before": int((g["lead_minutes"] > 0).sum())}
                       for k, g in table.groupby("pattern_type")},
    }
    reports = path("reports")
    (reports / f"replay_{version}.json").write_text(json.dumps(
        {**result, "rings_table": table.astype(str).to_dict("records")}, indent=2), encoding="utf-8")
    lines = [f"# Replay: caught in time ({dataset}, model {version})", "",
             f"Synthetic data. Steps every {result['step_hours']} h through the test period, so lead times are "
             f"accurate to about {result['step_hours']} h. Rings: {result['ring_definition']}.", "",
             "| Measure | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in result.items() if k not in ("by_pattern",)]
    lines += ["", "| Pattern | Rings | Caught before cash-out |", "|---|---|---|"]
    lines += [f"| {k} | {v['rings']} | {v['before']} |" for k, v in result["by_pattern"].items()]
    (reports / f"replay_{version}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ibm_hi_small")
    p.add_argument("--evaluate-only", action="store_true")
    args = p.parse_args()
    if not args.evaluate_only:
        run_replay(args.dataset)
    print(json.dumps(evaluate(args.dataset), indent=2, default=str))


if __name__ == "__main__":
    main()
