"""F-14 Retrain with analyst feedback, model versions (docs/03-ARCHITECTURE.md 3.2, 05 section 11).

retrain:  latest decision per account -> one training row (its features from the run where the
          decision was made). Confirmed mule = 1, False positive = 0, Needs investigation ignored.
          Feedback overrides that account's original training labels. Trains with the parent's
          settings, saves models/v<n+1>/ and a model_versions row. Not activated automatically.
          eval_valid = 1 only if every decision used came from a TRAIN or VALIDATION split run.
simulate: offline value of feedback (05 section 11): a simulated analyst labels the top
          `alerts_per_round` validation alerts with the truth each round; retrain; test metrics.

Usage:
    python -m trace_aml.pipeline.retrain --dataset ibm_hi_small --by Admin
    python -m trace_aml.pipeline.retrain --dataset ibm_hi_small --simulate
"""

import argparse
import json
import re
import sqlite3

import numpy as np
import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.models import metrics
from trace_aml.models.approaches import fit_lightgbm
from trace_aml.models.rows import account_groups, load_period_rows, period_bounds
from trace_aml.pipeline.common import now_utc
from trace_aml.pipeline.train import full_columns, judged, load_training_data, run_train, score_period

LABEL = {"CONFIRMED_MULE": 1, "FALSE_POSITIVE": 0}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(path("db"))
    conn.executescript(c.SQLITE_SCHEMA)
    return conn


def run_manifest(run_id: str) -> dict | None:
    for root in (path("runs"), path("fake") / "runs"):
        file = root / run_id / "manifest.json"
        if file.exists():
            return json.loads(file.read_text(encoding="utf-8"))
    return None


def usable_feedback(conn: sqlite3.Connection) -> pd.DataFrame:
    """Latest decision per account, with its label and whether it is evaluation-safe."""
    fb = pd.read_sql("SELECT * FROM feedback ORDER BY created_at, feedback_id", conn)
    columns = ["account_id", "run_id", "decision", "label", "eval_safe"]
    if fb.empty:
        return pd.DataFrame(columns=columns)
    latest = fb.groupby("account_id").tail(1).copy()
    latest["label"] = latest["decision"].map(LABEL)
    manifests = {run: run_manifest(run) or {} for run in latest["run_id"].unique()}
    latest["eval_safe"] = latest["run_id"].map(lambda r: manifests[r].get("kind") == "SPLIT"
                                               and manifests[r].get("split") in ("TRAIN", "VALIDATION"))
    return latest[columns].reset_index(drop=True)


def feedback_summary(feedback: pd.DataFrame) -> dict:
    """Counts for the Models screen: usable decisions and whether a retrain would be evaluation-safe."""
    used = feedback.dropna(subset=["label"])
    return {"confirmed": int((used["label"] == 1).sum()), "false_positive": int((used["label"] == 0).sum()),
            "ignored": int(feedback["label"].isna().sum()),
            "eval_safe": bool(len(used)) and bool(used["eval_safe"].all())}


def feedback_rows(feedback: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Training rows from the runs' saved features, and their labels."""
    used = feedback.dropna(subset=["label"])
    parts = []
    for run_id, group in used.groupby("run_id"):
        folder = next((root / run_id for root in (path("runs"), path("fake") / "runs") if (root / run_id).exists()), None)
        if folder is None:
            continue
        feats = pd.read_parquet(folder / c.RUN_FEATURES.filename)
        rows = feats[feats["account_id"].isin(group["account_id"])].copy()
        rows["label"] = rows["account_id"].map(group.set_index("account_id")["label"])
        if run_manifest(run_id).get("dataset") == "demo":       # demo IDs back to original IDs
            account_map = pd.read_csv(path("demo") / "demo_account_map.csv")
            rows["account_id"] = rows["account_id"].map(dict(zip(account_map["demo_id"], account_map["original_id"])))
        parts.append(rows)
    if not parts:
        return pd.DataFrame(columns=["account_id", *columns]), pd.Series(dtype="int8")
    rows = pd.concat(parts, ignore_index=True)
    return rows[["account_id", *columns]], rows["label"].astype("int8")


def with_feedback(data: dict, rows: pd.DataFrame, labels: pd.Series) -> dict:
    """Training data with feedback: overrides the account's existing rows, then adds the new rows."""
    data = dict(data)
    override = dict(zip(rows["account_id"], labels))
    target = data["target"].copy()
    hit = data["train"]["account_id"].isin(override)
    target[hit] = data["train"].loc[hit, "account_id"].map(override).astype("int8").to_numpy()
    data["train"] = pd.concat([data["train"], rows[[col for col in rows.columns if col in data["train"].columns]]],
                              ignore_index=True)
    data["target"] = pd.concat([target, labels], ignore_index=True).astype("int8")
    return data


def next_version() -> str:
    numbers = [int(m.group(1)) for d in path("models").iterdir() if (m := re.fullmatch(r"v(\d+)", d.name))]
    return f"v{max(numbers, default=0) + 1}"


def register(conn: sqlite3.Connection, version: str, created_by: str, n_feedback: int, eval_valid: bool) -> None:
    """Insert (or refresh) a model_versions row from the model folder's meta.json."""
    folder = path("models") / version
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    k = 50
    conn.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?,?,?,?,?)", (
        version, meta.get("created_at", now_utc()), created_by, meta.get("parent_version"), n_feedback,
        int(eval_valid), meta["validation_metrics"]["pr_auc"], meta["validation_metrics"].get(f"precision_at_{k}"),
        k, str(folder.relative_to(path("models").parent))))
    conn.commit()


def register_existing(conn: sqlite3.Connection) -> None:
    """Make sure every model folder has a model_versions row (versions trained by the pipeline)."""
    known = {row[0] for row in conn.execute("SELECT version FROM model_versions")}
    for folder in sorted(path("models").iterdir()):
        if folder.is_dir() and folder.name not in known and (folder / "meta.json").exists():
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            register(conn, folder.name, "pipeline", int(meta.get("n_feedback_used", 0)),
                     bool(meta.get("eval_valid", True)))


def retrain(dataset: str, created_by: str, parent: str | None = None) -> dict:
    settings = load_settings()
    conn = connect()
    register_existing(conn)
    feedback = usable_feedback(conn)
    summary = feedback_summary(feedback)
    data = load_training_data(dataset, settings)
    columns = full_columns(data["train"])
    rows, labels = feedback_rows(feedback, columns)
    if len(rows):
        data = with_feedback(data, rows, labels)
    version = next_version()
    parent = parent or settings["model"]["version"]
    run_train(dataset, data=data, version=version, meta_extra={
        "parent_version": parent, "n_feedback_used": int(len(rows)), "eval_valid": summary["eval_safe"],
        "created_by": created_by})
    register(conn, version, created_by, int(len(rows)), summary["eval_safe"])
    return {"version": version, "parent": parent, "feedback_rows": int(len(rows)), **summary}


def simulate(dataset: str) -> dict:
    """Offline: does validation-period feedback help on test? (05 section 11)."""
    settings = load_settings()
    sim = settings["feedback_simulation"]
    params, seed = settings["model"]["lightgbm"], settings["random_seed"]
    k_values, fpr_share = settings["metrics"]["k_values"], settings["metrics"]["fpr_top_share"]
    data = load_training_data(dataset, settings)
    columns = full_columns(data["train"])
    test = load_period_rows(dataset, "TEST", data["split"], settings["scoring"]["snapshot_every_hours"])
    test_groups = account_groups(data["account_labels"], *period_bounds(data["split"], "TEST"))
    # Truth for validation accounts the analyst may review: 1 = became a mule by end of validation.
    val = data["validation_labelled"].assign(target=data["validation_target"].to_numpy())

    reviewed, train, target, results = set(), data["train"], data["target"], []
    for round_number in range(sim["rounds"] + 1):
        model = fit_lightgbm("LIGHTGBM", columns, train, target, data["validation_labelled"],
                             data["validation_target"], params, seed)
        t, s = judged(score_period(model, test, test_groups))
        m = metrics.main_metrics(t, s, k_values, fpr_share)
        results.append({"round": round_number, "feedback_rows": len(reviewed),
                        **{key: round(float(m[key]), 4) for key in ("pr_auc", "precision_at_50", "precision_at_100",
                                                                     "recall_at_500")}})
        print(f"  round {round_number}: {results[-1]}", flush=True)
        if round_number == sim["rounds"]:
            break
        scored = val.assign(score=model.score(val))
        best = scored.sort_values("score", ascending=False).drop_duplicates("account_id")
        picked = best[~best["account_id"].isin(reviewed)].head(sim["alerts_per_round"])
        reviewed |= set(picked["account_id"])
        rows = picked[["account_id", *columns]]
        data_round = with_feedback({"train": train, "target": target}, rows, picked["target"].astype("int8"))
        train, target = data_round["train"], data_round["target"]
    result = {"dataset": dataset, "alerts_per_round": sim["alerts_per_round"], "rounds": sim["rounds"],
              "seed": seed, "feedback_from": "VALIDATION period only (evaluation-safe)", "results": results}
    version = settings["model"]["version"]
    (path("reports") / f"feedback_sim_{version}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ibm_hi_small")
    p.add_argument("--by", default="Admin")
    p.add_argument("--simulate", action="store_true")
    args = p.parse_args()
    out = simulate(args.dataset) if args.simulate else retrain(args.dataset, args.by)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
