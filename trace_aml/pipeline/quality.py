"""P2 Quality gate: a pass / warn / fail report card for a dataset (F-02).

Runs checks C1-C7 from docs/02-DATA.md section 8 on the processed files and writes
reports/quality_<dataset>.md and .json (format: docs/04-CONTRACT.md section 8).

Usage:
    python -m trace_aml.pipeline.quality --dataset ibm_hi_small
"""

import argparse
import json
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from trace_aml.config import load_settings, path
from trace_aml.pipeline.common import now_utc, write_manifest

ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def status_high_is_bad(value: float, warn: float, fail: float) -> str:
    """FAIL if value >= fail, WARN if value >= warn, else PASS."""
    return "FAIL" if value >= fail else "WARN" if value >= warn else "PASS"


def status_low_is_bad(value: float, warn: float, fail: float) -> str:
    """FAIL if value < fail, WARN if value < warn, else PASS."""
    return "FAIL" if value < fail else "WARN" if value < warn else "PASS"


def check(check_id: str, name: str, status: str, value, detail: str) -> dict:
    return {"id": check_id, "name": name, "status": status, "value": value, "detail": detail}


def single_column_pr_auc(train_x: pd.Series, train_y: pd.Series,
                         test_x: pd.Series, test_y: pd.Series, seed: int) -> float:
    """How well ONE column predicts the label on held-out rows (PR-AUC).

    Text columns: score = the category's laundering rate learned on train (smoothed).
    Number columns: a shallow decision tree on that one number.
    """
    if test_y.sum() == 0:
        return float("nan")
    if pd.api.types.is_numeric_dtype(train_x):
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                      class_weight="balanced", random_state=seed)
        tree.fit(train_x.to_frame().fillna(0), train_y)
        scores = tree.predict_proba(test_x.to_frame().fillna(0))[:, 1]
    else:
        prior, weight = train_y.mean(), 20
        stats = train_y.groupby(train_x.astype(str)).agg(["sum", "count"])
        rate = (stats["sum"] + prior * weight) / (stats["count"] + weight)
        scores = test_x.astype(str).map(rate).fillna(prior).to_numpy()
    return float(average_precision_score(test_y, scores))


def best_single_threshold_f1(values: pd.Series, labels: pd.Series) -> float:
    """Best F1 any single cut-off on one number can reach (either direction)."""
    best = 0.0
    for scores in (values, -values):
        precision, recall, _ = precision_recall_curve(labels, scores.fillna(0))
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        best = max(best, float(np.nanmax(f1)))
    return best


def c1_format(counts: dict, limits: dict) -> dict:
    dropped = (counts.get("dropped_bad_timestamp", 0) + counts.get("dropped_non_positive_amount", 0)
               + counts.get("dropped_exact_duplicates", 0))
    share = dropped / counts["rows_read"]
    return check("C1", "Format", status_high_is_bad(share, limits["warn"], limits["fail"]), round(share, 6),
                 f"{dropped} of {counts['rows_read']:,} rows dropped as broken ({share:.4%}); "
                 f"{counts.get('dropped_after_cutoff', 0):,} tail rows removed on purpose")


def c2_labels(roles: pd.Series, limits: dict) -> dict:
    mules = int((roles == "MULE").sum())
    detail = ", ".join(f"{k} {v:,}" for k, v in roles.value_counts().items())
    return check("C2", "Labels", status_low_is_bad(mules, limits["warn"], limits["fail"]), mules, detail)


def c3_time_spread(mule_counts: dict, limits: dict) -> dict:
    periods = {"train": mule_counts["train"], "validation": mule_counts["validation"],
               "test": mule_counts["test_new"]}
    lowest = min(periods.values())
    return check("C3", "Time spread", status_low_is_bad(lowest, limits["warn"], limits["fail"]), lowest,
                 "new mules: " + ", ".join(f"{k} {v}" for k, v in periods.items())
                 + f"; known mules still active in test {mule_counts['test_known']}"
                 + f" (new share {mule_counts['test_new_share']})")


def c4_giveaway(txns: pd.DataFrame, accounts: pd.DataFrame, validation_end: pd.Timestamp,
                limits: dict, seed: int) -> dict:
    """Transaction level: train before validation_end, test after. Account level: 50/50 split."""
    txns = txns.assign(hour=txns["timestamp"].dt.hour,
                       sender_bank=txns["sender_id"].str.split("_").str[0],
                       receiver_bank=txns["receiver_id"].str.split("_").str[0])
    early = txns["timestamp"] < validation_end
    results = {}
    for col in ("amount", "channel", "currency_original", "hour", "sender_bank", "receiver_bank"):
        results[f"txn:{col}"] = single_column_pr_auc(
            txns.loc[early, col], txns.loc[early, "is_laundering"],
            txns.loc[~early, col], txns.loc[~early, "is_laundering"], seed)

    train, test = train_test_split(accounts, test_size=0.5, stratify=accounts["is_mule"], random_state=seed)
    for col in ("main_channel", "main_currency", "mean_amount", "bank"):
        results[f"account:{col}"] = single_column_pr_auc(
            train[col], train["is_mule"], test[col], test["is_mule"], seed)

    worst_col = max(results, key=lambda k: -1 if np.isnan(results[k]) else results[k])
    worst = results[worst_col]
    base_txn = txns.loc[~early, "is_laundering"].mean()
    base_acc = test["is_mule"].mean()
    detail = (f"highest single-column PR-AUC {worst:.3f} ({worst_col}); chance level is about "
              f"{base_txn:.4f} for transactions and {base_acc:.4f} for accounts. All: "
              + ", ".join(f"{k} {v:.3f}" for k, v in results.items()))
    return check("C4", "Giveaway test", status_high_is_bad(worst, limits["warn"], limits["fail"]),
                 round(worst, 4), detail)


def c5_overlap(accounts: pd.DataFrame, limits: dict) -> dict:
    results = {col: best_single_threshold_f1(accounts[col], accounts["is_mule"])
               for col in ("in_degree", "out_degree", "total_amount")}
    worst_col = max(results, key=results.get)
    worst = results[worst_col]
    return check("C5", "Overlap", status_high_is_bad(worst, limits["warn"], limits["fail"]), round(worst, 4),
                 f"best one-threshold F1 {worst:.3f} ({worst_col}). All: "
                 + ", ".join(f"{k} {v:.3f}" for k, v in results.items()))


def c6_graph(txns: pd.DataFrame, roles: pd.DataFrame, limits: dict) -> dict:
    edges = txns.loc[~txns["is_self"], ["sender_id", "receiver_id"]].drop_duplicates()
    graph = ig.Graph.DataFrame(edges, directed=False, use_vids=False)
    names = np.array(graph.vs["name"])
    membership = np.array(graph.connected_components().membership)
    largest = np.bincount(membership).argmax()
    in_largest = set(names[membership == largest])
    mules = roles.loc[roles["role"] == "MULE", "account_id"]
    share = float(mules.isin(in_largest).mean())

    degree = pd.Series(graph.degree(), index=names)
    top = degree[degree >= degree.quantile(0.99)].index
    role_of = roles.set_index("account_id")["role"]
    normal_share_top = float((role_of.reindex(top) == "NORMAL").mean())

    status = status_low_is_bad(share, limits["warn"], limits["fail"])
    if normal_share_top < 0.5:
        status = "FAIL"
    return check("C6", "Graph sanity", status, round(share, 4),
                 f"{share:.1%} of mules are in the largest connected part "
                 f"({len(in_largest):,} of {len(names):,} accounts); {normal_share_top:.1%} of the "
                 f"top-1% most connected accounts are normal (normal hubs exist)")


def c7_compatibility(dataset: str) -> dict:
    features = path("features") / dataset
    if not features.exists() or not any(features.glob("*.parquet")):
        return check("C7", "Compatibility", "WARN", None,
                     "not checked yet: no feature files exist (run again after the feature step)")
    newest = max(features.glob("[0-9]*.parquet"), key=lambda p: p.stat().st_mtime)
    df = pd.read_parquet(newest)
    feature_cols = [c for c in df.columns if c not in ("account_id", "cutoff")]
    constant = [c for c in feature_cols if df[c].nunique(dropna=True) <= 1]
    limits = load_settings()["quality_gate"]["c7_constant_features"]
    status = "FAIL" if len(constant) > limits["fail"] else "WARN" if len(constant) > limits["warn"] else "PASS"
    return check("C7", "Compatibility", status, len(constant),
                 f"{len(feature_cols)} features in {newest.name}; constant or empty: {constant or 'none'}")


def account_table(txns: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Simple per-account numbers for C4 and C5; MULE vs NORMAL only."""
    real = txns[~txns["is_self"]]
    sent = real.groupby("sender_id").agg(out_degree=("receiver_id", "nunique"), out_amount=("amount", "sum"))
    received = real.groupby("receiver_id").agg(in_degree=("sender_id", "nunique"), in_amount=("amount", "sum"))
    both = pd.concat([
        txns[["sender_id", "channel", "currency_original", "amount"]].rename(columns={"sender_id": "account_id"}),
        txns[["receiver_id", "channel", "currency_original", "amount"]].rename(columns={"receiver_id": "account_id"}),
    ])
    grouped = both.groupby("account_id")
    table = pd.DataFrame({
        "main_channel": grouped["channel"].agg(lambda s: s.mode().iat[0]),
        "main_currency": grouped["currency_original"].agg(lambda s: s.mode().iat[0]),
        "mean_amount": grouped["amount"].mean(),
    })
    table = table.join(sent).join(received).fillna({"out_degree": 0, "in_degree": 0,
                                                    "out_amount": 0, "in_amount": 0})
    table["total_amount"] = table["out_amount"] + table["in_amount"]
    table["bank"] = table.index.str.split("_").str[0]
    labelled = roles[roles["role"].isin(["MULE", "NORMAL"])].set_index("account_id")
    table = table.join(labelled["is_mule"], how="inner")
    table["is_mule"] = table["is_mule"].astype(int)
    return table


def write_report(dataset: str, checks: list[dict]) -> Path:
    overall = max((c["status"] for c in checks), key=ORDER.get)
    report = {"dataset": dataset, "overall": overall, "checks": checks}
    out = path("reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"quality_{dataset}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    lines = [f"# Quality report: {dataset}", "", f"**Overall: {icon[overall]} {overall}**", "",
             "| Check | Status | Value | Detail |", "|---|---|---|---|"]
    lines += [f"| {c['id']} {c['name']} | {icon[c['status']]} {c['status']} | {c['value']} | {c['detail']} |"
              for c in checks]
    lines += ["", "Thresholds: config/settings.yaml `quality_gate`. Meaning of each check: docs/02-DATA.md section 8."]
    (out / f"quality_{dataset}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def run_quality(dataset: str) -> Path:
    """Run P2 for one dataset and return the reports folder."""
    started = now_utc()
    settings = load_settings()
    limits = settings["quality_gate"]
    seed = settings["random_seed"]
    folder = path("processed") / dataset

    manifest = json.loads((folder / "manifest_ingest.json").read_text(encoding="utf-8"))
    splits = json.loads((folder / "splits.json").read_text(encoding="utf-8"))
    txns = pd.read_parquet(folder / "transactions.parquet")
    txns["is_laundering"] = pd.read_parquet(folder / "labels_txn.parquet", columns=["is_laundering"])["is_laundering"]
    roles = pd.read_parquet(folder / "account_labels.parquet")

    print("Building account table ...")
    accounts = account_table(txns, roles)
    checks = [c1_format(manifest["counts"], limits["c1_dropped_rows"]),
              c2_labels(roles["role"], limits["c2_min_mules"]),
              c3_time_spread(splits["mule_counts"], limits["c3_new_mules_per_period"])]
    print("C4 giveaway test ...")
    checks.append(c4_giveaway(txns, accounts, pd.Timestamp(splits["validation_end"]),
                              limits["c4_single_column_pr_auc"], seed))
    print("C5 overlap ...")
    checks.append(c5_overlap(accounts, limits["c5_single_threshold_f1"]))
    print("C6 graph sanity ...")
    checks.append(c6_graph(txns, roles, limits["c6_mules_in_largest_component"]))
    checks.append(c7_compatibility(dataset))

    out = write_report(dataset, checks)
    write_manifest(folder, "quality", started, dataset=dataset,
                   overall=max((c["status"] for c in checks), key=ORDER.get))
    for c in checks:
        print(f"[{c['status']:<4}] {c['id']} {c['name']}: {c['detail']}")
    print(f"Wrote {out / f'quality_{dataset}.md'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 Quality gate")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    run_quality(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
