"""P2 Quality gate: a report card every dataset must pass before training (F-02).

Reads data/processed/<dataset>/ (after ingest and labels) and writes
reports/quality_<dataset>.md and .json (docs/02-DATA.md section 8).

Checks that look at labels (C4, C5, C6) use only the train and validation periods,
so the test period stays untouched. This module is not feature code; it may read labels.

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

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.pipeline.common import file_info, now_utc, write_manifest
from trace_aml.pipeline.labels import labels_as_of

Check = dict  # {"id", "name", "status", "value", "detail"}


def low_is_bad(value: float, limits: dict) -> str:
    """PASS / WARN / FAIL when small values are the problem (e.g. number of mules)."""
    return "FAIL" if value < limits["fail"] else "WARN" if value < limits["warn"] else "PASS"


def high_is_bad(value: float, limits: dict) -> str:
    """PASS / WARN / FAIL when large values are the problem (e.g. a giveaway score)."""
    return "FAIL" if value >= limits["fail"] else "WARN" if value >= limits["warn"] else "PASS"


def worst(statuses: list[str]) -> str:
    """The worst status in a list."""
    return max(statuses, key=c.STATUSES.index)


def make_check(check_id: str, name: str, status: str, value: float, detail: str) -> Check:
    return {"id": check_id, "name": name, "status": status, "value": round(float(value), 4), "detail": detail}


# --- C1 Format ---------------------------------------------------------------

def check_format(transactions: pd.DataFrame, ingest_counts: dict, limits: dict) -> Check:
    """C1: required columns present, and the share of rows dropped as broken."""
    missing = [col.name for col in c.TRANSACTIONS.columns if col.name not in transactions.columns]
    if missing:
        return make_check("C1", "Format", "FAIL", 1.0, f"missing columns {missing}")
    dropped = sum(ingest_counts.get(key, 0) for key in
                  ("dropped_bad_timestamp", "dropped_non_positive_amount", "dropped_exact_duplicates"))
    share = dropped / max(ingest_counts.get("rows_read", len(transactions)), 1)
    detail = f"{share:.4%} of rows dropped as broken ({dropped:,} rows)"
    if ingest_counts.get("dropped_after_cutoff"):
        detail += f"; {ingest_counts['dropped_after_cutoff']:,} tail rows removed on purpose"
    return make_check("C1", "Format", high_is_bad(share, limits), share, detail)


# --- C2 Labels and C3 Time spread ---------------------------------------------

def check_labels(account_labels: pd.DataFrame, limits: dict) -> Check:
    """C2: how many accounts have each role at the end of the data."""
    counts = account_labels["role"].value_counts().to_dict()
    mules = counts.get("MULE", 0)
    detail = ", ".join(f"{counts.get(role, 0):,} {role.lower()}" for role in c.ROLES)
    return make_check("C2", "Labels", low_is_bad(mules, limits), mules, detail)


def check_time_spread(transactions: pd.DataFrame, account_labels: pd.DataFrame, splits: dict,
                      per_period: dict, new_share: dict) -> Check:
    """C3: mules in each period, and the share of test-period mules that training never saw."""
    counts = splits["mule_counts"]
    fewest = min(counts["train"], counts["validation"], counts["test_new"])

    train_end = pd.Timestamp(splits["train_end"])
    in_test = transactions["timestamp"] > pd.Timestamp(splits["validation_end"])
    active = set(transactions.loc[in_test, "sender_id"]) | set(transactions.loc[in_test, "receiver_id"])
    mules = account_labels[(account_labels["role"] == "MULE") & account_labels["account_id"].isin(active)]
    share_new = float((mules["mule_since"] > train_end).mean()) if len(mules) else 0.0

    status = worst([low_is_bad(fewest, per_period), low_is_bad(share_new, new_share)])
    detail = (f"mules: {counts['train']:,} train, {counts['validation']:,} new in validation, "
              f"{counts['test_new']:,} new in test; {share_new:.0%} of the {len(mules):,} mules active "
              f"in the test period were not mules in the train period")
    return make_check("C3", "Time spread", status, fewest, detail)


# --- C4 Giveaway test --------------------------------------------------------

def one_column_scores(train_values: pd.Series, train_target: pd.Series, eval_values: pd.Series,
                      bins: int) -> np.ndarray:
    """The tiny one-column model: the target rate seen in training for each value.

    Numbers are first cut into bins at the training quantiles. Values never seen in
    training get the overall training rate.
    """
    if pd.api.types.is_float_dtype(train_values):
        edges = np.unique(np.quantile(train_values.dropna(), np.linspace(0, 1, bins + 1)))
        train_key = pd.Series(np.searchsorted(edges, train_values.to_numpy()))
        eval_key = pd.Series(np.searchsorted(edges, eval_values.to_numpy()))
    else:
        train_key = train_values.astype("string").reset_index(drop=True)
        eval_key = eval_values.astype("string").reset_index(drop=True)
    target = train_target.reset_index(drop=True).astype("float64")
    rates = target.groupby(train_key).mean()
    return eval_key.map(rates).fillna(target.mean()).to_numpy(dtype="float64")


def one_column_pr_auc(train: pd.DataFrame, evaluate: pd.DataFrame, columns: list[str], target: str,
                      bins: int) -> dict[str, float]:
    """PR-AUC on the later period of a one-column model fitted on the earlier period."""
    if evaluate[target].sum() == 0 or train[target].sum() == 0:
        return {}
    return {col: float(average_precision_score(
        evaluate[target], one_column_scores(train[col], train[target], evaluate[col], bins)))
        for col in columns}


def add_simple_columns(transactions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add hour of day and (when account IDs carry a bank prefix) the bank columns."""
    out = transactions.copy()
    out["hour"] = out["timestamp"].dt.hour.astype("string")
    columns = ["amount", "channel", "currency_original", "hour"]
    if out["sender_id"].str.contains("_").all() and out["receiver_id"].str.contains("_").all():
        out["sender_bank"] = out["sender_id"].str.split("_", n=1).str[0]
        out["receiver_bank"] = out["receiver_id"].str.split("_", n=1).str[0]
        columns += ["sender_bank", "receiver_bank"]
    return out, columns


def account_profile(transactions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per account: median amount and its most common channel, currency, hour and bank."""
    sides = []
    for id_col, bank_col in (("sender_id", "sender_bank"), ("receiver_id", "receiver_bank")):
        side = transactions[[id_col, "amount", "channel", "currency_original", "hour"]].rename(
            columns={id_col: "account_id"})
        if bank_col in columns:
            side["bank"] = transactions[bank_col]
        sides.append(side)
    long = pd.concat(sides, ignore_index=True)
    profile = long.groupby("account_id")[["amount"]].median()
    for col in [name for name in long.columns if name not in ("account_id", "amount")]:
        counts = long.groupby(["account_id", col]).size().reset_index(name="n")
        top = counts.sort_values(["n", col], ascending=[False, True]).drop_duplicates("account_id")
        profile[col] = top.set_index("account_id")[col]
    return profile.reset_index()


def check_giveaway(transactions: pd.DataFrame, labels_txn: pd.DataFrame, account_labels: pd.DataFrame,
                   splits: dict, limits: dict, bins: int) -> Check:
    """C4: can one raw column alone pick out laundering (per transaction) or mules (per account)?"""
    train_end, validation_end = pd.Timestamp(splits["train_end"]), pd.Timestamp(splits["validation_end"])
    data, columns = add_simple_columns(transactions[~transactions["is_self"]])
    data = data.merge(labels_txn[["txn_id", "is_laundering"]], on="txn_id")
    early = data[data["timestamp"] <= train_end]
    later = data[(data["timestamp"] > train_end) & (data["timestamp"] <= validation_end)]

    results = {f"transaction:{col}": value for col, value in
               one_column_pr_auc(early, later, columns, "is_laundering", bins).items()}

    profiles = []
    for period, as_of in ((early, train_end), (later, validation_end)):
        roles = labels_as_of(account_labels, as_of)
        roles = roles[roles["is_mule"].notna()]
        profile = account_profile(period, columns).merge(roles[["account_id", "is_mule"]], on="account_id")
        profile["is_mule"] = profile["is_mule"].astype("int8")
        profiles.append(profile)
    account_columns = [col for col in profiles[0].columns if col not in ("account_id", "is_mule")]
    results.update({f"account:{col}": value for col, value in
                    one_column_pr_auc(profiles[0], profiles[1], account_columns, "is_mule", bins).items()})

    if not results:
        return make_check("C4", "Giveaway test", "WARN", 0.0,
                          "not enough laundering in the train or validation period to run this check")
    top = max(results, key=results.get)
    listed = ", ".join(f"{name} {value:.3f}" for name, value in sorted(results.items(), key=lambda x: -x[1]))
    detail = f"highest one-column PR-AUC: {top} = {results[top]:.3f} (fitted on train, scored on validation). All: {listed}"
    return make_check("C4", "Giveaway test", high_is_bad(results[top], limits), results[top], detail)


# --- C5 Overlap ----------------------------------------------------------------

def best_single_threshold_f1(values: np.ndarray, target: np.ndarray) -> float:
    """Best F1 any one cut on this number can reach, flagging either the high or the low side."""
    best = 0.0
    for signed in (values, -values):
        precision, recall, _ = precision_recall_curve(target, signed)
        f1 = 2 * precision * recall / np.where(precision + recall == 0, 1, precision + recall)
        best = max(best, float(f1.max()))
    return best


def account_simple_stats(transactions: pd.DataFrame) -> pd.DataFrame:
    """Per account: distinct senders, distinct receivers, and total money in + out."""
    in_degree = transactions.groupby("receiver_id")["sender_id"].nunique().rename("in_degree")
    out_degree = transactions.groupby("sender_id")["receiver_id"].nunique().rename("out_degree")
    total = (transactions.groupby("receiver_id")["amount"].sum()
             .add(transactions.groupby("sender_id")["amount"].sum(), fill_value=0).rename("total_amount"))
    stats = pd.concat([in_degree, out_degree, total], axis=1).fillna(0)
    stats.index.name = "account_id"
    return stats.reset_index()


def check_overlap(transactions: pd.DataFrame, account_labels: pd.DataFrame, splits: dict,
                  limits: dict) -> Check:
    """C5: can one threshold on one simple number separate mules from normal accounts?"""
    validation_end = pd.Timestamp(splits["validation_end"])
    known = transactions[~transactions["is_self"] & (transactions["timestamp"] <= validation_end)]
    roles = labels_as_of(account_labels, validation_end)
    roles = roles[roles["is_mule"].notna()]
    stats = account_simple_stats(known).merge(roles[["account_id", "is_mule"]], on="account_id")
    target = stats["is_mule"].astype("int8").to_numpy()
    if target.sum() == 0:
        return make_check("C5", "Overlap", "WARN", 0.0, "no mules up to the end of validation")
    results = {col: best_single_threshold_f1(stats[col].to_numpy(dtype="float64"), target)
               for col in ("in_degree", "out_degree", "total_amount")}
    top = max(results, key=results.get)
    listed = ", ".join(f"{name} {value:.3f}" for name, value in results.items())
    return make_check("C5", "Overlap", high_is_bad(results[top], limits), results[top],
                      f"best one-threshold F1 (data up to end of validation): {listed}")


# --- C6 Graph sanity ----------------------------------------------------------

def check_graph(transactions: pd.DataFrame, account_labels: pd.DataFrame, splits: dict,
                limits: dict, hub_top_n: int) -> Check:
    """C6: are mules inside the main network, and do normal accounts also act as hubs?"""
    validation_end = pd.Timestamp(splits["validation_end"])
    known = transactions[~transactions["is_self"] & (transactions["timestamp"] <= validation_end)]
    pairs = known[["sender_id", "receiver_id"]].drop_duplicates()
    codes, names = pd.factorize(pd.concat([pairs["sender_id"], pairs["receiver_id"]], ignore_index=True))
    half = len(pairs)
    graph = ig.Graph(n=len(names), edges=np.column_stack([codes[:half], codes[half:]]).tolist(), directed=False)

    membership = np.array(graph.connected_components().membership)
    largest = np.bincount(membership).argmax()
    roles = labels_as_of(account_labels, validation_end).set_index("account_id")["role"]
    node_role = pd.Series(names).map(roles).to_numpy()
    is_mule = node_role == "MULE"
    if is_mule.sum() == 0:
        return make_check("C6", "Graph sanity", "WARN", 0.0, "no mules up to the end of validation")
    share = float((membership[is_mule] == largest).mean())

    degree = np.array(graph.simplify().degree())
    hubs = np.argsort(-degree, kind="stable")[:hub_top_n]
    normal_hubs = int((node_role[hubs] == "NORMAL").sum())
    status = low_is_bad(share, limits)
    if normal_hubs == 0:
        status = "FAIL"
    detail = (f"{share:.0%} of mules are in the largest connected part; "
              f"{normal_hubs} of the {len(hubs)} best-connected accounts are normal")
    return make_check("C6", "Graph sanity", status, share, detail)


# --- C7 Compatibility ---------------------------------------------------------

def check_compatibility(features_folder: Path, limits: dict) -> Check:
    """C7: every feature column in the saved feature files has real, varying values."""
    files = [p for p in sorted(features_folder.glob("*.parquet")) if "_rules" not in p.name] \
        if features_folder.exists() else []
    if not files:
        return make_check("C7", "Compatibility", "WARN", 0.0,
                          "not checked yet: no feature files exist for this dataset (run P4, then rerun the gate)")
    frames = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    feature_columns = [col for col in frames.columns if col not in ("account_id", "cutoff")]
    empty = [col for col in feature_columns if frames[col].isna().all()]
    constant = [col for col in feature_columns if col not in empty and frames[col].nunique(dropna=True) <= 1]
    bad = len(empty) + len(constant)
    status = "FAIL" if empty or bad > limits["fail"] else "WARN" if bad > limits["warn"] else "PASS"
    detail = f"{len(feature_columns)} features in {len(files)} files; constant: {constant or 'none'}; all empty: {empty or 'none'}"
    return make_check("C7", "Compatibility", status, bad, detail)


# --- Report --------------------------------------------------------------------

def run_checks(transactions: pd.DataFrame, labels_txn: pd.DataFrame, account_labels: pd.DataFrame,
               splits: dict, ingest_counts: dict, features_folder: Path, gate: dict) -> list[Check]:
    """Run C1 to C7 and return them in order."""
    return [
        check_format(transactions, ingest_counts, gate["c1_dropped_rows"]),
        check_labels(account_labels, gate["c2_min_mules"]),
        check_time_spread(transactions, account_labels, splits, gate["c3_mules_per_period"],
                          gate["c3_test_new_share"]),
        check_giveaway(transactions, labels_txn, account_labels, splits,
                       gate["c4_single_column_pr_auc"], gate["c4_bins"]),
        check_overlap(transactions, account_labels, splits, gate["c5_single_threshold_f1"]),
        check_graph(transactions, account_labels, splits, gate["c6_mules_in_largest_component"],
                    gate["c6_hub_top_n"]),
        check_compatibility(features_folder, gate["c7_constant_features"]),
    ]


def report_markdown(dataset: str, overall: str, checks: list[Check]) -> str:
    """The human-readable report card."""
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
    lines = [f"# Data quality report: {dataset}", "",
             f"**Overall: {icon[overall]} {overall}**", "",
             "Label-based checks (C4, C5, C6) use only the train and validation periods.", "",
             "| # | Check | Status | Value | Detail |", "|---|---|---|---|---|"]
    lines += [f"| {k['id']} | {k['name']} | {icon[k['status']]} {k['status']} | {k['value']} | {k['detail']} |"
              for k in checks]
    return "\n".join(lines) + "\n"


def run_quality(dataset: str) -> str:
    """Run P2 for one dataset, write both report files and return the overall status."""
    started = now_utc()
    settings = load_settings()
    folder = path("processed") / dataset
    transactions = pd.read_parquet(folder / c.TRANSACTIONS.filename)
    labels_txn = pd.read_parquet(folder / c.LABELS_TXN.filename)
    account_labels = pd.read_parquet(folder / c.ACCOUNT_LABELS.filename)
    splits = json.loads((folder / c.SPLITS_FILENAME).read_text(encoding="utf-8"))
    ingest_manifest = folder / "manifest_ingest.json"
    ingest_counts = json.loads(ingest_manifest.read_text(encoding="utf-8"))["counts"] if ingest_manifest.exists() else {}

    checks = run_checks(transactions, labels_txn, account_labels, splits, ingest_counts,
                        path("features") / dataset, settings["quality_gate"])
    overall = worst([check["status"] for check in checks])

    reports = path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"quality_{dataset}.json").write_text(
        json.dumps({"dataset": dataset, "overall": overall, "checks": checks}, indent=2), encoding="utf-8")
    (reports / f"quality_{dataset}.md").write_text(report_markdown(dataset, overall, checks), encoding="utf-8")
    write_manifest(reports, f"quality_{dataset}", started, dataset=dataset, overall=overall,
                   inputs=[file_info(folder / c.TRANSACTIONS.filename), file_info(folder / c.ACCOUNT_LABELS.filename)])

    for check in checks:
        print(f"  {check['id']} {check['name']}: {check['status']} ({check['detail']})")
    print(f"Overall: {overall}")
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 Quality gate: report card for one dataset")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    run_quality(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
