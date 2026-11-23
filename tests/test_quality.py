"""Tests for the P2 quality gate on small hand-made data."""

import numpy as np
import pandas as pd

from trace_aml import contract as c
from trace_aml.pipeline import quality as q
from trace_aml.pipeline.labels import build_account_labels, build_splits

START = pd.Timestamp("2022-09-01 00:00")
SHARES = {"train": 0.6, "validation": 0.2, "test": 0.2}
GATE = {
    "c1_dropped_rows": {"warn": 0.001, "fail": 0.01},
    "c2_min_mules": {"warn": 100, "fail": 30},
    "c3_mules_per_period": {"warn": 30, "fail": 10},
    "c3_test_new_share": {"warn": 0.8, "fail": 0.5},
    "c4_single_column_pr_auc": {"warn": 0.3, "fail": 0.8},
    "c4_bins": 10,
    "c5_single_threshold_f1": {"warn": 0.5, "fail": 0.8},
    "c6_mules_in_largest_component": {"warn": 0.5, "fail": 0.25},
    "c6_hub_top_n": 10,
    "c7_constant_features": {"warn": 0, "fail": 3},
}


def make_dataset(laundering_channel: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """100 hours of random normal transfers plus one laundering chain S -> M -> K every 2 hours.

    If laundering_channel is given, every laundering transfer uses that channel and no
    normal transfer does: a deliberate giveaway.
    """
    rng = np.random.default_rng(7)
    normal = [f"1_N{n}" for n in range(150)]
    rows = []
    for _ in range(3000):
        sender, receiver = rng.choice(normal, size=2, replace=False)
        rows.append((float(rng.uniform(0, 100)), sender, receiver, float(rng.lognormal(4, 1)),
                     str(rng.choice(["ACH", "Wire"])), 0))
    for chain in range(49):
        hour = 1 + 2 * chain
        channel = laundering_channel or str(rng.choice(["ACH", "Wire"]))
        source, mule, sink = f"2_S{chain}", f"2_M{chain}", str(rng.choice(normal))
        amount = float(rng.lognormal(4, 1))
        rows.append((hour, source, mule, amount, channel, 1))
        rows.append((hour + 0.5, mule, sink, amount, channel, 1))
        rows.append((hour - 0.5, str(rng.choice(normal)), mule, float(rng.lognormal(4, 1)), "ACH", 0))
    df = pd.DataFrame(rows, columns=["hour", "sender_id", "receiver_id", "amount", "channel", "is_laundering"])
    df = df.sort_values("hour", kind="stable").reset_index(drop=True)
    transactions = pd.DataFrame({
        "txn_id": pd.Series(range(len(df)), dtype="int64"),
        "timestamp": (START + pd.to_timedelta(df["hour"], unit="h")).astype("datetime64[ns]"),
        "sender_id": df["sender_id"].astype("string"),
        "receiver_id": df["receiver_id"].astype("string"),
        "amount": df["amount"],
        "amount_original": df["amount"],
        "currency_original": pd.Series("US Dollar", index=df.index, dtype="string"),
        "channel": df["channel"].astype("string"),
        "is_self": pd.Series(False, index=df.index),
        "dataset": pd.Series("test", index=df.index, dtype="string"),
    })
    labels_txn = pd.DataFrame({
        "txn_id": transactions["txn_id"],
        "is_laundering": df["is_laundering"].astype("int8"),
        "pattern_type": pd.Series(pd.NA, index=df.index, dtype="string"),
        "pattern_id": pd.Series(pd.NA, index=df.index, dtype="string"),
    })
    return transactions, labels_txn


def run_gate(laundering_channel: str | None, tmp_path) -> dict[str, dict]:
    transactions, labels_txn = make_dataset(laundering_channel)
    account_labels = build_account_labels(transactions, labels_txn)
    splits = build_splits("test", transactions, account_labels, SHARES)
    counts = {"rows_read": len(transactions), "dropped_exact_duplicates": 0}
    checks = q.run_checks(transactions, labels_txn, account_labels, splits, counts, tmp_path / "none", GATE)
    return {check["id"]: check for check in checks}


def test_status_helpers():
    assert q.low_is_bad(5, {"warn": 30, "fail": 10}) == "FAIL"
    assert q.low_is_bad(20, {"warn": 30, "fail": 10}) == "WARN"
    assert q.low_is_bad(30, {"warn": 30, "fail": 10}) == "PASS"
    assert q.high_is_bad(0.9, {"warn": 0.3, "fail": 0.8}) == "FAIL"
    assert q.high_is_bad(0.1, {"warn": 0.3, "fail": 0.8}) == "PASS"
    assert q.worst(["PASS", "WARN", "PASS"]) == "WARN"
    assert q.worst(["WARN", "FAIL"]) == "FAIL"


def test_one_column_model_finds_a_perfect_giveaway():
    train = pd.DataFrame({"col": pd.Series(["a", "a", "b", "b"], dtype="string"), "y": [0, 0, 1, 1]})
    later = pd.DataFrame({"col": pd.Series(["a", "b", "new"], dtype="string"), "y": [0, 1, 0]})
    scores = q.one_column_scores(train["col"], train["y"], later["col"], bins=5)
    assert scores.tolist() == [0.0, 1.0, 0.5]          # unseen value gets the overall rate
    assert q.one_column_pr_auc(train, later, ["col"], "y", bins=5)["col"] == 1.0


def test_one_column_model_bins_numbers():
    train = pd.DataFrame({"x": np.arange(100.0), "y": (np.arange(100) >= 90).astype(int)})
    assert q.one_column_pr_auc(train, train, ["x"], "y", bins=10)["x"] == 1.0


def test_best_single_threshold_f1_works_on_both_sides():
    target = np.array([0, 0, 0, 1, 1])
    assert q.best_single_threshold_f1(np.array([1.0, 2, 3, 8, 9]), target) == 1.0
    assert q.best_single_threshold_f1(np.array([9.0, 8, 7, 2, 1]), target) == 1.0   # mules are the low side
    assert q.best_single_threshold_f1(np.array([1.0, 9, 5, 5, 5]), target) < 1.0


def test_all_seven_checks_run_and_report(tmp_path):
    checks = run_gate(None, tmp_path)
    assert list(checks) == ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    assert all(check["status"] in c.STATUSES for check in checks.values())
    assert checks["C1"]["status"] == "PASS"
    assert checks["C2"]["value"] == 49                  # one mule per chain
    assert checks["C7"]["status"] == "WARN"             # no feature files yet
    report = q.report_markdown("test", q.worst([k["status"] for k in checks.values()]), list(checks.values()))
    assert "| C4 | Giveaway test |" in report


def test_giveaway_channel_fails_c4(tmp_path):
    assert run_gate("Bitcoin", tmp_path)["C4"]["status"] == "FAIL"
    assert "channel" in run_gate("Bitcoin", tmp_path)["C4"]["detail"].split("=")[0]


def test_no_giveaway_channel_does_not_fail_on_channel(tmp_path):
    detail = run_gate(None, tmp_path)["C4"]["detail"]
    channel_value = float(detail.split("transaction:channel ")[1].split(",")[0])
    assert channel_value < 0.3


def test_c5_catches_a_trivial_degree_shortcut():
    transactions, labels_txn = make_dataset(None)
    account_labels = build_account_labels(transactions, labels_txn)
    splits = build_splits("test", transactions, account_labels, SHARES)
    check = q.check_overlap(transactions, account_labels, splits, GATE["c5_single_threshold_f1"])
    # Mules here have exactly 2 senders and 1 receiver while normal accounts have dozens.
    assert check["status"] in ("WARN", "FAIL")


def test_c7_flags_constant_and_empty_features(tmp_path):
    folder = tmp_path / "features"
    folder.mkdir()
    pd.DataFrame({
        "account_id": pd.Series(["a", "b", "c"], dtype="string"),
        "cutoff": pd.to_datetime(["2022-09-02"] * 3),
        "good": np.array([1.0, 2.0, 3.0], dtype="float32"),
        "constant": np.array([5.0, 5.0, 5.0], dtype="float32"),
        "empty": np.array([np.nan] * 3, dtype="float32"),
    }).to_parquet(folder / "20220902T0000.parquet")
    check = q.check_compatibility(folder, GATE["c7_constant_features"])
    assert check["status"] == "FAIL" and "constant" in check["detail"] and "empty" in check["detail"]


def test_c1_fails_when_a_column_is_missing():
    transactions, _ = make_dataset(None)
    check = q.check_format(transactions.drop(columns=["channel"]), {}, GATE["c1_dropped_rows"])
    assert check["status"] == "FAIL"
