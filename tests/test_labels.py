"""Tests for P3 labels and split on a tiny hand-made network."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_contract import check_dataframe  # noqa: E402
from trace_aml import contract as c  # noqa: E402
from trace_aml.pipeline.labels import (  # noqa: E402
    build_account_labels, build_splits, labels_as_of,
)
from trace_aml.pipeline.split import compute_split, snapshot_times  # noqa: E402

# (hour, sender, receiver, is_laundering, pattern_id). Hours count from 2022-09-01 00:00.
#   S -> M1 -> M2 -> K   a laundering chain in pattern P-0001 (hours 1, 2, 3)
#   L  receives dirty money at hour 4 and only forwards it at hour 90 (a "late" mule)
#   X <-> Y              two mules outside any pattern, linked to each other
#   N1 -> N2             normal transfers; N1 -> N1 is a laundering self-transfer (ignored)
ROWS = [
    (1, "S", "M1", 1, "P-0001"),
    (2, "M1", "M2", 1, "P-0001"),
    (3, "M2", "K", 1, "P-0001"),
    (4, "S", "L", 1, None),
    (5, "N1", "N2", 0, None),
    (6, "N1", "N1", 1, None),
    (10, "S", "X", 1, None),
    (11, "X", "Y", 1, None),
    (12, "Y", "X", 1, None),
    (90, "L", "K", 1, None),
    (100, "N2", "N1", 0, None),
]
START = pd.Timestamp("2022-09-01 00:00")
SHARES = {"train": 0.6, "validation": 0.2, "test": 0.2}


def make_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = pd.DataFrame({
        "txn_id": pd.Series(range(len(ROWS)), dtype="int64"),
        "timestamp": [START + pd.Timedelta(hours=row[0]) for row in ROWS],
        "sender_id": pd.Series([row[1] for row in ROWS], dtype="string"),
        "receiver_id": pd.Series([row[2] for row in ROWS], dtype="string"),
    })
    transactions["timestamp"] = transactions["timestamp"].astype("datetime64[ns]")
    transactions["is_self"] = transactions["sender_id"] == transactions["receiver_id"]
    labels_txn = pd.DataFrame({
        "txn_id": transactions["txn_id"],
        "is_laundering": pd.Series([row[3] for row in ROWS], dtype="int8"),
        "pattern_type": pd.Series(["CYCLE" if row[4] else None for row in ROWS], dtype="string"),
        "pattern_id": pd.Series([row[4] for row in ROWS], dtype="string"),
    })
    return transactions, labels_txn


def labels() -> pd.DataFrame:
    return build_account_labels(*make_inputs()).set_index("account_id")


def test_roles_at_end_of_data():
    roles = labels()["role"].to_dict()
    assert roles == {"S": "SOURCE", "K": "SINK", "M1": "MULE", "M2": "MULE", "L": "MULE",
                     "X": "MULE", "Y": "MULE", "N1": "NORMAL", "N2": "NORMAL"}


def test_is_mule_is_empty_for_sources_and_sinks():
    out = labels()
    assert out.loc["M1", "is_mule"] == 1
    assert out.loc["N2", "is_mule"] == 0
    assert pd.isna(out.loc["S", "is_mule"]) and pd.isna(out.loc["K", "is_mule"])


def test_mule_since_is_when_both_in_and_out_have_happened():
    out = labels()
    assert out.loc["M1", "mule_since"] == START + pd.Timedelta(hours=2)
    assert out.loc["L", "mule_since"] == START + pd.Timedelta(hours=90)
    assert pd.isna(out.loc["N1", "mule_since"]) and pd.isna(out.loc["S", "mule_since"])


def test_laundering_self_transfer_does_not_make_a_mule():
    assert labels().loc["N1", "role"] == "NORMAL"


def test_roles_as_of_never_use_the_future():
    out = build_account_labels(*make_inputs())
    early = labels_as_of(out, START + pd.Timedelta(hours=50)).set_index("account_id")
    assert early.loc["L", "role"] == "SINK"          # has not forwarded yet at hour 50
    assert pd.isna(early.loc["L", "is_mule"])
    assert early.loc["M1", "role"] == "MULE"
    before_anything = labels_as_of(out, START).set_index("account_id")
    assert set(before_anything["role"]) == {"NORMAL"}


def test_later_laundering_does_not_change_earlier_labels():
    transactions, labels_txn = make_inputs()
    cut = START + pd.Timedelta(hours=50)
    full = labels_as_of(build_account_labels(transactions, labels_txn), cut).set_index("account_id")
    early_rows = transactions["timestamp"] <= cut
    part = build_account_labels(transactions[early_rows], labels_txn[early_rows])
    part = labels_as_of(part, cut).set_index("account_id")
    shared = part.index
    assert full.loc[shared, "role"].tolist() == part["role"].tolist()


def test_rings_from_pattern_file_and_from_links():
    out = labels()
    assert out.loc[["S", "M1", "M2", "K"], "ring_id"].tolist() == ["P-0001"] * 4
    assert set(out.loc[["S", "M1", "M2", "K"], "ring_source"]) == {"PATTERN_FILE"}
    assert out.loc["X", "ring_id"] == out.loc["Y", "ring_id"] == "C-0001"
    assert out.loc["X", "ring_source"] == "CONNECTED"
    assert pd.isna(out.loc["L", "ring_id"]) and pd.isna(out.loc["N1", "ring_id"])


def test_split_boundaries_are_whole_hours_in_order():
    transactions, _ = make_inputs()
    split = compute_split(transactions, SHARES)
    assert split["data_start"] == START + pd.Timedelta(hours=1)
    assert split["test_end"] == START + pd.Timedelta(hours=100)
    assert split["data_start"] < split["train_end"] < split["validation_end"] < split["test_end"]
    assert all(value == value.floor("h") for value in split.values())


def test_mule_counts_per_period():
    transactions, labels_txn = make_inputs()
    splits = build_splits("test", transactions, build_account_labels(transactions, labels_txn), SHARES)
    # train ends at hour 60, validation at hour 80; L becomes a mule at hour 90 (new in test).
    assert splits["mule_counts"] == {"train": 4, "validation": 0, "test_new": 1, "test_known": 4}


def test_output_matches_contract():
    out = build_account_labels(*make_inputs())
    assert all(ok for ok, _ in check_dataframe(out, c.ACCOUNT_LABELS, "account_labels"))


def test_snapshots_every_24_hours_plus_period_ends():
    transactions, _ = make_inputs()
    split = compute_split(transactions, SHARES)
    times = snapshot_times(split, every_hours=24)
    assert times[0] == split["data_start"] + pd.Timedelta(hours=24)
    assert {split["train_end"], split["validation_end"], split["test_end"]} <= set(times)
    assert times == sorted(times)
