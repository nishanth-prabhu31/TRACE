"""Tests for P3 labels on a hand-made mini network with known answers.

Network (L = laundering):
  S -L-> M1 -L-> M2 -L-> K        S, M1, M2 in pattern P-0001 (M2 -> K has no pattern)
  X -L-> Y -L-> Z                 no pattern: should become connected ring C-0001
  N1 -> N2                        normal
  N3 -L-> N3                      laundering self-transfer: must stay NORMAL
  M1 -> N1 on 10 Sept             a known mule still active in the test period
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_contract import check_dataframe  # noqa: E402
from trace_aml import contract as c  # noqa: E402
from trace_aml.pipeline.labels import (  # noqa: E402
    account_roles, assign_rings, mule_counts, time_split, to_contract,
)

ROWS = [  # sender, receiver, time, laundering, pattern_id
    ("S", "M1", "2022-09-01 01:00", 1, "P-0001"),
    ("M1", "M2", "2022-09-02 01:00", 1, "P-0001"),
    ("M2", "K", "2022-09-02 02:00", 1, None),
    ("N1", "N2", "2022-09-03 10:00", 0, None),
    ("N3", "N3", "2022-09-04 10:00", 1, None),
    ("X", "Y", "2022-09-09 09:00", 1, None),
    ("Y", "Z", "2022-09-09 10:00", 1, None),
    ("M1", "N1", "2022-09-10 22:30", 0, None),
]


def data():
    df = pd.DataFrame(ROWS, columns=["sender_id", "receiver_id", "timestamp", "is_laundering", "pattern_id"])
    df["txn_id"] = range(len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_self"] = df["sender_id"] == df["receiver_id"]
    transactions = df[["txn_id", "timestamp", "sender_id", "receiver_id", "is_self"]]
    labels = df[["txn_id", "is_laundering", "pattern_id"]].assign(pattern_type=None)
    return transactions, labels


def labelled():
    transactions, labels = data()
    roles = assign_rings(account_roles(transactions, labels), transactions, labels)
    return to_contract(roles).set_index("account_id"), transactions


def test_roles():
    out, _ = labelled()
    assert out.loc["M1", "role"] == "MULE"
    assert out.loc["M2", "role"] == "MULE"
    assert out.loc["Y", "role"] == "MULE"
    assert out.loc["S", "role"] == "SOURCE"
    assert out.loc["X", "role"] == "SOURCE"
    assert out.loc["K", "role"] == "SINK"
    assert out.loc["Z", "role"] == "SINK"
    assert out.loc["N1", "role"] == "NORMAL"


def test_laundering_self_transfer_stays_normal():
    out, _ = labelled()
    assert out.loc["N3", "role"] == "NORMAL"


def test_is_mule_and_mule_since():
    out, _ = labelled()
    assert out.loc["M1", "is_mule"] == 1
    assert out.loc["N1", "is_mule"] == 0
    assert pd.isna(out.loc["S", "is_mule"])  # sources and sinks are set aside
    assert out.loc["M1", "mule_since"] == pd.Timestamp("2022-09-01 01:00")
    assert pd.isna(out.loc["N1", "mule_since"])


def test_rings():
    out, _ = labelled()
    for account in ("S", "M1", "M2"):
        assert out.loc[account, "ring_id"] == "P-0001"
        assert out.loc[account, "ring_source"] == "PATTERN_FILE"
    for account in ("X", "Y", "Z"):
        assert out.loc[account, "ring_id"] == "C-0001"
        assert out.loc[account, "ring_source"] == "CONNECTED"
    assert pd.isna(out.loc["K", "ring_id"])  # only linked to an account that already has a ring
    assert pd.isna(out.loc["N1", "ring_id"])


def test_time_split_whole_hours():
    _, transactions = labelled()
    cuts = time_split(transactions, {"train": 0.6, "validation": 0.2, "test": 0.2})
    assert cuts["data_start"] == pd.Timestamp("2022-09-01 01:00")
    assert cuts["test_end"] == pd.Timestamp("2022-09-10 23:00")
    assert cuts["train_end"] < cuts["validation_end"] < cuts["test_end"]


def test_mule_counts():
    out, transactions = labelled()
    cuts = {"train_end": pd.Timestamp("2022-09-07"), "validation_end": pd.Timestamp("2022-09-09")}
    counts = mule_counts(out.reset_index(), transactions, cuts)
    assert counts["train"] == 2         # M1, M2
    assert counts["validation"] == 0
    assert counts["test_new"] == 1      # Y
    assert counts["test_known"] == 1    # M1 is still active on 10 Sept
    assert counts["test_new_share"] == 0.5


def test_matches_contract():
    out, _ = labelled()
    results = check_dataframe(out.reset_index(), c.ACCOUNT_LABELS, "account_labels")
    assert all(ok for ok, _ in results), [m for ok, m in results if not ok]
