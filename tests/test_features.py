"""Tests for P4 features: known answers on a mini network, the cut-off rule, and the contract."""

import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_contract import check_dataframe  # noqa: E402
from trace_aml import contract as c  # noqa: E402
from trace_aml.config import load_settings  # noqa: E402
from trace_aml.features.catalog import BY_NAME, describe  # noqa: E402
from trace_aml.pipeline.features import compute_features  # noqa: E402

CUTOFF = pd.Timestamp("2022-09-03 00:00")
ROWS = [  # sender, receiver, time, amount, channel
    ("B", "A", "2022-09-01 10:00", 100.0, "ACH"),     # A's history starts here
    ("C", "A", "2022-09-02 23:10", 100.0, "ACH"),     # inside 1 h before cut-off
    ("E", "A", "2022-09-02 23:30", 1000.0, "ACH"),    # big inflow, inside 1 h
    ("A", "D", "2022-09-02 23:50", 1150.0, "ACH"),    # forwards it all
    ("A", "A", "2022-09-02 12:00", 5.0, "ACH"),       # self-transfer
    ("F", "G", "2022-09-02 03:00", 50.0, "Wire"),     # night transaction, other accounts
]


def transactions(extra=()):
    df = pd.DataFrame(list(ROWS) + list(extra),
                      columns=["sender_id", "receiver_id", "timestamp", "amount", "channel"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df["txn_id"] = range(len(df))
    df["amount_original"] = df["amount"]
    df["is_self"] = df["sender_id"] == df["receiver_id"]
    return df


def features(extra=()):
    return compute_features(transactions(extra), CUTOFF, load_settings()).set_index("account_id")


def test_window_counts():
    f = features()
    assert f.loc["A", "in_cnt_1h"] == 2   # C and E
    assert f.loc["A", "in_cnt_24h"] == 2  # B was 37 h ago
    assert f.loc["A", "out_cnt_1h"] == 1
    assert f.loc["A", "in_cnt_6h"] == 2


def test_age_and_burstiness():
    f = features()
    assert f.loc["A", "age_hours"] == 38  # first transaction 2022-09-01 10:00
    assert pd.notna(f.loc["A", "burstiness"])  # 4 non-self transactions
    assert pd.isna(f.loc["D", "burstiness"])   # only 1 transaction


def test_night_share():
    f = features()
    assert f.loc["F", "night_share"] == 1.0
    assert f.loc["A", "night_share"] == 0.0


def test_relative_amounts():
    f = features()
    # A's amounts: 100, 100, 1000, 1150 -> median 550; largest inflow 1000
    assert abs(f.loc["A", "max_in_vs_own_median"] - 1000 / 550) < 1e-4
    assert f.loc["A", "channel_entropy"] == 0.0  # only ACH
    assert 0 < f.loc["A", "self_txn_share"] < 1
    assert f.loc["A", "round_amount_share"] == 0.75  # 100, 100, 1000 are multiples of 100; 1150 is not


def test_structural_basics():
    f = features()
    assert f.loc["A", "in_degree"] == 3   # B, C, E
    assert f.loc["A", "out_degree"] == 1  # D
    assert f.loc["A", "counterparties"] == 4
    assert f.loc["A", "pagerank"] > f.loc["C", "pagerank"]  # A receives from many
    assert f.loc["A", "nbr_new_share"] == 0.75  # C, E, D first seen in the last 24 h; B is older
    assert 0 <= f.loc["A", "community_internal_flow"] <= 1


def test_fifo_pass_through_and_holding():
    f = features()
    # A receives 100 (Sep 1 10:00), 100 (23:10), 1000 (23:30), sends 1150 at 23:50.
    # Oldest first: 100 held 2,270 min (~38 h), 100 held 40 min, 950 held 20 min.
    assert abs(f.loc["A", "pass_through_1h"] - 1050 / 1200) < 1e-4
    assert abs(f.loc["A", "pass_through_24h"] - 1050 / 1200) < 1e-4  # the 38 h piece is too slow for 24 h too
    assert f.loc["A", "holding_median_min"] == 40
    assert abs(f.loc["A", "net_flow_ratio"] - (1200 - 1150) / 2350) < 1e-4
    assert f.loc["A", "fan_in_max_1h"] == 2  # C and E in the 23:00 hour


def test_chains_and_cycles():
    from trace_aml.features.flow import chains_and_cycles
    rows = [("P", "Q", "2022-09-02 10:00", 100.0),   # chain P -> Q -> R -> S
            ("Q", "R", "2022-09-02 10:30", 95.0),
            ("R", "S", "2022-09-02 11:00", 90.0),
            ("U", "V", "2022-09-02 12:00", 50.0),    # round trip U -> V -> U
            ("V", "U", "2022-09-02 13:00", 49.0)]
    df = pd.DataFrame(rows, columns=["sender_id", "receiver_id", "timestamp", "amount"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["txn_id"] = range(len(df))
    df["is_self"] = False
    out = chains_and_cycles(df, CUTOFF, window_hours=48, max_gap_hours=24, min_keep_share=0.5)
    assert out.loc["Q", "chain_count"] == 1  # Q is the middle of P -> Q -> R -> S
    assert out.loc["V", "cycle_count"] == 1


def test_inactive_account_not_scored():
    f = features()
    assert "B" not in f.index  # B's only transaction was 37 h before the cut-off


def test_cutoff_rule_no_future_leakage():
    """A huge transaction 1 second after the cut-off must change nothing."""
    future = [("Z", "A", "2022-09-03 00:00:01", 9_999_999.0, "Wire"),
              ("A", "Z", "2022-09-03 00:00:02", 9_999_999.0, "Wire")]
    pdt.assert_frame_equal(features(), features(future))


def test_every_feature_is_in_catalog():
    f = features()
    missing = [col for col in f.columns if col != "cutoff" and col not in BY_NAME]
    assert not missing, missing
    assert describe("night_share", 0.5) == "50% at night"


def test_matches_contract():
    table = compute_features(transactions(), CUTOFF, load_settings())
    results = check_dataframe(table, c.FEATURES, "features")
    assert all(ok for ok, _ in results), [m for ok, m in results if not ok]
