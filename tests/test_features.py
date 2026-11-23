"""Feature tests on hand-made mini networks with known answers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_contract import check_dataframe  # noqa: E402
from trace_aml import contract as c  # noqa: E402
from trace_aml.config import load_settings  # noqa: E402
from trace_aml.features.catalog import BY_NAME, CATALOG, FEATURE_SETS, value_text  # noqa: E402
from trace_aml.pipeline.features import compute_features, feature_columns  # noqa: E402

START = pd.Timestamp("2022-09-01 00:00")
CUTOFF = START + pd.Timedelta(hours=24)
SETTINGS = load_settings()

# (minute, sender, receiver, amount)
#   A -> X -> C -> D -> A   a fast chain that closes into a 4-hop cycle
#   N1 <-> N2               slow ordinary transfers; N1 also has a self-transfer
#   S1..S6 -> H -> R1..R6   H collects from six senders in an hour, then pays six receivers
ROWS = [
    (1, "A", "X", 100.0), (10, "X", "C", 100.0), (20, "C", "D", 90.0), (30, "D", "A", 80.0),
    (40, "N1", "N2", 50.0), (41, "N1", "N1", 5.0), (500, "N2", "N1", 20.0),
    *[(600 + i, f"S{i}", "H", 10.0) for i in range(1, 7)],
    *[(700 + i, "H", f"R{i}", 10.0) for i in range(1, 7)],
]


def make_transactions(rows=ROWS) -> pd.DataFrame:
    """Rows in file order; channels alternate ACH, Wire."""
    df = pd.DataFrame(rows, columns=["minute", "sender_id", "receiver_id", "amount"])
    df = df.sort_values("minute", kind="stable").reset_index(drop=True)
    out = pd.DataFrame({
        "txn_id": pd.Series(range(len(df)), dtype="int64"),
        "timestamp": (START + pd.to_timedelta(df["minute"], unit="m")).astype("datetime64[ns]"),
        "sender_id": df["sender_id"].astype("string"),
        "receiver_id": df["receiver_id"].astype("string"),
        "amount": df["amount"].astype("float64"),
        "amount_original": df["amount"].astype("float64"),
        "channel": pd.Series([("ACH", "Wire")[i % 2] for i in range(len(df))], dtype="string"),
    })
    out["is_self"] = out["sender_id"] == out["receiver_id"]
    return out


def features_for(rows) -> pd.DataFrame:
    return compute_features(make_transactions(rows), CUTOFF, SETTINGS).set_index("account_id")


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return features_for(ROWS)


def test_output_matches_the_contract_and_the_catalogue(features):
    table = features.reset_index()
    assert all(ok for ok, _ in check_dataframe(table, c.FEATURES, "features"))
    assert list(table.columns[2:]) == feature_columns(SETTINGS)
    assert set(table.columns[2:]) <= set(BY_NAME)


def test_pure_pass_through_account(features):
    assert features.loc["X", "pass_through_1h"] == 1.0
    assert features.loc["X", "holding_median_min"] == 9.0
    assert features.loc["X", "net_flow_ratio"] == 0.0


def test_fifo_matches_oldest_money_first():
    out = features_for([(0, "P", "M", 100.0), (60, "Q", "M", 100.0), (90, "M", "Z", 150.0)])
    # 100 from P was held 90 min, then 50 of Q's money was held 30 min.
    assert out.loc["M", "pass_through_1h"] == pytest.approx(50 / 200)
    assert out.loc["M", "pass_through_24h"] == pytest.approx(150 / 200)
    assert out.loc["M", "holding_median_min"] == 60.0            # median of 90 and 30


def test_money_arriving_and_leaving_in_the_same_minute_still_matches():
    out = features_for([(5, "M", "Z", 100.0), (5, "P", "M", 100.0)])   # file order has the send first
    assert out.loc["M", "pass_through_1h"] == 1.0
    assert out.loc["M", "holding_median_min"] == 0.0


def test_account_that_never_forwards_has_no_holding_time(features):
    assert features.loc["R1", "pass_through_1h"] == 0.0
    assert np.isnan(features.loc["R1", "holding_median_min"])
    assert np.isnan(features.loc["S1", "pass_through_1h"])       # never received anything


def test_degrees_and_fan_bursts(features):
    assert features.loc["H", "in_degree"] == 6 and features.loc["H", "out_degree"] == 6
    assert features.loc["H", "counterparties"] == 12
    assert features.loc["H", "fan_in_max_1h"] == 6 and features.loc["H", "fan_out_max_1h"] == 6
    assert features.loc["X", "fan_in_max_1h"] == 1


def test_fan_burst_only_counts_senders_inside_one_hour():
    out = features_for([(0, "S1", "H", 1.0), (30, "S2", "H", 1.0), (100, "S3", "H", 1.0), (110, "S1", "H", 1.0)])
    assert out.loc["H", "fan_in_max_1h"] == 2
    assert out.loc["H", "in_degree"] == 3


def test_chains_and_cycles(features):
    assert features.loc["X", "chain_count"] == 1                 # A -> X -> C -> D
    assert features.loc["C", "chain_count"] == 1                 # X -> C -> D -> A
    assert features.loc["D", "chain_count"] == 0                 # nothing leaves A after D pays it
    assert features.loc["A", "cycle_count"] == 1                 # A -> X -> C -> D -> A
    assert features.loc["X", "cycle_count"] == 0                 # A paid X before the money came back
    assert features.loc["N1", "cycle_count"] == 1                # N1 -> N2 -> N1
    assert features.loc["H", "chain_count"] == 0 and features.loc["H", "cycle_count"] == 0


def test_chain_needs_the_amount_to_be_kept():
    out = features_for([(0, "A", "X", 100.0), (10, "X", "C", 100.0), (20, "C", "D", 10.0)])   # last hop keeps 10%
    assert out.loc["X", "chain_count"] == 0


def test_window_counts_and_self_transfers(features):
    assert features.loc["H", "in_cnt_24h"] == 6 and features.loc["H", "out_cnt_24h"] == 6
    assert features.loc["H", "in_cnt_1h"] == 0                   # all more than an hour before the cut-off
    assert features.loc["N1", "self_txn_share"] == pytest.approx(1 / 3)
    assert features.loc["N1", "in_cnt_24h"] == 1                 # the self-transfer is not counted


def test_burstiness_is_minus_one_for_evenly_spaced_activity():
    out = features_for([(i * 60, "E", f"T{i}", 1.0) for i in range(5)])
    assert out.loc["E", "burstiness"] == -1.0
    assert np.isnan(out.loc["T1", "burstiness"])                 # fewer than 3 transactions


def test_relative_amounts(features):
    assert features.loc["X", "max_in_vs_own_median"] == 1.0
    assert features.loc["X", "max_in_pctl"] == 100.0             # 100 is the largest amount in the data
    assert features.loc["H", "round_amount_share"] == 0.0
    assert features.loc["X", "round_amount_share"] == 1.0
    assert features.loc["H", "channel_entropy"] == 1.0           # half ACH, half Wire


def test_changing_the_currency_changes_no_feature():
    transactions = make_transactions()
    rupees = transactions.assign(amount=transactions["amount"] * 96.0)
    first = compute_features(transactions, CUTOFF, SETTINGS)
    second = compute_features(rupees, CUTOFF, SETTINGS)
    pd.testing.assert_frame_equal(first, second, check_exact=False, rtol=1e-5)


def test_only_recently_active_accounts_are_scored_unless_asked():
    late = START + pd.Timedelta(hours=40)
    extra = [*ROWS, (39 * 60, "N1", "N2", 5.0)]
    scored = compute_features(make_transactions(extra), late, SETTINGS)
    assert set(scored["account_id"]) == {"N1", "N2"}
    everyone = compute_features(make_transactions(extra), late, SETTINGS, score_all=True)
    assert len(everyone) == 19


def test_catalogue_is_complete_and_sets_are_consistent():
    names = [feature.name for feature in CATALOG]
    assert len(names) == len(set(names))
    assert set(FEATURE_SETS["TXN_ONLY"]) <= set(FEATURE_SETS["TEMPORAL"]) | set(FEATURE_SETS["RELATIVE"])
    assert set(FEATURE_SETS["FULL"]) == set(names)
    assert value_text("pass_through_1h", 0.93) == "forwarded 93% of inflow within 1 h"
    assert value_text("holding_median_min", float("nan")) == "not available"
    for feature in CATALOG:                                      # every template can be filled
        assert value_text(feature.name, 1.0)
