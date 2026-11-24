"""F-13 rings: label-free grouping of flagged accounts."""

import pandas as pd

from trace_aml import contract as c
from trace_aml.pipeline.rings import build_rings, evaluate_rings

T = pd.Timestamp
SNAP = T("2022-09-10 00:00")


def _inputs():
    scores = pd.DataFrame({
        "account_id": ["A", "B", "C", "D", "E", "N"],
        "risk": [90.0, 50.0, 10.0, 80.0, 70.0, 0.1],
        "verdict": ["LIKELY_MULE", "LIKELY_MULE", "WATCH", "LIKELY_MULE", "LIKELY_MULE", "NORMAL"],
    })
    # Trail of A goes A -> N -> C: links A and C (N is not flagged).
    trails = pd.DataFrame({
        "account_id": ["A", "A"], "trail_id": [1, 1], "hop": [1, 2], "txn_id": [10, 11],
        "from_id": ["A", "N"], "to_id": ["N", "C"],
        "timestamp": [T("2022-09-09 10:00"), T("2022-09-09 11:00")], "amount": [100.0, 95.0], "side": ["AFTER"] * 2,
    })
    view = pd.DataFrame({
        "txn_id": [10, 11, 12, 13],
        "timestamp": [T("2022-09-09 10:00"), T("2022-09-09 11:00"), T("2022-09-09 12:00"), T("2022-09-01 12:00")],
        "sender_id": ["A", "N", "B", "D"], "receiver_id": ["N", "C", "A", "E"],
        "amount": [100.0, 95.0, 40.0, 30.0],
    })
    snapshots = pd.Series(SNAP, index=scores["account_id"])
    return scores, trails, view, snapshots


def test_trail_and_recent_direct_money_make_one_ring():
    rings = build_rings(*_inputs(), within_hours=24)
    assert set(rings["account_id"]) == {"A", "B", "C"}        # D -> E was 9 days earlier: no link
    assert rings["ring_id"].unique().tolist() == ["R-0001"]
    row = rings.iloc[0]
    assert row["ring_size"] == 3 and row["ring_amount"] == 235.0 and row["max_risk"] == 90.0
    assert [col.name for col in c.RINGS.columns] == rings.columns.tolist()


def test_wider_window_links_old_money_and_numbers_by_risk():
    rings = build_rings(*_inputs(), within_hours=24 * 30)
    first = rings.groupby("ring_id")["max_risk"].first()
    assert first.tolist() == [90.0, 80.0]                     # R-0001 holds A (90), R-0002 holds D (80)


def test_evaluation_counts_purity_and_coverage():
    rings = build_rings(*_inputs(), within_hours=24)
    labels = pd.DataFrame({"account_id": ["A", "B", "C"], "role": ["MULE", "NORMAL", "SINK"],
                           "ring_id": ["P-1", None, "P-1"]})
    result = evaluate_rings(rings, labels)
    assert result["purity"] == round(2 / 3, 4) and result["coverage_mean"] == 1.0
