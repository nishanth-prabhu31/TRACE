"""F-11 demo builder: pattern picking and slicing."""

import pandas as pd

from trace_aml.pipeline.demo import candidate_patterns, slice_for

T = pd.Timestamp


def _data():
    txns = pd.DataFrame({
        "txn_id": [1, 2, 3, 4, 5],
        "timestamp": [T("2022-09-09 01:00"), T("2022-09-09 02:00"), T("2022-09-09 03:00"),
                      T("2022-09-08 01:00"), T("2022-09-09 04:00")],
        "sender_id": ["a", "b", "c", "x", "z"],
        "receiver_id": ["m", "m", "a", "y", "q"],
        "amount": [10.0, 20.0, 5.0, 1.0, 2.0],
    })
    labels = pd.DataFrame({
        "txn_id": [1, 2, 3, 4, 5],
        "is_laundering": [1, 1, 0, 1, 0],
        "pattern_type": ["FAN_IN", "FAN_IN", None, "CYCLE", None],
        "pattern_id": ["P-1", "P-1", None, "P-2", None],
    })
    return txns, labels


def test_only_patterns_fully_inside_window():
    txns, labels = _data()
    c = candidate_patterns(txns, labels, T("2022-09-09"), T("2022-09-11"))
    assert list(c["pattern_id"]) == ["P-1"]          # P-2 is before the window
    assert c.loc[0, "accounts"] == {"a", "b", "m"}


def test_slice_adds_direct_counterparties_only():
    txns, _ = _data()
    accounts, sl = slice_for(txns, {"a", "b", "m"})
    assert accounts == {"a", "b", "m", "c"}          # c sent to a; z/q untouched
    assert set(sl["txn_id"]) == {1, 2, 3}
