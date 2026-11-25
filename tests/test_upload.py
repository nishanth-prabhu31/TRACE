"""Upload format checks (docs/04-CONTRACT.md section 3)."""

import pandas as pd

from trace_aml import contract as c
from trace_aml.pipeline.upload import snapshot_times, validate_upload

SETTINGS = {"currency": {"base": "USD", "display": "INR", "display_units_per_base": 96.0,
                         "units_per_base": {"US Dollar": 1.0, "Euro": 0.85}}}


def frame(**columns):
    return pd.DataFrame(columns, dtype="string")


def good(n=3):
    return frame(timestamp=[f"2026-10-14 10:0{i}:00" for i in range(n)], sender_id=[f"A{i}" for i in range(n)],
                 receiver_id=[f"B{i}" for i in range(n)], amount=["9600"] * n)


def test_valid_file_becomes_trace_transactions():
    out, errors, _ = validate_upload(good(), SETTINGS)
    assert errors == [] and list(out.columns) == [col.name for col in c.TRANSACTIONS.columns]
    assert out["amount"].tolist() == [100.0] * 3                       # INR assumed, 9600 / 96
    assert (out["channel"] == "UNKNOWN").all() and out["txn_id"].tolist() == [0, 1, 2]


def test_missing_column_is_named():
    _, errors, _ = validate_upload(good().drop(columns="amount"), SETTINGS)
    assert "amount" in errors[0]


def test_errors_name_column_and_line_numbers():
    raw = good(7)
    raw.loc[[1, 4], "amount"] = ["-5", "abc"]
    raw.loc[2, "timestamp"] = "yesterday"
    raw.loc[3, "sender_id"] = " "
    _, errors, _ = validate_upload(raw, SETTINGS)
    text = " | ".join(errors)
    assert "amount: must be a number above 0 at line 3, 6" in text      # line 1 is the header
    assert "timestamp" in text and "line 4" in text
    assert "sender_id: empty at line 5" in text


def test_currency_codes_and_unknown_currency():
    raw = good().assign(currency=["usd", "", "Euro"])
    out, errors, _ = validate_upload(raw, SETTINGS)
    assert errors == [] and out["amount"].round(2).tolist() == [9600.0, 100.0, round(9600 / 0.85, 2)]
    _, errors, _ = validate_upload(good().assign(currency=["XYZ", "INR", "INR"]), SETTINGS)
    assert "currency" in errors[0] and "line 2" in errors[0]


def test_extra_columns_ignored_with_warning():
    out, errors, warnings = validate_upload(good().assign(is_laundering=["1", "0", "1"]), SETTINGS)
    assert errors == [] and "is_laundering" not in out.columns and any("Ignored" in w for w in warnings)


def test_snapshots_are_daily_plus_the_end():
    tx = pd.DataFrame({"timestamp": pd.to_datetime(["2022-09-01 05:00", "2022-09-03 12:00"])})
    assert snapshot_times(tx) == [pd.Timestamp("2022-09-02"), pd.Timestamp("2022-09-03"), pd.Timestamp("2022-09-03 12:00")]
