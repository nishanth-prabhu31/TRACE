"""Tests for the contract checker: a correct file passes, broken files fail."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_contract import check_dataframe, check_folder  # noqa: E402
from trace_aml import contract as c  # noqa: E402


def good_transactions() -> pd.DataFrame:
    return pd.DataFrame({
        "txn_id": pd.Series([0, 1], dtype="int64"),
        "timestamp": pd.to_datetime(["2022-09-01 00:20", "2022-09-01 00:25"]),
        "sender_id": pd.Series(["010_A", "020_B"], dtype="string"),
        "receiver_id": pd.Series(["020_B", "030_C"], dtype="string"),
        "amount": [100.0, 95.0],
        "amount_original": [100.0, 95.0],
        "currency_original": pd.Series(["US Dollar", "US Dollar"], dtype="string"),
        "channel": pd.Series(["ACH", "Wire"], dtype="string"),
        "is_self": [False, False],
        "dataset": pd.Series(["test", "test"], dtype="string"),
    })


def passed(results) -> bool:
    return all(ok for ok, _ in results)


def test_correct_file_passes():
    assert passed(check_dataframe(good_transactions(), c.TRANSACTIONS, "t"))


def test_missing_column_fails():
    df = good_transactions().drop(columns=["amount"])
    assert not passed(check_dataframe(df, c.TRANSACTIONS, "t"))


def test_negative_amount_fails():
    df = good_transactions()
    df.loc[0, "amount"] = -5.0
    assert not passed(check_dataframe(df, c.TRANSACTIONS, "t"))


def test_duplicate_key_fails():
    df = good_transactions()
    df.loc[1, "txn_id"] = 0
    assert not passed(check_dataframe(df, c.TRANSACTIONS, "t"))


def test_wrong_enum_value_fails():
    labels = pd.DataFrame({
        "account_id": pd.Series(["A"], dtype="string"),
        "role": pd.Series(["Mule"], dtype="string"),  # must be "MULE"
        "is_mule": pd.Series([1], dtype="Int8"),
        "mule_since": pd.to_datetime(["2022-09-01"]),
        "ring_id": pd.Series([None], dtype="string"),
        "ring_source": pd.Series([None], dtype="string"),
    })
    assert not passed(check_dataframe(labels, c.ACCOUNT_LABELS, "a"))


def test_folder_check_finds_processed_files(tmp_path):
    good_transactions().to_parquet(tmp_path / c.TRANSACTIONS.filename)
    results = check_folder(tmp_path)
    assert results and passed(results)
