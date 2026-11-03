"""Tests for P1 ingest on a tiny hand-made IBM-style file."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_contract import check_dataframe  # noqa: E402
from trace_aml import contract as c  # noqa: E402
from trace_aml.pipeline.ingest import clean_ibm, read_ibm_patterns, read_ibm_transactions  # noqa: E402

HEADER = ("Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
          "Amount Paid,Payment Currency,Payment Format,Is Laundering\n")
ROWS = [
    "2022/09/01 00:20,010,AAA,010,AAA,50.00,US Dollar,50.00,US Dollar,Reinvestment,0",  # self-transfer
    "2022/09/01 00:10,010,AAA,020,BBB,85.34,Euro,85.34,Euro,ACH,1",                     # euro, laundering
    "2022/09/01 00:30,020,BBB,030,CCC,100.00,US Dollar,100.00,US Dollar,Wire,1",
    "2022/09/01 00:30,020,BBB,030,CCC,100.00,US Dollar,100.00,US Dollar,Wire,1",        # exact duplicate
    "2022/09/01 00:40,030,CCC,040,DDD,0.00,US Dollar,0.00,US Dollar,Cash,0",            # zero amount
    "2022/09/12 01:00,040,DDD,050,EEE,10.00,US Dollar,10.00,US Dollar,ACH,1",           # in the tail
]
PATTERNS = """BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out
2022/09/01 00:10,010,AAA,020,BBB,85.34,Euro,85.34,Euro,ACH,1
END LAUNDERING ATTEMPT - FAN-OUT

BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops
2022/09/01 00:30,020,BBB,030,CCC,100.00,US Dollar,100.00,US Dollar,Wire,1
END LAUNDERING ATTEMPT - CYCLE
"""
RATES = {"US Dollar": 1.0, "Euro": 0.8534}


def run(tmp_path: Path):
    csv = tmp_path / "t.csv"
    csv.write_text(HEADER + "\n".join(ROWS) + "\n", encoding="utf-8")
    pat = tmp_path / "p.txt"
    pat.write_text(PATTERNS, encoding="utf-8")
    return clean_ibm(read_ibm_transactions(csv), read_ibm_patterns(pat), "test", RATES,
                     drop_after="2022-09-10 23:59:59")


def test_drops_bad_rows(tmp_path):
    txns, _, counts = run(tmp_path)
    assert counts["dropped_exact_duplicates"] == 1
    assert counts["dropped_non_positive_amount"] == 1
    assert counts["dropped_after_cutoff"] == 1
    assert len(txns) == 3


def test_ids_keep_leading_zeros_and_join_bank_and_account(tmp_path):
    txns, _, _ = run(tmp_path)
    assert "010_AAA" in set(txns["sender_id"])


def test_sorted_by_time_and_numbered(tmp_path):
    txns, _, _ = run(tmp_path)
    assert txns["timestamp"].is_monotonic_increasing
    assert list(txns["txn_id"]) == [0, 1, 2]


def test_amount_converted_to_base_currency(tmp_path):
    txns, _, _ = run(tmp_path)
    euro = txns[txns["currency_original"] == "Euro"].iloc[0]
    assert abs(euro["amount"] - 100.0) < 0.01  # 85.34 euro / 0.8534 = 100 USD


def test_self_transfer_flagged(tmp_path):
    txns, _, _ = run(tmp_path)
    assert txns["is_self"].sum() == 1


def test_patterns_attached(tmp_path):
    txns, labels, counts = run(tmp_path)
    assert counts["pattern_rows_matched"] == 2
    merged = txns.merge(labels, on="txn_id")
    assert set(merged["pattern_type"].dropna()) == {"FAN_OUT", "CYCLE"}
    assert set(merged["pattern_id"].dropna()) == {"P-0001", "P-0002"}


def test_outputs_match_contract(tmp_path):
    txns, labels, _ = run(tmp_path)
    assert all(ok for ok, _ in check_dataframe(txns, c.TRANSACTIONS, "t"))
    assert all(ok for ok, _ in check_dataframe(labels, c.LABELS_TXN, "l"))


def test_unknown_currency_raises(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text(HEADER + "2022/09/01 00:10,010,AAA,020,BBB,5,Zloty,5,Zloty,ACH,0\n", encoding="utf-8")
    try:
        clean_ibm(read_ibm_transactions(csv), None, "test", RATES, None)
    except ValueError as error:
        assert "Zloty" in str(error)
    else:
        raise AssertionError("expected a ValueError for an unknown currency")


def test_empty_pattern_columns_without_patterns_file(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text(HEADER + ROWS[1] + "\n", encoding="utf-8")
    _, labels, _ = clean_ibm(read_ibm_transactions(csv), None, "test", RATES, None)
    assert labels["pattern_type"].isna().all()
    assert isinstance(labels, pd.DataFrame)
