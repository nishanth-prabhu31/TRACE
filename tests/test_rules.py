"""Tests for P5 rules: each rule fires on its own pattern, never on a normal account."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_contract import check_dataframe  # noqa: E402
from trace_aml import contract as c  # noqa: E402
from trace_aml.config import load_settings  # noqa: E402
from trace_aml.pipeline.rules import compute_rules  # noqa: E402

CUTOFF = pd.Timestamp("2022-09-03 00:00")
D = "2022-09-02 "
ROWS = [
    # pass-through: PT receives 2 payments and forwards 95% within 20 min
    ("P1", "PT", D + "10:00", 100), ("P2", "PT", D + "10:10", 100), ("PT", "P3", D + "10:30", 190),
    # fan-in: 5 senders to FI within 2 h, 90% forwarded
    *[(f"S{i}", "FI", D + f"12:{i * 10:02d}", 100) for i in range(5)], ("FI", "O1", D + "13:30", 450),
    # fan-out: FO receives 500 then pays 5 receivers
    ("I1", "FO", D + "14:00", 500),
    *[("FO", f"R{i}", D + f"14:{10 + i * 10}", 90) for i in range(5)],
    # chain: C0 -> C1 -> C2 -> C3, each hop within 1 h keeping >= 70%
    ("C0", "C1", D + "18:00", 100), ("C1", "C2", D + "18:30", 90), ("C2", "C3", D + "19:00", 80),
    # cycle: Y1 -> Y2 -> Y3 -> Y1
    ("Y1", "Y2", D + "20:00", 100), ("Y2", "Y3", D + "20:30", 90), ("Y3", "Y1", D + "21:00", 85),
    # normal
    ("N1", "N2", D + "09:00", 50),
]


def run():
    df = pd.DataFrame(ROWS, columns=["sender_id", "receiver_id", "timestamp", "amount"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = df["amount"].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["txn_id"] = range(len(df))
    df["amount_original"] = df["amount"]
    df["channel"] = "ACH"
    df["is_self"] = False
    flags, evidence = compute_rules(df, CUTOFF, load_settings())
    return flags.set_index("account_id"), evidence


def test_each_rule_fires_on_its_pattern():
    flags, _ = run()
    assert flags.loc["PT", "rule_pass_through"] == 1
    assert flags.loc["FI", "rule_fan_in"] == 1
    assert flags.loc["FO", "rule_fan_out"] == 1
    assert flags.loc["C1", "rule_chain"] == 1
    assert flags.loc["Y1", "rule_cycle"] == 1


def test_normal_account_fires_nothing():
    flags, _ = run()
    cols = ["rule_chain", "rule_fan_in", "rule_fan_out", "rule_cycle", "rule_pass_through"]
    assert flags.loc[["N1", "N2"], cols].sum().sum() == 0
    assert flags.loc["N1", "rule_score"] == 0


def test_evidence_points_at_real_transactions():
    flags, evidence = run()
    fi = evidence[(evidence["account_id"] == "FI") & (evidence["rule"] == "FAN_IN")]
    assert len(fi) == 5  # the five incoming payments


def test_outputs_match_contract():
    flags, evidence = run()
    for df, spec in ((flags.reset_index(), c.RULE_FLAGS), (evidence, c.RULE_EVIDENCE)):
        results = check_dataframe(df, spec, spec.filename)
        assert all(ok for ok, _ in results), [m for ok, m in results if not ok]
