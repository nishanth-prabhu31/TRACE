"""Rule tests: each rule fires on a hand-made pattern and not on a normal account."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_contract import check_dataframe  # noqa: E402
from tests.test_features import CUTOFF, SETTINGS, make_transactions  # noqa: E402
from trace_aml import contract as c  # noqa: E402
from trace_aml.pipeline.rules import FLAG_COLUMNS, compute_rules  # noqa: E402

# (minute, sender, receiver, amount). Minutes count from 2022-09-01 00:00; cut-off is 24 h later.
NORMAL = [(100, "N1", "N2", 50.0), (900, "N2", "N3", 5.0)]
PASS_THROUGH = [(200, "P1", "PT", 100.0), (205, "P2", "PT", 100.0), (230, "PT", "P3", 195.0)]
FAN_IN = [*[(300 + i, f"FS{i}", "FI", 10.0) for i in range(5)], (380, "FI", "FZ", 45.0)]
FAN_OUT = [(400, "GA", "FO", 60.0), *[(410 + i, "FO", f"GR{i}", 10.0) for i in range(5)]]
CHAIN = [(500, "C1", "C2", 100.0), (520, "C2", "C3", 90.0), (540, "C3", "C4", 80.0)]
CYCLE = [(600, "Y1", "Y2", 30.0), (700, "Y2", "Y3", 30.0), (800, "Y3", "Y1", 30.0)]
ALL_ROWS = [*NORMAL, *PASS_THROUGH, *FAN_IN, *FAN_OUT, *CHAIN, *CYCLE]


def run(rows) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags, evidence = compute_rules(make_transactions(rows), CUTOFF, SETTINGS)
    return flags.set_index("account_id"), evidence


def test_outputs_match_the_contract():
    flags, evidence = compute_rules(make_transactions(ALL_ROWS), CUTOFF, SETTINGS)
    assert all(ok for ok, _ in check_dataframe(flags, c.RULE_FLAGS, "flags"))
    assert all(ok for ok, _ in check_dataframe(evidence, c.RULE_EVIDENCE, "evidence"))


def test_normal_accounts_fire_nothing():
    flags, _ = run(ALL_ROWS)
    assert flags.loc[["N1", "N2", "N3"], FLAG_COLUMNS].to_numpy().sum() == 0
    assert flags.loc["N1", "rule_score"] == 0.0


def test_pass_through_fires_and_names_its_transfers():
    flags, evidence = run(ALL_ROWS)
    assert flags.loc["PT", "rule_pass_through"] == 1
    assert flags.loc["P1", "rule_pass_through"] == 0
    found = evidence[(evidence["account_id"] == "PT") & (evidence["rule"] == "PASS_THROUGH")]
    assert len(found) == 3                                   # two arrivals and the forward


def test_pass_through_needs_speed_and_most_of_the_money():
    slow, _ = run([(200, "P1", "PT", 100.0), (205, "P2", "PT", 100.0), (400, "PT", "P3", 195.0)])
    assert slow.loc["PT", "rule_pass_through"] == 0          # forwarded after more than an hour
    partial, _ = run([(200, "P1", "PT", 100.0), (205, "P2", "PT", 100.0), (230, "PT", "P3", 100.0)])
    assert partial.loc["PT", "rule_pass_through"] == 0       # only half forwarded
    single, _ = run([(200, "P1", "PT", 100.0), (230, "PT", "P3", 100.0)])
    assert single.loc["PT", "rule_pass_through"] == 0        # only one arrival


def test_fan_in_fires_only_when_the_money_is_forwarded():
    flags, evidence = run(ALL_ROWS)
    assert flags.loc["FI", "rule_fan_in"] == 1
    assert len(evidence[(evidence["account_id"] == "FI") & (evidence["rule"] == "FAN_IN")]) == 6
    kept, _ = run(FAN_IN[:-1])                               # a merchant: collects and keeps
    assert kept.loc["FI", "rule_fan_in"] == 0
    few, _ = run([*FAN_IN[:4], FAN_IN[-1]])                  # only four senders
    assert few.loc["FI", "rule_fan_in"] == 0


def test_fan_out_fires_only_when_funded_just_before():
    flags, _ = run(ALL_ROWS)
    assert flags.loc["FO", "rule_fan_out"] == 1
    unfunded, _ = run(FAN_OUT[1:])                           # pays five accounts from its own money
    assert unfunded.loc["FO", "rule_fan_out"] == 0
    early, _ = run([(100, "GA", "FO", 60.0), *FAN_OUT[1:]])  # funded five hours earlier
    assert early.loc["FO", "rule_fan_out"] == 0


def test_chain_fires_for_the_middle_accounts_only():
    flags, evidence = run(ALL_ROWS)
    assert flags.loc["C2", "rule_chain"] == 1 and flags.loc["C3", "rule_chain"] == 1
    assert flags.loc["C1", "rule_chain"] == 0 and flags.loc["C4", "rule_chain"] == 0
    assert len(evidence[(evidence["account_id"] == "C2") & (evidence["rule"] == "CHAIN")]) == 3
    two_hops, _ = run(CHAIN[:2])
    assert two_hops.loc["C2", "rule_chain"] == 0
    leaky, _ = run([CHAIN[0], CHAIN[1], (540, "C3", "C4", 10.0)])   # last hop keeps too little
    assert leaky.loc["C2", "rule_chain"] == 0


def test_cycle_fires_for_the_account_whose_money_came_back():
    flags, evidence = run(ALL_ROWS)
    assert flags.loc[["Y1", "Y2", "Y3"], "rule_cycle"].tolist() == [1, 0, 0]   # only Y1's money came back in time order
    assert len(evidence[(evidence["account_id"] == "Y1") & (evidence["rule"] == "CYCLE")]) == 3
    open_path, _ = run(CYCLE[:2])
    assert open_path.loc["Y1", "rule_cycle"] == 0


def test_rules_ignore_transfers_older_than_the_window():
    late = CUTOFF + pd.Timedelta(hours=30)
    flags, _ = compute_rules(make_transactions([*ALL_ROWS, (50 * 60, "N1", "N2", 1.0)]), late, SETTINGS)
    assert set(flags["account_id"]) == {"N1", "N2"}
    assert flags[FLAG_COLUMNS].to_numpy().sum() == 0


def test_rule_score_is_the_share_of_rules_fired():
    flags, _ = run(ALL_ROWS)
    assert abs(flags.loc["PT", "rule_score"] - flags.loc["PT", FLAG_COLUMNS].sum() / 5) < 1e-6
