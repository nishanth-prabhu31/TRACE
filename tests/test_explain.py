"""Tests for calibration, verdict cut-offs, money trails, labels and summaries."""

import numpy as np
import pandas as pd

from tests.test_features import CUTOFF, START, make_transactions
from trace_aml.explain.summary import held_text, short_labels, summary_text
from trace_aml.explain.trails import build_trails
from trace_aml.models.calibrate import choose_cuts, fit_calibrator, to_risk, verdict_for

TRAIL_SETTINGS = {"max_hops": 3, "max_gap_hours": 24, "branch": 2, "max_trails": 3}
VERDICTS = {"likely_mule_min_precision": 0.5, "watch_min_recall": 0.8, "watch_fallback_gap": 10}

# (minute, sender, receiver, amount): S -> A -> X -> C -> D, plus small side transfers at X.
CHAIN = [(100, "S", "A", 500.0), (130, "A", "X", 480.0), (150, "X", "C", 470.0), (170, "C", "D", 450.0),
         (120, "Q", "X", 5.0), (160, "X", "R", 3.0)]


def trails_for(rows, account="X") -> pd.DataFrame:
    accounts = pd.DataFrame({"account_id": [account], "cutoff": [CUTOFF]})
    return build_trails(make_transactions(rows), accounts, TRAIL_SETTINGS)


def test_best_trail_follows_the_big_money_through_the_account():
    trails = trails_for(CHAIN)
    best = trails[trails["trail_id"] == 1].sort_values("hop")
    assert best["from_id"].tolist() == ["S", "A", "X", "C"]
    assert best["to_id"].tolist() == ["A", "X", "C", "D"]
    assert best["side"].tolist() == ["BEFORE", "BEFORE", "AFTER", "AFTER"]
    assert best["timestamp"].is_monotonic_increasing
    assert trails["trail_id"].max() <= 3


def test_trails_are_ranked_by_the_money_that_went_all_the_way():
    trails = trails_for(CHAIN)
    smallest = trails.groupby("trail_id")["amount"].min()
    assert smallest.is_monotonic_decreasing
    assert smallest.loc[1] == 450.0


def test_no_trail_without_money_both_in_and_out():
    assert trails_for(CHAIN, account="S").empty               # S only sends
    assert trails_for(CHAIN, account="D").empty               # D only receives
    assert trails_for(CHAIN, account="NOBODY").empty


def test_trail_never_goes_back_in_time():
    rows = [(200, "A", "X", 100.0), (100, "X", "C", 100.0)]   # X paid C before A paid X
    assert trails_for(rows).empty


def test_trail_uses_nothing_after_the_cutoff():
    after = int((CUTOFF - START).total_seconds() // 60) + 10
    rows = [(100, "A", "X", 100.0), (after, "X", "C", 100.0)]
    assert trails_for(rows).empty


def test_calibrated_risk_follows_the_observed_mule_rate():
    score = np.array([0.1] * 90 + [0.9] * 10)
    target = np.array([0] * 90 + [1] * 8 + [0] * 2)
    prob, risk = to_risk(fit_calibrator(target, score), np.array([0.1, 0.9]))
    assert prob.tolist() == [0.0, 0.8] and risk.tolist() == [0.0, 80.0]


def test_cut_offs_and_verdicts():
    risk = np.array([90.0] * 4 + [40.0] * 10 + [5.0] * 86)
    target = np.array([1, 1, 1, 0] + [1] * 3 + [0] * 7 + [1] + [0] * 85)
    cuts = choose_cuts(target, risk, VERDICTS)
    assert cuts["likely_mule_cut"] == 90.0 and cuts["precision_target_met"]   # 3 of 4; adding the 40s gives 6 of 14
    assert cuts["watch_cut"] == 40.0                                          # 6 of 7 mules are at 40 or above
    assert verdict_for(np.array([95.0, 40.0, 5.0]), cuts).tolist() == ["LIKELY_MULE", "WATCH", "NORMAL"]


def test_watch_cut_respects_analyst_capacity():
    risk = np.array([90.0] * 2 + [40.0] * 6 + [10.0] * 20 + [1.0] * 72)
    target = np.array([1, 1] + [1] + [0] * 5 + [1] * 4 + [0] * 16 + [0] * 72)
    loose = choose_cuts(target, risk, VERDICTS)
    assert loose["likely_mule_cut"] == 90.0 and loose["watch_cut"] == 10.0    # 80% recall needs the 10s: 28 flagged
    capped = choose_cuts(target, risk, {**VERDICTS, "watch_max_share": 0.08})
    assert capped["watch_cut"] == 40.0                                        # at most 8 of 100 flagged


def test_cut_off_falls_back_to_best_f1_when_precision_is_never_reached():
    risk = np.array([30.0] * 10 + [1.0] * 90)
    target = np.array([1, 1] + [0] * 8 + [0] * 90)
    cuts = choose_cuts(target, risk, VERDICTS)
    assert not cuts["precision_target_met"] and cuts["likely_mule_cut"] == 30.0
    assert cuts["watch_cut"] == 20.0                                          # likely cut minus the gap


def test_short_label_comes_from_the_strongest_upward_reason():
    verdict = pd.Series({"A": "LIKELY_MULE", "B": "WATCH", "C": "NORMAL", "D": "WATCH"})
    reasons = pd.DataFrame({
        "account_id": ["A", "A", "B", "D"], "position": [1, 2, 1, 1],
        "feature": ["age_hours", "pass_through_1h", "cycle_count", "clustering"],
        "direction": ["DOWN", "UP", "UP", "UP"]})
    assert short_labels(verdict, reasons).to_dict() == {
        "A": "Pass-through", "B": "Layering hop", "C": "Normal activity", "D": "Unusual activity"}


def test_summary_is_filled_from_the_accounts_numbers():
    activity = pd.Series({"in_amount_24h": 1000.0, "n_senders": 2, "out_amount_24h": 950.0, "n_receivers": 1})
    features = pd.Series({"pass_through_24h": 0.93, "holding_median_min": 25.0})
    trail = trails_for(CHAIN)
    text = summary_text("LIKELY_MULE", activity, features, trail, rate=100.0)
    assert "received ₹1 L from 2 account(s)" in text and "sent ₹95,000 to 1 account(s)" in text
    assert "forwarded 93%" in text and "25 minutes" in text
    assert "4-hop chain from S to D" in text
    assert summary_text("NORMAL", activity, features, trail, rate=100.0) == "No unusual pattern found."
    assert held_text(210) == "3.5 hours"
