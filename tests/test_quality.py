"""Tests for the P2 quality gate building blocks."""

import numpy as np
import pandas as pd

from trace_aml.pipeline.quality import (
    best_single_threshold_f1, c3_time_spread, single_column_pr_auc,
    status_high_is_bad, status_low_is_bad,
)

RNG = np.random.default_rng(0)


def test_status_helpers():
    assert status_high_is_bad(0.1, warn=0.3, fail=0.8) == "PASS"
    assert status_high_is_bad(0.5, warn=0.3, fail=0.8) == "WARN"
    assert status_high_is_bad(0.9, warn=0.3, fail=0.8) == "FAIL"
    assert status_low_is_bad(200, warn=100, fail=30) == "PASS"
    assert status_low_is_bad(50, warn=100, fail=30) == "WARN"
    assert status_low_is_bad(10, warn=100, fail=30) == "FAIL"


def labels(n=4000, rate=0.05):
    return pd.Series((RNG.random(n) < rate).astype(int))


def test_giveaway_column_is_caught():
    y = labels()
    leaky = pd.Series(np.where(y == 1, "SPECIAL", "ACH"))  # channel that reveals the answer
    half = len(y) // 2
    score = single_column_pr_auc(leaky[:half], y[:half], leaky[half:], y[half:], seed=0)
    assert score > 0.95


def test_random_column_is_not_a_giveaway():
    y = labels()
    noise = pd.Series(RNG.normal(size=len(y)))
    half = len(y) // 2
    score = single_column_pr_auc(noise[:half], y[:half], noise[half:], y[half:], seed=0)
    assert score < 0.2


def test_single_threshold_f1_finds_perfect_split():
    y = labels()
    perfect = pd.Series(np.where(y == 1, 100.0, 1.0))
    assert best_single_threshold_f1(perfect, y) > 0.99
    assert best_single_threshold_f1(-perfect, y) > 0.99  # works in either direction


def test_c3_uses_lowest_new_mule_count():
    limits = {"warn": 30, "fail": 10}
    counts = {"train": 520, "validation": 210, "test_new": 65, "test_known": 602, "test_new_share": 0.097}
    result = c3_time_spread(counts, limits)
    assert result["status"] == "PASS" and result["value"] == 65
    counts["test_new"] = 20
    assert c3_time_spread(counts, limits)["status"] == "WARN"
