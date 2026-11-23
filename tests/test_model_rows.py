"""Tests for training-row labels, evaluation groups and metrics."""

import numpy as np
import pandas as pd

from trace_aml.models import metrics
from trace_aml.models.rows import (
    EXCLUDED, KNOWN_MULE, NEW_MULE, NORMAL, account_groups, best_snapshot_per_account, row_labels,
)

DAY = pd.Timedelta(days=1)
START = pd.Timestamp("2022-09-01")
NAT = pd.NaT

# first laundering received / sent, in days from START
ACCOUNT_LABELS = pd.DataFrame({
    "account_id": pd.Series(["EARLY", "LATE", "SINK", "SOURCE", "CLEAN", "SLOW"], dtype="string"),
    "first_laundering_in": [START + 1 * DAY, START + 8 * DAY, START + 2 * DAY, NAT, NAT, START + 3 * DAY],
    "first_laundering_out": [START + 2 * DAY, START + 8.5 * DAY, NAT, START + 1 * DAY, NAT, START + 9 * DAY],
})
TRAIN_END, VALIDATION_END, TEST_END = START + 6 * DAY, START + 8 * DAY, START + 10 * DAY


def rows_at(day: float) -> pd.DataFrame:
    return pd.DataFrame({"account_id": ACCOUNT_LABELS["account_id"], "cutoff": START + day * DAY})


def labels_at(day: float, **kwargs) -> dict:
    rows = rows_at(day)
    return dict(zip(rows["account_id"], row_labels(rows, ACCOUNT_LABELS, TRAIN_END, **kwargs).tolist()))


def test_training_labels_follow_the_as_of_rule():
    labels = labels_at(4)
    assert labels["EARLY"] == 1                       # a mule since day 2
    assert labels["CLEAN"] == 0
    assert labels["LATE"] == 0                        # not known to be anything at the end of train
    assert pd.isna(labels["SINK"]) and pd.isna(labels["SOURCE"])
    assert pd.isna(labels["SLOW"])                    # received dirty money by day 3: a sink so far


def test_a_mule_is_not_a_positive_before_it_became_one():
    assert pd.isna(labels_at(1.5)["EARLY"])           # row dropped, not labelled 0
    assert labels_at(2)["EARLY"] == 1


def test_known_mules_can_be_left_out_of_validation_rows():
    rows = rows_at(7)
    kept = row_labels(rows, ACCOUNT_LABELS, VALIDATION_END, drop_known_before=TRAIN_END)
    assert pd.isna(dict(zip(rows["account_id"], kept.tolist()))["EARLY"])


def test_evaluation_groups_for_the_test_period():
    groups = account_groups(ACCOUNT_LABELS.assign(), VALIDATION_END, TEST_END).to_dict()
    assert groups == {"EARLY": KNOWN_MULE, "LATE": NEW_MULE, "SLOW": NEW_MULE, "SINK": EXCLUDED,
                      "SOURCE": EXCLUDED, "CLEAN": NORMAL}


def test_account_score_is_its_highest_snapshot():
    rows = pd.concat([rows_at(9), rows_at(10)], ignore_index=True)
    score = pd.Series([0.1] * 6 + [0.9, 0.05, 0.1, 0.1, 0.1, 0.1])
    best = best_snapshot_per_account(rows, score).set_index("account_id")
    assert best.loc["EARLY", "score"] == 0.9 and best.loc["EARLY", "cutoff"] == START + 10 * DAY
    assert best.loc["LATE", "score"] == 0.1 and best.loc["LATE", "cutoff"] == START + 9 * DAY
    assert len(best) == 6


def test_precision_and_recall_at_k():
    target = np.array([1, 0, 1, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert metrics.precision_at_k(target, score, 2) == 0.5
    assert metrics.recall_at_k(target, score, 3) == 2 / 3
    assert metrics.precision_at_k(target, score, 100) == 3 / 8      # k larger than the list


def test_metrics_at_a_cut_and_top_share():
    target = np.array([1, 0, 1, 0, 0, 0, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    cut = metrics.at_cut(target, score, 0.7)
    assert cut["precision"] == 2 / 3 and cut["recall"] == 2 / 3 and cut["flagged"] == 3
    assert metrics.at_cut(target, score, 2.0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "flagged": 0}
    assert metrics.fpr_at_top_share(target, score, 0.25) == 1 / 5   # top 2 holds one normal account


def test_main_metrics_and_seed_summary():
    target = np.array([1, 0, 1, 0, 0, 0, 0, 0])
    perfect = metrics.main_metrics(target, np.array([9, 1, 8, 1, 1, 1, 1, 1.0]), [2], 0.25)
    assert perfect["pr_auc"] == 1.0 and perfect["precision_at_2"] == 1.0 and perfect["n_mules"] == 2
    summary = metrics.mean_and_sd([{"pr_auc": 0.2}, {"pr_auc": 0.4}])
    assert abs(summary["pr_auc"]["mean"] - 0.3) < 1e-9 and abs(summary["pr_auc"]["sd"] - 0.1) < 1e-9
