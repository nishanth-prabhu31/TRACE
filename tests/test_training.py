"""Tests for training rows, period assignment, evaluation targets and metrics."""

import numpy as np
import pandas as pd

from trace_aml.models.dataset import (evaluation_targets, label_training_rows, period_of,
                                      period_scores)
from trace_aml.models.metrics import evaluate, precision_recall_at_k

SPLITS = {"data_start": "2022-09-01", "train_end": "2022-09-07", "validation_end": "2022-09-09",
          "test_end": "2022-09-11"}


def test_snapshot_periods():
    assert period_of(pd.Timestamp("2022-09-07 00:00"), SPLITS) == "TRAIN"       # covers 6 Sept
    assert period_of(pd.Timestamp("2022-09-08 00:00"), SPLITS) == "VALIDATION"
    assert period_of(pd.Timestamp("2022-09-09 00:00"), SPLITS) == "VALIDATION"
    assert period_of(pd.Timestamp("2022-09-10 00:00"), SPLITS) == "TEST"


def rows():
    return pd.DataFrame({
        "account_id": ["M", "M", "LATE", "N", "S"],
        "cutoff": pd.to_datetime(["2022-09-03", "2022-09-05", "2022-09-03", "2022-09-03", "2022-09-03"]),
        "role": ["MULE", "MULE", "MULE", "NORMAL", "SOURCE"],
        "mule_since": pd.to_datetime(["2022-09-04", "2022-09-04", "2022-09-06", None, "2022-09-02"]),
        "period": "TRAIN",
    })


def test_training_labels():
    out = label_training_rows(rows())
    labelled = dict(zip(zip(out["account_id"], out["cutoff"].dt.day), out["y"]))
    assert labelled == {("M", 5): 1, ("N", 3): 0}  # M before 4 Sept dropped, LATE dropped, source dropped


def test_evaluation_targets_only_new_mules_and_normals():
    r = pd.DataFrame({
        "account_id": ["NEW", "OLD", "N", "SNK"], "period": "TEST",
        "role": ["MULE", "MULE", "NORMAL", "SINK"],
        "mule_since": pd.to_datetime(["2022-09-09 05:00", "2022-09-02 00:00", None, "2022-09-10 00:00"]),
    })
    t = evaluation_targets(r, "TEST", SPLITS)
    assert dict(t) == {"NEW": 1, "N": 0}


def test_period_score_is_highest_snapshot():
    r = pd.DataFrame({"account_id": ["A", "A", "B"], "period": ["TEST", "TEST", "TEST"]})
    s = period_scores(r, np.array([0.2, 0.9, 0.1]), "TEST")
    assert s["A"] == 0.9 and s["B"] == 0.1


def test_precision_recall_at_k():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    p, r = precision_recall_at_k(y, s, 2)
    assert p == 0.5 and r == 0.5


def test_perfect_ranking_scores_one():
    targets = pd.Series([1, 0, 0, 1], index=list("abcd"))
    scores = pd.Series([0.9, 0.1, 0.2, 0.8], index=list("abcd"))
    result = evaluate(targets, scores, [2], 0.25)
    assert result["pr_auc"] == 1.0 and result["precision@2"] == 1.0
