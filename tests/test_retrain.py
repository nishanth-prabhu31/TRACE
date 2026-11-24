"""F-14 retrain: feedback labels, overrides and the evaluation-safe rule."""

import pandas as pd

from trace_aml.pipeline.retrain import LABEL, feedback_summary, with_feedback


def test_feedback_overrides_existing_rows_and_adds_new_ones():
    data = {"train": pd.DataFrame({"account_id": ["A", "A", "B"], "f": [1.0, 2.0, 3.0]}),
            "target": pd.Series([0, 0, 1], dtype="int8")}
    rows = pd.DataFrame({"account_id": ["A", "C"], "f": [9.0, 8.0]})
    out = with_feedback(data, rows, pd.Series([1, 0], dtype="int8"))
    assert out["target"].tolist() == [1, 1, 1, 1, 0]          # A's old rows take the feedback label
    assert out["train"]["account_id"].tolist() == ["A", "A", "B", "A", "C"]
    assert data["target"].tolist() == [0, 0, 1]               # input left unchanged


def test_summary_marks_demo_only_when_any_decision_is_outside_train_or_validation():
    fb = pd.DataFrame({"account_id": ["A", "B", "C"],
                       "decision": ["CONFIRMED_MULE", "FALSE_POSITIVE", "NEEDS_INVESTIGATION"],
                       "eval_safe": [True, False, True]})
    fb["label"] = fb["decision"].map(LABEL)
    s = feedback_summary(fb)
    assert (s["confirmed"], s["false_positive"], s["ignored"]) == (1, 1, 1)
    assert not s["eval_safe"]
    assert feedback_summary(fb.iloc[[0, 2]])["eval_safe"]
