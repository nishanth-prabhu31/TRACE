"""Leakage guards (docs/03-ARCHITECTURE.md section 9)."""

import ast
from pathlib import Path

import pandas as pd

from tests.test_features import CUTOFF, ROWS, SETTINGS, START, make_transactions
from trace_aml.pipeline.features import compute_features

ROOT = Path(__file__).resolve().parents[1]
FEATURE_CODE = [*sorted((ROOT / "trace_aml" / "features").glob("*.py")),
                ROOT / "trace_aml" / "pipeline" / "features.py",
                ROOT / "trace_aml" / "pipeline" / "split.py"]
FORBIDDEN_WORDS = ("labels_txn", "account_labels", "is_laundering", "is_mule", "mule_since",
                   "LABELS_TXN", "ACCOUNT_LABELS")


def test_feature_code_never_imports_label_code():
    for file in FEATURE_CODE:
        tree = ast.parse(file.read_text(encoding="utf-8"))
        imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not [name for name in imported if "labels" in name or "quality" in name], file.name


def test_feature_code_never_mentions_label_files_or_columns():
    for file in FEATURE_CODE:
        text = file.read_text(encoding="utf-8")
        assert not [word for word in FORBIDDEN_WORDS if word in text], file.name


def test_a_transaction_after_the_cutoff_changes_nothing():
    before = compute_features(make_transactions(), CUTOFF, SETTINGS)
    minutes_after = int((CUTOFF - START).total_seconds() // 60) + 1
    planted = [*ROWS,
               (minutes_after, "X", "A", 1e12),          # extreme amount one minute after the cut-off
               (minutes_after, "NEW", "H", 1e12),
               (minutes_after + 5, "H", "X", 1e12)]
    after = compute_features(make_transactions(planted), CUTOFF, SETTINGS)
    pd.testing.assert_frame_equal(before, after)


def test_a_transaction_exactly_at_the_cutoff_is_used():
    at_cutoff = int((CUTOFF - START).total_seconds() // 60)
    planted = [*ROWS, (at_cutoff, "X", "A", 5.0)]
    before = compute_features(make_transactions(), CUTOFF, SETTINGS).set_index("account_id")
    after = compute_features(make_transactions(planted), CUTOFF, SETTINGS).set_index("account_id")
    assert after.loc["X", "out_cnt_1h"] == before.loc["X", "out_cnt_1h"] + 1
