"""Reasons: which features pushed an account's score up or down (docs/05-FEATURES-MODEL.md 6.3).

SHAP values are computed on the model's raw output (log-odds). Calibration only
stretches the score without changing its order, so the direction of each reason holds
for the calibrated risk too.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from trace_aml.features.catalog import BY_NAME, value_text


def shap_values(booster: lgb.Booster, features: pd.DataFrame) -> np.ndarray:
    """SHAP value of every feature for every row (rows x features)."""
    values = shap.TreeExplainer(booster).shap_values(features)
    return np.asarray(values[1] if isinstance(values, list) else values)


def top_reasons(booster: lgb.Booster, rows: pd.DataFrame, columns: list[str], top: int) -> pd.DataFrame:
    """reasons.parquet rows: the `top` largest-impact features for each account in `rows`."""
    if rows.empty:
        impacts = np.empty((0, len(columns)))
    else:
        impacts = shap_values(booster, rows[columns])
    records = []
    for position_in_rows, account_id in enumerate(rows["account_id"]):
        strongest = np.argsort(-np.abs(impacts[position_in_rows]), kind="stable")[:top]
        for position, column_index in enumerate(strongest, start=1):
            name = columns[column_index]
            value = float(rows.iloc[position_in_rows][name])
            impact = float(impacts[position_in_rows, column_index])
            records.append((account_id, position, name, BY_NAME[name].display_name, value,
                            value_text(name, value), impact, "UP" if impact > 0 else "DOWN"))
    out = pd.DataFrame(records, columns=["account_id", "position", "feature", "display_name", "value",
                                         "value_text", "impact", "direction"])
    return out.astype({"account_id": "string", "position": "int8", "feature": "string",
                       "display_name": "string", "value": "float32", "value_text": "string",
                       "impact": "float32", "direction": "string"})
