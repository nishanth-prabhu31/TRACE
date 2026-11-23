"""Training and evaluation rows (docs/05-FEATURES-MODEL.md sections 1.5 and 4).

Training: one row per (account, daily snapshot) in the train period.
Validation / test: one row per (account, snapshot) too; the account's period score is its
highest score over the period's snapshots (see `period_scores`).
"""

import json
from pathlib import Path

import pandas as pd

from trace_aml.config import path
from trace_aml.features.catalog import BY_NAME

RULE_FLAG_COLUMNS = ["rule_chain", "rule_fan_in", "rule_fan_out", "rule_cycle", "rule_pass_through"]


def load_snapshot(folder: Path, cutoff_file: str) -> pd.DataFrame:
    """Features + rule flags for one snapshot."""
    features = pd.read_parquet(folder / cutoff_file)
    rules = pd.read_parquet(folder / cutoff_file.replace(".parquet", "_rules.parquet"),
                            columns=["account_id", *RULE_FLAG_COLUMNS, "rule_score"])
    return features.merge(rules, on="account_id", how="left")


def period_of(cutoff: pd.Timestamp, splits: dict) -> str:
    """A snapshot at cutoff T covers the day before T; it belongs to the period containing T - 1 s."""
    moment = cutoff - pd.Timedelta(seconds=1)
    if moment < pd.Timestamp(splits["train_end"]):
        return "TRAIN"
    if moment < pd.Timestamp(splits["validation_end"]):
        return "VALIDATION"
    return "TEST"


def load_rows(dataset: str) -> tuple[pd.DataFrame, dict]:
    """All snapshot rows with their period and labels attached."""
    processed = path("processed") / dataset
    folder = path("features") / dataset
    splits = json.loads((processed / "splits.json").read_text(encoding="utf-8"))
    labels = pd.read_parquet(processed / "account_labels.parquet", columns=["account_id", "role", "mule_since"])

    frames = []
    for file in sorted(folder.glob("[0-9]*T[0-9][0-9][0-9][0-9].parquet")):
        snap = load_snapshot(folder, file.name)
        snap["period"] = period_of(snap["cutoff"].iloc[0], splits)
        frames.append(snap)
    rows = pd.concat(frames, ignore_index=True).merge(labels, on="account_id", how="left")
    return rows, splits


def label_training_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Section 4 labels: mule already acting -> 1, normal -> 0, everything else dropped."""
    mule_now = (rows["role"] == "MULE") & (rows["mule_since"] <= rows["cutoff"])
    keep = mule_now | (rows["role"] == "NORMAL")
    out = rows[keep].copy()
    out["y"] = mule_now[keep].astype(int)
    return out


def evaluation_targets(rows: pd.DataFrame, period: str, splits: dict) -> pd.Series:
    """Per account in a period: 1 = new mule (first laundering inside the period), 0 = normal.
    Known mules, sources and sinks are left out (section 7)."""
    bounds = {"VALIDATION": (splits["train_end"], splits["validation_end"]),
              "TEST": (splits["validation_end"], splits["test_end"])}[period]
    start, end = (pd.Timestamp(b) for b in bounds)
    accounts = rows.loc[rows["period"] == period, ["account_id", "role", "mule_since"]].drop_duplicates("account_id")
    new_mule = (accounts["role"] == "MULE") & (accounts["mule_since"] >= start) & (accounts["mule_since"] < end)
    keep = new_mule | (accounts["role"] == "NORMAL")
    return pd.Series(new_mule[keep].astype(int).to_numpy(), index=accounts.loc[keep, "account_id"].to_numpy())


def period_scores(rows: pd.DataFrame, scores, period: str) -> pd.Series:
    """Highest score per account over the period's snapshots."""
    mask = (rows["period"] == period).to_numpy()
    return pd.Series(scores[mask], index=rows.loc[mask, "account_id"].to_numpy()).groupby(level=0).max()


def feature_columns(rows: pd.DataFrame, family_set: str) -> list[str]:
    """Columns for a feature set from section 2.6 (FULL, TXN_ONLY, or FULL minus a family)."""
    base = [c for c in rows.columns if c in BY_NAME]
    if family_set == "FULL":
        return base + RULE_FLAG_COLUMNS
    if family_set == "TXN_ONLY":
        # own transactions only: no counterparty, graph or in-to-out matching information
        return [c for c in base if BY_NAME[c].family in ("TEMPORAL", "RELATIVE")
                and c != "new_counterparties_24h_ratio"]
    if family_set.startswith("FULL-"):
        drop = family_set.split("-", 1)[1]
        if drop == "RULES":
            return base
        return [c for c in base if BY_NAME[c].family != drop] + RULE_FLAG_COLUMNS
    raise ValueError(family_set)
