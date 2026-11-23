"""Build model rows: features + rule flags per snapshot, and their labels
(docs/05-FEATURES-MODEL.md sections 1.5 and 4).

This is training / evaluation code, so it may read labels. Labels are always taken
"as of" the end of the period they are used in (docs/02-DATA.md section 5.5).
"""

import json
from pathlib import Path

import pandas as pd

from trace_aml import contract as c
from trace_aml.config import path
from trace_aml.pipeline.features import cutoff_filename
from trace_aml.pipeline.labels import labels_as_of
from trace_aml.pipeline.rules import FLAG_COLUMNS
from trace_aml.pipeline.split import period_of, snapshot_times

NEW_MULE, KNOWN_MULE, NORMAL, EXCLUDED = "NEW_MULE", "KNOWN_MULE", "NORMAL", "EXCLUDED"


def load_split(dataset: str) -> dict[str, pd.Timestamp]:
    """Period boundaries from splits.json as timestamps."""
    raw = json.loads((path("processed") / dataset / c.SPLITS_FILENAME).read_text(encoding="utf-8"))
    return {key: pd.Timestamp(raw[key]) for key in ("data_start", "train_end", "validation_end", "test_end")}


def period_bounds(split: dict[str, pd.Timestamp], period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) of TRAIN, VALIDATION or TEST."""
    return {"TRAIN": (split["data_start"], split["train_end"]),
            "VALIDATION": (split["train_end"], split["validation_end"]),
            "TEST": (split["validation_end"], split["test_end"])}[period]


def load_snapshot(features_folder: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One snapshot: the feature file joined with its rule flags (rule_score left out)."""
    stem = cutoff_filename(cutoff).removesuffix(".parquet")
    features = pd.read_parquet(features_folder / f"{stem}.parquet")
    flags = pd.read_parquet(features_folder / f"{stem}_rules.parquet", columns=["account_id", *FLAG_COLUMNS])
    rows = features.merge(flags, on="account_id", how="left", validate="one_to_one")
    rows[FLAG_COLUMNS] = rows[FLAG_COLUMNS].fillna(0).astype("float32")
    return rows


def load_period_rows(dataset: str, period: str, split: dict[str, pd.Timestamp], every_hours: int) -> pd.DataFrame:
    """Every snapshot row of one period, stacked."""
    folder = path("features") / dataset
    cutoffs = [t for t in snapshot_times(split, every_hours) if period_of(t, split) == period]
    return pd.concat([load_snapshot(folder, cutoff) for cutoff in cutoffs], ignore_index=True)


def row_labels(rows: pd.DataFrame, account_labels: pd.DataFrame, as_of: pd.Timestamp,
               drop_known_before: pd.Timestamp | None = None) -> pd.Series:
    """Label for each snapshot row, using roles as of `as_of`.

    1 = mule that is already a mule at the row's cut-off; 0 = normal account;
    empty = row not used (source, sink, or a mule that is not one yet at the cut-off).
    With drop_known_before, mules that were already mules by that time are also left out.
    """
    roles = labels_as_of(account_labels, as_of).set_index("account_id")
    role = rows["account_id"].map(roles["role"])
    since = rows["account_id"].map(roles["mule_since"])
    label = pd.Series(pd.NA, index=rows.index, dtype="Int8")
    label[role == "NORMAL"] = 0
    label[(role == "MULE") & (since <= rows["cutoff"])] = 1
    if drop_known_before is not None:
        label[(role == "MULE") & (since <= drop_known_before)] = pd.NA
    return label


def account_groups(account_labels: pd.DataFrame, period_start: pd.Timestamp, period_end: pd.Timestamp
                   ) -> pd.Series:
    """account_id -> NEW_MULE / KNOWN_MULE / NORMAL / EXCLUDED for one evaluation period.

    Roles are as of the period's end. NEW_MULE = became a mule inside the period (the
    positives). KNOWN_MULE = already a mule before it. EXCLUDED = source or sink.
    """
    roles = labels_as_of(account_labels, period_end).set_index("account_id")
    group = pd.Series(EXCLUDED, index=roles.index, dtype="string")
    group[roles["role"] == "NORMAL"] = NORMAL
    is_mule = roles["role"] == "MULE"
    group[is_mule & (roles["mule_since"] > period_start)] = NEW_MULE
    group[is_mule & (roles["mule_since"] <= period_start)] = KNOWN_MULE
    return group


def best_snapshot_per_account(rows: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    """One row per account: the snapshot where its score was highest (ties: the earliest)."""
    ranked = rows.assign(score=score.to_numpy()).sort_values(["score", "cutoff"], ascending=[False, True],
                                                             kind="stable")
    return ranked.drop_duplicates("account_id").reset_index(drop=True)
