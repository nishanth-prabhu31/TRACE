"""Check that files match the TRACE data contract (docs/04-CONTRACT.md).

Usage:
    python scripts/check_contract.py <folder>

Walks the folder and its subfolders, works out what each folder is (processed data,
features, a run, a model, or a database) and checks every file it recognises.
Prints one line per check and exits with code 0 only if everything passes.

Types are checked by kind (whole number, decimal, text, date-time, true/false),
not by exact width: an int32 column where the contract says int64 still passes.
"""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_aml import contract as c  # noqa: E402

Result = tuple[bool, str]


def _kind_matches(series: pd.Series, dtype: str) -> bool:
    """True if the column's data kind fits the contract type."""
    if dtype.startswith("datetime"):
        return ptypes.is_datetime64_any_dtype(series) and getattr(series.dt, "tz", None) is None
    if dtype.lower().startswith("int"):
        return ptypes.is_integer_dtype(series) and not ptypes.is_bool_dtype(series)
    if dtype.startswith("float"):
        return ptypes.is_float_dtype(series)
    if dtype == "string":
        return ptypes.is_string_dtype(series)
    if dtype == "bool":
        return ptypes.is_bool_dtype(series)
    raise ValueError(f"Unknown contract type: {dtype}")


def check_dataframe(df: pd.DataFrame, spec: c.FileSpec, label: str) -> list[Result]:
    """Check one table against its FileSpec; return (passed, message) per check."""
    results: list[Result] = []
    expected = [col.name for col in spec.columns]
    actual = list(df.columns)

    missing = [name for name in expected if name not in actual]
    if missing:
        return [(False, f"{label}: missing columns {missing}")]

    if actual[: len(expected)] != expected:
        results.append((False, f"{label}: columns out of order, expected {expected} first"))
    else:
        results.append((True, f"{label}: all {len(expected)} contract columns present, in order"))

    extras = actual[len(expected):]
    if extras and not spec.extra_float_columns:
        results.append((False, f"{label}: unexpected extra columns {extras}"))
    elif extras:
        not_float = [name for name in extras if not ptypes.is_float_dtype(df[name])]
        results.append((not not_float, f"{label}: {len(extras)} feature columns"
                        + (f", not decimals: {not_float}" if not_float else ", all decimals")))

    for col in spec.columns:
        series = df[col.name]
        where = f"{label}.{col.name}"
        if not _kind_matches(series, col.dtype):
            results.append((False, f"{where}: type is {series.dtype}, contract says {col.dtype}"))
            continue
        if not col.nullable and series.isna().any():
            results.append((False, f"{where}: {int(series.isna().sum())} empty values, none allowed"))
        values = series.dropna()
        if col.allowed is not None:
            bad = sorted(set(values.astype(str)) - set(col.allowed))
            if bad:
                results.append((False, f"{where}: values not allowed {bad[:5]}"))
        if col.min_value is not None and (values < col.min_value).any():
            results.append((False, f"{where}: values below {col.min_value}"))
        if col.max_value is not None and (values > col.max_value).any():
            results.append((False, f"{where}: values above {col.max_value}"))

    duplicates = int(df.duplicated(subset=list(spec.key)).sum())
    results.append((duplicates == 0, f"{label}: key {spec.key} "
                    + ("unique" if duplicates == 0 else f"has {duplicates} duplicates")))

    if all(ok for ok, _ in results):
        results.append((True, f"{label}: types, empties, allowed values and ranges OK"))
    return results


def _check_file(path: Path, spec: c.FileSpec) -> list[Result]:
    try:
        df = pd.read_parquet(path)
    except Exception as error:  # unreadable file is a failed check, not a crash
        return [(False, f"{path.name}: cannot be read ({error})")]
    return check_dataframe(df, spec, path.name)


def check_processed(folder: Path) -> list[Result]:
    results = _check_file(folder / c.TRANSACTIONS.filename, c.TRANSACTIONS)
    for spec in (c.LABELS_TXN, c.ACCOUNT_LABELS):
        if (folder / spec.filename).exists():
            results += _check_file(folder / spec.filename, spec)
    return results


def check_features(folder: Path) -> list[Result]:
    results: list[Result] = []
    for path in sorted(folder.glob("*.parquet")):
        if path.name.endswith("_rules_evidence.parquet"):
            results += _check_file(path, c.RULE_EVIDENCE)
        elif path.name.endswith("_rules.parquet"):
            results += _check_file(path, c.RULE_FLAGS)
        else:
            results += _check_file(path, c.FEATURES)
    return results


def check_run(folder: Path) -> list[Result]:
    results: list[Result] = []
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    missing_keys = [key for key in c.MANIFEST_REQUIRED_KEYS if key not in manifest]
    results.append((not missing_keys, "manifest.json: "
                    + (f"missing keys {missing_keys}" if missing_keys else "all keys present")))
    version = manifest.get("contract_version")
    results.append((version == c.CONTRACT_VERSION,
                    f"manifest.json: contract_version {version} (current {c.CONTRACT_VERSION})"))
    kind_ok = manifest.get("kind") in c.RUN_KINDS
    split_ok = manifest.get("kind") != "SPLIT" or manifest.get("split") in c.SPLITS
    results.append((kind_ok and split_ok, f"manifest.json: kind={manifest.get('kind')} split={manifest.get('split')}"))

    tables: dict[str, pd.DataFrame] = {}
    for spec in c.RUN_REQUIRED + c.RUN_OPTIONAL:
        path = folder / spec.filename
        if not path.exists():
            if spec in c.RUN_REQUIRED:
                results.append((False, f"{spec.filename}: required file missing"))
            continue
        results += _check_file(path, spec)
        try:
            tables[spec.filename] = pd.read_parquet(path)
        except Exception:
            pass

    scores = tables.get(c.SCORES.filename)
    if scores is not None:
        scored = set(scores["account_id"])
        for name in (c.REASONS.filename, c.TRAILS.filename, c.RINGS.filename):
            if name in tables and "account_id" in tables[name]:
                orphans = set(tables[name]["account_id"]) - scored
                results.append((not orphans, f"{name}: every account_id is in scores"
                                + ("" if not orphans else f" (missing {len(orphans)})")))
        view = tables.get(c.TRANSACTIONS_VIEW.filename)
        trails = tables.get(c.TRAILS.filename)
        if view is not None and trails is not None and "txn_id" in trails:
            unknown = set(trails["txn_id"]) - set(view["txn_id"])
            results.append((not unknown, "trails.parquet: every txn_id is in transactions_view"
                            + ("" if not unknown else f" (missing {len(unknown)})")))
    return results


def check_model(folder: Path) -> list[Result]:
    missing = [name for name in c.MODEL_FILES if not (folder / name).exists()]
    return [(not missing, f"model {folder.name}: " + (f"missing {missing}" if missing else "all files present"))]


def check_database(path: Path) -> list[Result]:
    results: list[Result] = []
    with sqlite3.connect(path) as conn:
        for table, expected in c.SQLITE_TABLES.items():
            columns = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))
            if not columns:
                results.append((False, f"{path.name}: table {table} missing"))
            else:
                results.append((columns == expected, f"{path.name}: table {table} "
                                + ("matches" if columns == expected else f"columns {columns}")))
    return results


def check_folder(root: Path) -> list[Result]:
    """Check every recognised folder and database under root."""
    results: list[Result] = []
    for folder in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if (folder / "manifest.json").exists() and (folder / c.SCORES.filename).exists():
            results += check_run(folder)
        elif (folder / c.TRANSACTIONS.filename).exists():
            results += check_processed(folder)
        elif (folder / "model.txt").exists():
            results += check_model(folder)
        elif folder.parent.name == "features" or folder.name == "features":
            results += check_features(folder)
    for db in sorted(root.rglob("*.db")):
        results += check_database(db)
    return results


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"[FAIL] folder not found: {root}")
        return 1
    results = check_folder(root)
    if not results:
        print(f"[FAIL] nothing recognisable to check in {root}")
        return 1
    for ok, message in results:
        print(("[ OK ] " if ok else "[FAIL] ") + message)
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results) - failed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
