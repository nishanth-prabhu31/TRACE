"""P1 Ingest: load a raw dataset and write it in TRACE's common format (F-01).

Writes data/processed/<dataset>/transactions.parquet and labels_txn.parquet,
following the cleaning steps in docs/02-DATA.md section 7.

Usage:
    python -m trace_aml.pipeline.ingest --dataset ibm_hi_small
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from trace_aml.config import PROJECT_ROOT, load_settings, path
from trace_aml.pipeline.common import file_info, now_utc, write_manifest

# Raw IBM columns, in file order. The second "Account" is the receiver; pandas would
# rename it "Account.1", so we give every column our own name while reading.
IBM_COLUMNS = [
    "ts", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "currency_received", "amount_paid", "currency_paid",
    "payment_format", "is_laundering",
]
IBM_DTYPES = {
    "ts": "string",
    "from_bank": "string",  # text, not number: bank "010" must keep its leading zero
    "from_account": "string",
    "to_bank": "string",
    "to_account": "string",
    "amount_received": "float64",
    "currency_received": "string",
    "amount_paid": "float64",
    "currency_paid": "string",
    "payment_format": "string",
    "is_laundering": "int8",
}
IBM_TIME_FORMAT = "%Y/%m/%d %H:%M"

# Columns that identify one transaction, used to match pattern-file rows to the main file.
MATCH_KEY = ["ts", "from_bank", "from_account", "to_bank", "to_account",
             "amount_paid", "currency_paid", "payment_format"]

BEGIN_RE = re.compile(r"^BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")


def read_ibm_transactions(csv_path: Path) -> pd.DataFrame:
    """Read the IBM transactions CSV with explicit types."""
    return pd.read_csv(csv_path, header=0, names=IBM_COLUMNS, dtype=IBM_DTYPES)


def read_ibm_patterns(patterns_path: Path) -> pd.DataFrame:
    """Parse the IBM patterns file into one row per pattern transaction.

    The file is blocks of: a BEGIN line naming the pattern, transaction rows in the
    same 11-column format, then an END line. Attempts have no ID, so they are
    numbered in file order (P-0001, P-0002, ...).
    """
    rows: list[list[str]] = []
    pattern_type, attempt = None, 0
    with open(patterns_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            begin = BEGIN_RE.match(line)
            if begin:
                attempt += 1
                pattern_type = begin.group(1).replace("-", "_")
            elif line.startswith("END LAUNDERING ATTEMPT"):
                pattern_type = None
            elif pattern_type is not None:
                rows.append(line.split(",") + [pattern_type, f"P-{attempt:04d}"])
    patterns = pd.DataFrame(rows, columns=IBM_COLUMNS + ["pattern_type", "pattern_id"])
    patterns["amount_paid"] = patterns["amount_paid"].astype("float64")
    for col in MATCH_KEY:
        if col != "amount_paid":
            patterns[col] = patterns[col].astype("string")
    return patterns[MATCH_KEY + ["pattern_type", "pattern_id"]]


def clean_ibm(raw: pd.DataFrame, patterns: pd.DataFrame | None, dataset: str,
              units_per_base: dict[str, float], drop_after: str | None
              ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply docs/02-DATA.md section 7 to raw IBM rows.

    Returns (transactions, labels, counts), with transactions and labels in the
    contract format (docs/04-CONTRACT.md section 2).
    """
    counts: dict = {"rows_read": len(raw)}
    df = raw.copy()

    # Step 2: parse timestamps; drop rows that fail.
    df["timestamp"] = pd.to_datetime(df["ts"], format=IBM_TIME_FORMAT, errors="coerce")
    bad_time = df["timestamp"].isna()
    counts["dropped_bad_timestamp"] = int(bad_time.sum())
    df = df[~bad_time]

    # Step 3: drop rows whose paid amount is not positive.
    bad_amount = ~(df["amount_paid"] > 0)
    counts["dropped_non_positive_amount"] = int(bad_amount.sum())
    df = df[~bad_amount]

    # Step 4: drop exact duplicate rows.
    duplicated = df.duplicated(subset=IBM_COLUMNS)
    counts["dropped_exact_duplicates"] = int(duplicated.sum())
    df = df[~duplicated]

    # Step 4a: drop the sparse tail after the main period (docs/02-DATA.md section 13).
    if drop_after:
        late = df["timestamp"] > pd.Timestamp(drop_after)
        counts["dropped_after_cutoff"] = int(late.sum())
        counts["dropped_after_cutoff_laundering"] = int(df.loc[late, "is_laundering"].sum())
        df = df[~late]

    # Step 5: account IDs (bank + account) and amounts in base currency.
    unknown = sorted(set(df["currency_paid"].dropna()) - set(units_per_base))
    if unknown:
        raise ValueError(f"No exchange rate in settings for currencies: {unknown}")
    df["sender_id"] = df["from_bank"] + "_" + df["from_account"]
    df["receiver_id"] = df["to_bank"] + "_" + df["to_account"]
    df["amount"] = df["amount_paid"] / df["currency_paid"].map(units_per_base).astype("float64")

    # Step 6: flag self-transfers (kept, but left out of graph features later).
    df["is_self"] = df["sender_id"] == df["receiver_id"]
    counts["self_transfers"] = int(df["is_self"].sum())

    # Attach pattern labels by matching pattern-file rows to transactions.
    if patterns is not None:
        unique_patterns = patterns.drop_duplicates(subset=MATCH_KEY)
        df = df.merge(unique_patterns, on=MATCH_KEY, how="left")
        counts["pattern_rows_in_file"] = len(patterns)
        counts["pattern_rows_matched"] = int(df["pattern_id"].notna().sum())
    else:
        df["pattern_type"] = pd.NA
        df["pattern_id"] = pd.NA

    # Step 7: sort by time (stable, so same-minute rows keep file order) and number them.
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["txn_id"] = pd.Series(range(len(df)), dtype="int64")

    transactions = pd.DataFrame({
        "txn_id": df["txn_id"],
        "timestamp": df["timestamp"].astype("datetime64[ns]"),
        "sender_id": df["sender_id"].astype("string"),
        "receiver_id": df["receiver_id"].astype("string"),
        "amount": df["amount"].astype("float64"),
        "amount_original": df["amount_paid"].astype("float64"),
        "currency_original": df["currency_paid"].astype("string"),
        "channel": df["payment_format"].astype("string"),
        "is_self": df["is_self"].astype("bool"),
        "dataset": pd.Series(dataset, index=df.index, dtype="string"),
    })
    labels = pd.DataFrame({
        "txn_id": df["txn_id"],
        "is_laundering": df["is_laundering"].astype("int8"),
        "pattern_type": df["pattern_type"].astype("string"),
        "pattern_id": df["pattern_id"].astype("string"),
    })

    counts["rows_written"] = len(transactions)
    counts["laundering_written"] = int(labels["is_laundering"].sum())
    counts["accounts"] = int(pd.concat([transactions["sender_id"], transactions["receiver_id"]]).nunique())
    counts["time_range"] = [str(transactions["timestamp"].min()), str(transactions["timestamp"].max())]
    return transactions, labels, counts


def ingest(dataset: str) -> Path:
    """Run P1 for one dataset and return the output folder."""
    started = now_utc()
    settings = load_settings()
    ds = settings["datasets"][dataset]
    if not dataset.startswith("ibm_"):
        raise NotImplementedError(f"Ingest for {dataset} is not written yet (only IBM so far)")

    raw_dir = PROJECT_ROOT / ds["raw_dir"]
    csv_path = raw_dir / ds["transactions_file"]
    patterns_path = raw_dir / ds["patterns_file"] if ds.get("patterns_file") else None

    print(f"Reading {csv_path.name} ...")
    raw = read_ibm_transactions(csv_path)
    patterns = None
    if patterns_path and patterns_path.exists():
        patterns = read_ibm_patterns(patterns_path)
    else:
        print("No patterns file found; pattern labels will be empty.")

    print("Cleaning ...")
    transactions, labels, counts = clean_ibm(
        raw, patterns, dataset,
        units_per_base=settings["currency"]["units_per_base"],
        drop_after=ds.get("drop_after"),
    )

    out = path("processed") / dataset
    out.mkdir(parents=True, exist_ok=True)
    transactions.to_parquet(out / "transactions.parquet", index=False)
    labels.to_parquet(out / "labels_txn.parquet", index=False)

    inputs = [file_info(csv_path)] + ([file_info(patterns_path)] if patterns is not None else [])
    write_manifest(out, "ingest", started, dataset=dataset, inputs=inputs, counts=counts)

    for key, value in counts.items():
        print(f"  {key}: {value}")
    print(f"Wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 Ingest: raw dataset to common format")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    ingest(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
