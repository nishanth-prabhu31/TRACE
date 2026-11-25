"""Score an uploaded CSV of transfers (docs/04-CONTRACT.md section 3, PRD F-10).

validate_upload: raw CSV -> TRACE transactions, or a list of errors that name the column and the
                 first 5 failing line numbers. Required: timestamp, sender_id, receiver_id, amount.
                 Optional: currency (missing = INR, converted with the display rate), channel
                 (missing = UNKNOWN). Any other column, including labels, is ignored.
score_upload:    score every account in the file with a saved model and write an UPLOAD run.
                 Snapshots: one per day through the file (accounts active in the 24 h before
                 each) plus one at the end covering every account; each account keeps its
                 riskiest snapshot, exactly as split runs do. No labels are read anywhere.
"""

import json
import time
from pathlib import Path

import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.pipeline.common import now_utc, settings_hash
from trace_aml.pipeline.features import compute_features
from trace_aml.pipeline.rules import FLAG_COLUMNS, compute_rules
from trace_aml.pipeline.score import build_run, load_model, write_run

REQUIRED = ["timestamp", "sender_id", "receiver_id", "amount"]
OPTIONAL = ["currency", "channel"]
SHOW_ROWS = 5
MIN_ACCOUNTS = 30           # below this the network and "compared to others" features say little
MAX_DAILY_SNAPSHOTS = 14


def currency_rates(settings: dict) -> dict[str, float]:
    """Accepted currency names (upper case) -> units per 1 base currency unit."""
    rates = {name.upper(): float(rate) for name, rate in settings["currency"]["units_per_base"].items()}
    rates[settings["currency"]["base"].upper()] = 1.0
    rates[settings["currency"]["display"].upper()] = float(settings["currency"]["display_units_per_base"])
    return rates


def _lines(mask: pd.Series) -> str:
    """'line 3, 7, 12' for the first failing rows (line 1 is the header)."""
    failing = (mask[mask].index[:SHOW_ROWS] + 2).tolist()
    more = int(mask.sum()) - len(failing)
    return "line " + ", ".join(map(str, failing)) + (f" (and {more} more)" if more > 0 else "")


def validate_upload(raw: pd.DataFrame, settings: dict | None = None) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """(transactions in TRACE format or None, errors, warnings)."""
    settings = settings or load_settings()
    errors, warnings = [], []
    raw = raw.rename(columns=lambda name: str(name).strip().lower())
    missing = [name for name in REQUIRED if name not in raw.columns]
    if missing:
        return None, [f"Missing required column(s): {', '.join(missing)}. "
                      f"The file needs: {', '.join(REQUIRED)} (optional: {', '.join(OPTIONAL)})."], []
    if raw.empty:
        return None, ["The file has a header but no rows."], []
    ignored = [name for name in raw.columns if name not in REQUIRED + OPTIONAL]
    if ignored:
        warnings.append(f"Ignored column(s): {', '.join(ignored)}.")

    data = raw.reset_index(drop=True)
    text = {name: data[name].astype("string").str.strip() for name in data.columns}

    timestamp = pd.to_datetime(text["timestamp"], errors="coerce", format="ISO8601")
    bad = timestamp.isna()
    if bad.any():
        errors.append(f"timestamp: not a date-time like 2026-10-14 10:02:00 at {_lines(bad)}.")
    for name in ("sender_id", "receiver_id"):
        bad = text[name].isna() | (text[name] == "")
        if bad.any():
            errors.append(f"{name}: empty at {_lines(bad)}.")
    amount = pd.to_numeric(text["amount"].str.replace(",", "", regex=False), errors="coerce")
    bad = amount.isna() | (amount <= 0)
    if bad.any():
        errors.append(f"amount: must be a number above 0 at {_lines(bad)}.")

    rates = currency_rates(settings)
    display = settings["currency"]["display"].upper()
    currency = (text["currency"].str.upper() if "currency" in text else pd.Series(pd.NA, index=data.index, dtype="string"))
    currency = currency.fillna(display).replace("", display)
    bad = ~currency.isin(list(rates))
    if bad.any():
        errors.append(f"currency: unknown value(s) {sorted(currency[bad].unique().tolist())[:5]} at {_lines(bad)}. "
                      f"Use INR, USD or leave it empty for INR.")
    if errors:
        return None, errors, warnings

    channel = (text["channel"] if "channel" in text else pd.Series(pd.NA, index=data.index, dtype="string"))
    channel = channel.fillna("UNKNOWN").replace("", "UNKNOWN")
    out = pd.DataFrame({
        "timestamp": timestamp.astype("datetime64[ns]"),
        "sender_id": text["sender_id"].astype("string"),
        "receiver_id": text["receiver_id"].astype("string"),
        "amount": (amount / currency.map(rates).astype("float64")).astype("float64"),
        "amount_original": amount.astype("float64"),
        "currency_original": currency.astype("string"),
        "channel": channel.astype("string"),
    })
    out = out.sort_values(["timestamp"], kind="stable").reset_index(drop=True)
    out.insert(0, "txn_id", pd.Series(range(len(out)), dtype="int64"))
    out["is_self"] = out["sender_id"] == out["receiver_id"]
    out["dataset"] = pd.Series("upload", index=out.index, dtype="string")

    accounts = pd.concat([out["sender_id"], out["receiver_id"]]).nunique()
    if accounts < MIN_ACCOUNTS:
        warnings.append(f"Only {accounts} accounts. TRACE judges an account against the others around it, "
                        f"so with fewer than {MIN_ACCOUNTS} accounts the scores are not reliable.")
    return out[[col.name for col in c.TRANSACTIONS.columns]], errors, warnings


def snapshot_times(transactions: pd.DataFrame) -> list[pd.Timestamp]:
    """Daily cut-offs through the file (at most MAX_DAILY_SNAPSHOTS) plus the end of the file."""
    start, end = transactions["timestamp"].min(), transactions["timestamp"].max()
    days = list(pd.date_range(start.normalize() + pd.Timedelta(days=1), end, freq="1D"))[-MAX_DAILY_SNAPSHOTS:]
    return [t for t in days if t < end] + [end]


def score_tables(transactions: pd.DataFrame, version: str | None = None, progress=None) -> dict[str, pd.DataFrame]:
    """Score every account in `transactions` in memory; returns the run tables (scores, reasons,
    trails, features incl. rule flags, transactions view). Nothing is written."""
    settings = load_settings()
    version = version or settings["model"]["version"]
    model = load_model(path("models") / version)
    cutoffs = snapshot_times(transactions)
    parts = []
    for number, cutoff in enumerate(cutoffs, start=1):
        if progress:
            progress(number / (len(cutoffs) + 1), f"Computing behaviour up to {cutoff:%d %b %H:%M} ({number} of {len(cutoffs)})")
        final = number == len(cutoffs)
        features = compute_features(transactions, cutoff, settings, score_all=final)
        if features.empty:
            continue
        flags, _ = compute_rules(transactions, cutoff, settings)
        rows = features.merge(flags[["account_id", *FLAG_COLUMNS]], on="account_id", how="left")
        rows[FLAG_COLUMNS] = rows[FLAG_COLUMNS].fillna(0).astype("float32")
        parts.append(rows)
    if progress:
        progress(len(cutoffs) / (len(cutoffs) + 1), "Scoring accounts and writing explanations")
    return build_run(transactions, pd.concat(parts, ignore_index=True), model, settings)


def score_upload(transactions: pd.DataFrame, source_name: str, version: str | None = None,
                 progress=None) -> Path:
    """Score the uploaded transactions and write data/runs/upload__<stamp>__<version>/."""
    settings = load_settings()
    version = version or settings["model"]["version"]
    tables = score_tables(transactions, version, progress)
    scores = tables[c.SCORES.filename]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    manifest = {
        "contract_version": c.CONTRACT_VERSION,
        "run_id": f"upload__{stamp}__{version}",
        "dataset": "upload",
        "kind": "UPLOAD",
        "split": None,
        "cutoff": transactions["timestamp"].max().strftime("%Y-%m-%dT%H:%M:%S"),
        "model_version": version,
        "created_at": now_utc(),
        "settings_hash": settings_hash(),
        "source_file": source_name,
        "n_accounts": int(len(scores)),
        "n_transactions": int(len(transactions)),
        "verdict_counts": {verdict: int((scores["verdict"] == verdict).sum()) for verdict in c.VERDICTS},
    }
    folder = write_run(tables, manifest, path("runs"))
    from trace_aml.pipeline.rings import run_rings
    run_rings(manifest["run_id"])
    return folder


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Score a CSV in the upload format")
    parser.add_argument("csv")
    parser.add_argument("--version")
    args = parser.parse_args()
    transactions, errors, warnings = validate_upload(pd.read_csv(args.csv, dtype=str, keep_default_na=False))
    for line in warnings:
        print("warning:", line)
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps({"run": str(score_upload(transactions, Path(args.csv).name, args.version))}))


if __name__ == "__main__":
    main()
