"""Write a small fake run in the exact contract format, for building the app early.

Creates data/fake/runs/fake__demo__v0/ (every file in docs/04-CONTRACT.md section 5)
and data/fake/trace.db. Nothing here comes from a model: scores, reasons and trails are
invented so the screens have something realistic to show (docs/04-CONTRACT.md section 9).

Usage:
    python scripts/make_fake_data.py
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_aml import contract as c  # noqa: E402
from trace_aml.config import load_settings, path  # noqa: E402

RUN_ID = "fake__demo__v0"
MODEL_VERSION = "v0"
N_ACCOUNTS = 300
CUTOFF = pd.Timestamp("2026-10-14 18:00:00")
RING = [f"ACC-{n:04d}" for n in range(1, 8)]      # source, 5 mules, sink
MULES = RING[1:6]
WATCH = [f"ACC-{n:04d}" for n in range(8, 12)]
MERCHANT = "ACC-0300"
FEATURES = ["pass_through_1h", "holding_median_min", "in_degree", "out_degree",
            "fan_in_max_1h", "chain_count", "net_flow_ratio", "burstiness"]
DISPLAY = {
    "pass_through_1h": ("Pass-through (1 h)", "forwarded {:.0%} of inflow within 1 h"),
    "holding_median_min": ("Holding time", "money held for {:.0f} min (median)"),
    "in_degree": ("Senders", "received from {:.0f} accounts"),
    "out_degree": ("Receivers", "sent to {:.0f} accounts"),
    "fan_in_max_1h": ("Fan-in burst", "{:.0f} senders within 1 h"),
    "chain_count": ("Chains", "part of {:.0f} time-ordered chains"),
    "net_flow_ratio": ("Net flow", "net flow ratio {:.2f}"),
    "burstiness": ("Burstiness", "burstiness {:.2f}"),
}


def make_transactions(rng: np.random.Generator) -> pd.DataFrame:
    """Background transfers, a merchant hub, repeated pairs, a ring and some fan-ins."""
    accounts = [f"ACC-{n:04d}" for n in range(1, N_ACCOUNTS + 1)]
    ordinary = [a for a in accounts if a not in RING and a != MERCHANT]
    rows: list[tuple[pd.Timestamp, str, str, float, str]] = []

    def minutes_before_cutoff(low: int, high: int) -> pd.Timestamp:
        return CUTOFF - pd.Timedelta(minutes=int(rng.integers(low, high)))

    for sender in ordinary:                                  # everyone sends at least once
        for _ in range(int(rng.integers(1, 6))):
            receiver = str(rng.choice([a for a in ordinary if a != sender]))
            rows.append((minutes_before_cutoff(60, 4300), sender, receiver,
                         float(rng.lognormal(4.0, 1.0)), str(rng.choice(["ACH", "Credit Card", "Cheque"]))))
    for sender in rng.choice(ordinary, size=90, replace=False):   # high-degree normal merchant
        rows.append((minutes_before_cutoff(60, 4300), str(sender), MERCHANT,
                     float(rng.lognormal(3.0, 0.5)), "Credit Card"))
    for _ in range(4):                                       # repeated transfers, same pair
        rows.append((minutes_before_cutoff(60, 2000), "ACC-0020", "ACC-0021", 250.0, "ACH"))
    for _ in range(3):
        rows.append((minutes_before_cutoff(60, 2000), "ACC-0021", "ACC-0020", 80.0, "ACH"))

    amount, when = 5000.0, CUTOFF - pd.Timedelta(hours=5)    # the ring: 6 fast hops
    for sender, receiver in zip(RING[:-1], RING[1:]):
        rows.append((when, sender, receiver, round(amount, 2), "Wire"))
        amount *= 0.97
        when += pd.Timedelta(minutes=int(rng.integers(4, 15)))
    rows.append((CUTOFF - pd.Timedelta(hours=30), RING[-1], "ACC-0030", 40.0, "ACH"))

    for watched in WATCH:                                    # several senders, then one payout
        start = minutes_before_cutoff(300, 1500)
        total = 0.0
        for sender in rng.choice(ordinary, size=6, replace=False):
            paid = float(rng.uniform(150, 400))
            total += paid
            rows.append((start + pd.Timedelta(minutes=int(rng.integers(0, 50))), str(sender),
                         watched, round(paid, 2), "ACH"))
        rows.append((start + pd.Timedelta(minutes=90), watched, "ACC-0040", round(total * 0.8, 2), "Wire"))

    df = pd.DataFrame(rows, columns=["timestamp", "sender_id", "receiver_id", "amount", "channel"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return pd.DataFrame({
        "txn_id": pd.Series(range(len(df)), dtype="int64"),
        "timestamp": df["timestamp"].astype("datetime64[ns]"),
        "sender_id": df["sender_id"].astype("string"),
        "receiver_id": df["receiver_id"].astype("string"),
        "amount": df["amount"].round(2).astype("float64"),
        "amount_original": df["amount"].round(2).astype("float64"),
        "currency_original": pd.Series("US Dollar", index=df.index, dtype="string"),
        "channel": df["channel"].astype("string"),
        "is_self": pd.Series(False, index=df.index, dtype="bool"),
        "dataset": pd.Series("fake", index=df.index, dtype="string"),
    })


def make_scores(transactions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One invented score per account, with totals taken from the transactions."""
    sent = transactions.groupby("sender_id").agg(out_amount=("amount", "sum"), n_out=("txn_id", "size"),
                                                 first_out=("timestamp", "min"))
    got = transactions.groupby("receiver_id").agg(in_amount=("amount", "sum"), n_in=("txn_id", "size"),
                                                  first_in=("timestamp", "min"))
    table = sent.join(got, how="outer")
    table.index.name = "account_id"
    table = table.reset_index()

    prob = rng.beta(1.0, 30.0, size=len(table))
    prob[table["account_id"].isin(MULES)] = rng.uniform(0.85, 0.97, size=len(MULES))
    prob[table["account_id"].isin(WATCH)] = rng.uniform(0.35, 0.50, size=len(WATCH))
    table["risk"] = np.round(100 * prob, 1)
    table["prob"] = table["risk"] / 100
    table["verdict"] = "NORMAL"
    table.loc[table["account_id"].isin(WATCH), "verdict"] = "WATCH"
    table.loc[table["account_id"].isin(MULES), "verdict"] = "LIKELY_MULE"
    table = table.sort_values(["risk", "account_id"], ascending=[False, True]).reset_index(drop=True)

    label = table["verdict"].map({"LIKELY_MULE": "Pass-through", "WATCH": "Fan-in collector",
                                  "NORMAL": "Normal activity"})
    summary = table["verdict"].map({
        "LIKELY_MULE": "Received money and forwarded 97% of it within 15 minutes. Part of a 6-hop chain ending at ACC-0007.",
        "WATCH": "Received money from 6 accounts within an hour, then forwarded 80% of it in one transfer.",
        "NORMAL": "No unusual pattern found.",
    })
    return pd.DataFrame({
        "account_id": table["account_id"].astype("string"),
        "risk": table["risk"].astype("float32"),
        "prob": table["prob"].astype("float32"),
        "verdict": table["verdict"].astype("string"),
        "rank": pd.Series(range(1, len(table) + 1), dtype="int32"),
        "short_label": label.astype("string"),
        "summary": summary.astype("string"),
        "has_trail": table["account_id"].isin(MULES).astype("bool"),
        "in_amount": table["in_amount"].fillna(0.0).astype("float64"),
        "out_amount": table["out_amount"].fillna(0.0).astype("float64"),
        "n_txn": (table["n_in"].fillna(0) + table["n_out"].fillna(0)).astype("int32"),
        "first_seen": table[["first_in", "first_out"]].min(axis=1).astype("datetime64[ns]"),
    })


def make_features(scores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """An invented feature row per account (risky accounts get risky-looking values)."""
    risky = scores["verdict"].ne("NORMAL").to_numpy()
    n = len(scores)
    values = {
        "pass_through_1h": np.where(risky, rng.uniform(0.8, 0.99, n), rng.uniform(0.0, 0.3, n)),
        "holding_median_min": np.where(risky, rng.uniform(4, 60, n), rng.uniform(300, 3000, n)),
        "in_degree": np.where(risky, rng.integers(1, 7, n), rng.integers(0, 5, n)),
        "out_degree": rng.integers(1, 6, n),
        "fan_in_max_1h": np.where(risky, rng.integers(1, 7, n), rng.integers(0, 2, n)),
        "chain_count": np.where(risky, rng.integers(1, 5, n), 0),
        "net_flow_ratio": np.where(risky, rng.uniform(-0.05, 0.05, n), rng.uniform(-1, 1, n)),
        "burstiness": rng.uniform(-1, 1, n),
    }
    out = pd.DataFrame({
        "account_id": scores["account_id"],
        "cutoff": pd.Series(CUTOFF, index=scores.index).astype("datetime64[ns]"),
        "snapshot": pd.Series(CUTOFF, index=scores.index).astype("datetime64[ns]"),
    })
    for name in FEATURES:
        out[name] = values[name].astype("float32")
    return out


def make_reasons(scores: pd.DataFrame, features: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Five invented reasons for every LIKELY_MULE and WATCH account."""
    rows = []
    flagged = scores[scores["verdict"] != "NORMAL"]["account_id"]
    by_account = features.set_index("account_id")
    for account in flagged:
        impacts = sorted(rng.uniform(0.2, 2.0, size=5), reverse=True)
        for position, (name, impact) in enumerate(zip(FEATURES[:5], impacts), start=1):
            value = float(by_account.loc[account, name])
            signed = -impact if name == "out_degree" else impact
            rows.append((account, position, name, DISPLAY[name][0], value,
                         DISPLAY[name][1].format(value), signed, "UP" if signed > 0 else "DOWN"))
    out = pd.DataFrame(rows, columns=[col.name for col in c.REASONS.columns])
    return out.astype({"account_id": "string", "position": "int8", "feature": "string",
                       "display_name": "string", "value": "float32", "value_text": "string",
                       "impact": "float32", "direction": "string"})


def make_trails(transactions: pd.DataFrame) -> pd.DataFrame:
    """One trail per ring mule: up to 3 hops before it and 3 hops after it."""
    chain = transactions[transactions["sender_id"].isin(RING[:-1]) & transactions["receiver_id"].isin(RING[1:])
                         & (transactions["channel"] == "Wire")].sort_values("timestamp")
    hops = list(chain.itertuples(index=False))
    rows = []
    for mule in MULES:
        arriving = next(i for i, hop in enumerate(hops) if hop.receiver_id == mule)
        picked = hops[max(0, arriving - 2): arriving + 4]
        for number, hop in enumerate(picked, start=1):
            side = "BEFORE" if hop.timestamp <= hops[arriving].timestamp else "AFTER"
            rows.append((mule, 1, number, hop.txn_id, hop.sender_id, hop.receiver_id,
                         hop.timestamp, hop.amount, side))
    out = pd.DataFrame(rows, columns=[col.name for col in c.TRAILS.columns])
    return out.astype({"account_id": "string", "trail_id": "int8", "hop": "int8", "txn_id": "int64",
                       "from_id": "string", "to_id": "string", "timestamp": "datetime64[ns]",
                       "amount": "float64", "side": "string"})


def make_database(db_path: Path, scores: pd.DataFrame) -> None:
    """Create trace.db with the contract tables, one model version and two decisions."""
    if db_path.exists():
        db_path.unlink()
    risk = scores.set_index("account_id")["risk"]
    with sqlite3.connect(db_path) as conn:
        conn.executescript(c.SQLITE_SCHEMA)
        conn.execute("INSERT INTO model_versions (version, created_at, created_by, path) VALUES (?,?,?,?)",
                     (MODEL_VERSION, "2026-10-14T18:30:00Z", "pipeline", f"models/{MODEL_VERSION}/"))
        conn.executemany("INSERT INTO app_state (key, value) VALUES (?,?)",
                         [("active_model_version", MODEL_VERSION), ("active_run_id", RUN_ID)])
        conn.executemany(
            "INSERT INTO feedback (account_id, run_id, model_version, decision, analyst_name, analyst_role,"
            " risk_at_decision, note, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            [(MULES[0], RUN_ID, MODEL_VERSION, "CONFIRMED_MULE", "Analyst A", "ANALYST",
              float(risk[MULES[0]]), "Clear pass-through", "2026-10-14T19:00:00Z"),
             (WATCH[0], RUN_ID, MODEL_VERSION, "NEEDS_INVESTIGATION", "Analyst B", "ANALYST",
              float(risk[WATCH[0]]), None, "2026-10-14T19:05:00Z")])


def make_fake_data(out_root: Path) -> Path:
    """Write the fake run and database under out_root; return the run folder."""
    settings = load_settings()
    rng = np.random.default_rng(settings["random_seed"])
    run = out_root / "runs" / RUN_ID
    run.mkdir(parents=True, exist_ok=True)

    transactions = make_transactions(rng)
    scores = make_scores(transactions, rng)
    features = make_features(scores, rng)
    reasons = make_reasons(scores, features, rng)
    trails = make_trails(transactions)
    view = transactions.sort_values(["sender_id", "timestamp"], kind="stable").reset_index(drop=True)

    scores.to_parquet(run / c.SCORES.filename, index=False)
    reasons.to_parquet(run / c.REASONS.filename, index=False)
    trails.to_parquet(run / c.TRAILS.filename, index=False)
    features.to_parquet(run / c.RUN_FEATURES.filename, index=False)
    view.to_parquet(run / c.TRANSACTIONS_VIEW.filename, index=False)

    manifest = {
        "contract_version": c.CONTRACT_VERSION,
        "run_id": RUN_ID,
        "dataset": "fake",
        "kind": "DEMO",
        "split": None,
        "cutoff": CUTOFF.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_version": MODEL_VERSION,
        "created_at": "2026-10-14T18:30:00Z",
        "settings_hash": "fake",
        "n_accounts": len(scores),
        "n_transactions": len(transactions),
        "verdict_counts": {v: int((scores["verdict"] == v).sum()) for v in c.VERDICTS},
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_database(out_root / "trace.db", scores)
    return run


if __name__ == "__main__":
    print(f"Wrote {make_fake_data(path('fake'))}")
