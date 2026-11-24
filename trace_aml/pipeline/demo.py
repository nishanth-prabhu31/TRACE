"""F-11 Demo data file: a test-period IBM slice restyled to rupees / Indian channels.

Steps follow docs/02-DATA.md section 10 (1-6):
1. pick one laundering pattern group that lies fully in the test period,
2. take its accounts plus their direct counterparties in the same window (200-500 accounts),
3. rename accounts to ACC-xxxx (mapping kept),
4. convert amounts to rupees with the display rate,
5. relabel channels with the fixed table in settings (cosmetic only),
6. save the CSV in the upload format; labels go to a separate file,
7. cut the pre-scored DEMO run (data/runs/demo__demo__<version>) out of the full test run.

Needs the test run (score --split TEST) for step 7 and to pick the main ring by risk.

Writes into data/demo/:
    demo_transactions.csv   upload format (docs/04-CONTRACT.md section 3)
    demo_labels.csv         row number -> is_laundering, pattern (for checking only)
    demo_account_map.csv    ACC-xxxx -> original account id
    manifest_demo.json

Usage:
    python -m trace_aml.pipeline.demo --dataset ibm_hi_small
    python -m trace_aml.pipeline.demo --dataset ibm_hi_small --pattern-id 123
"""

import argparse
import json

import numpy as np
import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.pipeline.common import file_info, now_utc, write_manifest
from trace_aml.pipeline.score import write_run

# Shapes that are easy to show on screen, best first.
PREFERRED_SHAPES = ["FAN_IN", "FAN_OUT", "SCATTER_GATHER", "GATHER_SCATTER", "STACK", "CYCLE", "BIPARTITE", "RANDOM"]
PATTERN_SIZE = (3, 12)          # accounts in the pattern itself
DEMO_RUN_ID = "demo__demo__{version}"


def source_run_id(dataset: str, version: str | None = None) -> str:
    """The full test run the demo run is cut from."""
    return f"{dataset}__test__{version or load_settings()['model']['version']}"


def recut_demo_run(dataset: str, version: str):
    """Cut the DEMO run again from another model version's test run, keeping the same demo
    slice and ACC numbers (used when a model version is activated, F-14)."""
    account_map = pd.read_csv(path("demo") / "demo_account_map.csv")
    n_slice = json.loads((path("demo") / "manifest_demo.json").read_text(encoding="utf-8"))["accounts"]
    in_slice = account_map.sort_values("demo_id").head(n_slice)
    mapping = dict(zip(in_slice["original_id"], in_slice["demo_id"]))
    return build_demo_run(dataset, mapping, in_slice, path("demo") / "demo_account_map.csv", version)


def test_window(splits: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(splits["validation_end"]), pd.Timestamp(splits["test_end"])


def candidate_patterns(txns: pd.DataFrame, labels: pd.DataFrame, start, end) -> pd.DataFrame:
    """Pattern groups fully inside [start, end), with their size and shape."""
    merged = txns[["txn_id", "timestamp", "sender_id", "receiver_id"]].merge(labels, on="txn_id")
    merged = merged[merged["pattern_id"].notna() & (merged["pattern_id"].astype(str) != "")]
    rows = []
    for pid, g in merged.groupby("pattern_id"):
        if g["timestamp"].min() < start or g["timestamp"].max() >= end:
            continue
        accounts = set(g["sender_id"]) | set(g["receiver_id"])
        shape = str(g["pattern_type"].iloc[0]).upper()
        rows.append({"pattern_id": pid, "pattern_type": shape, "n_accounts": len(accounts),
                     "n_txns": len(g), "accounts": accounts})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    rank = {s: i for i, s in enumerate(PREFERRED_SHAPES)}
    out["shape_rank"] = out["pattern_type"].map(lambda s: rank.get(s, len(rank)))
    return out.sort_values(["shape_rank", "n_txns"], ascending=[True, False]).reset_index(drop=True)


def slice_for(window_txns: pd.DataFrame, pattern_accounts: set) -> tuple[set, pd.DataFrame]:
    """Pattern accounts + direct counterparties, and all transfers among that set."""
    touch = window_txns["sender_id"].isin(pattern_accounts) | window_txns["receiver_id"].isin(pattern_accounts)
    accounts = set(pattern_accounts) | set(window_txns.loc[touch, "sender_id"]) | set(window_txns.loc[touch, "receiver_id"])
    inside = window_txns["sender_id"].isin(accounts) & window_txns["receiver_id"].isin(accounts)
    return accounts, window_txns[inside]


def build(dataset: str, pattern_id=None) -> dict:
    s = load_settings()
    started = now_utc()
    proc = path("processed") / dataset
    txns = pd.read_parquet(proc / "transactions.parquet")
    labels = pd.read_parquet(proc / "labels_txn.parquet")
    splits = json.loads((proc / "splits.json").read_text(encoding="utf-8"))
    start, end = test_window(splits)
    window = txns[(txns["timestamp"] >= start) & (txns["timestamp"] < end)]

    cands = candidate_patterns(txns, labels, start, end)
    if cands.empty:
        raise SystemExit("No pattern group lies fully in the test period.")
    lo, hi = s["demo"]["target_accounts"]
    if pattern_id is not None:
        cands = cands[cands["pattern_id"].astype(str) == str(pattern_id)]
        if cands.empty:
            raise SystemExit(f"Pattern {pattern_id} is not fully inside the test period.")
    else:
        cands = cands[cands["n_accounts"].between(*PATTERN_SIZE)]

    if cands.empty:
        raise SystemExit("No pattern of the wanted size in the test period; pass --pattern-id.")
    all_cands = candidate_patterns(txns, labels, start, end)
    all_cands["slice"] = [len(slice_for(window, a)[0]) for a in all_cands["accounts"]]
    cands = cands.merge(all_cands[["pattern_id", "slice"]], on="pattern_id")
    # Main pattern: the one whose riskiest member scores highest in the full test run, so the
    # demo story opens on a ring the model flags. The file still keeps rings it misses.
    scores_file = path("runs") / source_run_id(dataset) / "scores.parquet"
    if scores_file.exists():
        risk = pd.read_parquet(scores_file, columns=["account_id", "risk"]).set_index("account_id")["risk"]
        cands["top_risk"] = [float(risk.reindex(list(a)).max()) for a in cands["accounts"]]
        row = cands.sort_values(["top_risk", "slice"], ascending=[False, False]).iloc[0]
    else:
        row = cands.sort_values(["shape_rank", "slice"], ascending=[True, False]).iloc[0]

    # A single test pattern + counterparties is far below `lo` accounts in IBM HI-Small
    # (max about 75), so other test patterns' slices are added, biggest first, until the
    # file reaches lo..hi accounts (02-DATA section 13 records this).
    pattern_accounts = set(row["accounts"])
    extra = []
    accounts, _ = slice_for(window, pattern_accounts)
    for _, other in all_cands.sort_values("slice", ascending=False).iterrows():
        if len(accounts) >= lo:
            break
        if other["pattern_id"] == row["pattern_id"]:
            continue
        trial, _ = slice_for(window, pattern_accounts | set(other["accounts"]))
        if len(trial) > hi:
            continue
        pattern_accounts |= set(other["accounts"])
        accounts = trial
        extra.append(str(other["pattern_id"]))
    accounts, sl = slice_for(window, pattern_accounts)

    # 3. Rename: shuffled order so ACC numbers say nothing about the pattern.
    rng = np.random.default_rng(s["random_seed"])
    ordered = sorted(accounts)
    rng.shuffle(ordered)
    mapping = {acc: f"ACC-{i + 1:04d}" for i, acc in enumerate(ordered)}

    sl = sl.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
    rate = float(s["currency"]["display_units_per_base"])
    relabel = s["demo"]["channel_relabel"]
    demo = pd.DataFrame({
        "timestamp": sl["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "sender_id": sl["sender_id"].map(mapping),
        "receiver_id": sl["receiver_id"].map(mapping),
        "amount": (sl["amount"] * rate).round(2),            # 4. rupees
        "currency": s["currency"]["display"],
        "channel": sl["channel"].map(lambda c: relabel.get(c, c)),  # 5. cosmetic
    })
    demo = demo[demo["amount"] > 0].reset_index()            # upload rule: amount > 0
    lab = sl[["txn_id"]].merge(labels, on="txn_id", how="left").loc[demo["index"]]
    demo = demo.drop(columns="index")
    demo_labels = pd.DataFrame({
        "row": range(1, len(demo) + 1),
        "is_laundering": lab["is_laundering"].fillna(0).astype(int).to_numpy(),
        "pattern_type": lab["pattern_type"].fillna("").to_numpy(),
        "pattern_id": lab["pattern_id"].fillna("").astype(str).to_numpy(),
        "source_txn_id": lab["txn_id"].to_numpy(),
    })
    acc_map = pd.DataFrame({"demo_id": list(mapping.values()), "original_id": list(mapping.keys()),
                            "in_main_pattern": [a in row["accounts"] for a in mapping],
                            "in_any_pattern": [a in pattern_accounts for a in mapping]})

    out = path("demo")
    out.mkdir(parents=True, exist_ok=True)
    files = {"transactions": out / "demo_transactions.csv", "labels": out / "demo_labels.csv",
             "account_map": out / "demo_account_map.csv"}
    demo.to_csv(files["transactions"], index=False, encoding="utf-8")
    demo_labels.to_csv(files["labels"], index=False, encoding="utf-8")
    acc_map.sort_values("demo_id").to_csv(files["account_map"], index=False, encoding="utf-8")

    run_folder = build_demo_run(dataset, mapping, acc_map, files["account_map"])

    summary = {"run": str(run_folder) if run_folder else None,
               "dataset": dataset, "pattern_id": str(row["pattern_id"]), "pattern_type": row["pattern_type"],
               "pattern_accounts": int(row["n_accounts"]), "extra_patterns": extra,
               "accounts": len(accounts),
               "transactions": len(demo), "laundering_rows": int(demo_labels["is_laundering"].sum()),
               "window": [str(start), str(end)], "display_rate": rate}
    write_manifest(out, "demo", started, **summary,
                   outputs={k: file_info(v) for k, v in files.items()})
    return summary


def build_demo_run(dataset: str, mapping: dict, acc_map: pd.DataFrame, map_file,
                   version: str | None = None) -> "Path | None":
    """Step 7: cut the pre-scored DEMO run out of the full test run (same scores, reasons,
    trails), renaming every account. Amounts stay in base currency, as in every run; the app
    shows rupees. Accounts reached only by a trail get ACC numbers after the slice's."""
    source = path("runs") / source_run_id(dataset, version)
    if not (source / c.SCORES.filename).exists():
        print(f"No test run at {source}; demo run skipped (run score first).")
        return None
    keep = set(mapping)
    scores = pd.read_parquet(source / c.SCORES.filename)
    scores = scores[scores["account_id"].isin(keep)].copy()
    reasons = pd.read_parquet(source / c.REASONS.filename)
    reasons = reasons[reasons["account_id"].isin(keep)].copy()
    trails = pd.read_parquet(source / c.TRAILS.filename)
    trails = trails[trails["account_id"].isin(keep)].copy()
    feats = pd.read_parquet(source / c.RUN_FEATURES.filename)
    feats = feats[feats["account_id"].isin(keep)].copy()

    # Trail hops may leave the slice: give those accounts new ACC numbers too.
    full = dict(mapping)
    outside = sorted((set(trails["from_id"]) | set(trails["to_id"])) - keep)
    for acc in outside:
        full[acc] = f"ACC-{len(full) + 1:04d}"
    if outside:
        extra = pd.DataFrame({"demo_id": [full[a] for a in outside], "original_id": outside,
                              "in_main_pattern": False, "in_any_pattern": False})
        pd.concat([acc_map, extra]).sort_values("demo_id").to_csv(map_file, index=False, encoding="utf-8")

    view = pd.read_parquet(source / c.TRANSACTIONS_VIEW.filename)
    ids = set(full)
    view = view[view["sender_id"].isin(ids) & view["receiver_id"].isin(ids)].copy()   # includes trail hops
    view["sender_id"] = view["sender_id"].map(full)
    view["receiver_id"] = view["receiver_id"].map(full)
    view["dataset"] = "demo"
    view = view.sort_values(["sender_id", "timestamp"], kind="stable").reset_index(drop=True)

    for table in (scores, reasons, trails, feats):
        table["account_id"] = table["account_id"].map(full)
    trails["from_id"] = trails["from_id"].map(full)
    trails["to_id"] = trails["to_id"].map(full)
    scores = scores.sort_values("risk", ascending=False, kind="stable").reset_index(drop=True)
    scores["rank"] = np.arange(1, len(scores) + 1, dtype="int32")

    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    version = source_manifest["model_version"]
    manifest = {**source_manifest, "run_id": DEMO_RUN_ID.format(version=version), "dataset": "demo",
                "kind": "DEMO", "split": None, "created_at": now_utc(), "source_run": source.name,
                "note": "Restyled IBM test-period data; scores cut from the full test run.",
                "n_accounts": int(len(scores)), "n_transactions": int(len(view)),
                "verdict_counts": {v: int((scores["verdict"] == v).sum())
                                   for v in ("LIKELY_MULE", "WATCH", "NORMAL")}}
    tables = {c.SCORES.filename: scores, c.REASONS.filename: reasons.reset_index(drop=True),
              c.TRAILS.filename: trails.reset_index(drop=True), c.RUN_FEATURES.filename: feats.reset_index(drop=True),
              c.TRANSACTIONS_VIEW.filename: view}
    folder = write_run(tables, manifest, path("runs"))
    from trace_aml.pipeline.rings import run_rings      # rings from the demo run's own flags
    run_rings(manifest["run_id"])
    return folder


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ibm_hi_small")
    p.add_argument("--pattern-id", default=None)
    args = p.parse_args()
    print(json.dumps(build(args.dataset, args.pattern_id), indent=2))


if __name__ == "__main__":
    main()
