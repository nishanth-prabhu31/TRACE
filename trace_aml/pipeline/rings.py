"""P10 Rings: group flagged accounts into rings (F-13, docs/05-FEATURES-MODEL.md section 6.6).

Built from TRACE's own output only (no labels):
1. take every LIKELY_MULE and WATCH account in the run,
2. link two of them if they are on the same money trail, or if money passed directly
   between them within `rings.link_within_hours` before either one's scoring snapshot,
3. each connected group of 2+ accounts is a ring, numbered R-0001... by highest member risk.

Writes data/runs/<run_id>/rings.parquet (docs/04-CONTRACT.md section 5.4).
`--evaluate` then compares the rings with the ground-truth rings; only that part reads labels.

Usage:
    python -m trace_aml.pipeline.rings --run ibm_hi_small__test__v2
    python -m trace_aml.pipeline.rings --run ibm_hi_small__test__v2 --evaluate
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path

FLAGGED = ("LIKELY_MULE", "WATCH")


def ring_links(scores: pd.DataFrame, trails: pd.DataFrame, view: pd.DataFrame,
               snapshots: pd.Series, within_hours: float) -> tuple[nx.Graph, pd.DataFrame]:
    """Graph of flagged accounts plus the transfers behind each link (txn_id, a, b, amount, timestamp)."""
    flagged = set(scores.loc[scores["verdict"].isin(FLAGGED), "account_id"])
    graph = nx.Graph()
    graph.add_nodes_from(flagged)
    used = []

    # Same trail: every flagged account on one trail (owner included) is linked to the owner.
    on_trail = trails[trails["account_id"].isin(flagged)]
    for (owner, _), trail in on_trail.groupby(["account_id", "trail_id"]):
        members = (set(trail["from_id"]) | set(trail["to_id"])) & flagged
        members.discard(owner)
        if members:
            graph.add_edges_from((owner, other) for other in members)
            used.append(trail[["txn_id", "from_id", "to_id", "amount", "timestamp"]])

    # Direct money within the window before either account's snapshot.
    direct = view[view["sender_id"].isin(flagged) & view["receiver_id"].isin(flagged)
                  & (view["sender_id"] != view["receiver_id"])]
    window = pd.Timedelta(hours=within_hours)
    near = pd.Series(False, index=direct.index)
    for side in ("sender_id", "receiver_id"):
        snap = direct[side].map(snapshots)
        near |= (direct["timestamp"] <= snap) & (direct["timestamp"] > snap - window)
    direct = direct[near]
    graph.add_edges_from(zip(direct["sender_id"], direct["receiver_id"]))
    used.append(direct.rename(columns={"sender_id": "from_id", "receiver_id": "to_id"})
                [["txn_id", "from_id", "to_id", "amount", "timestamp"]])

    links = pd.concat(used, ignore_index=True).drop_duplicates("txn_id") if used else pd.DataFrame(
        columns=["txn_id", "from_id", "to_id", "amount", "timestamp"])
    return graph, links


def build_rings(scores: pd.DataFrame, trails: pd.DataFrame, view: pd.DataFrame,
                snapshots: pd.Series, within_hours: float) -> pd.DataFrame:
    """rings.parquet rows: one per (ring, member)."""
    graph, links = ring_links(scores, trails, view, snapshots, within_hours)
    risk = scores.set_index("account_id")["risk"]
    groups = [g for g in nx.connected_components(graph) if len(g) >= 2]
    groups.sort(key=lambda g: (-float(risk.reindex(list(g)).max()), -len(g), min(g)))
    rows = []
    for number, members in enumerate(groups, start=1):
        inside = links[links["from_id"].isin(members) | links["to_id"].isin(members)]
        member_risk = risk.reindex(sorted(members))
        for account in sorted(members):
            rows.append({"ring_id": f"R-{number:04d}", "account_id": account, "ring_size": len(members),
                         "ring_amount": float(inside["amount"].sum()),
                         "first_txn": inside["timestamp"].min(), "last_txn": inside["timestamp"].max(),
                         "max_risk": float(member_risk.max()), "mean_risk": float(member_risk.mean())})
    out = pd.DataFrame(rows, columns=[col.name for col in c.RINGS.columns])
    return out.astype({"ring_size": "int32", "max_risk": "float32", "mean_risk": "float32",
                       "first_txn": "datetime64[ns]", "last_txn": "datetime64[ns]"})


def evaluate_rings(rings: pd.DataFrame, account_labels: pd.DataFrame, id_map: dict | None = None) -> dict:
    """Compare with ground-truth rings (evaluation only).

    coverage = for each ground-truth ring with 2+ members flagged, the largest share of its
    flagged members that sit inside one TRACE ring; purity = share of TRACE ring members that
    are laundering accounts (mule, source or sink)."""
    labels = account_labels.set_index("account_id")
    members = rings.assign(original=rings["account_id"].map(id_map) if id_map else rings["account_id"])
    members["truth_ring"] = members["original"].map(labels["ring_id"])
    members["role"] = members["original"].map(labels["role"]).fillna("NORMAL")
    coverage = []
    for truth, group in members.dropna(subset=["truth_ring"]).groupby("truth_ring"):
        if len(group) >= 2:
            coverage.append(group["ring_id"].value_counts().iloc[0] / len(group))
    return {"rings": int(rings["ring_id"].nunique()), "members": int(len(rings)),
            "purity": round(float((members["role"] != "NORMAL").mean()), 4) if len(members) else None,
            "truth_rings_seen": len(coverage),
            "coverage_mean": round(float(np.mean(coverage)), 4) if coverage else None,
            "rings_with_any_laundering": int(members.groupby("ring_id")["role"].apply(
                lambda r: (r != "NORMAL").any()).sum()) if len(members) else 0}


def run_rings(run_id: str, evaluate: bool = False) -> dict:
    folder = path("runs") / run_id
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    scores = pd.read_parquet(folder / c.SCORES.filename)
    trails = pd.read_parquet(folder / c.TRAILS.filename)
    view = pd.read_parquet(folder / c.TRANSACTIONS_VIEW.filename,
                           columns=["txn_id", "timestamp", "sender_id", "receiver_id", "amount"])
    snapshots = pd.read_parquet(folder / c.RUN_FEATURES.filename,
                                columns=["account_id", "snapshot"]).set_index("account_id")["snapshot"]
    rings = build_rings(scores, trails, view, snapshots, load_settings()["rings"]["link_within_hours"])
    rings.to_parquet(folder / c.RINGS.filename, index=False)
    summary = {"run_id": run_id, "rings": int(rings["ring_id"].nunique()), "members": int(len(rings)),
               "largest": int(rings["ring_size"].max()) if len(rings) else 0}
    if evaluate:
        dataset = manifest["dataset"]
        id_map = None
        if dataset == "demo":
            account_map = pd.read_csv(path("demo") / "demo_account_map.csv")
            id_map = dict(zip(account_map["demo_id"], account_map["original_id"]))
            dataset = manifest.get("source_run", "ibm_hi_small__").split("__")[0]
        labels = pd.read_parquet(path("processed") / dataset / c.ACCOUNT_LABELS.filename,
                                 columns=["account_id", "role", "ring_id"])
        summary["evaluation"] = evaluate_rings(rings, labels, id_map)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--evaluate", action="store_true")
    args = p.parse_args()
    print(json.dumps(run_rings(args.run, args.evaluate), indent=2, default=str))


if __name__ == "__main__":
    main()
