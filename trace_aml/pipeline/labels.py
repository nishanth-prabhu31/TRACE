"""P3 Labels and time split: turn transaction labels into account labels (F-03).

Writes data/processed/<dataset>/account_labels.parquet and splits.json,
following docs/02-DATA.md sections 5 and 6.

Usage:
    python -m trace_aml.pipeline.labels --dataset ibm_hi_small
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from trace_aml.config import load_settings, path
from trace_aml.pipeline.common import now_utc, write_manifest


def account_roles(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per account with role, is_mule and mule_since (docs/02-DATA.md 5.1-5.2).

    Self-transfers are ignored: money sent to yourself is not passing it on.
    """
    accounts = pd.Index(pd.concat([transactions["sender_id"], transactions["receiver_id"]]).unique())
    dirty = transactions.merge(labels[["txn_id", "is_laundering"]], on="txn_id")
    dirty = dirty[(dirty["is_laundering"] == 1) & ~dirty["is_self"]]

    sent = dirty.groupby("sender_id").size()
    received = dirty.groupby("receiver_id").size()
    first_out = dirty.groupby("sender_id")["timestamp"].min()
    first_in = dirty.groupby("receiver_id")["timestamp"].min()

    out = pd.DataFrame(index=accounts)
    has_out = out.index.isin(sent.index)
    has_in = out.index.isin(received.index)
    out["role"] = "NORMAL"
    out.loc[has_in & has_out, "role"] = "MULE"
    out.loc[has_out & ~has_in, "role"] = "SOURCE"
    out.loc[has_in & ~has_out, "role"] = "SINK"

    out["is_mule"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out.loc[out["role"] == "MULE", "is_mule"] = 1
    out.loc[out["role"] == "NORMAL", "is_mule"] = 0

    since = pd.concat([first_out, first_in], axis=1).min(axis=1)
    out["mule_since"] = since.reindex(out.index)
    out.loc[out["role"] == "NORMAL", "mule_since"] = pd.NaT
    out.index.name = "account_id"
    return out.reset_index()


def assign_rings(roles: pd.DataFrame, transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Add ring_id and ring_source (docs/02-DATA.md 5.3).

    1. Pattern rings: accounts in the same pattern attempt share its pattern_id.
       An account in several attempts takes its earliest one.
    2. Connected rings: remaining laundering accounts linked by laundering transfers
       form groups C-0001, C-0002, ...; a lone account gets no ring.
    """
    roles = roles.copy()
    dirty = transactions.merge(labels, on="txn_id")
    dirty = dirty[(dirty["is_laundering"] == 1) & ~dirty["is_self"]]

    in_pattern = dirty[dirty["pattern_id"].notna()]
    members = pd.concat([
        in_pattern[["sender_id", "pattern_id", "timestamp"]].rename(columns={"sender_id": "account_id"}),
        in_pattern[["receiver_id", "pattern_id", "timestamp"]].rename(columns={"receiver_id": "account_id"}),
    ])
    first_attempt = (members.sort_values(["timestamp", "pattern_id"])
                     .drop_duplicates("account_id")
                     .set_index("account_id")["pattern_id"])

    ring_id = roles["account_id"].map(first_attempt)
    ring_source = pd.Series(pd.NA, index=roles.index, dtype="string")
    ring_source[ring_id.notna()] = "PATTERN_FILE"

    laundering_accounts = set(roles.loc[roles["role"] != "NORMAL", "account_id"])
    without_ring = laundering_accounts - set(first_attempt.index)
    edges = dirty[dirty["sender_id"].isin(without_ring) & dirty["receiver_id"].isin(without_ring)]
    graph = nx.Graph()
    graph.add_edges_from(zip(edges["sender_id"], edges["receiver_id"]))
    groups = sorted((sorted(g) for g in nx.connected_components(graph) if len(g) >= 2),
                    key=lambda g: g[0])
    connected = {account: f"C-{i:04d}" for i, group in enumerate(groups, start=1) for account in group}

    fill = roles["account_id"].map(connected)
    use_connected = ring_id.isna() & fill.notna()
    ring_id[use_connected] = fill[use_connected]
    ring_source[use_connected] = "CONNECTED"

    roles["ring_id"] = ring_id.astype("string")
    roles["ring_source"] = ring_source
    return roles


def time_split(transactions: pd.DataFrame, shares: dict[str, float]) -> dict[str, pd.Timestamp]:
    """Cut the timeline into train / validation / test at whole hours (docs/02-DATA.md 6)."""
    start = transactions["timestamp"].min().floor("h")
    end = transactions["timestamp"].max().floor("h") + pd.Timedelta(hours=1)
    hours = (end - start) / pd.Timedelta(hours=1)
    train_end = start + pd.Timedelta(hours=round(hours * shares["train"]))
    validation_end = start + pd.Timedelta(hours=round(hours * (shares["train"] + shares["validation"])))
    return {"data_start": start, "train_end": train_end, "validation_end": validation_end, "test_end": end}


def mule_counts(roles: pd.DataFrame, transactions: pd.DataFrame, cuts: dict) -> dict:
    """How many mules each period has; "new" = first laundering inside that period."""
    mules = roles[roles["role"] == "MULE"]
    since = mules["mule_since"]
    in_test = transactions["timestamp"] >= cuts["validation_end"]
    active_in_test = set(transactions.loc[in_test, "sender_id"]) | set(transactions.loc[in_test, "receiver_id"])
    test_new = int((since >= cuts["validation_end"]).sum())
    test_known = int(((since < cuts["validation_end"]) & mules["account_id"].isin(active_in_test)).sum())
    return {
        "train": int((since < cuts["train_end"]).sum()),
        "validation": int(((since >= cuts["train_end"]) & (since < cuts["validation_end"])).sum()),
        "test_new": test_new,
        "test_known": test_known,
        "test_new_share": round(test_new / (test_new + test_known), 3) if test_new + test_known else None,
    }


def to_contract(roles: pd.DataFrame) -> pd.DataFrame:
    """Columns and types exactly as docs/04-CONTRACT.md section 2.3."""
    return pd.DataFrame({
        "account_id": roles["account_id"].astype("string"),
        "role": roles["role"].astype("string"),
        "is_mule": roles["is_mule"].astype("Int8"),
        "mule_since": pd.to_datetime(roles["mule_since"]).astype("datetime64[ns]"),
        "ring_id": roles["ring_id"].astype("string"),
        "ring_source": roles["ring_source"].astype("string"),
    }).sort_values("account_id").reset_index(drop=True)


def make_labels(dataset: str) -> Path:
    """Run P3 for one dataset and return the output folder."""
    started = now_utc()
    settings = load_settings()
    folder = path("processed") / dataset
    transactions = pd.read_parquet(folder / "transactions.parquet",
                                   columns=["txn_id", "timestamp", "sender_id", "receiver_id", "is_self"])
    labels = pd.read_parquet(folder / "labels_txn.parquet")

    print("Assigning roles ...")
    roles = account_roles(transactions, labels)
    print("Grouping rings ...")
    roles = assign_rings(roles, transactions, labels)
    account_labels = to_contract(roles)

    cuts = time_split(transactions, settings["split"])
    counts = mule_counts(account_labels, transactions, cuts)
    splits = {
        "dataset": dataset,
        **{key: value.isoformat() for key, value in cuts.items()},
        "shares": settings["split"],
        "mule_counts": counts,
    }

    account_labels.to_parquet(folder / "account_labels.parquet", index=False)
    (folder / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    summary = {
        "roles": account_labels["role"].value_counts().to_dict(),
        "rings": account_labels.groupby("ring_source")["ring_id"].nunique().to_dict(),
        "laundering_accounts_without_ring": int(((account_labels["role"] != "NORMAL")
                                                 & account_labels["ring_id"].isna()).sum()),
        "largest_ring": int(account_labels["ring_id"].value_counts().max()),
    }
    write_manifest(folder, "labels", started, dataset=dataset, summary=summary, splits=splits)

    print(json.dumps({"summary": summary, "splits": splits}, indent=2))
    print(f"Wrote {folder}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 Labels and time split")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    make_labels(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
