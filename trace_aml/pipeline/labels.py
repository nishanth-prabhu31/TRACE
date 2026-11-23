"""P3 Labels and split: account roles, rings and the time split (F-03).

Reads data/processed/<dataset>/transactions.parquet and labels_txn.parquet, writes
account_labels.parquet and splits.json next to them (docs/02-DATA.md sections 5-6).

Roles are "as of" a time: only laundering transactions up to that time count, so a
training label never uses what happens later (docs/02-DATA.md section 5.5).
This module is labelling code; feature code must never import it.

Usage:
    python -m trace_aml.pipeline.labels --dataset ibm_hi_small
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.pipeline.common import file_info, now_utc, write_manifest
from trace_aml.pipeline.split import compute_split

TIME_COLUMNS = ["first_laundering_in", "first_laundering_out"]


def laundering_edges(transactions: pd.DataFrame, labels_txn: pd.DataFrame) -> pd.DataFrame:
    """Laundering transfers between two different accounts, with their pattern group."""
    merged = transactions[["txn_id", "timestamp", "sender_id", "receiver_id", "is_self"]].merge(
        labels_txn, on="txn_id", how="inner", validate="one_to_one")
    keep = (merged["is_laundering"] == 1) & ~merged["is_self"]
    return merged.loc[keep].drop(columns=["is_self", "is_laundering"]).reset_index(drop=True)


def first_laundering_times(all_accounts: pd.Series, edges: pd.DataFrame) -> pd.DataFrame:
    """Per account: when it first received and first sent laundering money (empty if never)."""
    first_in = edges.groupby("receiver_id")["timestamp"].min()
    first_out = edges.groupby("sender_id")["timestamp"].min()
    times = pd.DataFrame({"account_id": all_accounts.astype("string")})
    times["first_laundering_in"] = times["account_id"].map(first_in).astype("datetime64[ns]")
    times["first_laundering_out"] = times["account_id"].map(first_out).astype("datetime64[ns]")
    return times


def labels_as_of(account_labels: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Roles using only laundering transactions with timestamp <= as_of.

    Returns account_id, role, is_mule and mule_since. MULE = has both received and
    sent laundering by then; mule_since = the moment both became true.
    """
    got_in = account_labels["first_laundering_in"] <= as_of
    sent_out = account_labels["first_laundering_out"] <= as_of
    role = pd.Series("NORMAL", index=account_labels.index, dtype="string")
    role[got_in & ~sent_out] = "SINK"
    role[~got_in & sent_out] = "SOURCE"
    role[got_in & sent_out] = "MULE"

    is_mule = pd.Series(pd.NA, index=account_labels.index, dtype="Int8")
    is_mule[role == "MULE"] = 1
    is_mule[role == "NORMAL"] = 0

    both = account_labels[TIME_COLUMNS].max(axis=1)
    mule_since = both.where(role == "MULE").astype("datetime64[ns]")
    return pd.DataFrame({
        "account_id": account_labels["account_id"],
        "role": role,
        "is_mule": is_mule,
        "mule_since": mule_since,
    })


def pattern_rings(edges: pd.DataFrame) -> pd.Series:
    """account_id -> pattern_id, for every account in a pattern group.

    An account in several groups gets the one it appears in first.
    """
    in_pattern = edges[edges["pattern_id"].notna()]
    long = pd.concat([
        in_pattern[["sender_id", "timestamp", "pattern_id"]].rename(columns={"sender_id": "account_id"}),
        in_pattern[["receiver_id", "timestamp", "pattern_id"]].rename(columns={"receiver_id": "account_id"}),
    ])
    long = long.sort_values(["timestamp", "pattern_id"], kind="stable")
    return long.drop_duplicates("account_id").set_index("account_id")["pattern_id"]


def connected_rings(edges: pd.DataFrame, mules: set[str], already_in_ring: set[str]) -> pd.Series:
    """account_id -> ring id for mules outside any pattern group (fallback rule).

    Mules that sent laundering money to each other are linked; each linked group of
    two or more is a ring, numbered C-0001, C-0002, ... by its earliest account ID.
    """
    candidates = mules - already_in_ring
    between = edges[edges["sender_id"].isin(candidates) & edges["receiver_id"].isin(candidates)]
    graph = nx.Graph()
    graph.add_edges_from(zip(between["sender_id"], between["receiver_id"]))
    groups = sorted((sorted(group) for group in nx.connected_components(graph) if len(group) >= 2),
                    key=lambda group: group[0])
    ring_of = {account: f"C-{number:04d}" for number, group in enumerate(groups, start=1)
               for account in group}
    return pd.Series(ring_of, dtype="string")


def build_account_labels(transactions: pd.DataFrame, labels_txn: pd.DataFrame) -> pd.DataFrame:
    """One row per account in the contract format, with roles as of the end of the data."""
    all_accounts = pd.Series(pd.concat([transactions["sender_id"], transactions["receiver_id"]]).unique())
    all_accounts = all_accounts.sort_values(ignore_index=True)
    edges = laundering_edges(transactions, labels_txn)

    out = first_laundering_times(all_accounts, edges)
    final = labels_as_of(out, transactions["timestamp"].max())
    out[["role", "is_mule", "mule_since"]] = final[["role", "is_mule", "mule_since"]]

    by_pattern = pattern_rings(edges)
    mules = set(out.loc[out["role"] == "MULE", "account_id"])
    by_link = connected_rings(edges, mules, set(by_pattern.index))
    from_pattern = out["account_id"].map(by_pattern)
    from_link = out["account_id"].map(by_link)
    out["ring_id"] = from_pattern.fillna(from_link).astype("string")
    out["ring_source"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out.loc[from_pattern.notna(), "ring_source"] = "PATTERN_FILE"
    out.loc[from_pattern.isna() & from_link.notna(), "ring_source"] = "CONNECTED"

    return out[[col.name for col in c.ACCOUNT_LABELS.columns]]


def count_mules(account_labels: pd.DataFrame, split: dict[str, pd.Timestamp]) -> dict[str, int]:
    """Mules per period, each counted with what is known at that period's end."""
    at_train = labels_as_of(account_labels, split["train_end"])
    at_validation = labels_as_of(account_labels, split["validation_end"])
    at_test = labels_as_of(account_labels, split["test_end"])
    return {
        "train": int((at_train["role"] == "MULE").sum()),
        "validation": int((at_validation["mule_since"] > split["train_end"]).sum()),
        "test_new": int((at_test["mule_since"] > split["validation_end"]).sum()),
        "test_known": int((at_test["mule_since"] <= split["validation_end"]).sum()),
    }


def build_splits(dataset: str, transactions: pd.DataFrame, account_labels: pd.DataFrame,
                 shares: dict[str, float]) -> dict:
    """The splits.json content (docs/04-CONTRACT.md section 2.4)."""
    split = compute_split(transactions, shares)
    return {
        "dataset": dataset,
        **{key: value.strftime("%Y-%m-%dT%H:%M:%S") for key, value in split.items()},
        "shares": dict(shares),
        "mule_counts": count_mules(account_labels, split),
    }


def run_labels(dataset: str) -> Path:
    """Run P3 for one dataset and return the output folder."""
    started = now_utc()
    settings = load_settings()
    folder = path("processed") / dataset
    transactions_file = folder / c.TRANSACTIONS.filename
    labels_file = folder / c.LABELS_TXN.filename

    transactions = pd.read_parquet(
        transactions_file, columns=["txn_id", "timestamp", "sender_id", "receiver_id", "is_self"])
    labels_txn = pd.read_parquet(labels_file)

    account_labels = build_account_labels(transactions, labels_txn)
    splits = build_splits(dataset, transactions, account_labels, settings["split"])

    account_labels.to_parquet(folder / c.ACCOUNT_LABELS.filename, index=False)
    (folder / c.SPLITS_FILENAME).write_text(json.dumps(splits, indent=2), encoding="utf-8")

    self_laundering = int((labels_txn["is_laundering"] == 1).sum()) - len(laundering_edges(transactions, labels_txn))
    counts = {
        "accounts": len(account_labels),
        "roles": {role: int((account_labels["role"] == role).sum()) for role in c.ROLES},
        "rings": {source: int(account_labels.loc[account_labels["ring_source"] == source, "ring_id"].nunique())
                  for source in c.RING_SOURCES},
        "laundering_self_transfers_ignored": self_laundering,
        "mule_counts": splits["mule_counts"],
    }
    write_manifest(folder, "labels", started, dataset=dataset,
                   inputs=[file_info(transactions_file), file_info(labels_file)], counts=counts)

    for key, value in counts.items():
        print(f"  {key}: {value}")
    print(f"  split: train_end={splits['train_end']} validation_end={splits['validation_end']} "
          f"test_end={splits['test_end']}")
    print(f"Wrote {folder}")
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 Labels and split: account roles, rings, time split")
    parser.add_argument("--dataset", required=True, help="dataset name from settings.yaml")
    run_labels(parser.parse_args().dataset)


if __name__ == "__main__":
    main()
