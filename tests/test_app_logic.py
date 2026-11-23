"""Tests for the app's logic layer on the fake run (no Streamlit needed)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from app.logic import data_access, feedback  # noqa: E402
from app.logic.formatting import format_inr, indian_grouping  # noqa: E402
from app.logic.neighbourhood import account_transfers, build_network, build_neighbourhood, merged_edges  # noqa: E402
from make_fake_data import MERCHANT, MULES, RUN_ID, make_fake_data  # noqa: E402


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    root = tmp_path_factory.mktemp("fake")
    folder = make_fake_data(root)
    run = data_access.load_run(folder)
    return {"root": root, "folder": folder, "run": run, "network": build_network(run["transactions"]),
            "scores": run["scores"].set_index("account_id")}


def test_list_runs_finds_the_fake_run(fake):
    runs = data_access.list_runs([fake["root"] / "runs", fake["root"] / "missing"])
    assert [run["run_id"] for run in runs] == [RUN_ID]
    assert data_access.run_label(runs[0]) == "fake · demo · v0"


def test_account_transactions_have_direction_and_counterparty(fake):
    mule = MULES[2]
    rows = account_transfers(fake["network"], mule)
    assert set(rows["direction"]) == {"in", "out"}
    assert rows["timestamp"].is_monotonic_decreasing
    assert (rows["counterparty"] != mule).all()


def test_database_is_created_and_state_round_trips(tmp_path):
    conn = data_access.connect(tmp_path / "trace.db")
    assert data_access.get_state(conn, "active_run_id") is None
    data_access.set_state(conn, "active_run_id", "a")
    data_access.set_state(conn, "active_run_id", "b")
    assert data_access.get_state(conn, "active_run_id") == "b"


def test_decisions_are_kept_and_latest_one_wins(tmp_path):
    conn = data_access.connect(tmp_path / "trace.db")
    args = ("ACC-1", "run", "v1")
    feedback.save_decision(conn, *args, "NEEDS_INVESTIGATION", "Analyst A", "ANALYST", 91.5)
    feedback.save_decision(conn, *args, "CONFIRMED_MULE", "Analyst B", "ANALYST", 91.5, note=" sure ")
    conn.close()

    conn = data_access.connect(tmp_path / "trace.db")          # survives reopening
    history = feedback.decision_history(conn, "ACC-1")
    assert history["decision"].tolist() == ["CONFIRMED_MULE", "NEEDS_INVESTIGATION"]
    assert history.loc[0, "note"] == "sure"
    current = feedback.current_decisions(conn)
    assert current.loc[0, "decision"] == "CONFIRMED_MULE" and current.loc[0, "analyst_name"] == "Analyst B"


def test_unknown_decision_is_rejected(tmp_path):
    conn = data_access.connect(tmp_path / "trace.db")
    with pytest.raises(ValueError):
        feedback.save_decision(conn, "ACC-1", "run", "v1", "FREEZE_ACCOUNT", "Analyst A", "ANALYST", 50)


def test_repeat_transfers_merge_into_one_arrow(fake):
    edges = merged_edges(fake["run"]["transactions"]).set_index(["sender_id", "receiver_id"])
    assert edges.loc[("ACC-0020", "ACC-0021"), "n_txn"] == 4
    assert edges.loc[("ACC-0020", "ACC-0021"), "amount"] == 1000.0
    assert edges.loc[("ACC-0021", "ACC-0020"), "n_txn"] == 3      # the other direction is its own arrow


def test_neighbourhood_of_a_mule_shows_its_chain(fake):
    run, mule = fake["run"], MULES[2]
    result = build_neighbourhood(fake["network"], fake["scores"], run["trails"], mule,
                                 hops=1, max_nodes=40, hover_max_txns=5)
    nodes = result["nodes"].set_index("account_id")
    assert nodes.loc[mule, "is_centre"]
    assert {MULES[1], MULES[3]} <= set(nodes.index)
    assert nodes.loc[MULES[1], "verdict"] == "LIKELY_MULE" and nodes.loc[MULES[1], "on_trail"]
    assert result["hidden"] == 0
    assert all(len(sample) <= 5 for sample in result["edges"]["sample"])


def test_neighbourhood_is_capped_and_says_how_many_are_hidden(fake):
    run = fake["run"]
    result = build_neighbourhood(fake["network"], fake["scores"], run["trails"], MERCHANT,
                                 hops=1, max_nodes=10, hover_max_txns=5)
    assert len(result["nodes"]) == 10
    assert result["hidden"] >= 50
    shown = set(result["nodes"]["account_id"])
    assert set(result["edges"]["sender_id"]) | set(result["edges"]["receiver_id"]) <= shown


def test_flagged_neighbours_are_kept_before_ordinary_ones(fake):
    run = fake["run"]
    result = build_neighbourhood(fake["network"], fake["scores"], run["trails"], "ACC-0040",
                                 hops=1, max_nodes=5, hover_max_txns=5)
    kept = result["nodes"][~result["nodes"]["is_centre"]]
    assert (kept["verdict"] == "WATCH").sum() == 4               # all four watch accounts pay ACC-0040


def test_rupee_formatting():
    assert indian_grouping(125000) == "1,25,000"
    assert indian_grouping(999) == "999"
    assert format_inr(125000) == "₹1,25,000"
    assert format_inr(435000, short=True) == "₹4.35 L"
    assert format_inr(12000000, short=True) == "₹1.2 Cr"
    assert format_inr(950, short=True) == "₹950"
