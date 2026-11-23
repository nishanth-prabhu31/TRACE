"""The fake run must pass the contract checker and contain what the app needs to show."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_contract import check_folder  # noqa: E402
from make_fake_data import MERCHANT, MULES, make_fake_data  # noqa: E402


def test_fake_run_passes_the_contract_checker(tmp_path):
    make_fake_data(tmp_path)
    results = check_folder(tmp_path)
    failed = [message for ok, message in results if not ok]
    assert results and not failed, failed


def test_fake_run_has_the_demo_ingredients(tmp_path):
    run = make_fake_data(tmp_path)
    scores = pd.read_parquet(run / "scores.parquet").set_index("account_id")
    trails = pd.read_parquet(run / "trails.parquet")
    view = pd.read_parquet(run / "transactions_view.parquet")

    assert len(scores) == 300
    assert set(scores.index[scores["verdict"] == "LIKELY_MULE"]) == set(MULES)
    assert (scores["verdict"] == "WATCH").sum() >= 3
    assert set(trails["account_id"]) == set(MULES)
    assert view[view["receiver_id"] == MERCHANT]["sender_id"].nunique() >= 50   # normal hub
    assert view.groupby(["sender_id", "receiver_id"]).size().max() >= 4          # arrows to merge
    assert scores.loc[MERCHANT, "verdict"] == "NORMAL"


def test_fake_data_is_repeatable(tmp_path):
    first = pd.read_parquet(make_fake_data(tmp_path / "a") / "scores.parquet")
    second = pd.read_parquet(make_fake_data(tmp_path / "b") / "scores.parquet")
    assert first.equals(second)
