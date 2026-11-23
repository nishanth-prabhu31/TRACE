"""Smoke test: the app starts on the fake run and opens an account without errors.

Note: after switch_page the test harness runs the page file alone, without the sidebar
in main.py, so these tests cover the pages; the sidebar is covered by the first test.
"""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from app.logic import data_access, navigation  # noqa: E402
from make_fake_data import MULES, make_fake_data  # noqa: E402


@pytest.fixture()
def fake_app(tmp_path, monkeypatch):
    """The app pointed at a fresh fake run and an empty database in a temp folder."""
    make_fake_data(tmp_path)
    real_path = data_access.path
    monkeypatch.setattr(data_access, "run_roots", lambda: [tmp_path / "runs"])
    monkeypatch.setattr(data_access, "path", lambda key: tmp_path / "app.db" if key == "db" else real_path(key))
    return AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=60)


def test_overview_shows_the_run_totals(fake_app):
    app = fake_app.run()
    assert not app.exception
    assert app.title[0].value == "Overview"
    assert {metric.label: metric.value for metric in app.metric}["Accounts"] == "300"


def test_alert_queue_lists_flagged_accounts(fake_app):
    app = fake_app.run()
    app.switch_page("pages/alert_queue.py").run()
    assert not app.exception
    assert app.title[0].value == "Alert queue"
    assert any("9 accounts" in caption.value for caption in app.caption)


def test_account_page_opens_a_mule_with_reasons_and_summary(fake_app):
    app = fake_app.run()
    navigation.open_account(app.session_state, MULES[0])
    app.switch_page("pages/account.py").run()
    assert not app.exception
    text = " ".join(item.value for item in [*app.markdown, *app.caption, *app.header])
    assert MULES[0] in text and "Likely mule" in text and "Pass-through" in text


def test_saving_a_decision_shows_it_in_the_history(fake_app):
    app = fake_app.run()
    navigation.open_account(app.session_state, MULES[0])
    app.switch_page("pages/account.py").run()
    next(button for button in app.button if button.label == "False positive").click().run()
    assert not app.exception
    assert any("False positive · Analyst A" in caption.value for caption in app.caption)
