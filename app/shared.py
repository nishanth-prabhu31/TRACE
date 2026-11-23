"""Small helpers every page uses: the cached run, the database, the chosen currency."""

import sqlite3

import pandas as pd
import streamlit as st

from app.logic import data_access, feedback
from app.logic.neighbourhood import build_network
from app.logic.formatting import DECISION_TEXT, VERDICT_TEXT, format_money
from trace_aml.config import load_settings

VERDICT_SYMBOL = {"LIKELY_MULE": "▲", "WATCH": "◆", "NORMAL": "●"}


@st.cache_resource(show_spinner="Loading run…")
def cached_run(folder: str) -> dict:
    """A run's files, read once and kept in memory.

    Cached as a shared resource (no copy per page load): pages must treat it as read-only.
    """
    return data_access.load_run(folder)


@st.cache_resource(show_spinner="Preparing network…")
def cached_network(folder: str) -> dict:
    """The run's network look-up tables, prepared once (app/logic/neighbourhood.py)."""
    return build_network(cached_run(folder)["transactions"])


@st.cache_resource(show_spinner=False)
def cached_scores_by_account(folder: str) -> pd.DataFrame:
    """The run's scores indexed by account ID."""
    return cached_run(folder)["scores"].set_index("account_id")


@st.cache_resource
def database() -> sqlite3.Connection:
    """One shared connection to trace.db (created with empty tables if missing)."""
    conn = sqlite3.connect(data_access.path("db"), check_same_thread=False)
    conn.executescript(data_access.c.SQLITE_SCHEMA)
    return conn


def current_run() -> dict | None:
    """The run picked in the sidebar, or None (after showing why) if there is none."""
    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info("No scored runs yet. Run the pipeline or upload a file.")
        return None
    folder = st.session_state["runs"][run_id]["folder"]
    return {**cached_run(folder), "folder": folder}


def current_user() -> tuple[str, str]:
    """(name, role) of the analyst picked in the sidebar; the first analyst until one is picked."""
    people = {person["name"]: person["role"] for person in data_access.analysts()}
    name = st.session_state.get("user_name")
    if name not in people:
        name = next(iter(people))
    return name, people[name]


def chosen_currency() -> str:
    """The currency picked in the sidebar (the display currency until one is picked)."""
    return st.session_state.get("currency") or load_settings()["currency"]["display"]


def money(amount: float, short: bool = False) -> str:
    """An amount in the currency chosen in the sidebar."""
    return format_money(float(amount), chosen_currency(), short)


def verdict_label(verdict: str) -> str:
    """Verdict as shown everywhere: a shape plus words, never colour alone."""
    return f"{VERDICT_SYMBOL[verdict]} {VERDICT_TEXT[verdict]}"


def decisions_by_account() -> dict[str, str]:
    """account_id -> 'Confirmed mule · Analyst A' for accounts with a decision."""
    current = feedback.current_decisions(database())
    return {row.account_id: f"{DECISION_TEXT[row.decision]} · {row.analyst_name}" for row in current.itertuples()}
