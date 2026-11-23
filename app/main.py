"""TRACE dashboard entry point: sidebar and page routing (docs/06-DASHBOARD.md sections 1-2).

Run from the repo root:
    streamlit run app/main.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.logic import data_access, navigation  # noqa: E402
from app.shared import database  # noqa: E402
from trace_aml.config import load_settings  # noqa: E402

st.set_page_config(page_title="TRACE", layout="wide")


def on_run_change() -> None:
    """A different run means a different set of accounts: forget the open one."""
    navigation.clear(st.session_state)
    st.session_state.pop("queue_page", None)


def sidebar() -> None:
    """Who am I, which run, which model, which currency."""
    settings = load_settings()
    st.sidebar.title("TRACE")
    st.sidebar.caption("Transaction Risk Analytics and Chain Evaluation")

    people = {person["name"]: person["role"] for person in data_access.analysts()}
    name = st.sidebar.selectbox("Logged in as", list(people), key="user_name")
    st.sidebar.caption(f"Role: {people[name].title()} · role selection, not security")

    runs = data_access.list_runs()
    st.session_state["runs"] = {run["run_id"]: run for run in runs}
    if not runs:
        st.session_state["run_id"] = None
        st.sidebar.info("No scored runs yet.")
    else:
        # The picker lists readable labels; the run ID is added only if two labels would clash.
        plain = [data_access.run_label(run) for run in runs]
        labels = {(label if plain.count(label) == 1 else f"{label} · {run['run_id']}"): run["run_id"]
                  for label, run in zip(plain, runs)}
        st.session_state["run_labels"] = {run_id: label for label, run_id in labels.items()}
        if st.session_state.get("run_choice") not in labels:
            saved = data_access.get_state(database(), data_access.ACTIVE_RUN_KEY)
            start = saved if saved in st.session_state["run_labels"] else runs[0]["run_id"]
            st.session_state["run_choice"] = st.session_state["run_labels"][start]
        st.sidebar.selectbox("Run", list(labels), key="run_choice", on_change=on_run_change)
        st.session_state["run_id"] = labels[st.session_state["run_choice"]]
        run = st.session_state["runs"][st.session_state["run_id"]]
        active = data_access.get_state(database(), data_access.ACTIVE_MODEL_KEY) or run["model_version"]
        st.sidebar.text(f"Active model: {active}")

    currency = settings["currency"]
    st.sidebar.radio("Currency", [currency["display"], currency["base"]], key="currency", horizontal=True)
    st.sidebar.caption("Synthetic data. TRACE ranks and explains; the analyst decides.")


sidebar()
pages = [
    st.Page("pages/overview.py", title="Overview", default=True),
    st.Page("pages/alert_queue.py", title="Alert queue"),
    st.Page("pages/account.py", title="Account investigation"),
    st.Page("pages/data_quality.py", title="Data quality"),
]
st.navigation(pages).run()
