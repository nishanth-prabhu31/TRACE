"""Overview: totals and run information (docs/06-DASHBOARD.md section 3)."""

import streamlit as st

from app.logic import navigation
from app.shared import current_run, money

st.title("Overview")


def open_demo_run(run_id: str) -> None:
    """Switch the sidebar to the pre-scored demo run."""
    st.session_state["run_choice"] = st.session_state["run_labels"][run_id]
    navigation.clear(st.session_state)


demo_runs = [run for run in st.session_state.get("runs", {}).values() if run.get("kind") == "DEMO"]
if demo_runs:
    # Prefer the demo run of the active model version (F-14 keeps one demo run per version).
    from app.logic import data_access
    from app.shared import database
    active = data_access.get_state(database(), data_access.ACTIVE_MODEL_KEY)
    demo_runs.sort(key=lambda run: run.get("model_version") != active)
    st.button("Open demo run", on_click=open_demo_run, args=(demo_runs[0]["run_id"],))

run = current_run()
if run is not None:
    manifest, scores, transactions = run["manifest"], run["scores"], run["transactions"]
    counts = manifest["verdict_counts"]

    top = st.columns(5)
    top[0].metric("Transactions", f"{manifest['n_transactions']:,}")
    top[1].metric("Accounts", f"{manifest['n_accounts']:,}")
    top[2].metric("Likely mule", f"{counts.get('LIKELY_MULE', 0):,}")
    top[3].metric("Watch", f"{counts.get('WATCH', 0):,}")
    top[4].metric("Money in this run", money(transactions["amount"].sum(), short=True))

    st.subheader("Run")
    kind = {"SPLIT": "split", "UPLOAD": "upload", "DEMO": "demo", "REPLAY": "replay"}[manifest["kind"]]
    if manifest.get("split"):
        kind += f" ({manifest['split'].lower()} period)"
    st.table({
        "Dataset": [manifest["dataset"]],
        "Kind": [kind],
        "Cut-off time": [manifest["cutoff"].replace("T", " ")],
        "Model version": [manifest["model_version"]],
        "Created": [manifest["created_at"].replace("T", " ").replace("Z", " UTC")],
    })
    if manifest["dataset"] == "fake":
        st.warning("This is the fake run used to build the screens. Its scores, reasons and trails "
                   "are invented, not produced by a model.")
    if manifest["kind"] == "UPLOAD":
        st.caption("Accounts with little history in this file are scored less reliably.")

    st.page_link("pages/alert_queue.py", label="Go to the alert queue →")
