"""Overview: totals and run information (docs/06-DASHBOARD.md section 3)."""

import json

import streamlit as st

from app.components import charts
from app.logic import navigation
from trace_aml.config import path
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

    top = st.container(horizontal=True)
    top.metric("Transactions", f"{manifest['n_transactions']:,}", border=True)
    top.metric("Accounts", f"{manifest['n_accounts']:,}", border=True)
    top.metric("Likely mule", f"{counts.get('LIKELY_MULE', 0):,}", border=True)
    top.metric("Watch", f"{counts.get('WATCH', 0):,}", border=True)
    top.metric("Money in this run", money(transactions["amount"].sum(), short=True), border=True)

    flagged = scores[scores["verdict"].isin(["LIKELY_MULE", "WATCH"])]
    cuts_file = path("models") / manifest["model_version"] / "thresholds.json"
    row = st.columns(2)
    with row[0].container(border=True):
        st.markdown("**Who got flagged**")
        st.altair_chart(charts.verdict_donut(scores))
        st.caption(f"{len(flagged):,} of {len(scores):,} accounts ({len(flagged) / max(len(scores), 1):.1%}) "
                   "need a look. TRACE flags; the analyst decides.")
    with row[1].container(border=True):
        st.markdown("**How risk is spread**")
        if cuts_file.exists():
            cuts = json.loads(cuts_file.read_text(encoding="utf-8"))
            st.altair_chart(charts.risk_histogram(scores, cuts))
            st.caption("Almost every account sits near 0; the few right of the dashed lines are flagged.")
        else:
            st.caption("No model cut-offs for this run.")
    if len(flagged):
        row = st.columns(2)
        reasons = run["reasons"]
        top_reasons = reasons[reasons["account_id"].isin(flagged["account_id"]) & (reasons["position"] <= 3)
                              & (reasons["direction"] == "UP")]
        with row[0].container(border=True):
            st.markdown("**Most common reasons for a flag**")
            st.altair_chart(charts.count_bars(top_reasons["display_name"],
                                              "Flagged accounts (reason in their top 3)", colour=charts.ACCENT))
        with row[1].container(border=True):
            st.markdown("**What the flagged accounts do**")
            st.altair_chart(charts.count_bars(flagged["short_label"], "Flagged accounts"))

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
