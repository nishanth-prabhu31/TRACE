"""Alert queue: the ranked list of accounts to review (docs/06-DASHBOARD.md section 4)."""

import math

import pandas as pd
import streamlit as st

from app.logic import navigation
from app.logic.formatting import VERDICT_TEXT
from app.shared import current_run, decisions_by_account, money, verdict_label
from trace_aml.config import load_settings

DEFAULT_VERDICTS = ["LIKELY_MULE", "WATCH"]


def reset_filters() -> None:
    st.session_state["queue_verdicts"] = DEFAULT_VERDICTS
    st.session_state["queue_decided"] = "All"
    st.session_state["queue_search"] = ""
    st.session_state["queue_page"] = 1


st.title("Alert queue")
run = current_run()
if run is not None:
    page_size = load_settings()["app"]["queue_page_size"]
    scores = run["scores"].copy()
    decisions = decisions_by_account()
    scores["decision"] = pd.Series([decisions.get(account) for account in scores["account_id"]],
                                   index=scores.index, dtype="object")

    controls = st.columns([3, 2, 3, 2])
    verdicts = controls[0].multiselect("Verdict", list(VERDICT_TEXT), default=DEFAULT_VERDICTS,
                                       format_func=VERDICT_TEXT.get, key="queue_verdicts")
    decided = controls[1].selectbox("Decision", ["All", "Undecided", "Decided"], key="queue_decided")
    search = controls[2].text_input("Search account ID", key="queue_search").strip()
    order = controls[3].selectbox("Sort", ["Risk (highest first)", "Rank"], key="queue_sort")

    shown = scores[scores["verdict"].isin(verdicts)]
    if decided == "Undecided":
        shown = shown[shown["decision"].isna()]
    elif decided == "Decided":
        shown = shown[shown["decision"].notna()]
    if search:
        shown = shown[shown["account_id"].str.contains(search, case=False, regex=False)]
    shown = shown.sort_values(["risk", "rank"], ascending=[False, True]) if order.startswith("Risk") \
        else shown.sort_values("rank")

    if shown.empty:
        st.info("No accounts match these filters.")
        st.button("Reset filters", on_click=reset_filters)
    else:
        pages = math.ceil(len(shown) / page_size)
        page = st.number_input(f"Page (of {pages})", min_value=1, max_value=pages, step=1, key="queue_page") \
            if pages > 1 else 1
        rows = shown.iloc[(page - 1) * page_size: page * page_size].reset_index(drop=True)

        table = pd.DataFrame({
            "Rank": rows["rank"],
            "Account": rows["account_id"],
            "Risk": rows["risk"],
            "Verdict": rows["verdict"].map(verdict_label),
            "Behaviour": rows["short_label"],
            "Decision": rows["decision"].fillna("—"),
            "Total in": rows["in_amount"].map(money),
            "Total out": rows["out_amount"].map(money),
        })
        st.caption(f"{len(shown):,} accounts · select a row to open it")
        picked = st.dataframe(
            table, hide_index=True, on_select="rerun", selection_mode="single-row",
            key=f"queue_table_{st.session_state.get('queue_opened', 0)}_{page}",
            column_config={
                "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f"),
                "Total in": st.column_config.TextColumn("Total in"),
                "Total out": st.column_config.TextColumn("Total out"),
            })
        if picked.selection.rows:
            st.session_state["queue_opened"] = st.session_state.get("queue_opened", 0) + 1   # clears the selection
            navigation.open_account(st.session_state, rows.loc[picked.selection.rows[0], "account_id"])
            st.switch_page("pages/account.py")
