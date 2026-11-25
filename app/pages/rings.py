"""Rings: flagged accounts grouped into cases (docs/06-DASHBOARD.md section 7, F-13)."""

import pandas as pd
import streamlit as st

from app.components.graph import render_legend, render_network
from app.logic import navigation
from app.logic.neighbourhood import build_ring_graph
from app.shared import cached_network, cached_scores_by_account, chosen_currency, current_run, money, verdict_label
from trace_aml.config import load_settings

st.title("Rings")
run = current_run()
if run is not None:
    rings = run["rings"]
    if rings is None or rings.empty:
        st.info("No rings in this run: no two flagged accounts are linked by a money trail or direct money "
                "within 24 h. (Older runs were scored before rings existed.)")
    else:
        app = load_settings()["app"]
        st.caption("Flagged accounts linked by a money trail, or by money passing directly between them within "
                   "24 h. Built from TRACE's own flags, not from labels. Decisions stay per account.")
        table = (rings.groupby("ring_id", sort=False)
                 .agg(members=("ring_size", "first"), moved=("ring_amount", "first"), first=("first_txn", "first"),
                      last=("last_txn", "first"), max_risk=("max_risk", "first"), mean_risk=("mean_risk", "first"))
                 .sort_values(["max_risk", "members"], ascending=False).reset_index())
        shown = pd.DataFrame({
            "Ring": table["ring_id"], "Members": table["members"], "Money moved": table["moved"].map(money),
            "First transfer": table["first"].dt.strftime("%d %b %H:%M"),
            "Last transfer": table["last"].dt.strftime("%d %b %H:%M"),
            "Highest risk": table["max_risk"].round(1), "Mean risk": table["mean_risk"].round(1),
        })
        st.write(f"**{len(table)} rings**, {len(rings)} accounts · select a ring to see it")
        picked = st.dataframe(shown, hide_index=True, on_select="rerun", selection_mode="single-row",
                              key="ring_table", height=min(38 + 35 * len(shown), 400))
        chosen = table.loc[picked.selection.rows[0], "ring_id"] if picked.selection.rows else table.loc[0, "ring_id"]

        members = rings.loc[rings["ring_id"] == chosen, "account_id"].tolist()
        info = table[table["ring_id"] == chosen].iloc[0]
        st.subheader(f"{chosen}: {int(info['members'])} accounts, {money(info['moved'], short=True)} moved")
        scores = cached_scores_by_account(run["folder"])
        graph = build_ring_graph(cached_network(run["folder"]), scores, members,
                                 app.get("graph_max_nodes_more", 40), app["hover_max_txns"])
        left, right = st.columns([3, 2])
        with left:
            if graph["hidden"]:
                st.caption(f"Showing the {len(graph['nodes'])} riskiest members; {graph['hidden']} more in the list.")
            clicked = render_network(graph["nodes"], graph["edges"], chosen_currency(), key=f"ring_graph_{chosen}")
            render_legend()
        with right:
            member_rows = scores.reindex(members).sort_values("risk", ascending=False).reset_index()
            st.dataframe(pd.DataFrame({
                "Account": member_rows["account_id"], "Risk": member_rows["risk"].round(1),
                "Verdict": member_rows["verdict"].map(verdict_label), "Behaviour": member_rows["short_label"],
            }), hide_index=True, on_select="rerun", selection_mode="single-row", key=f"ring_members_{chosen}")
            selected = st.session_state.get(f"ring_members_{chosen}")
            if selected and selected.selection.rows:
                clicked = member_rows.loc[selected.selection.rows[0], "account_id"]
        if clicked:
            navigation.open_account(st.session_state, clicked)
            st.switch_page("pages/account.py")
