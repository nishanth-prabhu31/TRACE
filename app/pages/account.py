"""Account investigation: score, reasons, network, money trail, transactions, decision
(docs/06-DASHBOARD.md section 5)."""

import pandas as pd
import streamlit as st

from app.components.graph import render_legend, render_network
from app.logic import feedback, navigation
from app.logic.formatting import DECISION_TEXT
from app.logic.neighbourhood import account_transfers, build_neighbourhood
from app.shared import cached_network, cached_scores_by_account, chosen_currency, current_run, current_user, database, money, verdict_label
from trace_aml.config import load_settings

APP = load_settings()["app"]
state = st.session_state


def breadcrumb_bar() -> None:
    """Back button and the path taken so far; earlier items are clickable."""
    items = navigation.breadcrumb(state, APP["breadcrumb_length"])
    columns = st.columns([1] + [2] * len(items) + [max(1, 8 - 2 * len(items))])
    columns[0].button("← Back", on_click=navigation.go_back, args=(state,),
                      disabled=not state.get(navigation.HISTORY))
    for column, (position, account) in zip(columns[1:], items):
        if position is None:
            column.markdown(f"**{account}**")
        else:
            column.button(f"{account} →", key=f"crumb_{position}", on_click=navigation.jump_back_to,
                          args=(state, position))


def score_card(row: pd.Series, model_version: str) -> None:
    left, right = st.columns([1, 3])
    left.metric("Risk", f"{row['risk']:.1f}")
    right.markdown(f"### {verdict_label(row['verdict'])}")
    right.caption(f"{row['short_label']} · model {model_version}")
    st.markdown(f"First seen **{row['first_seen']:%d %b %Y %H:%M}** · **{int(row['n_txn']):,}** transactions · "
                f"total in **{money(row['in_amount'], short=True)}** · "
                f"total out **{money(row['out_amount'], short=True)}**")


def network_tab(run: dict, account: str) -> None:
    two_hops = st.toggle("2 hops", key="graph_two_hops")
    result = build_neighbourhood(cached_network(run["folder"]), cached_scores_by_account(run["folder"]),
                                 run["trails"], account, hops=2 if two_hops else APP["graph_default_hops"],
                                 max_nodes=APP["graph_max_nodes"], hover_max_txns=APP["hover_max_txns"])
    if len(result["nodes"]) == 1:
        st.info("This account has no transfers with other accounts in this run.")
        return
    clicked = render_network(result["nodes"], result["edges"], chosen_currency(),
                             key=f"graph_{account}_{two_hops}_{state.get(navigation.VIEW, 0)}")
    render_legend()
    if result["hidden"]:
        st.caption(f"+{result['hidden']} more accounts not shown")
    if clicked and clicked != account:
        navigation.hop_to(state, clicked)
        st.rerun()

    others = result["nodes"].loc[~result["nodes"]["is_centre"], "account_id"].tolist()
    target = st.selectbox("Go to account…", others, index=None, placeholder="Pick a neighbour to hop to",
                          key=f"goto_{account}_{state.get(navigation.VIEW, 0)}")
    if target:
        navigation.hop_to(state, target)
        st.rerun()


def trail_tab(run: dict, account: str, flagged: bool) -> None:
    trails = run["trails"][run["trails"]["account_id"] == account]
    if trails.empty:
        st.info("No suspicious money trail found for this account."
                + (" See \"Why this score\" for what raised its risk." if flagged else ""))
        return
    ids = sorted(trails["trail_id"].unique())
    trail_id = st.radio("Trail", ids, horizontal=True, format_func=lambda i: f"Trail {i}") if len(ids) > 1 else ids[0]
    hops = trails[trails["trail_id"] == trail_id].sort_values("hop")

    lines = ["digraph {", "  rankdir=LR; node [shape=box, style=\"rounded,filled\", fillcolor=\"#EEF1F4\", "
             "fontname=Helvetica, fontsize=11]; edge [fontname=Helvetica, fontsize=10];"]
    lines.append(f'  "{account}" [fillcolor="#2C3E50", fontcolor=white];')
    for hop in hops.itertuples():
        lines.append(f'  "{hop.from_id}" -> "{hop.to_id}" '
                     f'[label="{money(hop.amount, short=True)}\\n{hop.timestamp:%d %b %H:%M}"];')
    st.graphviz_chart("\n".join(lines + ["}"]))

    minutes = (hops["timestamp"].max() - hops["timestamp"].min()).total_seconds() / 60
    st.caption(f"{len(hops)} hops in {minutes:.0f} minutes, from {hops['timestamp'].min():%d %b %H:%M} "
               f"to {hops['timestamp'].max():%d %b %H:%M}")
    chain = [member for member in dict.fromkeys([*hops["from_id"], *hops["to_id"]]) if member != account]
    target = st.selectbox("Open an account on this trail", chain, index=None, placeholder="Pick an account to hop to",
                          key=f"trail_goto_{account}_{trail_id}_{state.get(navigation.VIEW, 0)}")
    if target:
        navigation.hop_to(state, target)
        st.rerun()


def transactions_tab(run: dict, account: str) -> None:
    rows = account_transfers(cached_network(run["folder"]), account)
    cutoff = pd.Timestamp(run["manifest"]["cutoff"])
    filters = st.columns(2)
    direction = filters[0].radio("Direction", ["All", "In", "Out"], horizontal=True, key="txn_direction")
    period = filters[1].radio("Period", ["All", "Last 24 h"], horizontal=True, key="txn_period")
    if direction != "All":
        rows = rows[rows["direction"] == direction.lower()]
    if period == "Last 24 h":
        rows = rows[rows["timestamp"] > cutoff - pd.Timedelta(hours=24)]
    rows = rows.reset_index(drop=True)
    if rows.empty:
        st.info("No transactions match these filters.")
        return
    on_trail = set(run["trails"].loc[run["trails"]["account_id"] == account, "txn_id"])
    table = pd.DataFrame({
        "Time": rows["timestamp"].dt.strftime("%d %b %Y %H:%M"),
        "Direction": rows["direction"],
        "Counterparty": rows["counterparty"],
        "Amount": rows["amount"].map(money),
        "Channel": rows["channel"],
        "On money trail": rows["txn_id"].isin(on_trail).map({True: "✓", False: ""}),
    })
    st.caption("Select a row to hop to its counterparty.")
    picked = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row",
                          key=f"txn_table_{account}_{state.get(navigation.VIEW, 0)}")
    if picked.selection.rows:
        navigation.hop_to(state, rows.loc[picked.selection.rows[0], "counterparty"])
        st.rerun()


def reasons_panel(run: dict, account: str, verdict: str) -> None:
    st.subheader("Why this score")
    reasons = run["reasons"][run["reasons"]["account_id"] == account].sort_values("position")
    if reasons.empty:
        st.caption("Reasons are stored for Likely mule and Watch accounts only."
                   if verdict == "NORMAL" else "No reasons were stored for this account.")
        return
    largest = float(reasons["impact"].abs().max()) or 1.0
    for reason in reasons.itertuples():
        arrow = "▲ raises risk" if reason.direction == "UP" else "▼ lowers risk"
        st.markdown(f"**{reason.display_name}**: {reason.value_text}")
        st.progress(min(abs(reason.impact) / largest, 1.0), text=arrow)
    st.caption("Bars show how much each signal moved this account's score up or down.")


def decision_panel(run: dict, account: str, risk: float) -> None:
    st.subheader("Decision")
    note = st.text_input("Note (optional)", key=f"note_{account}")
    for decision, text in DECISION_TEXT.items():
        if st.button(text, key=f"decide_{decision}_{account}", width="stretch"):
            name, role = current_user()
            feedback.save_decision(database(), account, run["manifest"]["run_id"],
                                   run["manifest"]["model_version"], decision, name, role, risk, note)
            st.toast(f"Saved: {text} by {name}.")
    history = feedback.decision_history(database(), account)
    for row in history.itertuples():
        when = pd.Timestamp(row.created_at).strftime("%d %b %H:%M")
        st.caption(f"{DECISION_TEXT[row.decision]} · {row.analyst_name} · {when} UTC"
                   + (f" · “{row.note}”" if row.note else ""))
    st.caption("TRACE records the decision only. It never acts on an account.")


st.title("Account investigation")
run = current_run()
if run is not None:
    scores = cached_scores_by_account(run["folder"])
    if state.get(navigation.SELECTED) is None:
        st.info("Pick an account here, or open one from the alert queue.")
        typed = st.text_input("Account ID", placeholder="Type all or part of an account ID").strip()
        if typed:
            matches = [account for account in scores.index if typed.lower() in account.lower()][:APP["search_max_matches"]]
            if not matches:
                st.warning("No account in this run matches that text.")
            choice = st.selectbox("Matches", matches, index=None, placeholder="Pick an account") if matches else None
            if choice:
                navigation.open_account(state, choice)
                st.rerun()
    else:
        account = state[navigation.SELECTED]
        breadcrumb_bar()
        main_area, side_area = st.columns([2, 1], gap="large")
        if account not in scores.index:
            main_area.header(account)
            main_area.warning("This account has no score in this run.")
            with main_area:
                network_tab(run, account)
        else:
            row = scores.loc[account]
            with main_area:
                st.header(account)
                score_card(row, run["manifest"]["model_version"])
                network, trail, transactions = st.tabs(["Network", "Follow the money", "Transactions"])
                with network:
                    network_tab(run, account)
                with trail:
                    trail_tab(run, account, flagged=row["verdict"] != "NORMAL")
                with transactions:
                    transactions_tab(run, account)
            with side_area:
                reasons_panel(run, account, row["verdict"])
                st.subheader("In plain words")
                st.write(row["summary"])
                decision_panel(run, account, float(row["risk"]))
