"""Caught in time: replay one laundering ring hour by hour (evaluation view, F-12).

Uses the hourly replay (data/replay/...) scored with the frozen model from data up to each hour
only. Ring membership comes from the labels, so this is an evaluation view, not an analyst screen.
"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from app.components import charts
from app.logic.formatting import VERDICT_TEXT
from trace_aml import contract as c
from trace_aml.config import load_settings, path

DATASET = "ibm_hi_small"


@st.cache_data(max_entries=2, show_spinner="Loading the replay…")
def load(version: str):
    report = json.loads((path("reports") / f"replay_{version}.json").read_text(encoding="utf-8"))
    rings = pd.DataFrame(report["rings_table"])
    for column in ("start", "cashout", "first_alert"):
        rings[column] = pd.to_datetime(rings[column].replace({"NaT": None, "None": None}))
    rings["lead_minutes"] = pd.to_numeric(rings["lead_minutes"], errors="coerce")
    alerts = pd.read_parquet(path("runs").parent / "replay" / f"{DATASET}__{version}" / "alerts.parquet")
    proc = path("processed") / DATASET
    labels = pd.read_parquet(proc / c.LABELS_TXN.filename)
    txns = pd.read_parquet(proc / c.TRANSACTIONS.filename, columns=["txn_id", "sender_id", "receiver_id"])
    pattern = txns.merge(labels[labels["pattern_id"].isin(rings["pattern_id"])], on="txn_id")
    members = (pd.concat([pattern[["pattern_id", "sender_id"]].rename(columns={"sender_id": "account_id"}),
                          pattern[["pattern_id", "receiver_id"]].rename(columns={"receiver_id": "account_id"})])
               .drop_duplicates())
    steps = sorted(alerts["step"].unique())
    alerts = alerts[alerts["account_id"].isin(members["account_id"])]
    cuts = json.loads((path("models") / version / "thresholds.json").read_text(encoding="utf-8"))
    return report, rings, members, alerts, steps, cuts


def describe(row) -> str:
    if pd.isna(row["lead_minutes"]):
        outcome = "never flagged"
    elif row["lead_minutes"] <= 0:
        outcome = "flagged only after cash-out"
    elif row["first_alert"] < row["start"]:
        outcome = f"a member was already flagged before it started"
    else:
        outcome = f"flagged {row['lead_minutes'] / 60:.0f} h before cash-out"
    return f"{row['pattern_id']} · {row['pattern_type'].replace('_', ' ').lower()} · {row['members']} accounts · {outcome}"


version = load_settings()["model"]["version"]
st.title("Caught in time")
st.caption("Hour-by-hour replay of the test period: at each hour TRACE saw only the transfers up to that hour. "
           "Pick a laundering ring and drag through time. Evaluation view: ring membership comes from the "
           "dataset's labels, which TRACE itself never reads.")
if not (path("reports") / f"replay_{version}.json").exists():
    st.info("No replay yet. Run: python -m trace_aml.pipeline.replay --dataset ibm_hi_small")
    st.stop()

report, rings, members, alerts, steps, cuts = load(version)
# Rings genuinely caught in time first (flagged after starting, before cash-out), bigger rings first.
rings["members"] = pd.to_numeric(rings["members"])
strict = (rings["lead_minutes"] > 0) & (rings["first_alert"] >= rings["start"])
rings = rings.assign(_strict=strict).sort_values(["_strict", "members", "lead_minutes"], ascending=[False, False, False])
pick = st.selectbox("Ring", rings.index, format_func=lambda i: describe(rings.loc[i]), key="replay_ring")
ring = rings.loc[pick]
accounts = members.loc[members["pattern_id"] == ring["pattern_id"], "account_id"].tolist()
mine = alerts[alerts["account_id"].isin(accounts)]

timeline = pd.DataFrame({"step": steps})
timeline["Highest member risk"] = timeline["step"].map(mine.groupby("step")["risk"].max()).fillna(0.0)
when = st.select_slider("Time", options=steps, value=steps[0], format_func=lambda t: pd.Timestamp(t).strftime("%d %b %H:%M"),
                        key=f"replay_time_{pick}")

line = alt.Chart(timeline).mark_line(interpolate="step-after", color=charts.NEUTRAL).encode(
    x=alt.X("step:T", title=None, axis=alt.Axis(format="%d %b %H:%M")),
    y=alt.Y("Highest member risk:Q", scale=alt.Scale(domain=[0, 100]), title="Highest risk in the ring"),
    tooltip=[alt.Tooltip("step:T", format="%d %b %H:%M"), alt.Tooltip("Highest member risk:Q", format=".1f")])
marks = pd.DataFrame({"t": [ring["start"], ring["cashout"], pd.Timestamp(when)], "what": ["Ring starts", "Cash-out", "Now"]})
marks = marks[marks["t"].between(pd.Timestamp(steps[0]) - pd.Timedelta(hours=1), pd.Timestamp(steps[-1]))]
rules = alt.Chart(marks).mark_rule(strokeDash=[4, 3]).encode(
    x="t:T", color=alt.Color("what:N", scale=alt.Scale(domain=["Ring starts", "Cash-out", "Now"], range=["#8A94A3", charts.ACCENT, "#1A2433"]),
                             legend=alt.Legend(orient="bottom", title=None)))
cut = alt.Chart(pd.DataFrame({"y": [cuts["likely_mule_cut"]]})).mark_rule(color="#A23B2A", opacity=0.35).encode(y="y:Q")
st.altair_chart((line + cut + rules).properties(height=260))
st.caption(f"Faint red line: Likely mule from risk {cuts['likely_mule_cut']:g}. The ring started "
           f"{ring['start']:%d %b %H:%M} and cashed out {ring['cashout']:%d %b %H:%M}. Risk below the Watch "
           f"cut-off ({cuts['watch_cut']:g}) is shown as 0.")

now = mine[mine["step"] <= pd.Timestamp(when)].sort_values("step").groupby("account_id").last()["risk"]
state = pd.DataFrame({"Account": accounts})
state["Risk now"] = state["Account"].map(now).fillna(0.0)
state["Verdict now"] = state["Risk now"].map(lambda r: VERDICT_TEXT["LIKELY_MULE"] if r >= cuts["likely_mule_cut"]
                                             else VERDICT_TEXT["WATCH"] if r >= cuts["watch_cut"] else "Below Watch")
first = mine[mine["risk"] >= cuts["likely_mule_cut"]].groupby("account_id")["step"].min()
state["First Likely mule"] = state["Account"].map(first).dt.strftime("%d %b %H:%M").fillna("—")
counts = state["Verdict now"].value_counts()
facts = st.container(horizontal=True)
facts.metric("Accounts in the ring", len(state), border=True)
facts.metric(f"Likely mule at {pd.Timestamp(when):%d %b %H:%M}", int(counts.get(VERDICT_TEXT["LIKELY_MULE"], 0)), border=True)
facts.metric("Watch", int(counts.get(VERDICT_TEXT["WATCH"], 0)), border=True)
facts.metric("Cashed out", "yes" if pd.Timestamp(when) >= ring["cashout"] else "not yet", border=True)
st.dataframe(state.sort_values("Risk now", ascending=False), hide_index=True, column_config={
    "Risk now": st.column_config.ProgressColumn("Risk now", min_value=0, max_value=100, format="%.1f")})
