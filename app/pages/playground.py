"""Try transfers: add a few transfers to the sample network and see what TRACE makes of them."""

import pandas as pd
import streamlit as st

from app.components.graph import render_legend, render_network
from app.logic import data_access
from app.logic.formatting import VERDICT_TEXT
from app.logic.playground import RULE_NAMES, SCENARIOS, involved, overall, scenario, with_background
from app.shared import chosen_currency, database
from trace_aml.config import load_settings
from trace_aml.pipeline.rules import FLAG_COLUMNS
from trace_aml.pipeline.upload import score_tables

OVERALL_ICON = {"Likely mule": "▲", "Watch": "◆", "No concerns found": "●"}


@st.cache_data(max_entries=8, show_spinner=False)
def score(csv_text: str, version: str) -> dict:
    """Background + the user's transfers, scored once per distinct table."""
    edited = pd.read_csv(pd.io.common.StringIO(csv_text), dtype=str, keep_default_na=False)
    transactions, errors, _ = with_background(edited)
    if errors:
        return {"errors": errors}
    tables = score_tables(transactions, version)
    return {"errors": [], "scores": tables["scores.parquet"], "reasons": tables["reasons.parquet"],
            "features": tables["features.parquet"][["account_id", *FLAG_COLUMNS]]}


def load(name: str) -> None:
    st.session_state["play_rows"] = scenario(name)
    st.session_state["play_version"] = st.session_state.get("play_version", 0) + 1
    st.session_state.pop("play_result", None)


st.title("Try transfers")
st.caption("Add transfers to a sample network of 229 accounts (10 days of restyled IBM data, in ₹) and score them "
           "with the TRACE model and the rules. Two accounts on their own cannot be judged: TRACE compares each "
           "account with its own past and with the accounts around it, so new transfers are added to this network.")

if "play_rows" not in st.session_state:
    load(SCENARIOS[0])
choice = st.segmented_control("Start from", SCENARIOS, default=SCENARIOS[0], key="play_scenario")
if choice and choice != st.session_state.get("play_loaded"):
    load(choice)
    st.session_state["play_loaded"] = choice

st.markdown("**Transfers** (edit any cell, add or delete rows; times are on 10 Sep 2022, the network's last day)")
edited = st.data_editor(
    st.session_state["play_rows"], num_rows="dynamic", hide_index=True, key=f"play_editor_{st.session_state['play_version']}",
    column_config={
        "timestamp": st.column_config.TextColumn("Time", help="YYYY-MM-DD HH:MM:SS", required=True),
        "sender_id": st.column_config.TextColumn("From", help="Any name, or an existing ACC-0001…ACC-0229", required=True),
        "receiver_id": st.column_config.TextColumn("To", required=True),
        "amount": st.column_config.NumberColumn("Amount (₹)", min_value=1, format="%.0f", required=True),
        "channel": st.column_config.SelectboxColumn("Channel", options=["UPI", "IMPS", "NEFT", "Cheque", "Cash"]),
    })

settings = load_settings()
version = data_access.get_state(database(), data_access.ACTIVE_MODEL_KEY) or settings["model"]["version"]
if st.button("Score these transfers", type="primary", icon=":material/play_arrow:"):
    with st.spinner("Scoring the network with your transfers…"):
        st.session_state["play_result"] = (edited.copy(), score(edited.to_csv(index=False), version))

if "play_result" not in st.session_state:
    st.stop()
rows, result = st.session_state["play_result"]
if result["errors"]:
    st.error("\n\n".join(f"- {line}" for line in result["errors"]), title="Fix these first", icon=":material/error:")
    st.stop()

names = involved(rows)
scores = result["scores"].set_index("account_id")
flags = result["features"].set_index("account_id")
table = []
for account in names:
    s = scores.loc[account] if account in scores.index else None
    fired = [RULE_NAMES[col] for col in FLAG_COLUMNS if account in flags.index and flags.loc[account, col] > 0]
    verdict = s["verdict"] if s is not None else "NORMAL"
    result_text = overall(verdict, fired)
    table.append({"Account": account, "Overall": f"{OVERALL_ICON[result_text]} {result_text}",
                  "Model risk": float(s["risk"]) if s is not None else 0.0,
                  "Model verdict": VERDICT_TEXT[verdict], "Rules fired": ", ".join(fired) or "—",
                  "Behaviour": s["short_label"] if s is not None else "—"})
table = pd.DataFrame(table)

st.subheader("What TRACE says")
st.dataframe(table, hide_index=True, column_config={
    "Model risk": st.column_config.ProgressColumn("Model risk", min_value=0, max_value=100, format="%.1f")})
st.caption(f"Model {version}. Overall = Likely mule if the model says so; Watch if the model or any rule raises a "
           "concern; otherwise no concerns found. TRACE flags; an analyst decides. In testing, the model misses fast "
           "rings through brand-new accounts that the rules catch, so both are shown.")

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown("**Your transfers in the network**")
    rate = float(settings["currency"]["display_units_per_base"])
    money = rows.assign(amount=pd.to_numeric(rows["amount"], errors="coerce") / rate,
                        timestamp=pd.to_datetime(rows["timestamp"], errors="coerce"))
    edges = (money.groupby(["sender_id", "receiver_id"])
             .agg(amount=("amount", "sum"), n_txn=("amount", "size"),
                  sample=("timestamp", lambda ts: [{"timestamp": t, "amount": a} for t, a in
                                                   zip(ts, money.loc[ts.index, "amount"])][:5]))
             .reset_index())
    known = scores.reindex(names)
    nodes = pd.DataFrame({"account_id": names, "is_centre": False, "on_trail": False,
                          "risk": known["risk"].to_numpy(), "verdict": known["verdict"].to_numpy(),
                          "short_label": known["short_label"].to_numpy(), "n_txn": known["n_txn"].fillna(0).to_numpy(),
                          "in_amount": known["in_amount"].fillna(0).to_numpy(), "out_amount": known["out_amount"].fillna(0).to_numpy()})
    nodes["has_score"] = nodes["verdict"].notna()
    # Colour by the overall result (model or rules), the same as the table above.
    code = {"Likely mule": "LIKELY_MULE", "Watch": "WATCH", "No concerns found": "NORMAL"}
    nodes.loc[nodes["has_score"], "verdict"] = [code[text.split(" ", 1)[1]] for text in table.loc[nodes["has_score"].to_numpy(), "Overall"]]
    render_network(nodes, edges, chosen_currency(), key=f"play_graph_{hash(rows.to_csv())}")
    render_legend()
with right:
    st.markdown("**Why the model scored them**")
    reasons = result["reasons"]
    shown = False
    for account in names:
        mine = reasons[reasons["account_id"] == account].sort_values("position").head(3)
        if len(mine):
            shown = True
            with st.container(border=True):
                st.markdown(f"**{account}**")
                for item in mine.itertuples():
                    st.markdown(f"- {item.display_name}: {item.value_text}")
    if not shown:
        st.caption("The model raised no account above its Watch cut-off, so it has no reasons to show. "
                   "See \"Rules fired\" in the table for what the rules noticed.")
