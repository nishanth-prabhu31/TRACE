"""Upload & score: check a CSV of transfers, score every account in it, open the result as a run."""

import io

import pandas as pd
import streamlit as st

from app.logic import data_access, navigation
from app.shared import database
from trace_aml.config import load_settings, path
from trace_aml.pipeline.demo import UPLOAD_SAMPLE
from trace_aml.pipeline.upload import MIN_ACCOUNTS, OPTIONAL, REQUIRED, score_upload, validate_upload

SAMPLE = path("demo") / UPLOAD_SAMPLE
KEY = "upload_source"                       # (file name, raw bytes) of the file being looked at


def use_sample() -> None:
    st.session_state[KEY] = (SAMPLE.name, SAMPLE.read_bytes())


@st.cache_data(max_entries=4, show_spinner=False)
def check(raw_bytes: bytes):
    """Parse and validate once per file (re-runs of the page reuse the result)."""
    try:
        raw = pd.read_csv(io.BytesIO(raw_bytes), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except Exception as error:                            # not a readable CSV at all
        return None, [f"Could not read the file as CSV: {error}"], []
    return validate_upload(raw)


st.title("Upload & score")
st.caption("Score a CSV of money transfers with the saved TRACE model. Each row is one transfer; "
           "TRACE scores every account that appears in the file.")

with st.expander("File format", icon=":material/description:"):
    st.table(pd.DataFrame({
        "Column": REQUIRED + OPTIONAL,
        "Required": ["yes"] * len(REQUIRED) + ["no"] * len(OPTIONAL),
        "Example": ["2022-09-10 10:02:00", "ACC-0012", "ACC-0076", "45000", "INR", "UPI"],
        "Rule": ["date and time", "not empty", "not empty", "number above 0",
                 "INR (default) or USD", "any text; UNKNOWN if empty"],
    }))
    st.caption(f"Other columns are ignored, including any label column. For reliable scores give at least "
               f"{MIN_ACCOUNTS} accounts and a few days of history: TRACE compares each account with its own "
               "past and with the accounts around it.")
    if SAMPLE.exists():
        st.download_button("Download the sample file", SAMPLE.read_bytes(), file_name=SAMPLE.name,
                           mime="text/csv", icon=":material/download:")

pick = st.container(horizontal=True, vertical_alignment="bottom")
uploaded = pick.file_uploader("CSV file", type=["csv"], key="upload_file")
if SAMPLE.exists():
    pick.button("Use the sample file", on_click=use_sample, icon=":material/dataset:",
                help="229 accounts, 10 days of restyled IBM transfers in ₹")
if uploaded is not None:
    st.session_state[KEY] = (uploaded.name, uploaded.getvalue())

source = st.session_state.get(KEY)
if source is None:
    st.info("Choose a CSV file, or use the sample file to see how it works.")
    st.stop()

name, raw_bytes = source
transactions, errors, warnings = check(raw_bytes)
st.subheader(name)
if errors:
    st.error("\n\n".join(f"- {line}" for line in errors), title="File not accepted", icon=":material/error:")
    st.stop()
for line in warnings:
    st.warning(line, icon=":material/warning:")

settings = load_settings()
rate = float(settings["currency"]["display_units_per_base"])
accounts = pd.concat([transactions["sender_id"], transactions["receiver_id"]]).nunique()
days = (transactions["timestamp"].max() - transactions["timestamp"].min()).total_seconds() / 86400
facts = st.container(horizontal=True)
facts.metric("Transfers", f"{len(transactions):,}", border=True)
facts.metric("Accounts", f"{accounts:,}", border=True)
facts.metric("Period", f"{days:.1f} days", border=True)
facts.metric("Money moved", f"₹{transactions['amount'].sum() * rate / 1e7:,.2f} Cr", border=True)
st.caption(f"From {transactions['timestamp'].min():%d %b %Y %H:%M} to {transactions['timestamp'].max():%d %b %Y %H:%M}. "
           "First 10 transfers:")
preview = transactions.head(10)
st.dataframe(pd.DataFrame({
    "Time": preview["timestamp"].dt.strftime("%d %b %Y %H:%M"), "From": preview["sender_id"], "To": preview["receiver_id"],
    "Amount": preview["amount_original"].map(lambda v: f"{v:,.2f}"), "Currency": preview["currency_original"],
    "Channel": preview["channel"]}), hide_index=True)

version = data_access.get_state(database(), data_access.ACTIVE_MODEL_KEY) or settings["model"]["version"]
if st.button(f"Score this file with model {version}", type="primary", icon=":material/play_arrow:"):
    with st.status("Scoring…", expanded=True) as status:
        bar = st.progress(0.0)
        try:
            folder = score_upload(transactions, name, version, progress=lambda share, text: bar.progress(share, text))
        except Exception as error:
            status.update(label="Scoring failed", state="error")
            st.error(f"{type(error).__name__}: {error}")
            st.stop()
        bar.progress(1.0, "Done")
        status.update(label="Scored", state="complete")
    run_id = folder.name
    data_access.set_state(database(), data_access.ACTIVE_RUN_KEY, run_id)
    navigation.clear(st.session_state)
    st.session_state.pop("run_choice", None)
    st.session_state.pop(KEY, None)
    st.switch_page("pages/overview.py")
