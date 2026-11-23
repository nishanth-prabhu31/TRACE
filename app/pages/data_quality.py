"""Data quality: the report cards written by the quality gate (docs/06-DASHBOARD.md section 6)."""

import json

import streamlit as st

from trace_aml.config import path

ICON = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}

st.title("Data quality")
st.caption("Every dataset goes through these checks before it is used for training or testing.")

reports = sorted(path("reports").glob("quality_*.json"))
if not reports:
    st.info("No quality reports yet. Run: python -m trace_aml.pipeline.quality --dataset <name>")
for report_file in reports:
    report = json.loads(report_file.read_text(encoding="utf-8"))
    with st.container(border=True):
        st.subheader(f"{report['dataset']}: {ICON[report['overall']]} {report['overall']}")
        st.table([{"Check": f"{check['id']} {check['name']}",
                   "Status": f"{ICON[check['status']]} {check['status']}",
                   "Value": check["value"],
                   "Detail": check["detail"]} for check in report["checks"]])
        full = report_file.with_suffix(".md")
        if full.exists():
            with st.expander("Full report"):
                st.markdown(full.read_text(encoding="utf-8"))
