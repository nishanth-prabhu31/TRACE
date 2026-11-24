"""Models: versions, retrain and activate (docs/06-DASHBOARD.md section 8, F-14)."""

import json

import pandas as pd
import streamlit as st

from app.logic import data_access, navigation
from app.shared import current_run, current_user, database
from trace_aml.config import path
from trace_aml.pipeline import retrain as rt

st.title("Models")
conn = database()
rt.register_existing(conn)
versions = pd.read_sql("SELECT * FROM model_versions ORDER BY version", conn)
run = current_run()
active = data_access.get_state(conn, data_access.ACTIVE_MODEL_KEY) or (run["manifest"]["model_version"] if run else None)
_, role = current_user()
admin = role == "ADMIN"

if versions.empty:
    st.info("No trained models yet. Run the pipeline's train step first.")
    st.stop()

table = pd.DataFrame({
    "Version": versions["version"] + versions["version"].map(lambda v: "  ● active" if v == active else ""),
    "Created": versions["created_at"].str.slice(0, 16).str.replace("T", " "),
    "By": versions["created_by"], "Parent": versions["parent_version"].fillna("—"),
    "Feedback rows": versions["n_feedback_used"],
    "Validation PR-AUC": versions["val_pr_auc"].round(3),
    "Validation precision@50": versions["val_precision_at_k"].round(2),
    "Use": versions["eval_valid"].map({1: "evaluation-safe", 0: "demo only"}),
})
st.dataframe(table, hide_index=True)
st.caption("Validation metrics are from the validation period (synthetic data). \"Demo only\" versions used "
           "feedback from test, demo or upload runs and never appear in reported results.")

if not admin:
    st.info("Only Admin can manage models. Pick Admin in the sidebar (role selection, not security).")
    st.stop()

st.subheader("Retrain with analyst feedback")
summary = rt.feedback_summary(rt.usable_feedback(conn))
kind = "evaluation-safe" if summary["eval_safe"] else "demo only"
outcome = (f"The new version will be **{kind}**." if summary["confirmed"] + summary["false_positive"]
           else "Nothing to retrain with yet: record Confirmed mule / False positive decisions on accounts first.")
st.write(f"Usable feedback: **{summary['confirmed']} Confirmed mule, {summary['false_positive']} False positive**; "
         f"{summary['ignored']} Needs investigation ignored. {outcome}")
if st.button("Retrain", type="primary", disabled=summary["confirmed"] + summary["false_positive"] == 0):
    with st.spinner("Retraining (about a minute: features are already saved, only the model is refitted)…"):
        made = rt.retrain(run["manifest"]["dataset"] if run and run["manifest"]["dataset"] != "demo" else "ibm_hi_small",
                          created_by=current_user()[0], parent=active)
    st.session_state["models_last_trained"] = made["version"]
    st.rerun()

last = st.session_state.get("models_last_trained")
if last and active and last != active:
    pair = versions.set_index("version").reindex([active, last])
    st.write(f"**{last}** compared with the active **{active}** (validation period):")
    st.dataframe(pd.DataFrame({"Version": [active, last], "PR-AUC": pair["val_pr_auc"].round(3).tolist(),
                               "Precision@50": pair["val_precision_at_k"].round(2).tolist()}), hide_index=True)

st.subheader("Activate a version")
choices = [v for v in versions["version"] if v != active]
if choices:
    pick = st.selectbox("Version", choices)
    st.caption("Re-scores the test period with this version (as a new run …__" + pick + "), cuts the demo run "
               "again and selects it. Switching back is the same action.")
    if st.button(f"Activate {pick}"):
        from trace_aml.pipeline.demo import recut_demo_run
        from trace_aml.pipeline.score import run_score
        with st.spinner(f"Scoring the test period with {pick}…"):
            dataset = "ibm_hi_small"
            if not (path("runs") / f"{dataset}__test__{pick}" / "scores.parquet").exists():
                run_score(dataset, "TEST", pick)
            demo_folder = recut_demo_run(dataset, pick) if (path("demo") / "demo_account_map.csv").exists() else None
        data_access.set_state(conn, data_access.ACTIVE_MODEL_KEY, pick)
        new_run = json.loads(((demo_folder or path("runs") / f"{dataset}__test__{pick}") / "manifest.json")
                             .read_text(encoding="utf-8"))["run_id"]
        data_access.set_state(conn, data_access.ACTIVE_RUN_KEY, new_run)
        st.cache_resource.clear()
        navigation.clear(st.session_state)
        st.session_state.pop("run_choice", None)
        st.rerun()
