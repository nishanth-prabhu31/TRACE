"""Results: how well TRACE works, from the saved evaluation reports (synthetic IBM data)."""

import json

import altair as alt
import pandas as pd
import streamlit as st

from app.components import charts
from trace_aml.config import load_settings, path

NAMES = {"LIGHTGBM": "TRACE model (LightGBM)", "RANDOM_FOREST": "Random forest", "TXN_ONLY": "Transaction-only model",
         "LOGREG": "Logistic regression", "ISOLATION_FOREST": "Isolation forest (no labels)", "RULES": "Rules only"}


@st.cache_data(max_entries=8)
def report(name: str) -> dict | None:
    file = path("reports") / name
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


version = load_settings()["model"]["version"]
st.title("Results")
st.caption(f"Model {version}, IBM HI-Small test period (synthetic data). Every number here was measured on "
           "data the model never trained on. Mean over 5 seeds where stated.")

evaluation = report(f"eval_{version}.json")
if evaluation is None:
    st.info("No evaluation report yet. Run the pipeline's evaluate step.")
    st.stop()

test = {name: values["TEST"] for name, values in evaluation["approaches"].items()}
n_mules = int(test["LIGHTGBM"]["n_mules"]["mean"])
n_accounts = int(test["LIGHTGBM"]["n_accounts"]["mean"])

kpi = st.container(horizontal=True)
kpi.metric("Hidden mules in the test period", f"{n_mules}", border=True)
kpi.metric("Accounts searched", f"{n_accounts:,}", border=True)
kpi.metric("Real mules in TRACE's top 50", f"{test['LIGHTGBM']['precision_at_50']['mean'] * 50:.0f} of 50", border=True)
kpi.metric("Same 50 picked at random", f"{n_mules / n_accounts * 50:.2f} of 50", border=True)

row = st.columns(2)
with row[0].container(border=True):
    st.markdown("**Of the 50 accounts each approach ranks highest, how many are real mules?**")
    data = pd.DataFrame({"Approach": [NAMES[k] for k in test],
                         "Real mules in top 50": [v["precision_at_50"]["mean"] * 50 for v in test.values()]})
    st.altair_chart(charts.labelled_bars(data, "Real mules in top 50", "Approach", "Real mules (of 50)",
                                         highlight=NAMES["LIGHTGBM"], fmt=".0f"))
with row[1].container(border=True):
    st.markdown("**Ranking quality over all accounts (PR-AUC, higher is better)**")
    data = pd.DataFrame({"Approach": [NAMES[k] for k in test], "PR-AUC": [v["pr_auc"]["mean"] for v in test.values()]})
    st.altair_chart(charts.labelled_bars(data, "PR-AUC", "Approach", "PR-AUC on the test period",
                                         highlight=NAMES["LIGHTGBM"], fmt=".3f"))
    st.caption(f"Random guessing scores about {n_mules / n_accounts:.4f}.")

replay = report(f"replay_{version}.json")
if replay:
    with st.container(border=True):
        st.markdown("**Was the ring flagged before the money was cashed out?** (hour-by-hour replay)")
        strict = replay["strict_caught_after_start_before_cashout"]
        parts = pd.DataFrame({
            "Outcome": ["Before cash-out, after the ring started", "Before cash-out, account already flagged earlier",
                        "Only after cash-out", "Never flagged"],
            "Rings": [strict, replay["alert_before_ring_start"], replay["caught_after_cashout"], replay["never_caught"]],
            "order": [0, 1, 2, 3]})
        bar = alt.Chart(parts).mark_bar(height=34).encode(
            x=alt.X("Rings:Q", stack="zero", title=f"{replay['rings']} complete laundering rings"),
            color=alt.Color("Outcome:N", sort=parts["Outcome"].tolist(), legend=alt.Legend(orient="bottom", title=None, columns=2),
                            scale=alt.Scale(range=["#2E7D5B", "#7FB89A", "#B7791F", "#B0B7BF"])),
            order="order:Q", tooltip=["Outcome", "Rings"]).properties(height=110)
        st.altair_chart(bar)
        st.caption(f"Strictly counted, {strict} of {replay['rings']} rings ({replay['strict_share']:.0%}) were flagged in time, "
                   f"a median {replay['strict_lead_minutes_median'] / 60:.0f} hours before cash-out. "
                   f"Lead times are accurate to about {replay['step_hours']} h.")

row = st.columns(2)
feedback = report(f"feedback_sim_{version}.json")
if feedback:
    with row[0].container(border=True):
        st.markdown("**Does analyst feedback help?** (simulated analyst, 50 reviews per round)")
        data = pd.DataFrame(feedback["results"])
        line = alt.Chart(data).mark_line(point=True, color=charts.ACCENT).encode(
            x=alt.X("feedback_rows:Q", title="Reviewed accounts fed back"),
            y=alt.Y("pr_auc:Q", title="Test PR-AUC", scale=alt.Scale(zero=False)),
            tooltip=["round", "feedback_rows", alt.Tooltip("pr_auc:Q", format=".3f")]).properties(height=240)
        st.altair_chart(line)
        st.caption("A modest gain, measured on one seed.")
stress = report(f"stress_{version}.json")
if stress:
    with row[1].container(border=True):
        st.markdown("**Fast, small UPI-style rings** (injected stress test, never trained on)")
        rows = [{"Mules": variant.title(), "Detector": label, "Caught": stress[variant][key]["recall"]}
                for variant in ("FRESH", "AGED")
                for key, label in (("model_watch", "Model"), ("rules_any", "Rules"), ("either", "Model or rules"))]
        data = pd.DataFrame(rows)
        detectors = ["Model", "Rules", "Model or rules"]
        data["Bar"] = data["Mules"] + " · " + data["Detector"]
        order = [f"{m} · {d}" for m in ("Fresh", "Aged") for d in detectors]
        base = alt.Chart(data).encode(
            y=alt.Y("Bar:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            x=alt.X("Caught:Q", axis=alt.Axis(format="%"), title="Injected mules caught", scale=alt.Scale(domain=[0, 1.2])))
        bars = (base.mark_bar().encode(
                    color=alt.Color("Detector:N", legend=None,
                                    scale=alt.Scale(domain=detectors, range=[charts.NEUTRAL, charts.ACCENT, "#8A94A3"])),
                    tooltip=["Mules", "Detector", alt.Tooltip("Caught:Q", format=".0%")])
                + base.mark_text(align="left", dx=4, fontSize=12).encode(text=alt.Text("Caught:Q", format=".0%"))
                ).properties(height=210)
        st.altair_chart(bars)
        st.caption(f"The IBM-trained model misses them; the rules catch them but also flag about "
                   f"{stress['FRESH']['rules_any']['real_accounts_also_flagged']:,} real accounts. "
                   "Real UPI data would need retraining.")
