"""Charts for the Overview and Results pages. Plain Altair (Vega-Lite); each chart answers one question."""

import altair as alt
import pandas as pd

from app.logic.formatting import VERDICT_TEXT
from trace_aml.config import load_settings

ORDER = ["LIKELY_MULE", "WATCH", "NORMAL"]
NEUTRAL = "#5A6B7F"
ACCENT = "#A23B2A"


def _colours() -> alt.Scale:
    colours = load_settings()["app"]["verdict_colours"]
    return alt.Scale(domain=[VERDICT_TEXT[v] for v in ORDER], range=[colours[v] for v in ORDER])


def verdict_donut(scores: pd.DataFrame) -> alt.Chart:
    """How many accounts got each verdict."""
    counts = scores["verdict"].value_counts().reindex(ORDER, fill_value=0)
    data = pd.DataFrame({"Verdict": [VERDICT_TEXT[v] for v in ORDER], "Accounts": counts.to_numpy()})
    data["Share"] = data["Accounts"] / max(int(data["Accounts"].sum()), 1)
    base = alt.Chart(data).encode(
        theta=alt.Theta("Accounts:Q", stack=True),
        color=alt.Color("Verdict:N", scale=_colours(), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Verdict", alt.Tooltip("Accounts:Q", format=","), alt.Tooltip("Share:Q", format=".1%")])
    return base.mark_arc(innerRadius=62, outerRadius=105, stroke="white", strokeWidth=2).properties(height=260)


def risk_histogram(scores: pd.DataFrame, cuts: dict) -> alt.LayerChart:
    """Risk scores of all accounts (log count): most sit near 0, a few stand out. Lines = verdict cut-offs."""
    bins = alt.Bin(extent=[0, 100], step=2)
    bars = alt.Chart(scores[["risk"]]).mark_bar(color=NEUTRAL).encode(
        x=alt.X("risk:Q", bin=bins, title="Risk score (0–100)"),
        y=alt.Y("count():Q", scale=alt.Scale(type="symlog"), title="Accounts (log scale)"),
        tooltip=[alt.Tooltip("risk:Q", bin=bins, title="Risk"), alt.Tooltip("count():Q", title="Accounts", format=",")])
    lines = pd.DataFrame({"risk": [cuts["watch_cut"], cuts["likely_mule_cut"]],
                          "label": [f"Watch from {cuts['watch_cut']:g}", f"Likely mule from {cuts['likely_mule_cut']:g}"]})
    rules = alt.Chart(lines).mark_rule(color=ACCENT, strokeDash=[4, 3]).encode(x="risk:Q", tooltip=["label"])
    text = alt.Chart(lines).mark_text(align="left", dx=4, dy=-6, color=ACCENT, fontSize=11).encode(
        x="risk:Q", y=alt.value(10), text="label")
    return (bars + rules + text).properties(height=260)


def count_bars(values: pd.Series, title: str, colour: str = NEUTRAL, top: int = 8) -> alt.Chart:
    """Horizontal bars for the most common values, largest at the top."""
    counts = values.value_counts().head(top)
    data = pd.DataFrame({"Item": counts.index.astype(str), "Accounts": counts.to_numpy()})
    return alt.Chart(data).mark_bar(color=colour).encode(
        x=alt.X("Accounts:Q", title=title, axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Item:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        tooltip=["Item", "Accounts"]).properties(height=max(120, 30 * len(data)))


def labelled_bars(data: pd.DataFrame, x: str, y: str, x_title: str, highlight: str | None = None,
                  fmt: str = ".2f") -> alt.LayerChart:
    """Horizontal bars with the value written at the end; `highlight` row drawn in the accent colour."""
    data = data.assign(_hl=data[y] == highlight)
    base = alt.Chart(data).encode(y=alt.Y(f"{y}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
                                  x=alt.X(f"{x}:Q", title=x_title))
    bars = base.mark_bar().encode(color=alt.condition("datum._hl", alt.value(ACCENT), alt.value(NEUTRAL)),
                                  tooltip=[y, alt.Tooltip(f"{x}:Q", format=fmt)])
    text = base.mark_text(align="left", dx=4, fontSize=12).encode(text=alt.Text(f"{x}:Q", format=fmt))
    return (bars + text).properties(height=max(140, 34 * len(data)))
