"""Draw the network around an account and report which account was clicked.

Uses the interactive graph component when it is installed. If it is missing or
fails, a static drawing is shown instead and the page's "Go to account" dropdown
does the hopping (docs/06-DASHBOARD.md 5.2).
"""

import json
import math

import pandas as pd
import streamlit as st

from app.logic.formatting import VERDICT_TEXT, format_money
from trace_aml.config import load_settings

try:
    from streamlit_agraph import Config, Edge, Node, _agraph
    HAS_INTERACTIVE_GRAPH = True
except Exception:  # the component is optional; the fallback below covers its absence
    HAS_INTERACTIVE_GRAPH = False

NO_SCORE_COLOUR = "#B0B7BF"
EDGE_COLOUR = "#8A939C"
DOT_SHAPES = {"triangle": "triangle", "diamond": "diamond", "dot": "ellipse", "box": "box"}


def edge_width(amount: float, smallest: float, largest: float, limits: list[int]) -> float:
    """Arrow thickness on a log scale between limits[0] and limits[1] pixels."""
    low, high = limits
    if largest <= smallest or amount <= 0:
        return float(low)
    position = (math.log(amount) - math.log(smallest)) / (math.log(largest) - math.log(smallest))
    return round(low + position * (high - low), 2)


def node_style(row: pd.Series, app: dict) -> tuple[str, str]:
    """(colour, shape) for a node: by verdict, and never by colour alone."""
    if not row["has_score"]:
        return NO_SCORE_COLOUR, "box"
    return app["verdict_colours"][row["verdict"]], app["verdict_shapes"][row["verdict"]]


def node_hover(row: pd.Series, currency: str) -> str:
    """Hover text for a node."""
    if not row["has_score"]:
        return f"{row['account_id']}\nNo score in this run"
    return (f"{row['account_id']}\nRisk {row['risk']:.1f} · {VERDICT_TEXT[row['verdict']]}\n"
            f"{row['short_label']}\n{int(row['n_txn'])} transactions\n"
            f"In {format_money(row['in_amount'], currency, short=True)} · "
            f"Out {format_money(row['out_amount'], currency, short=True)}")


def edge_label(row: pd.Series, currency: str) -> str:
    """Text on a merged arrow, e.g. '4 txns · ₹1.8 L'."""
    count = int(row["n_txn"])
    return f"{count} txn{'s' if count != 1 else ''} · {format_money(row['amount'], currency, short=True)}"


def edge_hover(row: pd.Series, currency: str) -> str:
    """Hover text for an arrow: its first few transfers."""
    lines = [f"{item['timestamp']:%d %b %H:%M}  {format_money(item['amount'], currency)}" for item in row["sample"]]
    extra = int(row["n_txn"]) - len(row["sample"])
    return "\n".join(lines + ([f"+{extra} more"] if extra > 0 else []))


def _interactive(nodes: pd.DataFrame, edges: pd.DataFrame, currency: str, key: str) -> str | None:
    app = load_settings()["app"]
    amounts = edges["amount"] if len(edges) else pd.Series([1.0])
    drawn_nodes = []
    for _, row in nodes.iterrows():
        colour, shape = node_style(row, app)
        label = row["account_id"] + (f"\n{row['short_label']}" if row["is_centre"] and row["has_score"] else "")
        drawn_nodes.append(Node(
            id=row["account_id"], label=label, title=node_hover(row, currency), shape=shape,
            size=app["node_size_px"][1] if row["is_centre"] else app["node_size_px"][0],
            color=colour, borderWidth=4 if row["is_centre"] else 1,
            font={"size": 13}))
    drawn_edges = [Edge(
        source=row["sender_id"], target=row["receiver_id"], label=edge_label(row, currency),
        title=edge_hover(row, currency), color=EDGE_COLOUR,
        width=edge_width(row["amount"], amounts.min(), amounts.max(), app["edge_width_px"]),
        font={"size": 11, "align": "top"}, smooth={"type": "curvedCW", "roundness": 0.15})
        for _, row in edges.iterrows()]
    config = Config(height=app["graph_height_px"], width=1200, directed=True, physics=True)
    config.width = "100%"
    data = json.dumps({"nodes": [n.to_dict() for n in drawn_nodes], "edges": [e.to_dict() for e in drawn_edges]})
    options = {name: value for name, value in config.__dict__.items() if value is not None}
    return _agraph(data=data, config=json.dumps(options), key=key)


def _static(nodes: pd.DataFrame, edges: pd.DataFrame, currency: str) -> None:
    app = load_settings()["app"]
    lines = ["digraph {", "  rankdir=LR; node [style=filled, fontname=Helvetica, fontsize=10];",
             "  edge [fontname=Helvetica, fontsize=9, color=\"#8A939C\"];"]
    amounts = edges["amount"] if len(edges) else pd.Series([1.0])
    for _, row in nodes.iterrows():
        colour, shape = node_style(row, app)
        lines.append(f'  "{row["account_id"]}" [fillcolor="{colour}", fontcolor=white, '
                     f'shape={DOT_SHAPES[shape]}, penwidth={4 if row["is_centre"] else 1}];')
    for _, row in edges.iterrows():
        width = edge_width(row["amount"], amounts.min(), amounts.max(), app["edge_width_px"])
        lines.append(f'  "{row["sender_id"]}" -> "{row["receiver_id"]}" '
                     f'[label="{edge_label(row, currency)}", penwidth={width}];')
    st.graphviz_chart("\n".join(lines + ["}"]))


def render_network(nodes: pd.DataFrame, edges: pd.DataFrame, currency: str, key: str) -> str | None:
    """Draw the network. Returns the clicked account ID, or None if nothing was clicked."""
    if HAS_INTERACTIVE_GRAPH:
        try:
            return _interactive(nodes, edges, currency, key)
        except Exception as error:
            st.caption(f"Interactive graph unavailable ({error}); showing a static drawing.")
    _static(nodes, edges, currency)
    return None


def render_legend() -> None:
    """Colours and shapes, and how to read and use the graph."""
    app = load_settings()["app"]
    symbols = {"triangle": "▲", "diamond": "◆", "dot": "●"}
    parts = [f"<span style='color:{app['verdict_colours'][v]}'>{symbols[app['verdict_shapes'][v]]}</span> {VERDICT_TEXT[v]}"
             for v in ("LIKELY_MULE", "WATCH", "NORMAL")]
    parts.append(f"<span style='color:{NO_SCORE_COLOUR}'>■</span> No score in this run")
    st.markdown(" &nbsp;·&nbsp; ".join(parts) + " &nbsp;·&nbsp; arrow width = money &nbsp;·&nbsp; click an account to hop",
                unsafe_allow_html=True)
