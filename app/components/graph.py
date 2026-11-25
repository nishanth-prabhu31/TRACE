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
TRAIL_COLOUR = "#A23B2A"                 # the suspicious money path
FADED_EDGE = "rgba(138,147,156,0.45)"    # context transfers
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


def _faded(hex_colour: str) -> str:
    """The same colour at low opacity, for accounts that are only context."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},0.35)"


def flow_columns(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, int]:
    """Column for each account so money reads left to right.

    With a centre account: senders to it go left (-1, -2 for their senders...), receivers right.
    Without one (a ring): mostly-sending accounts left, mostly-receiving right, the rest middle.
    The layout is computed here because automatic layered layouts break on money cycles."""
    ids = nodes["account_id"].tolist()
    centre = nodes.loc[nodes["is_centre"], "account_id"].tolist()
    if centre:
        column, frontier = {centre[0]: 0}, [centre[0]]
        while frontier:
            nxt = []
            for account in frontier:
                for _, e in edges[edges["sender_id"] == account].iterrows():
                    if e["receiver_id"] not in column:
                        column[e["receiver_id"]] = column[account] + 1
                        nxt.append(e["receiver_id"])
                for _, e in edges[edges["receiver_id"] == account].iterrows():
                    if e["sender_id"] not in column:
                        column[e["sender_id"]] = column[account] - 1
                        nxt.append(e["sender_id"])
            frontier = nxt
        return {account: column.get(account, 0) for account in ids}
    sent = edges.groupby("sender_id")["amount"].sum()
    got = edges.groupby("receiver_id")["amount"].sum()
    out = {}
    for account in ids:
        s, g = float(sent.get(account, 0.0)), float(got.get(account, 0.0))
        share = s / (s + g) if s + g else 0.5
        out[account] = -1 if share > 0.65 else (1 if share < 0.35 else 0)
    return out


COLUMN_GAP, ROW_GAP = 230, 120


def _interactive(nodes: pd.DataFrame, edges: pd.DataFrame, currency: str, key: str) -> str | None:
    """Money flows left to right (senders left, receivers right). The account, its trail and
    flagged accounts are drawn strong; plain context accounts are faded. Arrow text is on hover."""
    app = load_settings()["app"]
    amounts = edges["amount"] if len(edges) else pd.Series([1.0])
    focus = set(nodes.loc[nodes["is_centre"] | nodes["on_trail"], "account_id"])
    columns = flow_columns(nodes, edges)
    rows_in = {}
    for account in sorted(columns, key=lambda a: (columns[a], a)):
        rows_in.setdefault(columns[account], []).append(account)
    position = {a: (col * COLUMN_GAP, (i - (len(group) - 1) / 2) * ROW_GAP)
                for col, group in rows_in.items() for i, a in enumerate(group)}
    drawn_nodes = []
    for _, row in nodes.iterrows():
        colour, shape = node_style(row, app)
        strong = row["is_centre"] or row["on_trail"] or row.get("verdict") in ("LIKELY_MULE", "WATCH")
        x, y = position[row["account_id"]]
        drawn_nodes.append(Node(
            id=row["account_id"], label=row["account_id"], title=node_hover(row, currency), shape=shape,
            size=app["node_size_px"][1] if row["is_centre"] else app["node_size_px"][0],
            color=colour if strong else _faded(colour), borderWidth=4 if row["is_centre"] else 1,
            x=x, y=y,
            font={"size": 22 if row["is_centre"] else 17, "color": "#1A2433" if strong else "#8A94A3",
                  "strokeWidth": 4, "strokeColor": "#FFFFFF"}))
    drawn_edges = []
    for _, row in edges.iterrows():
        on_trail = row["sender_id"] in focus and row["receiver_id"] in focus and len(focus) > 1
        drawn_edges.append(Edge(
            source=row["sender_id"], target=row["receiver_id"], label="",
            title=edge_label(row, currency) + "\n" + edge_hover(row, currency),
            color=TRAIL_COLOUR if on_trail else FADED_EDGE,
            width=edge_width(row["amount"], amounts.min(), amounts.max(), app["edge_width_px"]),
            smooth={"type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.4}))
    config = Config(height=app["graph_height_px"], width=1200, directed=True, physics=False,
                    interaction={"hover": True, "tooltipDelay": 120})
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
    parts.append(f"<span style='color:{TRAIL_COLOUR}'>━</span> money trail")
    st.caption(" &nbsp;·&nbsp; ".join(parts) + " &nbsp;·&nbsp; money flows left → right, thicker = more "
               "&nbsp;·&nbsp; hover an arrow for amounts &nbsp;·&nbsp; click an account to open it",
               unsafe_allow_html=True)
