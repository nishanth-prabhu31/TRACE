"""Save and read analyst decisions (F-10, docs/04-CONTRACT.md section 6).

TRACE only records what the analyst decided. Nothing here acts on an account.
"""

import sqlite3
from datetime import datetime, timezone

import pandas as pd

from trace_aml import contract as c

FEEDBACK_COLUMNS = list(c.SQLITE_TABLES["feedback"])


def save_decision(conn: sqlite3.Connection, account_id: str, run_id: str, model_version: str,
                  decision: str, analyst_name: str, analyst_role: str, risk_at_decision: float,
                  note: str | None = None) -> int:
    """Add one decision row and return its id. Earlier rows are kept, never changed."""
    if decision not in c.DECISIONS:
        raise ValueError(f"Unknown decision {decision!r}; allowed: {c.DECISIONS}")
    if analyst_role not in c.ANALYST_ROLES:
        raise ValueError(f"Unknown role {analyst_role!r}; allowed: {c.ANALYST_ROLES}")
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cursor = conn.execute(
        "INSERT INTO feedback (account_id, run_id, model_version, decision, analyst_name, analyst_role,"
        " risk_at_decision, note, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (account_id, run_id, model_version, decision, analyst_name, analyst_role,
         float(risk_at_decision), (note or "").strip() or None, created_at))
    conn.commit()
    return int(cursor.lastrowid)


def decision_history(conn: sqlite3.Connection, account_id: str) -> pd.DataFrame:
    """Every decision for one account, newest first."""
    return pd.read_sql_query(
        "SELECT * FROM feedback WHERE account_id = ? ORDER BY feedback_id DESC", conn, params=(account_id,))


def current_decisions(conn: sqlite3.Connection) -> pd.DataFrame:
    """The latest decision per account: account_id, decision, analyst_name, created_at."""
    return pd.read_sql_query(
        "SELECT account_id, decision, analyst_name, created_at FROM feedback "
        "WHERE feedback_id IN (SELECT MAX(feedback_id) FROM feedback GROUP BY account_id)", conn)
