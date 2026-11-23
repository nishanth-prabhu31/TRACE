"""Read run folders and the SQLite database for the app (docs/04-CONTRACT.md sections 5-6).

No Streamlit imports here: pages add caching on top, and the same functions could
sit behind an API later (docs/03-ARCHITECTURE.md section 12).
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path

ACTIVE_RUN_KEY = "active_run_id"
ACTIVE_MODEL_KEY = "active_model_version"


def run_roots() -> list[Path]:
    """Folders that hold runs: real runs first, then the fake runs used to build the app."""
    return [path("runs"), path("fake") / "runs"]


def list_runs(roots: list[Path] | None = None) -> list[dict]:
    """Every complete run found, newest first. Each item is its manifest plus its folder."""
    runs = []
    for root in roots if roots is not None else run_roots():
        if not root.exists():
            continue
        for folder in root.iterdir():
            manifest_file = folder / "manifest.json"
            if not manifest_file.exists() or not (folder / c.SCORES.filename).exists():
                continue
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            runs.append({**manifest, "folder": str(folder)})
    return sorted(runs, key=lambda run: run.get("created_at", ""), reverse=True)


def run_label(run: dict) -> str:
    """Sidebar text for a run, e.g. 'ibm_hi_small · test · v1'."""
    kind = (run.get("split") or run.get("kind") or "").lower()
    return f"{run.get('dataset', '?')} · {kind} · {run.get('model_version', '?')}"


def load_run(folder: str | Path) -> dict:
    """All files of one run. 'rings' is None when the run has no rings file."""
    folder = Path(folder)
    run = {
        "manifest": json.loads((folder / "manifest.json").read_text(encoding="utf-8")),
        "scores": pd.read_parquet(folder / c.SCORES.filename),
        "reasons": pd.read_parquet(folder / c.REASONS.filename),
        "trails": pd.read_parquet(folder / c.TRAILS.filename),
        "transactions": pd.read_parquet(folder / c.TRANSACTIONS_VIEW.filename),
        "rings": None,
    }
    if (folder / c.RINGS.filename).exists():
        run["rings"] = pd.read_parquet(folder / c.RINGS.filename)
    return run


# --- SQLite ---------------------------------------------------------------------

def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open trace.db, creating the file and its tables if they are missing."""
    conn = sqlite3.connect(db_path if db_path is not None else path("db"))
    conn.executescript(c.SQLITE_SCHEMA)
    return conn


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    """A value from app_state, or None if it was never set."""
    row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Save a value in app_state (insert or replace)."""
    conn.execute("INSERT INTO app_state (key, value) VALUES (?, ?) "
                 "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
    conn.commit()


def analysts() -> list[dict]:
    """Names and roles for the sidebar picker (settings, not the database)."""
    return list(load_settings()["analysts"])
