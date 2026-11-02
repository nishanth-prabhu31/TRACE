"""TRACE data contract: the exact columns of every file, written so Python can check them.

This file must match docs/04-CONTRACT.md. Change both together, or neither.
"""

from dataclasses import dataclass

CONTRACT_VERSION = 1


@dataclass(frozen=True)
class Column:
    """One column in a file: its name, its type, and what values are allowed."""

    name: str
    dtype: str  # the pandas type writers must use, e.g. "int64", "string", "datetime64[ns]"
    nullable: bool = False  # True = empty values are allowed
    allowed: tuple[str, ...] | None = None  # for text columns with a fixed list of values
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class FileSpec:
    """One file: its name, its columns in order, and the column(s) that must be unique."""

    filename: str
    columns: tuple[Column, ...]
    key: tuple[str, ...]
    extra_float_columns: bool = False  # True = any further columns must be decimals (feature columns)


# --- Allowed values (enums) ---------------------------------------------------

ROLES = ("MULE", "SOURCE", "SINK", "NORMAL")

PATTERN_TYPES = (
    "FAN_OUT", "FAN_IN", "GATHER_SCATTER", "SCATTER_GATHER",
    "CYCLE", "RANDOM", "BIPARTITE", "STACK", "OTHER",
)

RING_SOURCES = ("PATTERN_FILE", "CONNECTED")

RULES = ("CHAIN", "FAN_IN", "FAN_OUT", "CYCLE", "PASS_THROUGH")

VERDICTS = ("LIKELY_MULE", "WATCH", "NORMAL")

DIRECTIONS = ("UP", "DOWN")

TRAIL_SIDES = ("BEFORE", "AFTER")

RUN_KINDS = ("SPLIT", "UPLOAD", "DEMO", "REPLAY")

SPLITS = ("TRAIN", "VALIDATION", "TEST")

DECISIONS = ("CONFIRMED_MULE", "FALSE_POSITIVE", "NEEDS_INVESTIGATION")

ANALYST_ROLES = ("ANALYST", "ADMIN")

STATUSES = ("PASS", "WARN", "FAIL")


# --- Processed dataset files (docs/04-CONTRACT.md section 2) -------------------

TRANSACTIONS = FileSpec(
    filename="transactions.parquet",
    columns=(
        Column("txn_id", "int64", min_value=0),
        Column("timestamp", "datetime64[ns]"),
        Column("sender_id", "string"),
        Column("receiver_id", "string"),
        Column("amount", "float64", min_value=0),
        Column("amount_original", "float64", min_value=0),
        Column("currency_original", "string"),
        Column("channel", "string"),
        Column("is_self", "bool"),
        Column("dataset", "string"),
    ),
    key=("txn_id",),
)

LABELS_TXN = FileSpec(
    filename="labels_txn.parquet",
    columns=(
        Column("txn_id", "int64", min_value=0),
        Column("is_laundering", "int8", min_value=0, max_value=1),
        Column("pattern_type", "string", nullable=True, allowed=PATTERN_TYPES),
        Column("pattern_id", "string", nullable=True),
    ),
    key=("txn_id",),
)

ACCOUNT_LABELS = FileSpec(
    filename="account_labels.parquet",
    columns=(
        Column("account_id", "string"),
        Column("role", "string", allowed=ROLES),
        Column("is_mule", "Int8", nullable=True, min_value=0, max_value=1),
        Column("mule_since", "datetime64[ns]", nullable=True),
        Column("ring_id", "string", nullable=True),
        Column("ring_source", "string", nullable=True, allowed=RING_SOURCES),
    ),
    key=("account_id",),
)


# --- Feature files (docs/04-CONTRACT.md section 4) -----------------------------

FEATURES = FileSpec(
    filename="<cutoff>.parquet",
    columns=(
        Column("account_id", "string"),
        Column("cutoff", "datetime64[ns]"),
    ),
    key=("account_id",),
    extra_float_columns=True,
)

RULE_FLAGS = FileSpec(
    filename="<cutoff>_rules.parquet",
    columns=(
        Column("account_id", "string"),
        Column("cutoff", "datetime64[ns]"),
        Column("rule_chain", "int8", min_value=0, max_value=1),
        Column("rule_fan_in", "int8", min_value=0, max_value=1),
        Column("rule_fan_out", "int8", min_value=0, max_value=1),
        Column("rule_cycle", "int8", min_value=0, max_value=1),
        Column("rule_pass_through", "int8", min_value=0, max_value=1),
        Column("rule_score", "float32", min_value=0, max_value=1),
    ),
    key=("account_id",),
)

RULE_EVIDENCE = FileSpec(
    filename="<cutoff>_rules_evidence.parquet",
    columns=(
        Column("account_id", "string"),
        Column("rule", "string", allowed=RULES),
        Column("txn_id", "int64", min_value=0),
    ),
    key=("account_id", "rule", "txn_id"),
)


# --- Run files: what the app reads (docs/04-CONTRACT.md section 5) -------------

SCORES = FileSpec(
    filename="scores.parquet",
    columns=(
        Column("account_id", "string"),
        Column("risk", "float32", min_value=0, max_value=100),
        Column("prob", "float32", min_value=0, max_value=1),
        Column("verdict", "string", allowed=VERDICTS),
        Column("rank", "int32", min_value=1),
        Column("short_label", "string"),
        Column("summary", "string"),
        Column("has_trail", "bool"),
        Column("in_amount", "float64", min_value=0),
        Column("out_amount", "float64", min_value=0),
        Column("n_txn", "int32", min_value=0),
        Column("first_seen", "datetime64[ns]"),
    ),
    key=("account_id",),
)

REASONS = FileSpec(
    filename="reasons.parquet",
    columns=(
        Column("account_id", "string"),
        Column("position", "int8", min_value=1, max_value=5),
        Column("feature", "string"),
        Column("display_name", "string"),
        Column("value", "float32", nullable=True),
        Column("value_text", "string"),
        Column("impact", "float32"),
        Column("direction", "string", allowed=DIRECTIONS),
    ),
    key=("account_id", "position"),
)

TRAILS = FileSpec(
    filename="trails.parquet",
    columns=(
        Column("account_id", "string"),
        Column("trail_id", "int8", min_value=1, max_value=3),
        Column("hop", "int8", min_value=1),
        Column("txn_id", "int64", min_value=0),
        Column("from_id", "string"),
        Column("to_id", "string"),
        Column("timestamp", "datetime64[ns]"),
        Column("amount", "float64", min_value=0),
        Column("side", "string", allowed=TRAIL_SIDES),
    ),
    key=("account_id", "trail_id", "hop"),
)

RINGS = FileSpec(
    filename="rings.parquet",
    columns=(
        Column("ring_id", "string"),
        Column("account_id", "string"),
        Column("ring_size", "int32", min_value=2),
        Column("ring_amount", "float64", min_value=0),
        Column("first_txn", "datetime64[ns]"),
        Column("last_txn", "datetime64[ns]"),
        Column("max_risk", "float32", min_value=0, max_value=100),
        Column("mean_risk", "float32", min_value=0, max_value=100),
    ),
    key=("ring_id", "account_id"),
)

RUN_FEATURES = FileSpec(
    filename="features.parquet",
    columns=(
        Column("account_id", "string"),
        Column("cutoff", "datetime64[ns]"),
        Column("snapshot", "datetime64[ns]"),
    ),
    key=("account_id",),
    extra_float_columns=True,
)

TRANSACTIONS_VIEW = FileSpec(
    filename="transactions_view.parquet",
    columns=TRANSACTIONS.columns,
    key=("txn_id",),
)

RUN_REQUIRED = (SCORES, REASONS, TRAILS, RUN_FEATURES, TRANSACTIONS_VIEW)
RUN_OPTIONAL = (RINGS,)

MANIFEST_REQUIRED_KEYS = (
    "contract_version", "run_id", "dataset", "kind", "split", "cutoff",
    "model_version", "created_at", "n_accounts", "n_transactions", "verdict_counts",
)


# --- Model folder (docs/04-CONTRACT.md section 7) ------------------------------

MODEL_FILES = ("model.txt", "calibrator.joblib", "features.json", "thresholds.json", "meta.json")


# --- SQLite database trace.db (docs/04-CONTRACT.md section 6) ------------------

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
  feedback_id      INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id       TEXT NOT NULL,
  run_id           TEXT NOT NULL,
  model_version    TEXT NOT NULL,
  decision         TEXT NOT NULL CHECK (decision IN ('CONFIRMED_MULE','FALSE_POSITIVE','NEEDS_INVESTIGATION')),
  analyst_name     TEXT NOT NULL,
  analyst_role     TEXT NOT NULL CHECK (analyst_role IN ('ANALYST','ADMIN')),
  risk_at_decision REAL NOT NULL,
  note             TEXT,
  created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_account ON feedback(account_id);

CREATE TABLE IF NOT EXISTS model_versions (
  version            TEXT PRIMARY KEY,
  created_at         TEXT NOT NULL,
  created_by         TEXT NOT NULL,
  parent_version     TEXT,
  n_feedback_used    INTEGER NOT NULL DEFAULT 0,
  eval_valid         INTEGER NOT NULL DEFAULT 1,
  val_pr_auc         REAL,
  val_precision_at_k REAL,
  k                  INTEGER,
  path               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_state (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

SQLITE_TABLES = {
    "feedback": (
        "feedback_id", "account_id", "run_id", "model_version", "decision", "analyst_name",
        "analyst_role", "risk_at_decision", "note", "created_at",
    ),
    "model_versions": (
        "version", "created_at", "created_by", "parent_version", "n_feedback_used",
        "eval_valid", "val_pr_auc", "val_precision_at_k", "k", "path",
    ),
    "app_state": ("key", "value"),
}
