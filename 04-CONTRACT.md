# TRACE — Data Contract

| | |
|---|---|
| Doc | 04 of the TRACE doc set |
| Purpose | The **exact** format of every file and table passed between parts of TRACE. If the pipeline writes it and the app reads it, it is defined here. Lets the data/ML side and the app side be built in parallel and plugged together without surprises |
| Contract version | **1** |
| Source of truth in code | `trace_aml/contract.py` must match this doc exactly. Change both together, or neither |
| Progress | Tracked only in `CHECKLIST.md` |

---

## 1. Rules of the contract

1. **Changing a format** = update this doc + `contract.py` + `scripts/make_fake_data.py` in the same change, and bump `contract_version` here and in `config/settings.yaml`. Tell the other team member.
2. **Adding a column** to the end of a file is allowed without breaking readers, but still needs a version bump and a line in section 11.
3. **Never rename or remove a column** without the steps in rule 1.
4. **Column order** in files follows the tables below.
5. **Types** use pandas / Arrow names. `string` = pandas `string` dtype. `Int8` (capital I) = nullable integer. `datetime` = `datetime64[ns]`, no time zone, in the dataset's own clock.
6. **Enums** are UPPER_SNAKE_CASE strings, exactly as listed.
7. **Every file has one key** (marked 🔑). The key is unique within the file.
8. `scripts/check_contract.py <folder>` validates any folder against this doc (section 10). Run it on fake data and on real outputs.

## 2. Processed dataset files

Folder: `data/processed/<dataset>/`. `<dataset>` is one of: `ibm_hi_small`, `ibm_li_small`, `saml_d`, `demo`, or a backup name from `02-DATA.md` section 9.

### 2.1 `transactions.parquet`
| Column | Type | Meaning / rule |
|---|---|---|
| 🔑 `txn_id` | int64 | 0, 1, 2… in timestamp order |
| `timestamp` | datetime | When the transfer happened |
| `sender_id` | string | Account ID (`02-DATA.md` 4.3) |
| `receiver_id` | string | Account ID |
| `amount` | float64 | Amount in base currency, > 0 |
| `amount_original` | float64 | Amount as in the source |
| `currency_original` | string | Currency as in the source |
| `channel` | string | Payment format as in the source |
| `is_self` | bool | sender_id == receiver_id |
| `dataset` | string | Dataset name |

### 2.2 `labels_txn.parquet` (only training, evaluation and labelling code may read this)
| Column | Type | Meaning |
|---|---|---|
| 🔑 `txn_id` | int64 | Same as transactions |
| `is_laundering` | int8 | 0 or 1 |
| `pattern_type` | string, nullable | e.g. `FAN_IN`, `CYCLE`, … if known |
| `pattern_id` | string, nullable | Pattern group if known |

`pattern_type` enum: `FAN_OUT`, `FAN_IN`, `GATHER_SCATTER`, `SCATTER_GATHER`, `CYCLE`, `RANDOM`, `BIPARTITE`, `STACK`, `OTHER` (SAML-D types are mapped to these or kept as `OTHER`, with the original name recorded in the manifest).

### 2.3 `account_labels.parquet`
| Column | Type | Meaning |
|---|---|---|
| 🔑 `account_id` | string | |
| `role` | string enum | `MULE`, `SOURCE`, `SINK`, `NORMAL` |
| `is_mule` | Int8, nullable | 1 = MULE, 0 = NORMAL, empty = SOURCE or SINK (excluded) |
| `mule_since` | datetime, nullable | First laundering transaction in or out; empty for NORMAL |
| `ring_id` | string, nullable | Ring the account belongs to |
| `ring_source` | string enum, nullable | `PATTERN_FILE` or `CONNECTED` (`02-DATA.md` 5.3) |

### 2.4 `splits.json`
```json
{
  "dataset": "ibm_hi_small",
  "data_start": "2022-09-01T00:00:00",
  "train_end": "2022-09-07T00:00:00",
  "validation_end": "2022-09-09T00:00:00",
  "test_end": "2022-09-11T00:00:00",
  "shares": {"train": 0.6, "validation": 0.2, "test": 0.2},
  "mule_counts": {"train": 0, "validation": 0, "test_new": 0, "test_known": 0}
}
```
(Dates above are placeholders; the real ones come from the data.)

## 3. Upload format (what an analyst can upload)

A **CSV** file, UTF-8, header row, max 200 MB.

| Column | Required | Type | Rule |
|---|---|---|---|
| `timestamp` | yes | text | ISO 8601, e.g. `2026-10-14 10:02:00` |
| `sender_id` | yes | text | Not empty |
| `receiver_id` | yes | text | Not empty |
| `amount` | yes | number | > 0 |
| `currency` | no | text | If missing, the display currency (`INR`) is assumed and converted to base with the display rate |
| `channel` | no | text | If missing, `UNKNOWN` |

Any other columns are ignored (including any label column). Error messages must name the column and the first 5 failing row numbers.

## 4. Feature files

Folder: `data/features/<dataset>/`.

### 4.1 `<cutoff>.parquet` (one file per cut-off; `<cutoff>` written as `YYYYMMDDTHHMM`)
| Column | Type | Meaning |
|---|---|---|
| 🔑 `account_id` | string | Accounts active in the look-back window before the cut-off |
| `cutoff` | datetime | Features use only transactions with `timestamp <= cutoff` |
| *feature columns* | float32 | One column per feature, named exactly as in the feature catalogue (`05-FEATURES-MODEL.md` section 2). Missing = NaN (LightGBM handles it) |

### 4.2 `<cutoff>_rules.parquet`
| Column | Type | Meaning |
|---|---|---|
| 🔑 `account_id` | string | |
| `cutoff` | datetime | |
| `rule_chain` | int8 | 0 / 1 |
| `rule_fan_in` | int8 | 0 / 1 |
| `rule_fan_out` | int8 | 0 / 1 |
| `rule_cycle` | int8 | 0 / 1 |
| `rule_pass_through` | int8 | 0 / 1 |
| `rule_score` | float32 | 0–1, rules-only baseline score (`05` section 3.3) |

### 4.3 `<cutoff>_rules_evidence.parquet`
| Column | Type | Meaning |
|---|---|---|
| 🔑 (`account_id`, `rule`, `txn_id`) | | |
| `account_id` | string | |
| `rule` | string enum | `CHAIN`, `FAN_IN`, `FAN_OUT`, `CYCLE`, `PASS_THROUGH` |
| `txn_id` | int64 | A transaction that made the rule fire |

## 5. Run files (what the app reads)

Folder: `data/runs/<run_id>/`. `run_id` = `<dataset>__<split or upload stamp>__<model_version>`, e.g. `ibm_hi_small__test__v1`.

### 5.1 `scores.parquet`
| Column | Type | Meaning |
|---|---|---|
| 🔑 `account_id` | string | Every scored account |
| `risk` | float32 | 0–100, one decimal |
| `prob` | float32 | Calibrated probability 0–1 (risk = 100 × prob) |
| `verdict` | string enum | `LIKELY_MULE`, `WATCH`, `NORMAL` |
| `rank` | int32 | 1 = highest risk |
| `short_label` | string | Main behaviour in 1–3 words, e.g. `Pass-through`, `Fan-in collector`, `Normal activity` (`05` section 6.4) |
| `summary` | string | Plain-English summary (`05` section 6.4) |
| `has_trail` | bool | Whether `trails.parquet` has a trail for this account |
| `in_amount` | float64 | Total received, base currency, up to cut-off |
| `out_amount` | float64 | Total sent, base currency, up to cut-off |
| `n_txn` | int32 | Transactions in + out |
| `first_seen` | datetime | First transaction in the data |

### 5.2 `reasons.parquet` (only for `LIKELY_MULE` and `WATCH` accounts)
| Column | Type | Meaning |
|---|---|---|
| 🔑 (`account_id`, `position`) | | |
| `account_id` | string | |
| `position` | int8 | 1–5, 1 = biggest impact |
| `feature` | string | Catalogue name |
| `display_name` | string | Human name from the catalogue |
| `value` | float32 | The account's value |
| `value_text` | string | Formatted for people, e.g. `forwarded 93% of inflow` |
| `impact` | float32 | SHAP value (positive = pushes risk up) |
| `direction` | string enum | `UP` or `DOWN` |

### 5.3 `trails.parquet`
| Column | Type | Meaning |
|---|---|---|
| 🔑 (`account_id`, `trail_id`, `hop`) | | |
| `account_id` | string | The flagged account the trail belongs to |
| `trail_id` | int8 | 1–3 (up to 3 trails per account) |
| `hop` | int8 | Order along the trail, starting at 1, earliest first |
| `txn_id` | int64 | |
| `from_id` | string | |
| `to_id` | string | |
| `timestamp` | datetime | |
| `amount` | float64 | Base currency |
| `side` | string enum | `BEFORE` (money arriving at the account) or `AFTER` (money leaving it) |

### 5.4 `rings.parquet` (Layer 2)
| Column | Type | Meaning |
|---|---|---|
| 🔑 (`ring_id`, `account_id`) | | |
| `ring_id` | string | e.g. `R-0003` |
| `account_id` | string | Member |
| `ring_size` | int32 | Members in the ring |
| `ring_amount` | float64 | Total money moved between members, base currency |
| `first_txn` | datetime | |
| `last_txn` | datetime | |
| `max_risk` | float32 | |
| `mean_risk` | float32 | |

### 5.5a `features.parquet`
The feature row behind each account's score (same columns as section 4.1, plus `snapshot` datetime = the cut-off that produced the score). Needed by retraining (`03-ARCHITECTURE.md` 3.2). One row per account, 🔑 `account_id`.

### 5.5 `transactions_view.parquet`
A copy of the transactions the run covers (`transactions.parquet` columns, no labels), sorted by `sender_id` then `timestamp`, so the app can build neighbourhoods fast without touching `data/processed`.

### 5.6 `manifest.json`
```json
{
  "contract_version": 1,
  "run_id": "ibm_hi_small__test__v1",
  "dataset": "ibm_hi_small",
  "kind": "SPLIT",
  "split": "TEST",
  "cutoff": "2022-09-11T00:00:00",
  "model_version": "v1",
  "created_at": "2026-11-05T14:02:11Z",
  "settings_hash": "…",
  "n_accounts": 0,
  "n_transactions": 0,
  "verdict_counts": {"LIKELY_MULE": 0, "WATCH": 0, "NORMAL": 0}
}
```
`kind` enum: `SPLIT`, `UPLOAD`, `DEMO`, `REPLAY`. `split` enum (only when kind is `SPLIT`): `TRAIN`, `VALIDATION`, `TEST`.

## 6. SQLite (`trace.db`)

```sql
CREATE TABLE feedback (
  feedback_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  account_id     TEXT    NOT NULL,
  run_id         TEXT    NOT NULL,
  model_version  TEXT    NOT NULL,
  decision       TEXT    NOT NULL CHECK (decision IN ('CONFIRMED_MULE','FALSE_POSITIVE','NEEDS_INVESTIGATION')),
  analyst_name   TEXT    NOT NULL,
  analyst_role   TEXT    NOT NULL CHECK (analyst_role IN ('ANALYST','ADMIN')),
  risk_at_decision REAL  NOT NULL,
  note           TEXT,
  created_at     TEXT    NOT NULL          -- ISO 8601 UTC
);
CREATE INDEX idx_feedback_account ON feedback(account_id);

CREATE TABLE model_versions (
  version        TEXT PRIMARY KEY,          -- 'v1', 'v2', …
  created_at     TEXT NOT NULL,
  created_by     TEXT NOT NULL,             -- analyst name or 'pipeline'
  parent_version TEXT,                      -- version it was retrained from
  n_feedback_used INTEGER NOT NULL DEFAULT 0,
  eval_valid     INTEGER NOT NULL DEFAULT 1, -- 0 = "demo only": trained with test/demo/upload feedback; never used in reported metrics
  val_pr_auc     REAL,
  val_precision_at_k REAL,
  k              INTEGER,
  path           TEXT NOT NULL              -- models/vN/
);

CREATE TABLE app_state (
  key   TEXT PRIMARY KEY,                   -- 'active_model_version', 'active_run_id'
  value TEXT NOT NULL
);
```

Rules:
- An account can have several feedback rows (decisions over time). The **latest** row is the current decision.
- Feedback rows are never deleted by the app.
- `model_versions` rows are never deleted; versions are switched, not removed.

## 7. Model folder

Folder: `models/<version>/`.

| File | Contents |
|---|---|
| `model.txt` | LightGBM model (its own text format) |
| `calibrator.joblib` | Fitted isotonic calibrator |
| `features.json` | Ordered list of feature column names the model expects |
| `thresholds.json` | `{"likely_mule_cut": 0–100, "watch_cut": 0–100}` |
| `meta.json` | Training data description, settings, seed, validation metrics, training time, parent version |

## 8. Reports

Folder: `reports/`. Every report = one `.md` for people + one `.json` for code, same name.

`quality_<dataset>.json`:
```json
{
  "dataset": "ibm_hi_small",
  "overall": "PASS",
  "checks": [
    {"id": "C1", "name": "Format", "status": "PASS", "value": 0.0004, "detail": "0.04% rows dropped"}
  ]
}
```
`status` / `overall` enum: `PASS`, `WARN`, `FAIL`. `overall` = worst status of all checks.

`eval_<version>.json`: one entry per approach (`RULES`, `TXN_ONLY`, `LOGREG`, `RANDOM_FOREST`, `ISOLATION_FOREST`, `LIGHTGBM`) and per ablation, each with every metric from `05-FEATURES-MODEL.md` section 7, as mean and standard deviation over seeds.

Replay, explanation-check and cross-dataset reports follow the same pattern; their metric names are defined in `05` sections 8–10.

## 9. Fake data for app work

`scripts/make_fake_data.py` writes `data/fake/runs/fake__demo__v0/` with every file in section 5 plus a matching `trace.db`, for about 300 accounts, including:
- one obvious 5-hop ring (5 `LIKELY_MULE` accounts with trails),
- a few `WATCH` accounts,
- normal accounts including a high-degree merchant,
- repeated transfers between the same pair (to test arrow merging).

The fake files must pass `check_contract.py`. The app developer uses only these until the first real run exists.

## 10. Contract checker

`python scripts/check_contract.py <folder>` checks, for every file it finds:
1. Required files exist for that folder type (processed / features / run / models).
2. All columns present, in order, with the right types.
3. Enum columns contain only allowed values.
4. Keys are unique.
5. Ranges: `risk` 0–100, `prob` 0–1, `amount` > 0, `position` 1–5.
6. Cross-file links: every `account_id` in reasons / trails / rings exists in scores; every trail `txn_id` exists in `transactions_view`.
7. `manifest.json` `contract_version` matches the current version.

Output: a ✅ / ❌ line per check; exit code 0 only if all pass.

## 11. Change log

| Version | Change | Date |
|---|---|---|
| 1 | First version | |
