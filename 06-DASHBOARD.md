# TRACE — Dashboard

| | |
|---|---|
| Doc | 06 of the TRACE doc set |
| Purpose | Every screen, what is on it, how each interaction behaves, every empty / loading / error state, and the demo script |
| Covers features | F-09, F-10, F-13 (screen), F-14 (screen), F-17 |
| Reads | Run files and SQLite exactly as in `04-CONTRACT.md` sections 5–6 |
| Visual style | **Parked (P-01).** This doc fixes *what* is shown and *how it behaves*, not the final look. Section 9 gives interim rules only |
| Progress | Tracked only in `CHECKLIST.md` |

🔧 = tunable starting value (`config/settings.yaml`). ⚠️ = confirm with a quick test.

---

## 1. Screen map

| Screen | Who sees it | Layer | Purpose |
|---|---|---|---|
| Sidebar (always visible) | Everyone | Core | Who am I, which run, which model |
| Overview | Everyone | Core | Totals, upload, run info |
| Alert queue | Everyone | Core | Ranked list of accounts to review |
| Account investigation | Everyone | Core | Score, reasons, summary, network, money trail, transactions, decision |
| Data quality | Everyone | Core | Report cards from the quality gate |
| Rings | Everyone | Layer 2 | Flagged accounts grouped into cases |
| Models | Admin only | Layer 2 | Versions, retrain, switch active version |
| Results *(optional)* | Everyone | Core+ | Evaluation tables and charts from `reports/`, for the viva |

Navigation: Streamlit multipage app (`app/pages/`). Opening an account from anywhere goes to Account investigation with that account selected.

## 2. Sidebar

| Element | Behaviour |
|---|---|
| **Logged in as** | Dropdown of names from `settings.yaml`; the role shows under it ("Analyst" / "Admin"). Default: first analyst. No password (PRD F-17). Small note: "Role selection, not security" |
| **Run** | Dropdown of folders in `data/runs/` (newest first), labelled by dataset, kind and model, e.g. "IBM HI-Small · test · v1". Default: `active_run_id` from `app_state` |
| **Active model** | Shows the active version (read-only here; changed on the Models screen) |
| **Currency** | Toggle ₹ (display currency) / base currency. All amounts on every screen follow it. ₹ uses Indian grouping (₹1,25,000; ₹4.35 L; ₹1.2 Cr) |

Changing the run clears the selected account and the hop history.

## 3. Overview

**Top row (numbers from `manifest.json` and `scores.parquet`):** transactions, accounts, Likely mule count, Watch count, total money in the run.

**Run info:** dataset, kind (split / upload / demo), cut-off time, model version, created at. For uploaded runs, a note: "Accounts with little history in this file are scored less reliably."

**Upload box:**
1. Accepts one CSV (`04-CONTRACT.md` section 3).
2. Validation errors are listed plainly, e.g. "Column `amount`: 3 rows are not positive numbers (rows 14, 92, 310)". Nothing is scored if validation fails.
3. On success, a progress display with named steps: Checking file → Building network → Computing features → Scoring → Writing explanations → Done.
4. When done, the new run becomes the selected run and a button opens the Alert queue.

**Demo shortcut:** a button "Open demo run" that switches to the pre-scored demo run, so the demo never depends on a live upload.

## 4. Alert queue

**Table columns:** rank · account · risk (number + coloured badge) · verdict · short label · current decision (if any, with analyst name) · total in · total out.

**Controls:**
| Control | Behaviour |
|---|---|
| Verdict filter | Multi-select; default: Likely mule + Watch |
| Decision filter | All / Undecided / Decided |
| Search | Partial match on account ID |
| Sort | By risk (default, highest first) or by rank |
| Pages | 50 rows per page 🔧 |
| Row click | Opens Account investigation for that account (resets hop history) |

**Empty states:** no run → "Pick a run in the sidebar or upload a file". Filters hide everything → "No accounts match these filters" + a reset button.

## 5. Account investigation

### 5.1 Layout
- **Header:** account ID, short label, first seen, number of transactions, total in / out.
- **Score card:** large risk number, verdict chip (Likely mule / Watch / Looks normal), model version.
- **Main area, three tabs:** Network · Follow the money · Transactions.
- **Side area:** Why this score · In plain words · Decision.
- **Breadcrumb + Back** above the tabs (section 5.3).

### 5.2 Network tab (the hop-around graph)
| Rule | Spec |
|---|---|
| Centre | The selected account, drawn larger, with a clear outline |
| Neighbours | Default 1 hop (direct counterparties). Toggle "2 hops" 🔧 |
| Size cap | At most 40 accounts 🔧. If there are more, keep: (1) accounts on this account's money trail, (2) Likely mule / Watch neighbours, (3) the rest by total money exchanged. Show "+N more accounts not shown" |
| Merged arrows | All transfers in one direction between the same two accounts = one arrow, labelled "4 txns · ₹1.8 L". Two directions = two arrows |
| Arrow thickness | By total money, log scale, 1–8 px 🔧 |
| Node colour | By verdict (three colours). **Also** by shape or icon, so colour is never the only signal |
| Node label | Account ID; short label underneath for the centre |
| Hover on node | Account ID, risk, verdict, short label, transactions, total in / out |
| Hover on arrow | Up to 5 underlying transactions (time, amount), "+N more" |
| Click on node | **Hop**: that account becomes the centre (section 5.3) |
| Legend | Colours/shapes, "arrow width = money", "click an account to hop" |

**Component:** ⚠️ `streamlit-agraph` is the first candidate because it returns the clicked node. Test it for 30 minutes before building the screen. **Fallback** (allowed by PRD F-09): draw with `pyvis` and hop using a dropdown of the neighbours ("Go to account…") under the graph. Both paths share the same neighbourhood builder (`app/logic/neighbourhood.py`).

### 5.3 Hopping, Back and breadcrumb
- Hop history is a list in `st.session_state["hop_history"]`.
- Hop to account *B* from *A*: push *A*, show *B*. All panels (score, reasons, summary, trail, decision) switch to *B*.
- **Back:** pop and show the previous account. Disabled when the list is empty.
- **Breadcrumb:** "ACC-1187 → ACC-4821 → ACC-7710"; shows the last 6 🔧; clicking an earlier item jumps back to it (and trims the list).
- Opening an account from the Alert queue or Rings screen starts a fresh history.
- If an account is not in the current run's scores (e.g. outside the cut-off), it can be shown in the graph but opening it says "This account has no score in this run".

### 5.4 Follow the money tab
- Uses `trails.parquet` for the account.
- If several trails: a selector "Trail 1 / 2 / 3", ordered as stored.
- Drawn left to right in time order: account boxes joined by arrows; above each arrow the amount, below it the time. The selected account is highlighted.
- Under the chain: total time from first to last hop, and the number of hops.
- Each account in the chain is clickable (hop, same as 5.3).
- No trail: "No suspicious money trail found for this account." plus, for Likely mule / Watch accounts, a line pointing to the reasons panel.

### 5.5 Transactions tab
Table of the account's transactions up to the cut-off: time · direction (in / out) · counterparty · amount · channel. Filters: direction, last 24 h / all. Counterparty is clickable (hop). Rows that are part of a trail or rule evidence are marked.

### 5.6 Why this score
- Up to 5 rows from `reasons.parquet`: display name, value text, a bar whose length is the size of the impact; bars pushing risk up and pulling it down are visually distinct.
- Caption: "Bars show how much each signal moved this account's score up or down."
- For `NORMAL` accounts (no stored reasons): a button "Explain this score" computes SHAP on demand for this one account.

### 5.7 In plain words
The `summary` from `scores.parquet`, amounts shown in the selected currency.

### 5.8 Decision panel (F-10)
- Three buttons: **Confirmed mule**, **False positive**, **Needs investigation**; optional note box.
- One click saves a row to `feedback` with account, run, model version, decision, analyst name and role, risk at the time, note, time. Toast: "Saved: False positive by Analyst A."
- Under the buttons: decision history for this account, newest first ("Confirmed mule · Analyst B · 05 Nov 14:20").
- Changing a decision adds a new row; the newest row is the current decision. Rows are never deleted from the app.

## 6. Data quality screen
- One card per dataset found in `reports/quality_*.json`.
- Each card: overall status, then checks C1–C7 with status icon (✅ / ⚠️ / ❌), measured value and one-line detail.
- Link to the full markdown report.

## 7. Rings screen (Layer 2, F-13)
- Table from `rings.parquet`: ring ID · members · money moved · first and last transaction · highest risk · mean risk. Sorted by highest risk.
- Selecting a ring shows: a graph of **only** the ring members and the transfers between them (same drawing rules as 5.2, no cap needed for small rings, cap 40 otherwise), and the member list with risk and verdict. Clicking a member opens Account investigation.
- Ring-level decision is not stored; analysts decide per account.

## 8. Models screen (Layer 2, F-14, Admin only)
- If the role is Analyst: the page shows "Only Admin can manage models" and the versions table read-only.
- **Versions table** from `model_versions`: version · created · by · parent · feedback rows used · validation PR-AUC · validation Precision@K · "demo only" tag when `eval_valid = 0` · active marker.
- **Retrain button:** shows first how much feedback is usable ("12 Confirmed mule, 7 False positive; 3 Needs investigation ignored"), and whether the result will be **evaluation-safe** or **demo only** (demo only if any feedback comes from test, demo or upload runs). Then runs the retrain path (`03-ARCHITECTURE.md` 3.2) with a progress display.
- After retraining: the new row appears with a side-by-side comparison against the active version (validation metrics).
- **Activate** button on any non-active version: switches `active_model_version`, re-scores the selected run with it (as a new run `…__vN`), and selects that run. Switching back is the same action.

## 9. Interim visual rules (until P-01)
- Use Streamlit's theme file (`.streamlit/config.toml`) for base colours and font; no heavy custom CSS yet.
- Three risk colours used **consistently** everywhere (badges, nodes, bars); never reuse them for anything else.
- Avoid the generic "AI-generated dashboard" look the team rejected: no glow effects, gradient text, glassmorphism or decorative animation.
- Numbers right-aligned in tables; amounts always with currency symbol; Indian number grouping for ₹.
- Every colour-coded thing also has text or a shape (accessibility).

## 10. Performance targets
| Action | Target |
|---|---|
| Open an account (score card + reasons + graph) | ≤ 3 s 🔧 |
| Hop to a neighbour | ≤ 2 s 🔧 |
| Alert queue page load | ≤ 2 s 🔧 |
| Upload of a 5,000-transaction file, end to end | ≤ 60 s 🔧 |

How: cache run files with `st.cache_data`; build neighbourhoods from `transactions_view.parquet` (sorted, indexed by account); never load `data/processed` in the app.

## 11. States and errors (summary)
| Situation | What the user sees |
|---|---|
| No runs exist | Overview: "No scored runs yet. Run the pipeline or upload a file." |
| Upload invalid | List of problems by column and row; nothing scored |
| Upload scoring fails midway | "Scoring failed at step: Computing features" + the error message; no half-written run is shown (write to a temp folder, rename when complete) |
| Account has no score in run | Message in place of the score card; graph still shown |
| No trail | Message in Follow the money (5.4) |
| Graph component fails to load | Automatic fallback to the dropdown hop (5.2) |
| SQLite locked or missing | "Could not save decision, try again"; the app creates `trace.db` with empty tables if missing |

## 12. Session state keys
| Key | Holds |
|---|---|
| `user_name`, `user_role` | From the sidebar picker |
| `run_id` | Selected run |
| `selected_account` | Account on the investigation screen |
| `hop_history` | List of previous accounts |
| `graph_hops` | 1 or 2 |
| `currency` | `INR` or base |
| `queue_filters` | Verdict, decision, search, page |

## 13. Demo script (for evaluators, about 5 minutes)
1. **Sidebar:** pick "Analyst A". Say: no real data, synthetic and restyled; decision support only.
2. **Overview:** open the pre-scored demo run. Optionally upload the demo CSV to show the upload flow (its scores come from the small file alone and can differ; `02-DATA.md` section 10). Point at totals and the number of Likely mule alerts.
3. **Alert queue:** "The analyst starts from the riskiest accounts." Open the top account.
4. **Account:** risk score and verdict → **Why this score** (read the top two reasons) → **In plain words**.
5. **Network:** show the merged arrows and colours; **hop** into a neighbour that is also red; hop again; press **Back**.
6. **Follow the money:** the time-ordered chain: "this all happened in N minutes".
7. **Decide:** mark Confirmed mule; show it saved with the analyst's name.
8. *(Layer 2)* **Rings:** the whole ring as one case. **Models** (switch to Admin): retrain with the feedback, compare v1 and v2. Say plainly that this v2 is "demo only"; the measured effect of feedback is in the results (simulated analyst).
9. **Data quality / Results:** one slide-worth of honesty: quality report card, LightGBM vs baselines, and (if done) caught-in-time and SAML-D numbers.

Backup plan if anything fails live: a screen recording of this exact script, made once the Core is done.
