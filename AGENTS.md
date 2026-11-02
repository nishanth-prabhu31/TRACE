# AGENTS.md — Working rules for TRACE

Rules for **anyone building TRACE**: either team member, or any coding assistant they use. Read this file first, every session.

TRACE: Transaction Risk Analytics and Chain Evaluation. An explainable money-mule detection tool: transaction network → features → LightGBM risk score → reasons + money trail → analyst dashboard with feedback.

---

## 1. Start of every session (in this order)

1. Read `CHECKLIST.md`. It is short on purpose. Find the next unchecked item, or the one marked in progress.
2. Read **only** the doc section(s) that item points to. Do not read the whole doc set unless the task needs it.
3. If something in the docs is unclear or contradicts another doc, **stop and ask the team**. Do not guess.

## 2. Doc set (what to read for what)

| Need | Read |
|---|---|
| What / why / what counts as done (feature IDs F-xx) | `docs/01-PRD.md` |
| Datasets, mapping, labels, time split, quality gate | `docs/02-DATA.md` |
| Parts, data flow, folders, tools, how to run | `docs/03-ARCHITECTURE.md` |
| Exact file and table formats | `docs/04-CONTRACT.md` |
| Features, rules, models, scores, explanations, metrics, experiments | `docs/05-FEATURES-MODEL.md` |
| Screens and interactions | `docs/06-DASHBOARD.md` |
| Who owns which files and tasks | `WORKSPLIT.md` |

## 3. Marking progress

- Progress is marked **only** in `CHECKLIST.md`, never inside the docs.
- Markers: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked (add a short reason).
- When finishing an item: change it to `[x]` and add the date and initials at the end of the line, e.g. `[x] … — 2026-11-05 NP`.
- An item is `[x]` only when its acceptance criteria in `01-PRD.md` section 7 (or the doc section it points to) are met **and** its tests pass.
- If the work revealed something the docs should record (a verified fact, a changed 🔧 value, a new decision), update the **change log** at the bottom of the relevant doc in the same change.

## 4. Hard rules (never break)

1. **No data leakage.** Features use only transactions up to the cut-off. Feature code never reads label files. The test period is only for final numbers. Feedback on test-period accounts is never used for retraining. (`03-ARCHITECTURE.md` section 9)
2. **The contract is the agreement.** Never change a file format without updating `docs/04-CONTRACT.md`, `trace_aml/contract.py` and `scripts/make_fake_data.py` together and bumping `contract_version`.
3. **No magic numbers.** Every tunable value lives in `config/settings.yaml`.
4. **One code path.** Feature, scoring and explanation logic exists once, in `trace_aml/`. The app imports it; it never copies it.
5. **Raw data is read-only** and never committed. `data/raw/`, `data/processed/`, `data/features/`, `data/runs/`, `models/` and `trace.db` are git-ignored.
6. **Never print or paste large data** (thousands of rows) into a chat or log. Use `.head()`, counts and summaries.
7. **Honest results.** Every reported number names its dataset, split, model version and number of seeds. Synthetic results are never described as real-world performance. Bad results are reported, not hidden.
8. **No scope changes without the team.** Anything not in `01-PRD.md` goes to the team first, then into the PRD if agreed.
9. **No tool, assistant, vendor or model names** in project files, code comments, docs or commit messages. Commits carry no co-author lines for tools.
10. **TRACE never acts on an account.** No feature may freeze, block or auto-decide; the analyst decides.

## 5. Code conventions

- Python 3.11. Package `trace_aml` (not `trace`; that clashes with a built-in module).
- Small functions with type hints and a one-line docstring. Plain names over clever ones.
- Every feature function takes `cutoff` explicitly.
- pandas with explicit dtypes; Parquet for data files.
- Each pipeline step: reads files, writes files and a `manifest.json`, can run alone from the command line.
- Fixed random seed from settings for anything random.
- Tests in `tests/` with tiny hand-made data; never depend on the full dataset in tests.
- Prefer clear and boring over fast and clever, until a measured slowness forces a change.

## 6. Definition of done (any task)

- [ ] Meets the acceptance criteria it points to
- [ ] Tests written and passing (`pytest`)
- [ ] `python scripts/check_contract.py <folder>` passes for any files it writes
- [ ] Settings in `config/settings.yaml`, not in code
- [ ] Relevant doc change log updated if anything was verified or changed
- [ ] `CHECKLIST.md` item marked `[x]` with date and initials
- [ ] Either team member could explain it in the viva

## 7. When stuck

- A step is too slow or uses too much memory → see `03-ARCHITECTURE.md` section 8 before inventing a fix.
- A ⚠️ item in the docs turns out different in reality → record the truth in that doc's change log, adjust, tell the team.
- Results look "too good" (e.g. PR-AUC near 1.0) → suspect leakage or a data shortcut first; rerun the quality gate checks C4 and C5 (`02-DATA.md` section 8).
