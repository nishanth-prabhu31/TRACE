# TRACE — Checklist

`[ ]` not started · `[~]` in progress · `[x]` done (add date + initials) · `[!]` blocked (add reason)
Rules: `AGENTS.md` section 3. Owners: `WORKSPLIT.md` (A = data + model, B = dashboard). Doc refs: `PRD` = 01, `DATA` = 02, `ARCH` = 03, `CON` = 04, `FM` = 05, `DASH` = 06.

## 0. Setup
- [x] Repo created with folder layout (ARCH §5), `.gitignore`, `requirements.txt`, README stub — 2026-11-02 NP
- [x] Python venv (`.venv`) with all libraries installed — 2026-11-02 NP
- [x] `config/settings.yaml` created with all 🔧 values (ARCH §6); exchange rates set (derived from IBM data) — 2026-11-02 NP
- [x] `trace_aml/contract.py` matches CON v1 — 2026-11-02 NP
- [x] `scripts/check_contract.py` works (CON §10); tests in `tests/test_contract.py` pass — 2026-11-02 NP
- [x] `scripts/make_fake_data.py` writes `data/fake/` that passes the checker (CON §9) — 2026-11-14 NP
- [x] Graph component test: does `streamlit-agraph` return clicked node? Yes; recorded in DASH §14 — 2026-11-14 NP
- [x] Decide team split (PRD Q1): A = data + model, B = dashboard; see `WORKSPLIT.md` — 2026-11-02 NP

## 1. Data
- [x] Download IBM HI-Small (+ patterns file) to `data/raw/` — 2026-11-02 NP
- [x] Download IBM LI-Small (+ patterns file); check its date tail (same tail, same cut; DATA §13) — 2026-11-14 NP
- [x] Verify IBM columns, currencies, patterns file format; update DATA §13 — 2026-11-02 NP
- [x] F-01 Ingest IBM → common format (DATA §4, §7); passes contract checker; 9 tests — 2026-11-03 NP
- [x] F-03 Account labels + rings + `splits.json` (DATA §5, §6); as-of labels, contract v2; 10 tests; HI-Small 795 mules (340 / 253 / 202) — 2026-11-14 NP
- [x] F-02 Quality gate C1–C7 + report card (DATA §8); 10 tests; IBM HI-Small result: PASS with ⚠️ C3 (60% of test-period mules are new since train) — 2026-11-14 NP
- [x] Quality gate on LI-Small; result: PASS with ⚠️ C3 (58% new) — 2026-11-14 NP

## 2. Model (Core)
- [x] F-04 Features: structural (FM §2.1) — 2026-11-14 NP
- [x] F-04 Features: temporal (FM §2.2) — 2026-11-14 NP
- [x] F-04 Features: fund-flow incl. FIFO matching (FM §2.3) — 2026-11-14 NP
- [x] F-04 Features: relative amounts (FM §2.4) — 2026-11-14 NP
- [x] Leakage tests pass (ARCH §9) — 2026-11-14 NP
- [x] F-05 Rule detectors + evidence + rules-only score (FM §3); 10 tests; weighted score computed in training — 2026-11-14 NP
- [x] Training rows / snapshots (FM §4) — 2026-11-14 NP
- [x] F-06 Baselines: RULES, TXN_ONLY, LOGREG, RANDOM_FOREST, ISOLATION_FOREST (FM §5) — 2026-11-14 NP
- [x] F-06 LightGBM main model (FM §5); settings changed, FM §13 — 2026-11-14 NP
- [x] Evaluation report, 5 seeds, HI-Small + LI-Small (FM §7): `reports/eval_v1.md`, `reports/eval_v1_ibm_li_small.md`; LI-Small is much weaker (FM §13) — 2026-11-14 NP
- [x] Minimum bar met: LightGBM beats RULES and TXN_ONLY on PR-AUC (PRD §9): test PR-AUC 0.165 vs 0.0011 and 0.081 (HI-Small, synthetic, 5 seeds); note Random Forest 0.276 beats LightGBM — 2026-11-14 NP
- [ ] Numeric targets set from first results; recorded in FM §13
- [ ] Try the stronger-regularised LightGBM from branch `nishanth-pipeline` (FM §13) on this pipeline; keep only if validation PR-AUC improves
- [x] Ablations (FM §5.1) — 2026-11-14 NP
- [~] F-07 Calibration + verdict cut-offs (FM §6.1–6.2): done and tested; WATCH rule gives 23K accounts, needs a team decision (FM §13)
- [x] F-08 SHAP reasons (FM §6.3) — 2026-11-14 NP
- [x] F-08 Money trails (FM §6.4) — 2026-11-14 NP
- [x] F-08 Short labels + summaries (FM §6.5) — 2026-11-14 NP
- [x] P7 writes a real run that passes the contract checker: `ibm_hi_small__test__v1` — 2026-11-14 NP

## 3. App (Core)
- [x] Sidebar + F-17 analyst picker and roles (DASH §2) — 2026-11-14 NP
- [~] Overview + upload validation + upload scoring (DASH §3, ARCH §3.1): overview done; upload waits for the scoring function (P7)
- [x] Alert queue (DASH §4) — 2026-11-14 NP
- [~] Account: score card, reasons, summary (DASH §5.1, §5.6, §5.7): done; "Explain this score" for NORMAL accounts waits for the model (P7)
- [x] Account: network view + hop + back + breadcrumb (DASH §5.2, §5.3) — 2026-11-14 NP
- [x] Account: follow the money (DASH §5.4) — 2026-11-14 NP
- [x] Account: transactions tab (DASH §5.5) — 2026-11-14 NP
- [x] F-10 Decision panel + feedback saved to SQLite (DASH §5.8, CON §6) — 2026-11-14 NP
- [x] Data quality screen (DASH §6); seen in the browser with the real HI-Small and LI-Small reports — 2026-11-14 NP
- [x] **First integration:** app on a real run instead of fake data (ARCH §11) — 2026-11-14 NP
- [~] Performance targets met (DASH §10): open/hop/queue met on the real run (DASH §14); first account open per run takes ~10 s once; upload target not measured (upload not built)

## 4. Demo (Core)
- [ ] F-11 Demo data file from IBM test slice, ₹ / UPI restyled (DATA §10)
- [ ] Pre-scored demo run + "Open demo run" button
- [ ] Demo script rehearsed (DASH §13); backup screen recording made
- [ ] **CORE DONE**

## 5. Layer 2
- [ ] F-12 Replay + lead time report (FM §8)
- [ ] F-13 Rings: pipeline step + screen (ARCH P10, DASH §7)
- [ ] F-14 Retrain path + model versions + Models screen (ARCH §3.2, DASH §8)
- [ ] Feedback evaluation with simulated analyst, 3 rounds (FM §11)
- [ ] F-18 UPI-style stress test: inject fast ₹ mule rings into IBM normal background; compare model vs rules vs model+rules on them (stress test only, never trained on, never the headline result; PRD §5)
- [ ] **LAYER 2 DONE**

## 6. Layer 3
- [ ] Download SAML-D; verify columns; update DATA §13
- [ ] SAML-D mapping + quality gate (DATA §3, §8)
- [ ] F-16 Cross-dataset report (FM §10)
- [ ] F-15 Explanation correctness + random-trail baseline (FM §9)
- [ ] **LAYER 3 DONE**

## 7. Wrap-up
- [ ] Results page or charts for the report (FM §12, DASH §1)
- [ ] Final report sections drafted from `reports/`
- [ ] README: what it is, how to run, screenshots, honest limitations
- [ ] Viva prep sheet: pipeline in 6 lines, likely questions + answers, key numbers; practice round
- [ ] Detailed project explanation doc: every step from raw data to dashboard, in plain words

## Parked (only after Layer 2+)
- [ ] P-01 Own visual design
- [ ] P-02 FastAPI + Docker (ARCH §12)
- [ ] P-03 Public README polish + hosted demo
- [ ] P-04 Real password login
- [ ] Confirm dataset licences before publishing anything (DATA §11)
