# TRACE — Checklist

`[ ]` not started · `[~]` in progress · `[x]` done (add date + initials) · `[!]` blocked (add reason)
Rules: `AGENTS.md` section 3. Doc refs: `PRD` = 01, `DATA` = 02, `ARCH` = 03, `CON` = 04, `FM` = 05, `DASH` = 06.

## 0. Setup
- [ ] Repo created with folder layout (ARCH §5), `.gitignore`, `requirements.txt`, README stub
- [ ] `config/settings.yaml` created with all 🔧 values (ARCH §6); exchange rates set
- [ ] `trace_aml/contract.py` matches CON v1
- [ ] `scripts/check_contract.py` works (CON §10)
- [ ] `scripts/make_fake_data.py` writes `data/fake/` that passes the checker (CON §9)
- [ ] Graph component test: does `streamlit-agraph` return clicked node? Record result in DASH §5.2 (⚠️)
- [ ] Decide team split (PRD Q1)

## 1. Data
- [ ] Download IBM HI-Small + LI-Small (+ patterns files) to `data/raw/`
- [ ] Verify IBM columns, currencies, patterns file format; update DATA §13
- [ ] F-01 Ingest IBM → common format (DATA §4, §7)
- [ ] F-03 Account labels + rings + `splits.json` (DATA §5, §6)
- [ ] F-02 Quality gate C1–C7 + report card (DATA §8); IBM HI-Small result: ____
- [ ] Quality gate on LI-Small; result: ____

## 2. Model (Core)
- [ ] F-04 Features: structural (FM §2.1)
- [ ] F-04 Features: temporal (FM §2.2)
- [ ] F-04 Features: fund-flow incl. FIFO matching (FM §2.3)
- [ ] F-04 Features: relative amounts (FM §2.4)
- [ ] Leakage tests pass (ARCH §9)
- [ ] F-05 Rule detectors + evidence + rules-only score (FM §3)
- [ ] Training rows / snapshots (FM §4)
- [ ] F-06 Baselines: RULES, TXN_ONLY, LOGREG, RANDOM_FOREST, ISOLATION_FOREST (FM §5)
- [ ] F-06 LightGBM main model (FM §5)
- [ ] Evaluation report, 5 seeds, HI-Small + LI-Small (FM §7)
- [ ] Minimum bar met: LightGBM beats RULES and TXN_ONLY on PR-AUC (PRD §9)
- [ ] Numeric targets set from first results; recorded in FM §13
- [ ] Ablations (FM §5.1)
- [ ] F-07 Calibration + verdict cut-offs (FM §6.1–6.2)
- [ ] F-08 SHAP reasons (FM §6.3)
- [ ] F-08 Money trails (FM §6.4)
- [ ] F-08 Short labels + summaries (FM §6.5)
- [ ] P7 writes a real run that passes the contract checker

## 3. App (Core)
- [ ] Sidebar + F-17 analyst picker and roles (DASH §2)
- [ ] Overview + upload validation + upload scoring (DASH §3, ARCH §3.1)
- [ ] Alert queue (DASH §4)
- [ ] Account: score card, reasons, summary (DASH §5.1, §5.6, §5.7)
- [ ] Account: network view + hop + back + breadcrumb (DASH §5.2, §5.3)
- [ ] Account: follow the money (DASH §5.4)
- [ ] Account: transactions tab (DASH §5.5)
- [ ] F-10 Decision panel + feedback saved to SQLite (DASH §5.8, CON §6)
- [ ] Data quality screen (DASH §6)
- [ ] **First integration:** app on a real run instead of fake data (ARCH §11)
- [ ] Performance targets met (DASH §10)

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

## Parked (only after Layer 2+)
- [ ] P-01 Own visual design
- [ ] P-02 FastAPI + Docker (ARCH §12)
- [ ] P-03 Public README polish + hosted demo
- [ ] P-04 Real password login
- [ ] Confirm dataset licences before publishing anything (DATA §11)
