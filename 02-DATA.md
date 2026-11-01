# TRACE — Data

| | |
|---|---|
| Doc | 02 of the TRACE doc set |
| Purpose | Which datasets TRACE uses, what is inside them, how they are converted to TRACE's common format, how mule labels are made, how data is split in time, how dataset quality is checked, and what the backups are |
| Covers features | F-01 Data loading, F-02 Quality gate, F-03 Account labels, F-11 Demo data file, F-16 Unseen-dataset test (data side) |
| Exact file formats | Defined in `04-CONTRACT.md`. This doc says *what* the data means; the contract says *exactly* how it is stored |
| Progress | Tracked only in `CHECKLIST.md` |

**Marking used in this doc**
- ✅ **Verified**: confirmed from the dataset's paper or page.
- ⚠️ **Verify on download**: believed correct but not yet confirmed; must be checked when the files are first opened, and this doc updated.
- 🔧 **Tunable**: a starting value chosen by us; may change after first results. Record any change in section 13.

---

## 1. Datasets at a glance

| Role | Dataset | Why |
|---|---|---|
| **Primary (train + test)** | IBM AMLworld, **HI-Small** | Published at NeurIPS 2023 by IBM; realistic normal behaviour; complete labels; made by someone else, so we are not grading our own exam |
| **Unseen-dataset test** (F-16) | SAML-D | Different makers, different design, many laundering types; tests whether the model works on data it never trained on |
| **Harder version of primary** | IBM AMLworld, **LI-Small** | Same format, about half the laundering rate; a free "harder" test with zero extra mapping work |
| **Backups** | AMLNet, Tide datasets, IBM Medium | Only if a dataset above fails the quality gate (section 9) |
| **Demo file** (F-11) | A slice of IBM data, restyled to ₹ / UPI | Realistic structure, clear story for the live demo |

There is **no public labelled Indian mule dataset**. This is a known limitation, stated in the report (PRD section 10).

## 2. Primary dataset: IBM AMLworld

### 2.1 What it is
- Synthetic financial transactions produced by a multi-agent simulation of people, companies and banks, with laundering behaviour mixed in. ✅
- Labels are **complete**: the generator knows where every rupee/dollar came from, so every laundering transaction is marked. ✅
- The laundering tag **carries forward**: money traced from an illicit source keeps the tag through later transfers, including when mixed with clean money. ✅ This matters for labelling (section 5).

### 2.2 Versions ✅
"HI" = higher illicit ratio, "LI" = lower illicit ratio (harder).

| Version | Days covered | Accounts | Transactions | Laundering transactions | Laundering rate |
|---|---|---|---|---|---|
| **HI-Small** ← start here | 10 | 515K | 5M | 3.6K | 1 in 981 |
| LI-Small | 10 | 705K | 7M | 4.0K | 1 in 1,942 |
| HI-Medium | 16 | 2,077K | 32M | 35K | 1 in 905 |
| LI-Medium | 16 | 2,028K | 31M | 16K | 1 in 1,948 |
| HI-Large | 97 | 2,116K | 180M | 223K | 1 in 807 |
| LI-Large | 97 | 2,064K | 176M | 100K | 1 in 1,750 |

**Why HI-Small first:** it fits on a 16 GB laptop and has the most laundering per transaction, so the model has enough examples to learn from. Medium and Large are too big for the laptop without heavy slicing.

**Important consequence:** HI-Small covers only **10 days**. The time split (section 6) and the replay test (F-12) work in **hours and days**, not weeks.

### 2.3 Columns in the transaction file ⚠️ Verify on download
File name: `HI-Small_Trans.csv` (✅ name confirmed by IBM's own code repository). Expected columns, in order:

| # | Column | Meaning |
|---|---|---|
| 1 | `Timestamp` | Date and time of the transfer |
| 2 | `From Bank` | Sender's bank ID |
| 3 | `Account` | Sender's account number |
| 4 | `To Bank` | Receiver's bank ID |
| 5 | `Account` (second one; pandas renames it `Account.1`) | Receiver's account number |
| 6 | `Amount Received` | Amount that arrived, in the receiving currency |
| 7 | `Receiving Currency` | Currency of the received amount |
| 8 | `Amount Paid` | Amount that left, in the paying currency |
| 9 | `Payment Currency` | Currency of the paid amount |
| 10 | `Payment Format` | How the money moved |
| 11 | `Is Laundering` | 1 = laundering, 0 = normal ✅ (binary 0/1) |

Account numbers are only unique **within a bank**, so an account's identity is **bank + account number** (section 4).

### 2.4 Payment formats ✅
Cheque, Credit Card, ACH, Cash, Reinvestment, Wire, Bitcoin.

### 2.5 Currencies ✅ (partly)
The paper names US dollars, euros, yuan, yen and bitcoin "and more". The full list is read from the data on download. ⚠️

### 2.6 Laundering patterns ✅
Eight shapes: **fan-out, fan-in, gather-scatter, scatter-gather, cycle, random, bipartite, stack.**

Important: **most laundering transactions are not part of a neat pattern.** In LI-Large, about 19K laundering transactions belong to these patterns, and about 81K are "other" laundering (money that is dirty but moved in ordinary-looking ways). ✅ So:
- The model must find mules beyond the eight shapes, and
- Anything that needs "which ring does this belong to" (F-12 lead time, F-13 ring view, F-15 explanation correctness) can only use the pattern subset as ground truth, or define rings itself (section 5.3).

### 2.7 Patterns file ⚠️ Verify on download
Published research describes a separate labelled patterns file (expected name `HI-Small_Patterns.txt`) that lists each laundering attempt, its pattern type and its transactions. One paper reports **370 pattern groups for HI-Small**. The exact file format is not confirmed. When downloaded:
1. Confirm it exists and record its exact format in section 13.
2. Write the parser for it (feeds `ring_id` in section 5.3).
3. If it does not exist, use the fallback ring definition in section 5.3.

### 2.8 Licence ⚠️ Verify on download
Reported to be published under a **Community Data License Agreement**. Confirm the exact licence on the Kaggle page and record it in section 11. Use for a student project is expected to be fine either way.

### 2.9 Where to get it
Kaggle: *IBM Transactions for Anti Money Laundering (AML)*, user `ealtman2019`. Needs a free Kaggle account to download. Download only the HI-Small and LI-Small files (and their patterns files) to start.

## 3. Unseen-dataset test: SAML-D

| Fact | Value |
|---|---|
| Full name | Synthetic Anti-Money Laundering Dataset (SAML-D) |
| Makers | Oztas et al., Bournemouth University; IEEE ICEBE 2023 ✅ |
| Size | ~9.5M transactions, ~677K accounts ✅ |
| Columns | 12 ✅ (exact names ⚠️ verify on download) |
| Typologies | 28: 11 normal, 17 suspicious ✅ |
| Design | Built with input from AML specialists, existing datasets and literature ✅ |
| Setting | UK-style (pounds), with sender / receiver bank locations ⚠️ |
| Where | Kaggle, `berkanoztas/synthetic-transaction-monitoring-dataset-aml` ✅ |
| Licence | ⚠️ Not confirmed. The papers about it are CC BY-NC; the data licence must be read on the Kaggle page |

**Expected columns** ⚠️ (from memory; confirm on download): Time, Date, Sender_account, Receiver_account, Amount, Payment_currency, Received_currency, Sender_bank_location, Receiver_bank_location, Payment_type, Is_laundering, Laundering_type.

**Rules for using SAML-D (F-16):**
- It is **never used for training**. The model trained on IBM scores it as-is.
- It must pass the quality gate (section 8) like any dataset.
- Features must be computed the same way (same code), only the mapping (section 4) differs.
- If 9.5M rows is too heavy, use a contiguous time slice, and say so in the results.

## 4. TRACE common format

Every dataset (IBM, SAML-D, backups, demo file) is converted to **one common format** before anything else touches it. After this step, no code knows or cares which dataset it came from. Exact column types are in `04-CONTRACT.md`.

### 4.1 Transactions (what the pipeline and the model see)
| Column | Meaning |
|---|---|
| `txn_id` | Unique ID we assign (row number after sorting) |
| `timestamp` | Date-time of the transfer |
| `sender_id` | Sender's account ID (see 4.3) |
| `receiver_id` | Receiver's account ID |
| `amount` | Amount in the **base currency** (see 4.4) |
| `amount_original` | Amount as given in the source data |
| `currency_original` | Currency as given in the source data |
| `channel` | Payment format / type, as given |
| `dataset` | Which dataset this row came from (`ibm_hi_small`, `saml_d`, `demo`, …) |

### 4.2 Labels (kept in a **separate file**)
| Column | Meaning |
|---|---|
| `txn_id` | Links to the transaction |
| `is_laundering` | 0 / 1 from the source data |
| `pattern_type` | Pattern name if known, else empty |
| `pattern_id` | Pattern group if known, else empty |

**Why separate:** the feature code must never be able to read the labels by accident. Only the training and evaluation steps open the labels file. This is a leakage guard.

### 4.3 Account IDs
- IBM: `"<bank>_<account>"`, e.g. `"011_8000EBD30"`. One bank + one account number = one account.
- SAML-D: the account number as given (it has no bank ID).
- Demo file: renamed to `ACC-xxxx` for readability, with the mapping kept.

### 4.4 Amounts and currency
1. A settings file holds **one fixed rate per currency** to a base currency (USD for IBM). 🔧
2. `amount` = the **paid amount** converted to base currency (the money that left the sender).
3. The dashboard shows rupees by multiplying with **one display rate** (₹ per USD) from the same settings file. 🔧 Set it to the current rate when the file is created.
4. Because every amount is multiplied by the same number, all relative features (PRD F-04) and therefore all scores stay exactly the same. Currency choice never changes a score.

### 4.5 Mapping tables

**IBM → TRACE**
| TRACE | IBM |
|---|---|
| `timestamp` | `Timestamp` |
| `sender_id` | `From Bank` + `Account` |
| `receiver_id` | `To Bank` + `Account.1` |
| `amount_original` / `currency_original` | `Amount Paid` / `Payment Currency` |
| `channel` | `Payment Format` |
| labels file `is_laundering` | `Is Laundering` |

**SAML-D → TRACE** ⚠️ (column names to confirm)
| TRACE | SAML-D |
|---|---|
| `timestamp` | `Date` + `Time` |
| `sender_id` / `receiver_id` | `Sender_account` / `Receiver_account` |
| `amount_original` / `currency_original` | `Amount` / `Payment_currency` |
| `channel` | `Payment_type` |
| labels file `is_laundering` / `pattern_type` | `Is_laundering` / `Laundering_type` |

## 5. Account labels (F-03)

The source data labels **transactions**. TRACE scores **accounts**. This section is the one rule that turns one into the other. It must be applied identically to every dataset.

### 5.1 Roles
For each account, look at its laundering transactions:

| Role | Rule | Used as |
|---|---|---|
| **Mule** | Has **both** received ≥ 1 and sent ≥ 1 laundering transaction | **Positive (1)**: what TRACE is trained to find |
| **Source** | Sent laundering but never received any | Excluded from training and main metrics; reported separately |
| **Sink** | Received laundering but never sent any | Excluded from training and main metrics; reported separately |
| **Normal** | No laundering transactions at all | **Negative (0)** |

**Why mules = "both in and out":** a mule is by definition a pass-through account. Sources and sinks are a different job (the fraudster's or the cash-out), and labelling them as "normal" would teach the model wrong behaviour, while labelling them "mule" would blur the target. So they are set aside. 🔧 Revisit only if the quality gate shows too few mules.

### 5.2 When an account became a mule
`mule_since` = time of the account's **first** laundering transaction (in or out). Used by the time split (section 6) and the replay test (F-12).

### 5.3 Rings
- **Primary:** from the patterns file. All accounts in the same pattern group share a `ring_id`.
- **Fallback** (no patterns file, or laundering outside the eight patterns): connect mule accounts that sent laundering money to each other; each connected group is a ring.
- Every reported number that uses rings says which definition was used.

### 5.4 Account labels file
One row per account: `account_id`, `role`, `is_mule` (1 / 0 / empty for excluded), `mule_since`, `ring_id`. Format in `04-CONTRACT.md`.

## 6. Time split

**Rule:** the model trains on earlier time and is tested on later time. Never random. Never k-fold.

| Period | Share of the timeline 🔧 | Used for |
|---|---|---|
| Train | first 60% | Fitting models |
| Validation | next 20% | Choosing settings, verdict cut-offs, alert threshold |
| Test | last 20% | Final numbers. Touched **once**, at the end |

Rules that come with it:
1. **Features at a cut-off use only transactions up to that cut-off.** An account scored at the end of the train period cannot see anything that happened later.
2. **An account's label in a period** = whether it was a mule *by the end of that period* (using `mule_since`).
3. **Test mules must be new:** the main test numbers count only mules whose `mule_since` falls **inside the test period**. Mules that started earlier (in train **or** validation) are "known" mules and are reported separately. The same rule applied to the validation period defines validation positives.
4. The split is by **timestamp**, cut at whole hours.
5. With HI-Small's 10 days, 60/20/20 ≈ 6 / 2 / 2 days. If the quality gate shows a period has too few mules, adjust the shares and record why in section 13.

## 7. Cleaning steps (in order)

1. Read with explicit column types (saves memory).
2. Parse timestamps; drop rows that fail (count them).
3. Drop rows with amount ≤ 0 (count them).
4. Drop exact duplicate rows (count them).
5. Build account IDs (section 4.3) and convert amounts (section 4.4).
6. **Self-transfers** (sender = receiver): kept in the file, flagged, and **left out of graph features** (they do not move money between accounts). 🔧
7. Sort by timestamp; assign `txn_id`.
8. Write transactions and labels to separate files.
9. Every count from steps 2–4 and 6 goes into the quality report.

## 8. Data quality gate (F-02)

A script that runs on **any** dataset in the common format and writes a **report card**. Every dataset must pass before training or testing on it. A ❌ blocks use until fixed or the dataset is replaced. Thresholds are 🔧 tunable starting values.

| # | Check | What it measures | ✅ Pass | ⚠️ Warn | ❌ Fail |
|---|---|---|---|---|---|
| C1 | **Format** | Required columns, parsed timestamps, positive amounts, duplicates | All present; < 0.1% rows dropped | 0.1–1% dropped | Column missing, or > 1% dropped |
| C2 | **Labels** | Count of mule / source / sink / normal accounts | ≥ 100 mules | 30–99 | < 30 |
| C3 | **Time spread** | Mules per period; share of test mules not seen in train | ≥ 30 mules in each period and ≥ 80% of test mules new | 10–29 in a period, or 50–80% new | < 10 in a period, or < 50% new |
| C4 | **Giveaway test** | Train a tiny model on **one column at a time** (amount, channel, currency, hour of day, bank); PR-AUC of each | Every single column < 0.3 | Any column 0.3–0.8 | Any column ≥ 0.8 |
| C5 | **Overlap** | Best possible F1 from **one threshold** on one simple stat (in-degree, out-degree, total amount) | All < 0.5 | Any 0.5–0.8 | Any ≥ 0.8 |
| C6 | **Graph sanity** | Share of mules inside the largest connected part; whether normal accounts also have high degree | ≥ 50% of mules in the largest part and normal hubs exist | 25–50% | < 25%, or no normal hubs |
| C7 | **Compatibility** | Every TRACE feature can be computed (no column all-empty or constant) | All features computable | Up to 3 features constant | More than 3 constant, or any missing |

**How C4 is run:** transaction-level columns (amount, channel, currency, hour of day, bank) are tested at **transaction level** against `is_laundering`; the same columns aggregated per account (e.g. most-used channel) are tested at **account level** against `is_mule`. Both results are reported.

**If C4 or C5 fails (escape path):**
1. The failing column is recorded as a **shortcut**.
2. Every TRACE feature built from it is removed (e.g. if `channel` fails, drop `channel_entropy` and any channel-based feature), and this is recorded in section 13 and `05-FEATURES-MODEL.md` section 13.
3. The gate is re-run without those features; if it passes, training continues.
4. Only if the shortcut cannot be removed (e.g. raw amount itself) does the dataset stay blocked; then try the next backup in section 9 and report the shortcut as a finding about the dataset.

**Why C4 and C5 matter most:** they catch the "friend's overfitting" problem *before* any modelling. If one column or one threshold already separates mules almost perfectly, a high model score would prove nothing.

**Output:** `reports/quality_<dataset>.md` (human-readable report card) and `reports/quality_<dataset>.json` (same numbers, for the checklist and the final report).

## 9. Backups

| Backup | Facts | Switch to it if… |
|---|---|---|
| IBM **LI-Small** | Same format as HI-Small, half the laundering rate (table 2.2) | Already planned as the harder test; becomes primary only if HI-Small fails C4 / C5 |
| IBM **HI-Medium** | 16 days, 32M transactions | HI-Small has too few mules per period (C3); use a time slice to fit memory |
| **AMLNet** | 1,090,173 transactions, 1,745 laundering (~0.16%), 195 days, AUD, CC BY-NC 4.0 (non-commercial; fine for a student project). Zenodo record 16736515; a newer version exists ⚠️ | SAML-D cannot be used (licence or quality gate) |
| **Tide** reference datasets | Published with the Tide generator paper (2026), Zenodo DOI 10.5281/zenodo.18804069 ⚠️ not opened | Need a dataset with adjustable difficulty later |

## 10. Demo data file (F-11)

**Approach (decided): a slice of real IBM test-period data, restyled.** Real structure looks more convincing than an invented ring.

**Where the demo scores come from:** network measures (PageRank, percentiles, communities) on a 200–500-account slice would differ from those on the full graph, and the ring might not score high. So the **pre-scored demo run is cut out of the full test run**: same scores, reasons and trails, with account IDs renamed and amounts in ₹. The demo CSV can still be uploaded live to show the upload flow, but its scores are computed from the small file alone and may differ; the demo story uses the pre-scored run.

Steps:
1. Pick one laundering pattern group from the **test period** with a clear shape (e.g. a 4–6 hop chain or a fan-in → fan-out).
2. Take its accounts plus their direct counterparties during the same window. Aim for **about 200–500 accounts**. 🔧
3. Rename accounts to `ACC-xxxx`.
4. Convert amounts to ₹ with the display rate.
5. Relabel channels for an Indian look with a fixed table in the settings file (e.g. Wire → IMPS, ACH → NEFT, Credit Card → UPI). 🔧 **This is cosmetic only.** The dashboard states the file is "restyled IBM data".
6. Save the CSV in the TRACE upload format (labels kept separately for checking).
7. Build the pre-scored demo run by slicing the full test run's files to these accounts and applying the same renaming and ₹ conversion (run kind `DEMO`). The demo never depends on a live upload (PRD section 6).

## 11. Licences

| Dataset | Licence | Status |
|---|---|---|
| IBM AMLworld | Reported: Community Data License Agreement | ⚠️ Confirm on Kaggle page |
| SAML-D | Unknown for the data | ⚠️ Confirm on Kaggle page |
| AMLNet | CC BY-NC 4.0 | Recorded from Zenodo |
| Tide datasets | Tide code is MIT; data licence unknown | ⚠️ Only if used |

Rules:
- For coursework use, an unconfirmed licence does **not** block work. Confirm it only **before publishing** (paper, public repo, hosted demo) or sharing the data.
- Raw dataset files are **never** committed to the repo or re-shared, whatever the licence. Only the small demo file is committed, and only after the source licence is confirmed.

## 12. Where data lives

| Folder | Contents | Rules |
|---|---|---|
| `data/raw/` | Downloaded files, exactly as downloaded | Never edited. Not committed to Git (too big) |
| `data/processed/` | Common-format transactions and labels files | Rebuilt by the pipeline at any time |
| `data/demo/` | Demo file and its pre-scored copy | Small enough to commit, once the IBM licence is confirmed (section 11) |
| `reports/` | Quality report cards | Committed |

Full folder layout is in `03-ARCHITECTURE.md`.

## 13. Change log and verification record

Fill in as things are confirmed or changed.

| Item | Status | Finding / change | Date |
|---|---|---|---|
| IBM column names (2.3) | ⚠️ open | | |
| IBM full currency list (2.5) | ⚠️ open | | |
| IBM patterns file format (2.7) | ⚠️ open | | |
| IBM licence (2.8) | ⚠️ open | | |
| SAML-D column names and setting (3) | ⚠️ open | | |
| SAML-D licence (3) | ⚠️ open | | |
| Base and display exchange rates (4.4) | 🔧 set when settings file is created | | |
| Time split shares (6) | 🔧 60/20/20 | | |

## 14. Sources

- Altman et al., "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models", NeurIPS 2023 Datasets & Benchmarks. https://arxiv.org/abs/2306.16424 (dataset sizes, patterns, labelling)
- IBM Multi-GNN repository (confirms `HI-Small_Trans.csv` name and Kaggle source). https://github.com/IBM/Multi-GNN
- Kaggle: IBM Transactions for AML. https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- Oztas et al., SAML-D, IEEE ICEBE 2023. https://eprints.bournemouth.ac.uk/40982/
- Kaggle: SAML-D. https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml
- Customer-level fraud benchmark paper using SAML-D (account count). https://arxiv.org/abs/2404.14746
- AMLNet on Zenodo. https://zenodo.org/records/16736515
- Tide generator paper. https://arxiv.org/abs/2603.01863
