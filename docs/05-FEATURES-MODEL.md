# TRACE — Features, Model and Evaluation

| | |
|---|---|
| Doc | 05 of the TRACE doc set |
| Purpose | Exactly what TRACE computes and how it is judged: every feature, every rule, how training rows are built, the models and baselines, the risk score and verdicts, explanations, every metric, and the Layer 2 / 3 experiments |
| Covers features | F-04 to F-08, F-12, F-14 (model side), F-15, F-16 |
| File formats | `04-CONTRACT.md` |
| Progress | Tracked only in `CHECKLIST.md` |

🔧 = tunable starting value, kept in `config/settings.yaml`. Any change is recorded in section 13.

---

## 1. Ground rules (apply to everything below)

1. **Cut-off rule:** every feature for an account at cut-off time *T* uses only transactions with `timestamp <= T`.
2. **Unit-free rule:** no feature may depend on the currency. Amounts appear only as ratios, percentiles or comparisons with the account's own history.
3. **Self-transfers** (`is_self = true`) are ignored by all features except `self_txn_share`.
4. **One code path:** the same feature functions run for training, testing, uploads, the demo file, replay and SAML-D.
5. **Who gets scored (one rule for every run kind):**
   - **Split runs (train / validation / test):** snapshots every 24 h 🔧 inside the period, plus one at the period's end. At each snapshot, accounts active in the 24 h before it are scored. An account's score for the period = its **highest** risk over the snapshots where it was active; its reasons and trail come from that snapshot. An account active at any time in the period therefore always gets a score, and every new mule in the test period (which by definition has a laundering transaction in it) is scored.
   - **Uploads and replay steps:** one cut-off; every account with any transaction in the file / window is scored.
   - Accounts with no activity in the period are not scored and do not count in metrics.

## 2. Feature catalogue

The catalogue lives in code (`trace_aml/features/catalog.py`): for each feature its `name`, `family`, `display_name`, a one-line `description`, and a `value_text` template used in explanations. This table is the spec.

Notation: *in* = transactions the account received, *out* = transactions it sent. Windows *w* ∈ {1 h, 6 h, 24 h} 🔧 end at *T*. "All" = everything up to *T*.

### 2.1 Structural (where the account sits in the network)
Built on the directed graph of all non-self transactions up to *T*, with repeat transfers between the same pair merged into one edge weighted by total amount.

| Name | Definition |
|---|---|
| `in_degree` | Number of distinct accounts that sent to it |
| `out_degree` | Number of distinct accounts it sent to |
| `in_out_degree_ratio` | `in_degree / (out_degree + 1)` |
| `counterparties` | Distinct accounts in or out |
| `pagerank` | Amount-weighted PageRank (igraph), × number of nodes so it does not shrink with graph size |
| `betweenness_local` | Betweenness with paths limited to 3 hops (igraph), normalised; dropped if too slow (`03` section 8) |
| `clustering` | Clustering coefficient on the undirected version |
| `kcore` | k-core number |
| `community_size_log` | log(1 + size of its Louvain community) |
| `community_internal_flow` | Share of the community's money that stays inside it |
| `nbr_new_share` | Share of counterparties first seen in the last 24 h |
| `nbr_mean_degree_log` | log(1 + mean degree of its counterparties) |

### 2.2 Temporal (how its activity behaves over time)
| Name | Definition |
|---|---|
| `in_cnt_{1h,6h,24h}` | Received transactions in the window |
| `out_cnt_{1h,6h,24h}` | Sent transactions in the window |
| `cnt_24h_vs_avg` | Transactions in last 24 h ÷ (average per 24 h over its history + 1) |
| `burstiness` | (σ − μ) / (σ + μ) of gaps between its transactions; −1 = regular, +1 = very bursty; NaN if < 3 transactions |
| `age_hours` | Hours since its first transaction **in the data** (the data has a start date, so this is "age in data", not true account age) |
| `active_hours_spread` | Number of distinct hours of the day it has been active (1–24) |
| `new_counterparties_24h_ratio` | New counterparties in the last 24 h ÷ (counterparties before that + 1) |
| `night_share` | Share of its transactions between 00:00 and 05:00 🔧 |

### 2.3 Fund-flow (what it does with the money)
**Matching inflow to outflow (FIFO):** walk the account's transactions in time order. Each received amount joins a queue. Each sent amount is taken from the oldest queued money first. A queued piece of money is "forwarded within Δ" if it leaves within Δ of arriving. This gives, per received rupee, whether and how fast it left.

| Name | Definition |
|---|---|
| `pass_through_{1h,24h}` | Share of all received money that was forwarded within Δ = 1 h / 24 h 🔧 |
| `holding_median_min` | Median minutes between money arriving and leaving (from FIFO matching); NaN if nothing left |
| `holding_p10_min` | 10th percentile of the same (catches the fastest forwards) |
| `net_flow_ratio` | (in − out) / (in + out) over all time; near 0 = everything that comes in goes out |
| `in_out_similarity` | 1 − \|in − out\| / (in + out) over the last 24 h |
| `fan_in_max_1h` | Most distinct senders within any 1 h window |
| `fan_out_max_1h` | Most distinct receivers within any 1 h window |
| `chain_count` | Number of time-ordered 3-hop paths through the account (A → this → C → D) where each next hop happens within 24 h 🔧 of the previous and keeps ≥ 50% 🔧 of the amount |
| `cycle_count` | Number of time-ordered cycles of length 2–4 through the account completed within 24 h 🔧 |
| `self_txn_share` | Share of its transactions that are self-transfers |

### 2.4 Relative amounts (unit-free amount signals)
| Name | Definition |
|---|---|
| `max_in_vs_own_median` | Largest received amount ÷ its own median transaction amount |
| `max_out_vs_own_median` | Same for sent |
| `max_in_pctl` | Percentile of its largest received amount among all transactions in the data up to *T* (0–100) |
| `mean_amount_pctl` | Percentile of its mean transaction amount |
| `round_amount_share` | Share of its amounts that are round (divisible by 100 in original currency) 🔧 |
| `channel_entropy` | How mixed its payment channels are (0 = one channel) |

### 2.5 Rule flags (as features)
The five rule outputs (section 3) are also fed to the model as features: `rule_chain`, `rule_fan_in`, `rule_fan_out`, `rule_cycle`, `rule_pass_through`.

### 2.6 Feature sets (used by baselines and ablations)
| Set | Contains |
|---|---|
| `TXN_ONLY` | Per-account totals and counts of its own transactions only: `in_cnt_*`, `out_cnt_*`, `cnt_24h_vs_avg`, `burstiness`, `age_hours`, `active_hours_spread`, `night_share`, all of 2.4. **No** counterparty, graph or in-to-out matching information |
| `STRUCTURAL` | 2.1 |
| `TEMPORAL` | 2.2 |
| `FLOW` | 2.3 |
| `RULES` | 2.5 |
| `FULL` | All of the above |

## 3. Rule detectors (F-05)

Each rule looks only at transactions in the last 24 h before *T* 🔧, outputs a 0/1 flag and the transaction IDs that made it fire (evidence). Thresholds are 🔧.

| Rule | Fires when |
|---|---|
| `PASS_THROUGH` | Received ≥ 2 transactions and forwarded ≥ 90% of the received amount within 1 h |
| `FAN_IN` | ≥ 5 distinct senders within 2 h, **and** ≥ 80% of that money forwarded within 2 h |
| `FAN_OUT` | ≥ 5 distinct receivers within 2 h, funded by money received in the 2 h before |
| `CHAIN` | Part of a time-ordered path of ≥ 3 hops, each hop within 1 h, each keeping ≥ 70% of the amount |
| `CYCLE` | Money leaves and returns to it through 2–4 hops within 24 h |

### 3.1 Why rules exist
1. **Baseline:** what a traditional system would do. The model must beat it.
2. **Evidence:** the transactions that made a rule fire are concrete things to show the analyst.
3. **Features:** the model may use them, and the ablation shows whether that helps.

### 3.2 Testing rules
Each rule has a test with a hand-made mini graph that must fire it and a normal account that must not.

### 3.3 Rules-only score
`rule_score` = weighted share of rules that fired, where each rule's weight is its precision on the **validation** period. Accounts are ranked by `rule_score` (ties broken by `in_cnt_24h`). This ranking is the `RULES` baseline.

## 4. Training rows

TRACE **detects** accounts that are already acting as mules; it does not predict who will become one.

- **Snapshots:** in the train period, take a cut-off every 24 h 🔧. At each snapshot, every account active in the 24 h before it is one row: its features at that cut-off + label.
- **Label of a training row:**
  - role MULE and `mule_since <= cut-off` → **1**
  - role NORMAL → **0**
  - role MULE but `mule_since > cut-off` (not yet acting as a mule) → **row dropped** (labelling it 0 would teach the model that early mule behaviour is normal)
  - role SOURCE or SINK → **row dropped**
- **Validation and test:** scored as in rule 1.5 (highest risk over the period's snapshots). **Positives are only "new" mules**: `mule_since` inside that period. Mules whose `mule_since` is before the period (train **or** validation, for the test) are "known" mules, removed from the main metrics and reported separately. SOURCE / SINK accounts are excluded from the main metrics (section 7, source / sink check).
- **Imbalance:** handled with class weights (section 5), not by throwing away rows. If memory is tight, keep all positives and a random 20% of negatives per snapshot 🔧, and record it.
- **Seeds:** every experiment is run with 5 seeds 🔧 (affects model randomness and any sampling); report mean ± standard deviation.

## 5. Models (F-06)

All approaches use the same rows, the same split and the same metrics.

| Approach | Features | Settings |
|---|---|---|
| `RULES` | Rules only | Section 3.3 |
| `TXN_ONLY` | `TXN_ONLY` set | LightGBM, same settings as main |
| `LOGREG` | `FULL` | Logistic Regression; features standardised; NaN → median + "was missing" flag; `class_weight="balanced"` |
| `RANDOM_FOREST` | `FULL` | 300 trees 🔧, `class_weight="balanced_subsample"`, NaN → median |
| `ISOLATION_FOREST` | `FULL` | Unsupervised (ignores labels); score = anomaly score. Shows how far "unusual" gets you without labels |
| **`LIGHTGBM`** (main) | `FULL` | See below |

**LightGBM starting settings** 🔧
| Setting | Value |
|---|---|
| Objective | Binary (log loss) |
| Class weight | `scale_pos_weight` = negatives ÷ positives in training rows |
| Learning rate | 0.05 |
| Max trees | 1,000, with early stopping after 50 rounds without improvement in validation PR-AUC |
| Leaves | 31 |
| Min rows per leaf | 20 |
| Feature fraction / bagging fraction | 0.8 / 0.8 (bagging every 1 round) |
| L1 / L2 | 0 / 0 |
| Tuning *(optional, only after defaults work)* | Optuna, 50 trials, maximise validation PR-AUC; leaves 15–127, min rows 10–100, L1/L2 0–5 |

### 5.1 Ablations
LightGBM retrained with one family removed at a time: `FULL − STRUCTURAL`, `FULL − TEMPORAL`, `FULL − FLOW`, `FULL − RULES`, `FULL − relative amounts`. The drop in PR-AUC shows what each family adds. This answers "is the graph worth it?" together with `TXN_ONLY`.

## 6. Risk score, verdicts and explanations (F-07, F-08)

### 6.1 Calibration and score
1. LightGBM outputs a raw probability.
2. An **isotonic calibrator**, fitted on validation rows, maps it to a calibrated probability `prob` (so "0.8" really means about 80% of such accounts are mules, on validation data).
3. `risk = round(100 × prob, 1)`.

### 6.2 Verdict bands
Cut-offs are chosen on the **validation** period, saved in `thresholds.json`:
- `likely_mule_cut` = lowest risk at which validation precision is ≥ 50% 🔧.
- `watch_cut` = risk at which validation recall reaches 80% 🔧 (if lower than `likely_mule_cut`; otherwise `watch_cut = likely_mule_cut − 10`).
- Verdict: `risk >= likely_mule_cut` → `LIKELY_MULE`; `>= watch_cut` → `WATCH`; else `NORMAL`.

### 6.3 Reasons (SHAP)
- SHAP TreeExplainer on the LightGBM model, for every `LIKELY_MULE` and `WATCH` account (and on demand in the app for any account).
- Keep the 5 features with the largest absolute SHAP value; record value, impact and direction (`UP` / `DOWN`).
- Show each with its catalogue `display_name` and `value_text`, e.g. "Pass-through (1 h) — forwarded 93% of inflow".

### 6.4 Money trail
For a flagged account *X* at cut-off *T*:
1. **Before:** walk backwards from *X* along received transactions, up to 3 hops 🔧, where each earlier hop happened before the next one and within 24 h 🔧 of it.
2. **After:** walk forwards from *X* along sent transactions, up to 3 hops, each later hop after the previous and within 24 h.
3. At each step keep only the 2 largest-amount options 🔧 to stop the search exploding.
4. Join a before-path and an after-path through *X* into a trail; rank trails by the smallest amount along them (the money that clearly flowed all the way); keep the top 3.
5. If no transaction qualifies, `has_trail = false` and the app shows "No suspicious money trail found".

### 6.5 Short label and summary
- `short_label`: from the top-impact behaviour: pass-through features → `Pass-through`; fan-in → `Fan-in collector`; fan-out → `Fan-out distributor`; chain/cycle → `Layering hop`; otherwise `Unusual activity`; `NORMAL` accounts → `Normal activity`.
- `summary`: filled from fixed templates, no language model. Example template for a pass-through account:
  > "Received {in_amount_24h} from {n_senders} accounts between {t_first} and {t_last}, then forwarded {pass_through_pct}% to {n_receivers} accounts within {holding_median}. {trail_sentence}"
  `trail_sentence`, when a trail exists: "Part of a {n_hops}-hop chain ending at {last_account}."
  Amounts are written in the display currency.

### 6.6 Grouping flagged accounts into rings (F-13)
Ground-truth rings (`02-DATA.md` 5.3) are for evaluation only. The rings the **analyst sees** are built from TRACE's own output, without labels:
1. Take all `LIKELY_MULE` and `WATCH` accounts in the run.
2. Link two of them if they appear on the same money trail, or if money passed directly between them within 24 h 🔧.
3. Each connected group of 2 or more accounts is a ring (`R-0001`, …), numbered by highest member risk.
4. For the report, compare these rings with the ground-truth rings (share of ground-truth ring members found inside one TRACE ring).

## 7. Metrics

Mules are rare (around 1 in 1,000 transactions), so accuracy is useless: flagging nobody would be 99.9% "accurate". All metrics are at **account level** on the **test** snapshot unless stated.

| Metric | Definition | Role |
|---|---|---|
| **PR-AUC** | Average precision (scikit-learn `average_precision_score`) | **Main metric** |
| Precision@K | Of the K highest-risk accounts, the share that are true mules. K = 50, 100, 500 🔧 | Main operational metric |
| Recall@K | Of all true (new) mules, the share inside the top K | Coverage |
| Precision / Recall / F1 at `likely_mule_cut` | Performance at the verdict cut-off | Threshold summary |
| False-positive rate at top 1% | Share of normal accounts inside the top 1% of risk | Customer impact |
| ROC-AUC | Standard | Secondary only (looks optimistic when positives are rare) |
| Per-pattern recall | Recall@500 split by `pattern_type` | Which shapes are caught or missed |
| Known-mule recall | Recall on mules already known before test | Reported separately |
| Source / sink check | Mean risk of SOURCE and SINK accounts vs NORMAL | Shows what the model does with excluded roles |

Every metric is reported as mean ± standard deviation over the 5 seeds, on **HI-Small** and on **LI-Small** (the harder version).

**Minimum bar (PRD section 9):** LightGBM beats `RULES` and `TXN_ONLY` on PR-AUC on the HI-Small test period. **Numeric targets are set only after this first comparison exists** and are recorded in section 13.

## 8. Replay and "caught in time" (F-12, Layer 2)

**Question:** how long before the money reached the end of the chain would TRACE have raised an alert?

1. Replay the **test period** in steps of 1 h 🔧. At each step *T*: recompute features for accounts active in the last 24 h (heavy graph measures refreshed every 6 h 🔧) and score them with the frozen model.
2. For each ring with `ring_id` whose laundering ends inside the test period:
   - `cashout_time` = time of the ring's last laundering transaction into a SINK account (if none, its last laundering transaction).
   - `first_alert_time` = first step at which **any** member's risk ≥ `likely_mule_cut`.
   - `lead_minutes` = `cashout_time − first_alert_time`. Positive = caught before cash-out.
3. Report: share of rings caught before cash-out, median and spread of `lead_minutes`, and rings never caught. State the step size, because it limits precision (with 1 h steps, lead times are accurate to about an hour).
4. Report which ring definition was used (`PATTERN_FILE` or `CONNECTED`).

## 9. Explanation correctness (F-15, Layer 3)

**Question:** when TRACE correctly flags a mule, does the money trail it shows point at real laundering accounts, or at innocent bystanders?

For every true positive (a real mule with verdict `LIKELY_MULE`) that has a trail:
- *T* = the set of other accounts on its trails.
- **Trail precision** = share of *T* whose role is MULE, SOURCE or SINK.
- **Trail hit** = 1 if at least one account in *T* is MULE, SOURCE or SINK.
- **Ring coverage** = share of the account's own ring members that appear in *T* (when `ring_id` exists).

**Comparison (what makes the number meaningful):** a random-trail baseline that picks the same number of accounts at random from the account's direct counterparties. TRACE's trails should beat it clearly. Also report the same numbers for rule evidence (section 3).

Report means and distributions; list 3 good and 3 bad examples for the report.

## 10. Unseen-dataset test (F-16, Layer 3)

1. SAML-D passes the quality gate (`02-DATA.md` section 8) and is converted to the common format.
2. Features computed with the same code; the time split is applied the same way, but **only the test period is scored**.
3. Scored with the **HI-Small-trained** model, without retraining or new features.
4. Report section 7 metrics side by side: IBM test vs SAML-D test. Also report with `likely_mule_cut` re-chosen on SAML-D validation (to separate "ranking still works" from "cut-off needs adjusting").
5. Optional extra: a LightGBM trained on SAML-D itself, as the "best possible" reference.

## 11. Feedback and retraining evaluation (F-14)

The retrain button (`03-ARCHITECTURE.md` section 3.2) is demonstrated live. Its **value** is measured offline with a **simulated analyst**:
1. Round 0: model v1 trained as normal.
2. Each round: the simulated analyst reviews the top 50 🔧 **validation-period** alerts and labels them with the true answer (Confirmed Mule / False Positive).
3. Retrain with those labels added; score the **test** period.
4. Three rounds 🔧. Report Precision@K and PR-AUC on test after each round.
Only validation-period feedback is used here (the "evaluation-safe" retrain, `03-ARCHITECTURE.md` 3.2); the live demo's retrain is a separate "demo only" version and is never part of these numbers. The honest finding may be "small gain" or "no gain"; both are reportable.

## 12. Reporting conventions

- Every table states: dataset, split, model version, number of seeds.
- Results on synthetic data are always labelled as such.
- Each experiment's report is in `reports/` (`.md` + `.json`, `04-CONTRACT.md` section 8).
- Charts to produce for the final report: PR curves (all approaches on one chart), ablation bar chart, Precision@K bars, per-pattern recall bars, lead-time distribution (L2), trail precision vs random baseline (L3), IBM vs SAML-D bars (L3), feedback rounds line (L2).

## 13. Change log

| Item | Change | Reason | Date |
|---|---|---|---|
| Numeric targets | Not set yet; set after first baseline comparison | | |
