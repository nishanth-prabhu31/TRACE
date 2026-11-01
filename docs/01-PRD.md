# TRACE — Product Requirements Document (PRD)

**TRACE: Transaction Risk Analytics and Chain Evaluation**

| | |
|---|---|
| Doc | 01 of the TRACE doc set |
| Purpose | Defines **what** TRACE is, **why** it exists, **who** it is for, and **what counts as done**. It does not say *how* to build it (see the doc map at the end). |
| Course | AIML Lab, Part B (IS353IA), team of 2 |
| Timeline | As fast as possible. No fixed dates in these docs. |
| Progress | Tracked only in `CHECKLIST.md`, not in this file |

---

## 1. The problem

Fraudsters rarely leave stolen money in one place. They push it through **money mule accounts**: ordinary-looking accounts, often rented from students or job-seekers, that receive money and pass it on within minutes. After a few hops (called **layering**), the trail is hard to follow and the money is withdrawn.

Most bank monitoring checks **one transaction at a time** against fixed rules ("flag anything above ₹X"). That approach:

- raises many false alarms on genuine customers, and
- misses the real signal, which only appears when you look at **how money moves between accounts**.

In India this matters more because UPI and instant transfers let money hop across several accounts in minutes. The Reserve Bank Innovation Hub built MuleHunter.AI for exactly this problem, which shows it is real and current.

## 2. What TRACE is (one paragraph)

TRACE treats transactions as a **network**: every account is a point (node), every transfer is an arrow (edge) with an amount and a time. For every account it measures where the account sits in that network, how bursty its activity is, and how much of the money it receives it forwards and how fast. A machine-learning model (LightGBM) turns those measurements into a **risk score from 0 to 100**. A fraud analyst sees a ranked list of alerts, opens any account, sees **why** it was flagged (top reasons, the money trail, a plain-English summary), explores the surrounding network interactively, and records a verdict. Those verdicts can be used to retrain the model.

## 3. Who it is for

**Primary user: the bank fraud analyst.**
- Works through a daily queue of alerts.
- Needs to decide quickly and defend that decision.
- Does not trust a bare number; needs evidence.
- Is the one who decides. TRACE never acts on an account by itself.

**Secondary audience: the evaluator / interviewer watching the demo.**
- Needs to understand the problem in under a minute.
- Needs to *see* the money flow and the verdict, not read logs.
- Will ask "why this algorithm?", "how do you know it's not overfitting?", "is this real data?". The project must have honest, ready answers to all three.

## 4. Goals, non-goals, anti-goals

### Goals
1. A **working, interactive demo** that runs on a normal laptop.
2. An ML model that **beats simple rules and a model without graph features**, shown with honest numbers.
3. Every alert comes with **evidence an analyst can check**: reasons, money trail, summary.
4. **Honest evaluation**: time-ordered testing, imbalance-aware metrics, a data quality gate, and testing on a dataset the model never trained on.
5. Strong enough to put on a **resume** (clean repo, clear README, demoable).

### Non-goals (deliberately not doing)
- Real-time streaming or production deployment.
- Using real bank or customer data.
- Deep learning / graph neural networks / anything that needs a GPU.
- Paid APIs or LLM-generated explanations.
- Building a large custom data generator.
- Mobile app, KYC, crypto or cross-border specifics.

### Anti-goals (would count as failure even if it "works")
- **Near-perfect scores that come from a data shortcut**: the model learning a giveaway in the data instead of real mule behaviour.
- Presenting results on synthetic data as if they were real-world performance.
- A dashboard that shows a score with **no reason** behind it.
- The system freezing, blocking or "punishing" an account automatically.
- A "hairball" graph nobody can read.
- A tangle where nobody on the team can explain a part of the system in the viva.

## 5. Scope: features by layer

Work is built in **layers**. Each layer is a complete, demoable project on its own; later layers only add to it. Feature IDs (F-xx) are used by every other doc and by `CHECKLIST.md`.

### Core (must have)
| ID | Feature | One-line description |
|---|---|---|
| F-01 | Data loading | Load the IBM AMLworld dataset and convert it to TRACE's common format |
| F-02 | Data quality gate | Report card (pass / warn / fail) that every dataset must go through before training |
| F-03 | Account labels | Turn the dataset's per-transaction laundering labels into per-account mule labels |
| F-04 | Feature engine | Compute structural, temporal and fund-flow features per account, all unit-free |
| F-05 | Rule detectors | Hand-written pattern rules (chain, fan-in, fan-out, cycle, pass-through). Used as a baseline and as evidence |
| F-06 | Models and baselines | LightGBM (main) compared against rules-only, transaction-only, Logistic Regression and Random Forest |
| F-07 | Risk score and verdict | Calibrated 0–100 score per account plus a verdict band |
| F-08 | Explanations | Top reasons (SHAP), money trail, plain-English summary for every flagged account |
| F-09 | Analyst dashboard | Upload file, alert queue, account detail, interactive network graph with hop and back, follow-the-money view |
| F-10 | Feedback capture | Confirmed Mule / False Positive / Needs Investigation buttons, saved with the model version |
| F-11 | Demo data file | Small script that produces a ₹ / UPI-styled demo file with one clear mule ring |
| F-17 | Analyst identity and roles | Sidebar picker "Logged in as" (no password). Analyst role reviews and gives feedback; Admin role can also retrain and switch model version |

### Layer 2 (the edge)
| ID | Feature | One-line description |
|---|---|---|
| F-12 | Replay and "caught in time" | Replay data in time order and measure how many minutes before cash-out each ring was first flagged |
| F-13 | Ring view | Group flagged accounts into rings and show each ring as one case ("Ring #3: 6 accounts, ₹12L moved") |
| F-14 | Retrain and model versions | Analyst presses "Retrain" when they choose; panel compares v1 / v2 / v3 and allows switching back |

### Layer 3 (research flex)
| ID | Feature | One-line description |
|---|---|---|
| F-15 | Explanation correctness | Measure whether the money trail shown to the analyst actually points at the real laundering accounts |
| F-16 | Unseen-dataset test | Train on IBM, test on SAML-D without redesigning features |

### Parked (later, only after the layers above)
| ID | Item |
|---|---|
| P-01 | Own visual design for the dashboard (current mockup look is rejected as too generic) |
| P-02 | FastAPI + Docker wrapper |
| P-03 | Public GitHub README polish and free hosted demo link |
| P-04 | Real password login (mainly needed if the demo goes on a public link) |

## 6. The demo story (main user journey)

This is the walkthrough the finished Core must support, in this order:

0. **Pick who you are.** The analyst selects their name and role in the sidebar.
1. **Upload.** The analyst uploads a transactions file in TRACE's format. TRACE checks the format and shows a clear error if it is wrong. (A pre-scored dataset is also available without uploading, so the demo never depends on a live upload.)
2. **Scoring.** TRACE computes features and scores every account. A header shows totals: transactions, accounts, number of high-risk alerts.
3. **Alert queue.** A list of accounts sorted by risk, each with score, verdict and a short label. Searchable.
4. **Open an account.** Shows the risk score, verdict, top reasons (what pushed the score up and what pulled it down) and a plain-English summary.
5. **Network view.** The account in the centre, its direct counterparties around it (capped so it stays readable). Arrow thickness = money; colour = risk; several transfers between the same two accounts merge into one labelled arrow ("4 txns · ₹1.8L"). Hover shows details.
6. **Hop.** Clicking any neighbour re-centres the graph on that account. A Back control and a breadcrumb show the path taken.
7. **Follow the money.** For a flagged account, shows the time-ordered chain the money took (A → B → C → D with times and amounts).
8. **Decide.** The analyst marks Confirmed Mule / False Positive / Needs Investigation. The decision is saved.
9. *(Layer 2)* **Retrain.** The analyst presses Retrain; a new model version appears next to the old one with its scores.

## 7. Requirements and acceptance criteria

"Done" for each feature means every acceptance line below is true. Technical detail lives in the docs named in brackets.

**F-01 Data loading** *(Data doc)*
- IBM AMLworld (starting with HI-Small) loads into the common format: sender, receiver, amount, timestamp, channel. Labels go into a **separate** file (leakage guard).
- Accounts are identified by bank + account number.
- Works on a time slice if the full file does not fit in memory.

**F-02 Data quality gate** *(Data doc)*
- Runs on any dataset in the common format and prints a report card with all checks: format, labels, time spread, giveaway test, overlap, graph sanity, compatibility.
- A dataset with a ❌ is not used for training until fixed or replaced.

**F-03 Account labels** *(Data doc)*
- Every account gets a label: mule / normal, with the rule used to derive it written down.
- Train, validation and test periods each contain mules; the main test numbers count only "new" mules, whose first laundering transaction falls inside the test period.

**F-04 Feature engine** *(Features & Model doc)*
- Features are computed only from transactions **up to** the time being scored (no future data).
- Amount features are relative (compared with the account's own history or the dataset), so currency does not change the score.
- Output saved to a file so it is not recomputed every run.

**F-05 Rule detectors** *(Features & Model doc)*
- One detector per pattern; each returns a flag plus the transactions that triggered it.
- Rules alone produce a ranked list so they can be scored as a baseline.

**F-06 Models and baselines** *(Features & Model doc)*
- All six approaches (rules-only, transaction-only, Logistic Regression, Random Forest, Isolation Forest as an unsupervised reference, LightGBM) are trained and tested on the same time-ordered split and reported side by side.
- Results are repeatable (fixed random seeds).

**F-07 Risk score and verdict** *(Features & Model doc)*
- Every account gets a score from 0 to 100.
- Verdict bands: **Likely mule**, **Watch**, **Looks normal**. Cut-offs are chosen on the validation period, not fixed in advance.

**F-08 Explanations** *(Features & Model doc)*
- 100% of flagged accounts have all three parts: top reasons, money trail (or a clear "no trail found"), summary.
- The summary is generated from a template, not a language model.

**F-09 Analyst dashboard** *(Dashboard doc)*
- Supports the full demo story in section 6, steps 1–8.
- Opening an account and its graph takes under about 3 seconds on a laptop.
- The graph never shows more than a readable number of accounts at once.
- If clicking a node cannot be supported reliably, a search / dropdown to pick the account is an accepted fallback.

**F-10 Feedback capture** *(Dashboard doc, Contract doc)*
- Each decision is stored with account, decision, analyst name, time and model version.
- Decisions survive closing and reopening the app.

**F-17 Analyst identity and roles** *(Dashboard doc)*
- The sidebar shows "Logged in as" with a name and role; there is no password.
- Analyst role: alert queue, account view, feedback. Admin role: everything, plus Retrain and switching the active model version.
- The docs and the demo state plainly that this is role separation for the workflow, **not security**.

**F-11 Demo data file** *(Data doc)*
- Produces a small file in TRACE's format with ₹ amounts and UPI / IMPS / NEFT channels, containing at least one clear mule ring.
- Uses one fixed exchange rate kept in a settings file.

**F-12 Replay and "caught in time"** *(Features & Model doc)*
- Data is replayed in time order with re-scoring at fixed steps.
- For every ring: first alert time, cash-out time and the lead time between them. Reported as an average and a distribution.

**F-13 Ring view** *(Dashboard doc)*
- Flagged accounts are grouped into rings; each ring shows member count, total money moved and its accounts.

**F-14 Retrain and model versions** *(Dashboard doc, Features & Model doc)*
- Retrain runs only when the analyst presses the button.
- Models retrained live from demo or test-period feedback are marked "demo only" and never used in reported metrics; the measured value of feedback comes from the offline simulated-analyst test.
- Each version is saved with its metrics; the analyst can switch the active version back.

**F-15 Explanation correctness** *(Features & Model doc)*
- For correctly flagged mules, measures how much of the shown money trail is made up of truly laundering accounts.

**F-16 Unseen-dataset test** *(Data doc, Features & Model doc)*
- SAML-D passes the quality gate, is mapped to the common format, and is scored by a model trained only on IBM. Results are reported next to the IBM results.

## 8. Non-functional requirements

| Area | Requirement |
|---|---|
| Hardware | Runs on a laptop CPU with 16 GB RAM. No GPU needed. Google Colab is an allowed backup for heavy steps |
| Cost | Free tools only. No paid APIs |
| Language and tools | Python. Streamlit, pandas, NetworkX, LightGBM, scikit-learn, SHAP, SQLite. Reasons are in the Architecture doc |
| Reproducibility | Fixed random seeds; every pipeline step saves its output to a file so any step can be rerun alone |
| No data leakage | Training never sees future transactions; this is tested automatically |
| Readability | Every part must be explainable by either team member in the viva |

## 9. How success is measured

Exact definitions and formulas are in the Features & Model doc. **No numeric targets are promised until the first baseline results exist**; targets are then set from real numbers.

| Question | Measured by |
|---|---|
| Does it find mules among many normal accounts? | **PR-AUC** (main metric), Precision@K, Recall@K |
| Is the graph worth it? | LightGBM with graph features vs transaction-only model vs rules |
| Why LightGBM? | Side-by-side with Logistic Regression and Random Forest |
| Does it catch mules in time? *(L2)* | Lead time before cash-out |
| Are the explanations right? *(L3)* | Explanation correctness score |
| Is it overfitting to one dataset? *(L3)* | IBM-trained model tested on SAML-D |
| Is the demo usable? | Account + graph opens in about 3 seconds; 100% explanation coverage |

**Minimum bar for "the model works":** LightGBM beats the rules-only baseline and the transaction-only model on PR-AUC on the IBM test period.

## 10. Ethics and honesty rules

1. **Decision support only.** TRACE ranks and explains; a human decides. No automatic action on any account.
2. **No real personal data.** All datasets are synthetic.
3. **No demographic inputs.** No age, gender, religion, location. Behaviour only.
4. **Honest labelling of results.** Every reported number states which dataset it came from. Synthetic results are never presented as real-world performance.
5. **Known limitation stated openly.** No public labelled Indian mule dataset exists; the India relevance comes from the problem, the features and the demo, not the training data.

## 11. Assumptions and constraints

- Two team members, both new to ML; Python is known or being learned separately.
- Real bank data is not available, and never will be for this project.
- The IBM dataset's file list, licence and pattern file still need to be confirmed on its Kaggle page (Data doc).
- Evaluators want a working demo first; a paper is optional.

## 12. Decisions already made

| Decision | Why |
|---|---|
| Train on IBM AMLworld, not a self-built generator | Credible, published at NeurIPS, made by someone else, so we are not grading our own exam |
| SAML-D as the unseen-dataset test | Directly answers "is it overfitting to one dataset?" |
| Only a tiny demo-file script, no big generator | Saves weeks; the demo still gets a clear ₹ story |
| LightGBM as the main model | Fast on CPU, strong on tables, works well with SHAP; published work shows graph features + boosted trees match or beat deep graph models |
| Rules are a baseline, not the product | Rules are decades old; the model must beat them |
| Streamlit, no React | Pure Python, much less to learn |
| SQLite, not PostgreSQL | One file, nothing to install; data size and single user fit it |
| No FastAPI / Docker at first | Adds learning time without changing what the evaluator sees |
| Unit-free features + one exchange rate in settings | Scores do not change with currency; also enables the SAML-D test |
| Retrain only on analyst's button press | Retraining is fast, but automatic retraining adds complexity for little gain |
| Bank fraud analyst as the user | Matches how mule detection works in practice and gives the demo a natural flow |
| Name and role picker, no real login | Local demo with no real data; the picker gives an audit trail (who decided what). Real login parked as P-04 |

## 13. Open questions

| # | Question | Owner |
|---|---|---|
| Q1 | Who does what between the two team members (default proposal: one on data + ML, one on dashboard, swapping in Layer 2) | Team |
| Q2 | Interim report dates | Team |
| Q3 | Final visual design direction (P-01) | Team, later |

## 14. Doc map

| # | Doc | Answers |
|---|---|---|
| 01 | `docs/01-PRD.md` (this doc) | What, why, for whom, what counts as done |
| 02 | `docs/02-DATA.md` | Which datasets, how they are converted and labelled, the quality gate, backups |
| 03 | `docs/03-ARCHITECTURE.md` | System design: components, data flow, folder layout, tool choices |
| 04 | `docs/04-CONTRACT.md` | Exact format of every file passed between parts |
| 05 | `docs/05-FEATURES-MODEL.md` | Every feature, rules, models, scoring, explanations, evaluation and metrics |
| 06 | `docs/06-DASHBOARD.md` | Every screen, interaction and state |
| — | `AGENTS.md` | Working rules for anyone (person or tool) building TRACE |
| — | `CHECKLIST.md` | Short progress tracker; the only place progress is marked |

## 15. Glossary

| Term | Plain meaning |
|---|---|
| Money mule | An account that receives illegal money and passes it on, often owned by someone paid to "rent" it |
| Layering | Moving money through many accounts so its origin is hard to trace |
| Fan-in / fan-out | Many accounts sending to one / one account sending to many |
| Pass-through | An account that forwards most of what it receives, quickly |
| Cash-out | The point where laundered money leaves the chain (the last account) |
| Ring | A group of mule accounts working together |
| Node / edge | Account / transfer in the network |
| Feature | One measured number about an account that the model learns from |
| LightGBM | The ML model: builds many small decision trees that vote together |
| SHAP | Method that shows how much each feature pushed a score up or down |
| PR-AUC | Score for how well rare cases (mules) are ranked at the top; better than accuracy when mules are rare |
| Precision@K | Of the top K alerts, how many are real mules |
| Recall@K | Of all real mules, how many are inside the top K alerts |
| Baseline | A simpler method the main model must beat |
| Data leakage | The model accidentally seeing information it would not have in real life (e.g. future transactions) |
| Time-ordered split | Train on earlier data, test on later data, never mixed |
| Synthetic data | Artificially generated data that imitates real transactions |
