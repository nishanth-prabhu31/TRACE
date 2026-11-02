# TRACE — Work Split

Who owns what, so the two of us can build in parallel without stepping on each other.

| Person | Owns | In one line |
|---|---|---|
| **A — Nishanth** | **Data + Model** | Turns raw transactions into scored, explained accounts, written to files in the contract format |
| **B — Teammate** | **Dashboard** | Turns those files into the screens the analyst uses, and saves the analyst's decisions |

The two halves meet at **one place only: the files defined in `docs/04-CONTRACT.md`.** A writes them, B reads them. As long as both stick to the contract, the halves plug together.

---

## 1. How we work in parallel

```
            ┌──────────────── shared base (first) ────────────────┐
            │  contract.py + check_contract.py (A)                │
            │  make_fake_data.py (B)                              │
            └─────────────────────────────────────────────────────┘
                     │                                   │
     A: data → features → model → scores        B: screens on FAKE data
                     │                                   │
                     └──────► first integration ◄────────┘
                       (B points the app at A's first real run)
                                     │
                         both: polish, demo, Layer 2 (swap roles)
```

- **B never waits for A.** B builds on `data/fake/`, which has exactly the same format as A's real output.
- **First integration** happens as soon as A produces one real run (checklist: "P7 writes a real run that passes the contract checker"), not at the end.
- **Layer 2: swap.** A builds a Layer 2 screen, B builds a Layer 2 pipeline piece, so both of us can explain both halves in the viva.

## 2. Ownership

### A — Data + Model
| Area | Files / folders | Checklist items | Docs to follow |
|---|---|---|---|
| Contract in code | `trace_aml/contract.py`, `scripts/check_contract.py` | 0. Setup | CON (all) |
| Data loading + cleaning | `trace_aml/pipeline/ingest.py` | F-01 | DATA §4, §7 |
| Labels + time split | `trace_aml/pipeline/labels.py` | F-03 | DATA §5, §6 |
| Quality gate | `trace_aml/pipeline/quality.py` | F-02 | DATA §8 |
| Features | `trace_aml/features/*`, `trace_aml/pipeline/features.py` | F-04 | FM §1–2 |
| Rules | `trace_aml/pipeline/rules.py` | F-05 | FM §3 |
| Training + baselines + evaluation | `trace_aml/models/*`, `pipeline/train.py`, `pipeline/evaluate.py` | F-06 | FM §4–5, §7 |
| Scores + explanations | `pipeline/score.py`, `pipeline/explain.py`, `trace_aml/explain/*` | F-07, F-08 | FM §6 |
| Demo file + demo run | `pipeline/demo_file.py` | F-11 | DATA §10 |
| Settings | `config/settings.yaml` (model, data, feature, rule values) | — | ARCH §6 |
| Tests for all of the above | `tests/test_ingest.py`, `test_labels.py`, `test_features.py`, `test_rules.py`, `test_leakage.py` | — | ARCH §9–10 |

### B — Dashboard
| Area | Files / folders | Checklist items | Docs to follow |
|---|---|---|---|
| Fake data | `scripts/make_fake_data.py`, `data/fake/` | 0. Setup | CON §9 |
| Graph component test | throwaway page in `notebooks/` | 0. Setup | DASH §5.2 |
| App entry + sidebar + roles | `app/main.py` | F-17 | DASH §2 |
| Reading run files + SQLite | `app/logic/data_access.py` | — | CON §5–6 |
| Overview + upload screen | `app/pages/`, `app/logic/upload.py` | F-09 | DASH §3, ARCH §3.1 |
| Alert queue | `app/pages/` | F-09 | DASH §4 |
| Account investigation (score, reasons, summary, network + hop, follow the money, transactions) | `app/pages/`, `app/logic/neighbourhood.py`, `app/components/graph.py` | F-09 | DASH §5 |
| Decisions / feedback | `app/logic/feedback.py` | F-10 | DASH §5.8, CON §6 |
| Data quality screen | `app/pages/` | — | DASH §6 |
| Look and feel | `.streamlit/config.toml` | — | DASH §9 |
| Settings | `config/settings.yaml` (`app:` and `analysts:` sections only) | — | ARCH §6 |
| Tests | `tests/test_app_smoke.py`, `tests/test_neighbourhood.py` | — | ARCH §10 |

### Shared (both)
| Item | Rule |
|---|---|
| `docs/04-CONTRACT.md` + `contract.py` | Changed only together, by agreement, with a version bump (CON §1) |
| `CHECKLIST.md` | Each person ticks their own items |
| `config/settings.yaml` | Each edits only their own sections; tell the other if you touch theirs |
| Demo script + rehearsal | Together (DASH §13) |
| Final report | A writes data, model and results; B writes system design, dashboard and demo; both review all |

### Hand-off points (where one person needs something from the other)
| When | From | To | What |
|---|---|---|---|
| Very start | A | B | `contract.py` + checker, so B's fake data can be validated |
| After fake data | B | A | Confirmation that the fake files pass the checker (proves the contract is usable) |
| Upload screen | A | B | A Python function B can call: "score this uploaded file" (same code as the pipeline; B never copies it) |
| First real run | A | B | Path to `data/runs/<real_run>/`; B switches from fake to real |
| SHAP on demand (NORMAL accounts) | A | B | A function "explain this one account" |
| Layer 2 retrain | A | B | A function "retrain with feedback" that the Models screen calls |

## 3. Layer 2 and Layer 3 split (with the swap)

| Item | Owner |
|---|---|
| F-12 Replay + caught-in-time | **B** (pipeline side, with A's help) |
| F-13 Rings: pipeline step | A |
| F-13 Rings: screen | B |
| F-14 Retrain function | A |
| F-14 Models screen | **A** (screen side, with B's help) |
| Feedback simulation (FM §11) | A |
| F-15 Explanation correctness | B |
| F-16 SAML-D test | A |

## 4. Git rules (so we don't overwrite each other)

1. **Pull before you start:** `git pull` every time you sit down.
2. **Only edit files you own** (tables above). Need a change in the other person's file? Ask them.
3. **Commit small and often** with a clear message: what changed, in plain words.
4. **Never commit data files.** `.gitignore` already blocks them; don't force them in.
5. If `git pull` says there's a **conflict**, stop and sort it out together; don't just pick one side.

## 5. Weekly sync (15 minutes)

- What did each of us finish (ticked in `CHECKLIST.md`)?
- Is anything blocked waiting on the other person?
- Did either of us need to change the contract?
- What's next for each of us?
