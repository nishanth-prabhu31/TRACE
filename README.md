# TRACE

**Transaction Risk Analytics and Chain Evaluation**: an explainable money-mule detection tool. Transactions become a network, each account gets a risk score from 0 to 100, and an analyst sees why: top reasons, the money trail and a plain-English summary.

Work in progress. Full README later.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Download IBM AMLworld HI-Small (`HI-Small_Trans.csv`, `HI-Small_Patterns.txt`) from Kaggle into `data/raw/ibm_hi_small/`. See `docs/02-DATA.md`.

## Where to start

`AGENTS.md` → `CHECKLIST.md` → `docs/`.
