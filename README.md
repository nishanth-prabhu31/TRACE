# TRACE

**Transaction Risk Analytics and Chain Evaluation**: an explainable money-mule detection tool. Transactions become a network, each account gets a risk score from 0 to 100, and an analyst sees why: top reasons, the money trail and a plain-English summary.

Work in progress. Full README later.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Download IBM AMLworld HI-Small (`HI-Small_Trans.csv`, `HI-Small_Patterns.txt`) from Kaggle into `data/raw/ibm_hi_small/`. See `docs/02-DATA.md`.

## Run

```bash
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts/make_fake_data.py
.venv\Scripts\python -m streamlit run app/main.py
```

Pipeline steps, in order (each runs alone):

```bash
.venv\Scripts\python -m trace_aml.pipeline.ingest --dataset ibm_hi_small
.venv\Scripts\python -m trace_aml.pipeline.labels --dataset ibm_hi_small
.venv\Scripts\python -m trace_aml.pipeline.quality --dataset ibm_hi_small
```

## Where to start

`AGENTS.md` → `CHECKLIST.md` → `docs/`.
