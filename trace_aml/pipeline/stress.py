"""F-18 UPI-style stress test (PRD section 5): inject fast, small-rupee mule rings into IBM's
normal background and see whether the IBM-trained model, the rules, or both catch them.

Stress test only: the injected rings are never trained on and never the headline result.

Ring shapes (hops within minutes, ₹2,000-₹50,000 per transfer, ~95% forwarded each hop):
  CHAIN    real normal sender -> 4-6 mules in a line -> real normal receiver (cash-out)
  FAN      6-10 real normal senders -> collector mule -> 3 mules -> one cash-out receiver
  CYCLE    real normal sender -> 3-4 mules in a loop that returns money to the first mule
Two variants: FRESH mules (no history) and AGED mules (a few small ordinary transfers in the
weeks before, so "new account" alone cannot give them away).

Scores once at the end of the test period (features and rules from data up to then only).
Writes reports/stress_<version>.{json,md}.

Usage:
    python -m trace_aml.pipeline.stress --dataset ibm_hi_small
"""

import argparse
import json

import numpy as np
import pandas as pd

from trace_aml import contract as c
from trace_aml.config import load_settings, path
from trace_aml.pipeline.replay import score_step
from trace_aml.pipeline.rules import FLAG_COLUMNS, compute_rules
from trace_aml.pipeline.score import load_model

RINGS_PER_SHAPE = 10


def make_rings(normal_ids: np.ndarray, day: pd.Timestamp, rupees_per_base: float, aged: bool,
               seed: int) -> tuple[pd.DataFrame, set[str]]:
    """Injected transfers (TRACE format, amounts in base currency) and the mule account IDs."""
    rng = np.random.default_rng(seed)
    rows, mules = [], set()

    def send(src, dst, when, rupees, channel="UPI"):
        rows.append({"timestamp": when, "sender_id": src, "receiver_id": dst,
                     "amount": round(rupees / rupees_per_base, 2), "channel": channel})

    def new_mule(tag):
        account = f"STRESS_{'A' if aged else 'F'}_{tag}_{len(mules):04d}"
        mules.add(account)
        if aged:                                           # a few ordinary transfers in earlier weeks
            for _ in range(rng.integers(3, 7)):
                when = day - pd.Timedelta(days=int(rng.integers(2, 9)), minutes=int(rng.integers(0, 1440)))
                other = rng.choice(normal_ids)
                if rng.random() < 0.5:
                    send(other, account, when, float(rng.uniform(300, 4000)))
                else:
                    send(account, other, when, float(rng.uniform(300, 4000)))
        return account

    def start_time():
        return day + pd.Timedelta(hours=int(rng.integers(6, 20)), minutes=int(rng.integers(0, 60)))

    def gap():
        return pd.Timedelta(minutes=int(rng.integers(1, 10)))

    for _ in range(RINGS_PER_SHAPE):                       # CHAIN
        hops = [new_mule("CHAIN") for _ in range(rng.integers(4, 7))]
        when, amount = start_time(), float(rng.uniform(15000, 50000))
        send(rng.choice(normal_ids), hops[0], when, amount)
        for a, b in zip(hops, hops[1:] + [rng.choice(normal_ids)]):
            when, amount = when + gap(), amount * float(rng.uniform(0.93, 0.98))
            send(a, b, when, amount)

    for _ in range(RINGS_PER_SHAPE):                       # FAN-IN -> FAN-OUT
        collector, spread = new_mule("FAN"), [new_mule("FAN") for _ in range(3)]
        when, total = start_time(), 0.0
        for victim in rng.choice(normal_ids, size=int(rng.integers(6, 11)), replace=False):
            amount = float(rng.uniform(2000, 12000))
            total += amount
            when = when + pd.Timedelta(minutes=int(rng.integers(0, 5)))
            send(victim, collector, when, amount)
        cashout = rng.choice(normal_ids)
        for mule in spread:
            when = when + gap()
            send(collector, mule, when, total * 0.95 / 3)
            send(mule, cashout, when + gap(), total * 0.93 / 3)

    for _ in range(RINGS_PER_SHAPE):                       # CYCLE
        loop = [new_mule("CYCLE") for _ in range(rng.integers(3, 5))]
        when, amount = start_time(), float(rng.uniform(10000, 40000))
        send(rng.choice(normal_ids), loop[0], when, amount)
        for a, b in zip(loop, loop[1:] + [loop[0]]):
            when, amount = when + gap(), amount * float(rng.uniform(0.95, 0.99))
            send(a, b, when, amount)
    return pd.DataFrame(rows), mules


def inject(transactions: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    extra = extra.assign(txn_id=np.arange(len(extra)) + int(transactions["txn_id"].max()) + 1,
                         amount_original=extra["amount"], currency_original="US Dollar",
                         is_self=extra["sender_id"] == extra["receiver_id"], dataset="stress")
    both = pd.concat([transactions, extra[transactions.columns].astype(transactions.dtypes.to_dict())],
                     ignore_index=True)
    return both.sort_values(["timestamp", "txn_id"], kind="stable").reset_index(drop=True)


def measure(scored: pd.DataFrame, rules: pd.DataFrame, mules: set[str], cuts: dict) -> dict:
    frame = scored.merge(rules[["account_id", *FLAG_COLUMNS]], on="account_id", how="left").fillna(0)
    frame["rules_any"] = frame[FLAG_COLUMNS].sum(axis=1) > 0
    frame["model_likely"] = frame["risk"] >= cuts["likely_mule_cut"]
    frame["model_watch"] = frame["risk"] >= cuts["watch_cut"]
    frame["either"] = frame["model_watch"] | frame["rules_any"]
    is_mule = frame["account_id"].isin(mules)
    out = {"injected_mules": len(mules), "mules_scored": int(is_mule.sum()),
           "mule_median_risk": round(float(frame.loc[is_mule, "risk"].median()), 2)}
    for name in ("model_likely", "model_watch", "rules_any", "either"):
        out[name] = {"mules_caught": int((frame[name] & is_mule).sum()),
                     "recall": round(float(frame.loc[is_mule, name].mean()), 4),
                     "real_accounts_also_flagged": int((frame[name] & ~is_mule).sum())}
    per_shape = {}
    for shape in ("CHAIN", "FAN", "CYCLE"):
        part = frame[is_mule & frame["account_id"].str.contains(f"_{shape}_")]
        per_shape[shape] = {name: round(float(part[name].mean()), 3)
                            for name in ("model_watch", "rules_any", "either")}
    out["recall_by_shape"] = per_shape
    return out


def run_stress(dataset: str) -> dict:
    settings = load_settings()
    version = settings["model"]["version"]
    model = load_model(path("models") / version)
    proc = path("processed") / dataset
    transactions = pd.read_parquet(proc / c.TRANSACTIONS.filename)
    splits = json.loads((proc / "splits.json").read_text(encoding="utf-8"))
    cutoff = pd.Timestamp(splits["test_end"])
    day = cutoff - pd.Timedelta(days=1)
    # Background accounts: ordinary accounts active on that day (labels used only to keep
    # known laundering accounts out of the victim / cash-out pool, so injected rings stay separate).
    roles = pd.read_parquet(proc / c.ACCOUNT_LABELS.filename, columns=["account_id", "role"])
    recent = transactions[transactions["timestamp"] >= day]
    active = pd.Index(pd.concat([recent["sender_id"], recent["receiver_id"]]).unique())
    normal_ids = active.intersection(roles.loc[roles["role"] == "NORMAL", "account_id"]).to_numpy()

    result = {"dataset": dataset, "model_version": version, "cutoff": str(cutoff), "cuts": model["cuts"],
              "rings_per_shape": RINGS_PER_SHAPE, "note": "Stress test only; never trained on."}
    for variant, aged in (("FRESH", False), ("AGED", True)):
        extra, mules = make_rings(normal_ids, day, settings["currency"]["display_units_per_base"], aged,
                                  settings["random_seed"])
        both = inject(transactions, extra)
        scored = score_step(both, cutoff, model, settings)
        rules, _ = compute_rules(both, cutoff, settings)
        result[variant] = {"injected_transfers": len(extra), **measure(scored, rules, mules, model["cuts"])}
        print(f"  {variant}: {json.dumps(result[variant])}", flush=True)

    reports = path("reports")
    (reports / f"stress_{version}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    lines = [f"# UPI-style stress test ({dataset}, model {version})", "",
             "Stress test only: injected rings were never trained on. Synthetic background (IBM), synthetic rings.", "",
             "| Variant | Detector | Mules caught | Recall | Real accounts also flagged |", "|---|---|---|---|---|"]
    for variant in ("FRESH", "AGED"):
        for name in ("model_likely", "model_watch", "rules_any", "either"):
            r = result[variant][name]
            lines.append(f"| {variant} | {name} | {r['mules_caught']} / {result[variant]['injected_mules']} | "
                         f"{r['recall']:.2f} | {r['real_accounts_also_flagged']:,} |")
    (reports / f"stress_{version}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ibm_hi_small")
    print(json.dumps(run_stress(p.parse_args().dataset), indent=2, default=str))


if __name__ == "__main__":
    main()
