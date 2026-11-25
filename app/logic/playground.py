"""Transaction playground: add a few transfers to the sample network and score them.

The sample network (229 accounts, 10 days, in ₹) gives the new transfers the context TRACE
needs: an account is judged against its own past and the accounts around it, so two accounts
on their own cannot be scored meaningfully.
"""

import re

import pandas as pd

from trace_aml.config import path
from trace_aml.pipeline.demo import UPLOAD_SAMPLE
from trace_aml.pipeline.upload import validate_upload

DAY = "2022-09-10"           # last day of the sample network
COLUMNS = ["timestamp", "sender_id", "receiver_id", "amount", "channel"]


def _rows(items):
    return pd.DataFrame([{"timestamp": f"{DAY} {t}", "sender_id": a, "receiver_id": b, "amount": float(x), "channel": ch}
                         for t, a, b, x, ch in items], columns=COLUMNS)


def scenario(name: str) -> pd.DataFrame:
    """Ready-made transfers. Victims and cash-out accounts are existing accounts of the sample network."""
    if name == "Money-mule chain":
        victims = ["ACC-0012", "ACC-0033", "ACC-0047", "ACC-0061", "ACC-0090", "ACC-0115"]
        items = [(f"10:{5 * i:02d}:00", v, "MULE-A", 48000 + 3500 * i, "UPI") for i, v in enumerate(victims)]
        items += [("11:10:00", "MULE-A", "MULE-B", 300000, "IMPS"), ("11:40:00", "MULE-B", "MULE-C", 290000, "IMPS"),
                  ("12:05:00", "MULE-C", "ACC-0201", 282000, "NEFT")]
        return _rows(items)
    if name == "Fan-out distributor":
        items = [("09:00:00", "ACC-0039", "SPLIT-X", 900000, "NEFT")]
        items += [(f"09:{10 + 4 * i:02d}:00", "SPLIT-X", r, 140000 + 2000 * i, "IMPS")
                  for i, r in enumerate(["ACC-0018", "ACC-0026", "ACC-0080", "ACC-0112", "ACC-0180", "ACC-0237"])]
        return _rows(items)
    if name == "Normal shopper":
        items = [("08:30:00", "ACC-0150", "SHOPPER-S", 65000, "NEFT")]
        items += [(t, "SHOPPER-S", shop, amount, "UPI") for t, shop, amount in
                  [("10:12:00", "ACC-0022", 420), ("13:45:00", "ACC-0057", 1280), ("18:20:00", "ACC-0099", 2350),
                   ("20:05:00", "ACC-0130", 760)]]
        return _rows(items)
    return _rows([("10:00:00", "PERSON-A", "PERSON-B", 25000, "UPI"), ("10:20:00", "PERSON-B", "ACC-0076", 24000, "UPI")])


SCENARIOS = ["Money-mule chain", "Fan-out distributor", "Normal shopper", "Person A and B"]


def with_background(edited: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Sample network + the user's transfers, validated as one upload (₹)."""
    mine = edited.dropna(how="all").copy()
    mine = mine.astype("string").fillna("")
    mine["currency"] = "INR"
    if mine.empty:
        return None, ["Add at least one transfer."], []
    # Check the user's rows on their own first, so errors name their table rows (line N = row N-1).
    _, errors, _ = validate_upload(mine.reset_index(drop=True))
    if errors:
        return None, [re.sub(r"line (\d+)", lambda m: f"row {int(m.group(1)) - 1}", e).replace("line ", "row ")
                      for e in errors], []
    background = pd.read_csv(path("demo") / UPLOAD_SAMPLE, dtype=str, keep_default_na=False)
    raw = pd.concat([background, mine[background.columns]], ignore_index=True)
    transactions, errors, _ = validate_upload(raw)
    return transactions, errors, []


RULE_NAMES = {"rule_chain": "chain", "rule_fan_in": "fan-in", "rule_fan_out": "fan-out",
              "rule_cycle": "cycle", "rule_pass_through": "pass-through"}


def overall(verdict: str, rules_fired: list[str]) -> str:
    """Model and rules together (the F-18 finding: the rules cover fast new-account rings the model misses)."""
    if verdict == "LIKELY_MULE":
        return "Likely mule"
    if verdict == "WATCH" or rules_fired:
        return "Watch"
    return "No concerns found"


def involved(edited: pd.DataFrame) -> list[str]:
    """Accounts named in the user's transfers, in order of first appearance."""
    seen = []
    for a, b in zip(edited["sender_id"], edited["receiver_id"]):
        for x in (a, b):
            if isinstance(x, str) and x.strip() and x.strip() not in seen:
                seen.append(x.strip())
    return seen
