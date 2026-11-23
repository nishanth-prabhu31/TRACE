"""Time split and snapshot times, worked out from transactions alone (docs/02-DATA.md section 6).

Kept apart from labels.py so feature code can use it without touching any label code.
"""

import pandas as pd


def compute_split(transactions: pd.DataFrame, shares: dict[str, float]) -> dict[str, pd.Timestamp]:
    """Period boundaries by share of the timeline, cut at whole hours.

    A period covers start < timestamp <= end (the first one also includes data_start).
    """
    data_start = transactions["timestamp"].min().floor("h")
    test_end = transactions["timestamp"].max().ceil("h")
    span = test_end - data_start
    train_end = (data_start + span * shares["train"]).round("h")
    validation_end = (data_start + span * (shares["train"] + shares["validation"])).round("h")
    return {"data_start": data_start, "train_end": train_end,
            "validation_end": validation_end, "test_end": test_end}


def snapshot_times(split: dict[str, pd.Timestamp], every_hours: int) -> list[pd.Timestamp]:
    """Cut-offs to score at: every `every_hours` from the start, plus each period's end."""
    step = pd.Timedelta(hours=every_hours)
    times = set(pd.date_range(split["data_start"] + step, split["test_end"], freq=step))
    times |= {split["train_end"], split["validation_end"], split["test_end"]}
    return sorted(times)


def period_of(cutoff: pd.Timestamp, split: dict[str, pd.Timestamp]) -> str:
    """Which period a snapshot belongs to: TRAIN, VALIDATION or TEST."""
    if cutoff <= split["train_end"]:
        return "TRAIN"
    return "VALIDATION" if cutoff <= split["validation_end"] else "TEST"
