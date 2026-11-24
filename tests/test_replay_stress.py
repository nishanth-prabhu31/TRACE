"""F-12 replay steps and F-18 stress-ring generator."""

import numpy as np
import pandas as pd

from trace_aml.pipeline.replay import steps_for
from trace_aml.pipeline.stress import RINGS_PER_SHAPE, make_rings


def test_replay_steps_cover_the_test_period_hourly():
    steps = steps_for({"validation_end": "2022-09-09T00:00:00", "test_end": "2022-09-11T00:00:00"}, 1)
    assert len(steps) == 48 and steps[0] == pd.Timestamp("2022-09-09 01:00") and steps[-1] == pd.Timestamp("2022-09-11")


def test_stress_rings_are_fast_small_and_on_the_given_day():
    day = pd.Timestamp("2022-09-10")
    normal = np.array([f"N{i}" for i in range(50)])
    rows, mules = make_rings(normal, day, rupees_per_base=96.0, aged=False, seed=1)
    assert len({m.split("_")[2] for m in mules}) == 3                    # CHAIN, FAN, CYCLE
    assert rows["timestamp"].between(day, day + pd.Timedelta(days=1)).all()
    assert (rows["amount"] * 96 <= 50000 * 1.0001).all()                 # small rupee amounts
    assert sum(m.startswith("STRESS_F_CHAIN") for m in mules) >= 4 * RINGS_PER_SHAPE
    aged_rows, _ = make_rings(normal, day, 96.0, aged=True, seed=1)
    assert (aged_rows["timestamp"] < day).any()                          # aged mules have history
