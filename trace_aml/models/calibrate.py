"""Turn raw model scores into a calibrated 0-100 risk and choose the verdict cut-offs
(docs/05-FEATURES-MODEL.md sections 6.1-6.2). Everything here is fitted on the validation period.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

RISK_SCALE = 100.0
RISK_DECIMALS = 1


def fit_calibrator(validation_target: np.ndarray, validation_score: np.ndarray) -> IsotonicRegression:
    """Isotonic calibrator: maps a raw score to the share of such accounts that were mules."""
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.asarray(validation_score, dtype="float64"), np.asarray(validation_target, dtype="float64"))
    return calibrator


def to_risk(calibrator: IsotonicRegression, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(prob, risk): calibrated probability and risk = 100 x prob, one decimal."""
    prob = calibrator.predict(np.asarray(score, dtype="float64"))
    return prob, np.round(RISK_SCALE * prob, RISK_DECIMALS)


def choose_cuts(validation_target: np.ndarray, validation_risk: np.ndarray, verdicts: dict) -> dict:
    """Pick likely_mule_cut and watch_cut on the validation period.

    likely_mule_cut = lowest risk at which the accounts at or above it are at least
    `likely_mule_min_precision` mules. If no cut reaches that, the cut with the best F1
    is used and `precision_target_met` is False.
    watch_cut = highest risk at which recall reaches `watch_min_recall`, if that is
    below likely_mule_cut; otherwise likely_mule_cut - `watch_fallback_gap`.
    """
    target = np.asarray(validation_target)
    risk = np.asarray(validation_risk, dtype="float64")
    cuts = np.unique(risk)[::-1]                              # highest first
    order = np.argsort(-risk, kind="stable")
    hits = np.cumsum(target[order])
    flagged_at = np.searchsorted(-risk[order], -cuts, side="right")   # accounts with risk >= cut
    precision = hits[flagged_at - 1] / flagged_at
    recall = hits[flagged_at - 1] / max(target.sum(), 1)

    reaches = precision >= verdicts["likely_mule_min_precision"]
    if reaches.any():
        likely = float(cuts[np.flatnonzero(reaches)[-1]])
    else:
        f1 = 2 * precision * recall / np.where(precision + recall == 0, 1, precision + recall)
        likely = float(cuts[int(np.argmax(f1))])

    enough = np.flatnonzero(recall >= verdicts["watch_min_recall"])
    watch = float(cuts[enough[0]]) if len(enough) else 0.0
    # Analyst capacity: WATCH + LIKELY_MULE may cover at most `watch_max_share` of accounts.
    # The stricter of the two rules wins (docs/05-FEATURES-MODEL.md section 13).
    max_share = verdicts.get("watch_max_share")
    if max_share:
        fits = np.flatnonzero(flagged_at <= max_share * len(risk))
        if len(fits):
            watch = max(watch, float(cuts[fits[-1]]))
    if watch >= likely:
        watch = max(likely - verdicts["watch_fallback_gap"], 0.0)
    return {"likely_mule_cut": likely, "watch_cut": watch, "precision_target_met": bool(reaches.any())}


def verdict_for(risk: np.ndarray, cuts: dict) -> np.ndarray:
    """LIKELY_MULE / WATCH / NORMAL for each risk value."""
    risk = np.asarray(risk)
    return np.where(risk >= cuts["likely_mule_cut"], "LIKELY_MULE",
                    np.where(risk >= cuts["watch_cut"], "WATCH", "NORMAL"))
