"""The six approaches compared in the evaluation (docs/05-FEATURES-MODEL.md section 5).

Every `fit_*` function returns an object with `.score(frame) -> numpy array`, where a
higher number means riskier. All of them see the same rows and the same split.
"""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trace_aml.pipeline.rules import FLAG_COLUMNS

TIE_BREAK_STEP = 1e-6   # far smaller than any gap between two rule scores; only orders equal scores


@dataclass
class Scorer:
    """A fitted approach: which columns it reads and how it turns them into a score."""

    name: str
    columns: list[str]
    model: object
    kind: str                      # "proba", "anomaly" or "rules"
    weights: dict | None = None    # rule weights, for kind == "rules"

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Risk score for every row of `frame` (higher = riskier)."""
        data = frame[self.columns]
        if self.kind == "proba":
            return self.model.predict_proba(data)[:, 1]
        if self.kind == "anomaly":
            return -self.model.score_samples(data)
        return rule_score(frame, self.weights)


def positive_weight(target: pd.Series, mode: str) -> float:
    """Extra weight on mule rows: "full" = negatives / positives, "sqrt" = its square root, "none" = 1."""
    ratio = float((target == 0).sum()) / max(float((target == 1).sum()), 1.0)
    return {"full": ratio, "sqrt": float(np.sqrt(ratio)), "none": 1.0}[mode]


def fit_lightgbm(name: str, columns: list[str], train: pd.DataFrame, target: pd.Series,
                 validation: pd.DataFrame, validation_target: pd.Series, params: dict, seed: int) -> Scorer:
    """Gradient-boosted trees. If early stopping is on, it stops adding trees when validation
    PR-AUC stops improving; otherwise it builds a fixed number of trees."""
    model = lgb.LGBMClassifier(
        objective=params["objective"], learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"], num_leaves=params["num_leaves"],
        min_child_samples=params["min_child_samples"], colsample_bytree=params["feature_fraction"],
        subsample=params["bagging_fraction"], subsample_freq=params["bagging_freq"],
        reg_alpha=params["lambda_l1"], reg_lambda=params["lambda_l2"],
        scale_pos_weight=positive_weight(target, params["positive_weight"]),
        random_state=seed, n_jobs=-1, verbose=-1)
    if params["early_stopping_rounds"] > 0:
        model.fit(train[columns], target, eval_set=[(validation[columns], validation_target)],
                  eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=False)])
    else:
        model.fit(train[columns], target)
    return Scorer(name, columns, model, "proba")


def fit_logreg(columns: list[str], train: pd.DataFrame, target: pd.Series, params: dict, seed: int) -> Scorer:
    """Logistic Regression: missing values -> median plus a "was missing" flag, then standardised."""
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=params["max_iter"], random_state=seed))
    model.fit(train[columns], target)
    return Scorer("LOGREG", columns, model, "proba")


def fit_random_forest(columns: list[str], train: pd.DataFrame, target: pd.Series, params: dict, seed: int) -> Scorer:
    """Random Forest with balanced class weights; missing values -> median."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=params["n_estimators"], min_samples_leaf=params["min_samples_leaf"],
                               max_samples=params["max_samples"], class_weight="balanced_subsample",
                               n_jobs=-1, random_state=seed))
    model.fit(train[columns], target)
    return Scorer("RANDOM_FOREST", columns, model, "proba")


def fit_isolation_forest(columns: list[str], train: pd.DataFrame, params: dict, seed: int) -> Scorer:
    """Isolation Forest: never sees labels; the score is how unusual a row looks."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        IsolationForest(n_estimators=params["n_estimators"], max_samples=params["max_samples"],
                        n_jobs=-1, random_state=seed))
    model.fit(train[columns])
    return Scorer("ISOLATION_FOREST", columns, model, "anomaly")


def fit_rule_weights(validation: pd.DataFrame, validation_target: pd.Series) -> Scorer:
    """RULES baseline: each rule is weighted by its precision on the validation period."""
    weights = {}
    for flag in FLAG_COLUMNS:
        fired = validation[flag] == 1
        weights[flag] = float(validation_target[fired].mean()) if fired.any() else 0.0
    return Scorer("RULES", [*FLAG_COLUMNS, "in_cnt_24h"], None, "rules", weights)


def rule_score(frame: pd.DataFrame, weights: dict) -> np.ndarray:
    """Weighted share of rules that fired; ties are broken by transactions received in 24 h."""
    total = sum(weights.values())
    weighted = sum(frame[flag].to_numpy(dtype="float64") * weight for flag, weight in weights.items())
    base = weighted / total if total > 0 else np.zeros(len(frame))
    busiest = frame["in_cnt_24h"].to_numpy(dtype="float64")
    return base + TIE_BREAK_STEP * busiest / (1.0 + busiest)
