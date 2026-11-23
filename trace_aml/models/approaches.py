"""The six approaches compared in docs/05-FEATURES-MODEL.md section 5.

Each `fit_*` returns a function: rows DataFrame -> numpy array of scores (higher = more suspicious).
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trace_aml.models.dataset import RULE_FLAG_COLUMNS


def fit_rules(val: pd.DataFrame):
    """Weighted rules: each rule weighted by its precision on labelled validation rows (section 3.3)."""
    weights = {}
    for col in RULE_FLAG_COLUMNS:
        fired = val[val[col] == 1]
        weights[col] = float(fired["y"].mean()) if len(fired) else 0.0
    total = sum(weights.values()) or 1.0

    def score(rows: pd.DataFrame) -> np.ndarray:
        s = sum(rows[col].fillna(0).to_numpy() * w for col, w in weights.items()) / total
        tie = rows["in_cnt_24h"].fillna(0).to_numpy() * 1e-9  # ties broken by recent inflows
        return s + tie

    score.weights = weights
    return score


def lgbm_params(cfg: dict, scale_pos_weight: float, seed: int) -> dict:
    return {
        "objective": cfg["objective"], "learning_rate": cfg["learning_rate"],
        "n_estimators": cfg["n_estimators"], "num_leaves": cfg["num_leaves"],
        "min_child_samples": cfg["min_child_samples"], "colsample_bytree": cfg["feature_fraction"],
        "subsample": cfg["bagging_fraction"], "subsample_freq": cfg["bagging_freq"],
        "reg_alpha": cfg["lambda_l1"], "reg_lambda": cfg["lambda_l2"],
        "scale_pos_weight": scale_pos_weight, "random_state": seed, "verbose": -1, "n_jobs": -1,
    }


def fit_lightgbm(train: pd.DataFrame, val: pd.DataFrame, columns: list[str], cfg: dict, seed: int):
    """LightGBM with the settings in settings.yaml (class weight and early stopping only if set there)."""
    pos = train["y"].sum()
    weight = cfg.get("scale_pos_weight") or (len(train) - pos) / max(pos, 1)
    model = lgb.LGBMClassifier(**lgbm_params(cfg, weight, seed))
    if cfg.get("early_stopping_rounds"):
        model.fit(train[columns], train["y"], eval_set=[(val[columns], val["y"])], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
    else:
        model.fit(train[columns], train["y"])

    def score(rows: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(rows[columns])[:, 1]

    score.model = model
    return score


def _impute_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)


def fit_logreg(train: pd.DataFrame, columns: list[str], seed: int):
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    model.fit(_impute_matrix(train, columns), train["y"])
    return lambda rows: model.predict_proba(_impute_matrix(rows, columns))[:, 1]


def fit_random_forest(train: pd.DataFrame, columns: list[str], n_trees: int, seed: int):
    """Each tree sees a 20% sample of rows to keep training time reasonable on ~1.5M rows."""
    model = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=n_trees, class_weight="balanced_subsample",
                                                 min_samples_leaf=20, max_samples=0.2, n_jobs=-1,
                                                 random_state=seed))
    model.fit(_impute_matrix(train, columns), train["y"])
    return lambda rows: model.predict_proba(_impute_matrix(rows, columns))[:, 1]


def fit_isolation_forest(train: pd.DataFrame, columns: list[str], seed: int):
    """Unsupervised: ignores labels, scores how unusual an account looks."""
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(_impute_matrix(train, columns))
    model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(x)
    return lambda rows: -model.score_samples(imputer.transform(_impute_matrix(rows, columns)))


def validation_pr_auc(score, val: pd.DataFrame) -> float:
    return float(average_precision_score(val["y"], score(val)))
