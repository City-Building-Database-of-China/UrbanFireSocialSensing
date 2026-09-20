"""Manuscript XGBoost, stratified OOF and rank-grade computational functions.

Classifier settings, fold splitting, and grading retain the research functions.
The entry points validate five-fold eligibility before calling these functions.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

MODEL_PARAMETERS = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, random_state=42,
                        eval_metric="logloss", n_jobs=-1)


def create_xgb_classifier():
    return xgb.XGBClassifier(**MODEL_PARAMETERS)


def get_oof_predictions(X, y, n_splits=5):
    y_series = pd.Series(y).reset_index(drop=True)
    if y_series.nunique() != 2 or y_series.value_counts().min() < n_splits:
        raise ValueError("Both classes must be represented in every validation fold.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probabilities = np.zeros(len(X), dtype=float)
    folds = np.zeros(len(X), dtype=int)
    for fold_id, (train, valid) in enumerate(splitter.split(X, y_series), start=1):
        print(f"OOF fold {fold_id}/{n_splits}", flush=True)
        model = create_xgb_classifier()
        model.fit(X.iloc[train], y_series.iloc[train])
        probabilities[valid] = model.predict_proba(X.iloc[valid])[:, 1]
        folds[valid] = fold_id
    return probabilities, folds


def fit_full_model(X, y):
    model = create_xgb_classifier()
    model.fit(X, y)
    return model


def assign_risk_grade(probabilities):
    rank_pct = pd.Series(probabilities).rank(method="first", pct=True)
    return pd.cut(rank_pct, bins=[0, .50, .75, .90, .975, 1.0],
                  labels=["L1", "L2", "L3", "L4", "L5"],
                  include_lowest=True).astype(str)
