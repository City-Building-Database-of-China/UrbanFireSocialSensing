"""Validate the supplied analysis-ready matrix without altering observations."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FEATURES = ["Projected_Area", "Height", "Plot_Ratio", "Effective_Age",
            "Mean_Age_Score", "Pop_Density", "Price", "Fee", "EUI"]
DEMO_COLUMNS = ["public_building_id", "City", "fire", *FEATURES,
                "Real_Total_Pop", "Total_Built_Area", "Usage"]


def load_matrix(path: Path, demo: bool = False):
    frame = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    required = set(FEATURES + ["fire"])
    if required - set(frame):
        raise ValueError(f"Required matrix columns: {sorted(required - set(frame))}")
    if "child_ratio" in frame:
        raise ValueError("The model has exactly nine specified predictors.")
    X = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(frame.fire, errors="raise")
    if not np.isfinite(X.to_numpy()).all() or not y.isin([0, 1]).all():
        raise ValueError("Predictors must be finite and fire must be binary; no rows are dropped.")
    if y.value_counts().reindex([0, 1], fill_value=0).min() < 5:
        raise ValueError("At least five observations of each class are needed for five-fold OOF.")
    positive = [f for f in FEATURES if f not in {"Fee", "EUI"}]
    if (X[positive] <= 0).any().any() or (X[["Fee", "EUI"]] < 0).any().any():
        raise ValueError("Predictors do not meet the supplied matrix validity criteria.")
    if demo:
        if list(frame.columns) != DEMO_COLUMNS:
            raise ValueError("Unexpected Shanghai demo schema.")
        if len(frame) != 10000 or int(y.sum()) != 191 or int(y.eq(0).sum()) != 9809:
            raise ValueError("The Shanghai sample must retain its original 10000/191/9809 composition.")
        if not frame.public_building_id.is_unique or set(frame.City) != {"Shanghai"}:
            raise ValueError("The demo requires unique anonymous Shanghai building identifiers.")
    return frame, X, y.astype(int)


def file_sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
