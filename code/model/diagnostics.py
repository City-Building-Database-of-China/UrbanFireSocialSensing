"""Numerical discrimination, calibration and rank-grade summaries."""
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss, roc_curve,
                             precision_recall_curve)


def model_metrics(y, probability, scope="OOF"):
    # The manuscript calls the recall-weighted precision statistic AUPRC.
    values = [roc_auc_score(y, probability), average_precision_score(y, probability),
              brier_score_loss(y, probability), log_loss(y, probability, labels=[0, 1])]
    methods = ["sklearn.metrics.roc_auc_score", "sklearn.metrics.average_precision_score",
               "sklearn.metrics.brier_score_loss", "sklearn.metrics.log_loss"]
    return pd.DataFrame(dict(metric=["AUROC", "AUPRC", "Brier score", "Log loss"],
                             value=values, calculation=methods, scope=scope))


def risk_grade_summary(y, grades):
    frame = pd.DataFrame({"fire": np.asarray(y), "risk_grade": np.asarray(grades)})
    out = frame.groupby("risk_grade", sort=True).fire.agg(observations="size", positives="sum").reset_index()
    out["incidence_per_1000"] = out.positives / out.observations * 1000
    out["sample_fraction"] = out.observations / len(frame)
    out["positive_capture_fraction"] = out.positives / frame.fire.sum()
    return out


def diagnostic_curves(y, probability):
    fpr, tpr, thresholds = roc_curve(y, probability)
    precision, recall, pr_thresholds = precision_recall_curve(y, probability)
    bins = pd.qcut(pd.Series(probability).rank(method="first"), 10, labels=False)
    calibration = pd.DataFrame({"bin": bins, "probability": probability, "fire": np.asarray(y)})
    calibration = calibration.groupby("bin").agg(observations=("fire", "size"),
        mean_predicted_probability=("probability", "mean"), observed_fraction=("fire", "mean")).reset_index()
    # A missing ROC threshold denotes the mathematical +infinity endpoint.
    return {"roc_curve.csv": pd.DataFrame({"fpr": fpr, "tpr": tpr,
                "threshold": np.where(np.isfinite(thresholds), thresholds, np.nan)}),
            "precision_recall_curve.csv": pd.DataFrame({"precision": precision, "recall": recall,
                "threshold": np.r_[pr_thresholds, np.nan]}), "calibration_bins.csv": calibration}
