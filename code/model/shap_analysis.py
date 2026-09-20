"""TreeExplainer and source tables using the manuscript's raw-margin scale."""
import numpy as np
import pandas as pd
import shap


def calculate_shap(model, X):
    explanation = shap.TreeExplainer(model)(X)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.shape != X.shape or not np.isfinite(values).all():
        raise ValueError("Unexpected SHAP shape or non-finite values.")
    base = np.asarray(explanation.base_values)
    if base.ndim == 2:
        base = base[:, 1]
    base = np.broadcast_to(base, (len(X),))
    margin = model.predict(X, output_margin=True)
    if not np.allclose(base + values.sum(axis=1), margin, atol=1e-4, rtol=1e-4):
        raise ValueError("SHAP additive reconstruction failed.")
    return values, base


def importance_table(X, values):
    return pd.DataFrame({"predictor": list(X.columns),
                         "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values(
                             "mean_abs_shap", ascending=False).reset_index(drop=True)


def dependence_tables(X, values):
    # Deliberately no original identities or spatial information.
    return {name: pd.DataFrame({"feature_value": X[name].to_numpy(),
                                "shap_value": values[:, j]})
            for j, name in enumerate(X.columns)}
