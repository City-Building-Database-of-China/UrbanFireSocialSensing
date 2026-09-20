"""Shared runner called by code/run_all.py and demo/run_demo.py."""
from pathlib import Path
import importlib.metadata
import json
import platform
import time
import pandas as pd
from .preprocessing import load_matrix, file_sha256, FEATURES
from .xgboost_model import get_oof_predictions, assign_risk_grade, fit_full_model, MODEL_PARAMETERS
from .diagnostics import model_metrics, risk_grade_summary, diagnostic_curves
from .shap_analysis import calculate_shap, importance_table, dependence_tables


def run_workflow(input_path, output_dir, stage="all", demo=False):
    started = time.perf_counter()
    input_path, output_dir = Path(input_path), Path(output_dir)
    before = file_sha256(input_path)
    frame, X, y = load_matrix(input_path, demo=demo)
    print(f"Validated {len(frame):,} records, {int(y.sum())} positives and {len(FEATURES)} predictors.", flush=True)
    if stage == "check":
        return {"rows": len(frame), "positives": int(y.sum()), "predictors": FEATURES}
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = frame.public_building_id if "public_building_id" in frame else pd.Series(
        [f"sample_{i+1:06d}" for i in range(len(frame))])
    if stage in {"all", "oof"}:
        probabilities, folds = get_oof_predictions(X, y)
        grades = assign_risk_grade(probabilities)
        pd.DataFrame({"public_building_id": ids, "fire": y, "fold": folds,
                      "oof_probability": probabilities, "risk_grade": grades}).to_csv(
                          output_dir / "oof_predictions.csv", index=False)
        metric_scope = "Shanghai demo OOF" if demo else "user-supplied matrix OOF"
        model_metrics(y, probabilities, scope=metric_scope).to_csv(output_dir / "model_metrics.csv", index=False)
        risk_grade_summary(y, grades).to_csv(output_dir / "risk_grade_summary.csv", index=False)
        for name, table in diagnostic_curves(y, probabilities).items():
            table.to_csv(output_dir / name, index=False)
    if stage in {"all", "shap"}:
        print("Fitting all supplied records and calculating SHAP.", flush=True)
        model = fit_full_model(X, y)
        values, base = calculate_shap(model, X)
        importance_table(X, values).to_csv(output_dir / "global_shap_importance.csv", index=False)
        shap_table = pd.DataFrame(values, columns=["SHAP_" + f for f in FEATURES])
        shap_table.insert(0, "public_building_id", ids.to_numpy())
        shap_table["base_value"] = base
        shap_table.to_csv(output_dir / "shap_values_or_public_summary.csv", index=False)
        dependence_dir = output_dir / "dependence"
        dependence_dir.mkdir(exist_ok=True)
        for feature, table in dependence_tables(X, values).items():
            table.to_csv(dependence_dir / f"{feature}.csv", index=False)
    if before != file_sha256(input_path):
        raise RuntimeError("The input file changed during execution.")
    dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "shap", "joblib"]
    report = {"status": "passed", "stage": stage, "scope": "Shanghai demo" if demo else "user-supplied matrix",
        "rows": len(frame), "positives": int(y.sum()), "negatives": int(y.eq(0).sum()),
        "predictors": FEATURES, "input_sha256": before, "python": platform.python_version(),
        "platform": platform.system(), "device": "CPU", "model_parameters": MODEL_PARAMETERS,
        "dependencies": {p: importlib.metadata.version(p) for p in dependencies},
        "runtime_seconds": round(time.perf_counter() - started, 3)}
    name = "run_metadata.json" if stage == "all" else f"run_metadata_{stage}.json"
    (output_dir / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] {stage} completed in {report['runtime_seconds']:.3f} s on CPU.", flush=True)
    return report
