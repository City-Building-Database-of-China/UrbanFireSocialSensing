"""Run the shared nine-predictor workflow or validate the public package."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--input", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "demo/output")
    p.add_argument("--stage", choices=["all", "check", "oof", "shap"], default="all")
    args = p.parse_args()
    if args.validate_only:
        from validation.validate_public_data import validate
        from validation.validate_figure_source_data import validate_figures
        from validation.privacy_check import scan
        validate(ROOT)
        validate_figures(ROOT)
        result = scan(ROOT)
        if result["findings"]:
            raise RuntimeError(str(result))
        print("[OK] Public data, figure source data and privacy checks passed.")
    else:
        from model.workflow import run_workflow
        path = args.input or ROOT / "demo/data/shanghai_buildings_anonymized_10k.csv"
        run_workflow(path, args.output_dir, stage=args.stage, demo=args.input is None)


if __name__ == "__main__":
    main()
