"""Run the fixed Shanghai 10k sample through the shared research model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["all", "check", "oof", "shap"], default="all")
    p.add_argument("--output-dir", type=Path, default=ROOT / "demo/output")
    args = p.parse_args()
    from model.workflow import run_workflow
    run_workflow(ROOT / "demo/data/shanghai_buildings_anonymized_10k.csv",
                 args.output_dir, stage=args.stage, demo=True)


if __name__ == "__main__":
    main()
