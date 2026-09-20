"""Recompute the Supplementary 2020 official-statistics validation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]


def compute(source):
    source = Path(source)
    monthly = pd.read_csv(source / "FigS7_temporal_monthly_2020.csv")
    province = pd.read_csv(source / "FigS7_provincial_2020.csv")
    meta = json.loads((source / "FigS7_provenance_2020.json").read_text(encoding="utf-8"))
    rows = []
    for label, frame, count in [("monthly", monthly, 12), ("provincial", province, 31)]:
        if len(frame) != count or set(frame.year) != {2020}:
            raise ValueError("Use the complete 2020 monthly and provincial inputs.")
        shares = frame[["official_share", "weibo_share"]]
        if not np.isfinite(shares).all().all() or not np.allclose(shares.sum(), 1):
            raise ValueError("Shares must be finite fractions summing to one.")
        r = float(pearsonr(frame.weibo_share, frame.official_share).statistic)
        rows.append(dict(analysis=label, year=2020, n=count, pearson_r=r,
                         pearson_r_squared=r*r,
                         spearman_rho=float(spearmanr(frame.weibo_share, frame.official_share).statistic)))
    if set(monthly.month) != set(range(1, 13)) or not province.province.is_unique:
        raise ValueError("Duplicate or absent month/province.")
    subset = province[province.weibo_dedup_events > 0]
    x = subset.weibo_dedup_events.to_numpy(float)
    y = subset.official_share.to_numpy(float) * meta["official_total_fire_count"]
    if not np.allclose(y, np.rint(y), rtol=0, atol=1e-7):
        raise ValueError("Shares do not recover integer official counts.")
    y = np.rint(y)
    beta = float(x @ y / (x @ x))
    return pd.DataFrame(rows), dict(year=2020, n=len(x), coefficient=beta,
        unit="official incidents per social-sensing event proxy",
        no_intercept_r_squared=float(1 - ((y-beta*x)**2).sum()/(y*y).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=ROOT / "figure/supplementary/figureS7")
    p.add_argument("--output-dir", type=Path, default=ROOT / "demo/output/official_validation")
    args = p.parse_args()
    metrics, coefficient = compute(args.source_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "correlations_2020.csv", index=False)
    (args.output_dir / "coefficient_2020.json").write_text(json.dumps(coefficient, indent=2)+"\n", encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"[OK] No-intercept coefficient: {coefficient['coefficient']:.6f}")


if __name__ == "__main__":
    main()
