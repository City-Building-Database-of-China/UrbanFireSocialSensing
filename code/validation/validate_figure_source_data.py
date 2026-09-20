"""Check panel coverage and scientific identities in processed figure tables."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from model.preprocessing import FEATURES


def validate_figures(root=ROOT):
    root = Path(root).resolve()
    manifest=pd.read_csv(root/'metadata/figure_data_manifest.csv').fillna('')
    required={f'Figure {n}' for n in range(1,7)}|{f'Figure S{n}' for n in range(1,10)}|{f'Table S{n}' for n in range(1,9)}
    if not required.issubset(set(manifest.Figure)):
        raise ValueError('Figure/table coverage is incomplete.')
    for row in manifest.itertuples():
        path=(root/row.Public_file).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Invalid public file reference: {row.Public_file}')
    for name in ['provenance.csv','data_dictionary.csv']:
        table=pd.read_csv(root/'metadata'/name)
        if 'Public_file' in table:
            for rel in table.Public_file.dropna().unique():
                if not (root/rel).is_file():
                    raise ValueError(f'Metadata file reference does not exist: {rel}')
    provenance=pd.read_csv(root/'metadata/provenance.csv')
    for row in provenance.itertuples():
        if hashlib.sha256((root/row.Public_file).read_bytes()).hexdigest()!=row.SHA256:
            raise ValueError(f'Source-file checksum changed: {row.Public_file}')
    source_files=list((root/'figure').rglob('*.csv'))
    for p in source_files:
        frame=pd.read_csv(p)
        if frame.empty:
            raise ValueError(f'Empty source table: {p.name}')
    age=pd.read_csv(root/'figure/figure3/a/Fig3a_age_fire_relationship.csv')
    if set(age.observation_start_year)!={2016} or set(age.observation_end_year)!={2025}:
        raise ValueError('Figure 3 uses 2016–2025.')
    if not np.allclose(age.normalized_fire_incidence_per_1000_building_years,1000*age.fire_count/age.exposure_building_years):
        raise ValueError('Age-bin rates do not match the supplied counts/exposure.')
    height=pd.read_csv(root/'figure/figure4/Fig4b_height_fire_relationship.csv')
    if not np.allclose(height.Fire_per_1000_buildings,1000*height.fire_count/height.building_count):
        raise ValueError('Height-bin count denominator error.')
    if not np.allclose(height.Per_Meter,height.Fire_per_1000_buildings/height.avg_height):
        raise ValueError('Height normalization error.')
    if not np.allclose(height.Per_Person,1000*height.fire_count_pop_valid/height.total_pop):
        raise ValueError('Population normalization error.')
    imp=pd.read_csv(root/'figure/figure5/a/Fig5a_global_importance.csv')
    table=pd.read_csv(root/'figure/supplementary/tables/TableS8.csv')
    if len(imp)!=9 or set(imp.Feature)!=set(FEATURES):
        raise ValueError('Figure 5 must use the nine predictors.')
    values=imp.set_index('Feature').Mean_abs_SHAP.reindex(table.predictor).to_numpy()
    if not np.allclose(np.round(values,3),table.mean_abs_shap):
        raise ValueError('Figure 5 importance and Table S8 disagree at reported precision.')
    points=list((root/'figure/figure5').rglob('Fig5_points_*.csv'))
    if len(points)!=9:
        raise ValueError('Expected nine individual feature plotting-point files.')
    for p in points:
        columns=pd.read_csv(p,nrows=0).columns
        if set(columns)!={'City','Feature','Feature_value','SHAP_value','used_in_panels'}:
            raise ValueError('Unexpected SHAP plotting-point schema.')
    nodes=pd.read_csv(root/'figure/figure6/Fig6_node_frequencies.csv')
    edges=pd.read_csv(root/'figure/figure6/Fig6_cooccurrence_edges.csv')
    if len(nodes)!=45 or len(edges)!=86:
        raise ValueError('Final mechanism plotting-table size differs.')
    if not np.allclose(nodes.groupby(['cluster_cn','layer']).display_pct.sum(),100):
        raise ValueError('Mechanism node layer percentages must sum to 100.')
    keys=set(zip(nodes.cluster_cn,nodes.layer,nodes.factor_cn))
    for row in edges.itertuples():
        if (row.cluster_cn,row.source_stage,row.source_term_cn) not in keys or (row.cluster_cn,row.target_stage,row.target_term_cn) not in keys:
            raise ValueError('Mechanism edge endpoint is absent from its profile node table.')
    metrics=pd.read_csv(root/'figure/supplementary/figureS8/FigS8_summary_metrics.csv')
    if dict(zip(metrics.metric,metrics.reported_value))!={'AUROC':.762,'AUPRC':.071,'Brier score':.001106,'Log loss':.00805}:
        raise ValueError('Incorrect manuscript diagnostic values.')
    grades=pd.read_csv(root/'figure/supplementary/figureS9/FigS9_risk_grades.csv')
    if grades.observations.sum()!=54132 or grades.positive_rows.sum()!=71:
        raise ValueError('Chongqing summary counts changed.')
    if not np.allclose(grades.incidence_per_1000_observations,1000*grades.positive_rows/grades.observations):
        raise ValueError('Chongqing incidence denominators changed.')
    return dict(status='passed',manifest_rows=len(manifest),source_tables=len(source_files),
                main_figures=6,supplementary_figures=9,supplementary_tables=8,fig6_nodes=len(nodes),fig6_edges=len(edges))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args()
    print(json.dumps(validate_figures(args.root),indent=2))


if __name__=='__main__':
    main()
