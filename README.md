# Coupled socio-physical systems govern urban fire risk

The data and code in this repository support the manuscript: **Coupled socio-physical systems govern urban fire risk**, by Zirui He, Minghao Deng, Rui Jing, Meng Wang, Chenghao Wang, Sarah Feron and Raúl R. Cordero.

Repository: [UrbanFireSocialSensing](https://github.com/City-Building-Database-of-China/UrbanFireSocialSensing).

## Purpose

This repository supports inspection of the numerical evidence and computational workflow linking urban fire risk to coupled building, demographic and socioeconomic conditions. The study covers Beijing, Shanghai, Guangzhou and Wuhan, with an external assessment in Chongqing.

The public package combines processed figure and supplementary source data with an executable, anonymized Shanghai example. The example demonstrates the same nine-predictor XGBoost and SHAP workflow used in the research analysis.

## Workflow overview

1. Social sensing identifies and validates fire-event evidence.
2. Building information and contextual attributes define nine predictors.
3. Stratified five-fold cross-validation produces out-of-fold (OOF) probabilities and performance diagnostics.
4. A model fitted to all supplied observations supports SHAP importance and dependence analysis.
5. Processed source tables document the manuscript figures, supplementary analyses and external assessment.

## Data availability and public release boundary

The public data include:

- Numerical source data and scope notes for Figures 1–6, Supplementary Figures S1–S9 and Tables S1–S8.
- The fixed **10,000-building Shanghai sample**, containing **191 positive** and **9,809 negative** labels.
- **420 anonymous Shanghai event aggregates** and **446 anonymous manual-validation records** (427 valid and 19 rejected).
- Field definitions, predictor definitions, provenance and file checksums in [`metadata/`](metadata/).

The complete four-city building-level model matrix and Chongqing building-level matrix are outside this public package. Raw social-media narratives, user identifiers, record URLs, original building identifiers, exact addresses, coordinates, location crosswalks and detailed GIS layers are not redistributed. Third-party building and geographic inputs remain subject to their providers' terms. Public anonymous identifiers have no released mapping to source identities or locations.

Processed plotting points and anonymous grid indices support inspection of figure values. They do not provide a geographic transform or a complete building-level model matrix. See [`DATA_USAGE.md`](DATA_USAGE.md) for reuse conditions.

## Repository layout

```text
UrbanFireSocialSensing/
├── README.md
├── CITATION.cff
├── LICENSE
├── DATA_USAGE.md
├── requirements.txt
├── requirements-lock.txt
├── code/
│   ├── run_all.py
│   ├── model/
│   ├── validation/
│   └── data/social_sensing/
├── demo/
│   ├── README.md
│   ├── run_demo.py
│   ├── data/
│   └── output/
├── figure/
│   ├── figure1/ … figure6/
│   └── supplementary/
│       ├── figureS1/ … figureS9/
│       └── tables/
├── metadata/
└── tests/test_public_release.py
```

## Core-model organization

Both [`code/run_all.py`](code/run_all.py) and [`demo/run_demo.py`](demo/run_demo.py) call the same implementation in [`code/model/`](code/model/).

| Module | Role |
| --- | --- |
| `preprocessing.py` | Validate the supplied matrix and select the nine predictors without resampling or recalculating them |
| `xgboost_model.py` | XGBoost specification, stratified five-fold OOF prediction, full-data fitting and rank-based risk grades |
| `diagnostics.py` | AUROC, AUPRC, Brier score, log loss, curves, calibration bins and risk-grade summaries |
| `shap_analysis.py` | Tree SHAP, additive-consistency check, mean absolute importance and dependence values |
| `workflow.py` | Shared input, computation, output and run metadata |

The predictor order is `Projected_Area`, `Height`, `Plot_Ratio`, `Effective_Age`, `Mean_Age_Score`, `Pop_Density`, `Price`, `Fee`, `EUI`. Definitions and units appear in [`metadata/predictor_dictionary.csv`](metadata/predictor_dictionary.csv).

The classifier uses 200 trees, maximum depth 6, learning rate 0.05, subsample 0.9, column subsample 0.9 and random seed 42. Cross-validation uses five shuffled stratified folds with seed 42. SHAP values explain the full-data model on its raw margin (log-odds) scale. Risk grades use ranks of OOF probabilities with boundaries at the 50th, 75th, 90th and 97.5th percentiles; ties are ordered by input row order.

## Python environment

Tested with **Python 3.12.14 on Windows, CPU only**, in a newly created virtual environment. From the repository root:

```bash
python -m venv .venv
```

Activate the environment using the command for your shell:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

Install the tested dependency versions:

```bash
python -m pip install -r requirements-lock.txt
```

[`requirements.txt`](requirements.txt) lists direct dependencies; [`requirements-lock.txt`](requirements-lock.txt) records the complete installed set used for validation. The numerical workflow exports CSV tables and does not require Matplotlib or statsmodels. No GPU, model download or private input file is required for the quick demo. Other operating systems have not been independently tested.

## Quick demo

```bash
python demo/run_demo.py
```

This validates the immutable Shanghai sample, fits five cross-validation models, calculates OOF diagnostics, fits one model to all 10,000 records and calculates SHAP for all nine predictors. Outputs are written to `demo/output/`.

The supplied CPU run produced AUROC **0.680652**, AUPRC **0.038255**, Brier score **0.018929** and log loss **0.095046**. Its computation took approximately **2 seconds after imports**, with additional environment/import startup time. Hardware and library builds affect runtime. Machine-readable results and timing are in [`demo/output/model_metrics.csv`](demo/output/model_metrics.csv) and [`demo/output/run_metadata.json`](demo/output/run_metadata.json).

The 10,000-building sample is **case-enriched and not prevalence-representative**. Its metrics demonstrate the analytical workflow and are not intended to reproduce the four-city manuscript metrics. Exact manuscript numerical results are provided separately as processed figure source data.

## Running the analysis workflow

The general entry point runs the same Shanghai example by default:

```bash
python code/run_all.py
```

To supply an independently authorized matrix, pass `--input` and `--output-dir`. Input must contain `fire` and the nine named numeric predictors; the loader checks finite values, valid labels and the minimum class size for five-fold validation. It does not impute, resample or derive missing predictors. Do not place private inputs or their outputs in a public repository.

Run all public-data and figure checks together:

```bash
python code/run_all.py --validate-only
```

## Step-by-step workflow

Validate the fixed sample:

```bash
python demo/run_demo.py --stage check
```

Run cross-validation and diagnostics:

```bash
python demo/run_demo.py --stage oof
```

Fit the full sample and calculate SHAP:

```bash
python demo/run_demo.py --stage shap
```

| Output in `demo/output/` | Contents |
| --- | --- |
| `model_metrics.csv` | OOF AUROC, AUPRC, Brier score and log loss |
| `oof_predictions.csv` | Anonymous IDs, labels, held-out folds, probabilities and risk grades |
| `risk_grade_summary.csv` | Counts, incidence and fire capture by grade |
| `roc_curve.csv`, `precision_recall_curve.csv`, `calibration_bins.csv` | Diagnostic plotting values |
| `global_shap_importance.csv` | Mean absolute SHAP values |
| `shap_values_or_public_summary.csv` | Anonymous sample SHAP values and base value |
| `dependence/*.csv` | Feature values and corresponding SHAP values for each predictor |
| `run_metadata*.json` | Input checksum, model parameters, dependency versions and runtime |

## Social-sensing and official-statistics validation

[`code/data/social_sensing/`](code/data/social_sensing/) contains anonymous event-level aggregates and separate manual-validation labels. Event aggregates, manually checked records and building-level labels have different units and denominators; their counts should not be substituted for one another.

Supplementary Figure S7 uses **2020 only**. Recompute the monthly/provincial correlations and the no-intercept provincial sampling coefficient with:

```bash
python code/validation/official_statistics.py
```

Results are written to `demo/output/official_validation/`. The provincial coefficient uses the 29 provinces with nonzero social-sensing events; the share comparison retains all 31 provinces.

## Figure source data

[`metadata/figure_data_manifest.csv`](metadata/figure_data_manifest.csv) gives panel-level paths, variables, units and public scope. The [`data dictionary`](metadata/data_dictionary.csv) documents table fields and the [`provenance table`](metadata/provenance.csv) records source-table checksums.

| Figure | Public material |
| --- | --- |
| 1 | [`figure1/`](figure/figure1/): conceptual-framework scope note |
| 2 | [`figure2/`](figure/figure2/): city counts, function composition and city-level spatial context; detailed location/KDE layers excluded |
| 3 | [`figure3/`](figure/figure3/): age-bin incidence for 2016–2025, anonymous age/fire grids, class counts and class ranges |
| 4 | [`figure4/`](figure/figure4/): conceptual threshold provenance and height-bin counts, denominators and rates |
| 5 | [`figure5/`](figure/figure5/): global importance, independently fitted city directional importance and nine single-predictor SHAP plotting-point tables |
| 6 | [`figure6/`](figure/figure6/): profile-specific node frequencies, co-occurrence edges and profile definitions; Top 10 factors for aging low-rise buildings and Top 6 for high-rise/super-high-rise buildings, with Others |

Figure 6 node percentages use within-layer term-frequency denominators. Edge weights count co-occurrences and are not node-frequency totals. Each figure directory supplies further interpretation where needed.

## Supplementary source data

| Item | Public material |
| --- | --- |
| Figure S1 | [`figureS1/`](figure/supplementary/figureS1/): predictor-map scope and links to definitions/demo |
| Figures S2–S5 | [`figureS2/`](figure/supplementary/figureS2/), [`figureS3/`](figure/supplementary/figureS3/), [`figureS4/`](figure/supplementary/figureS4/), [`figureS5/`](figure/supplementary/figureS5/): reported validation summaries and third-party/location scope notes |
| Figure S6 | [`figureS6/`](figure/supplementary/figureS6/): social-sensing workflow counts with explicit units |
| Figure S7 | [`figureS7/`](figure/supplementary/figureS7/): 2020 monthly/provincial source data and derived statistics |
| Figure S8 | [`figureS8/`](figure/supplementary/figureS8/): manuscript OOF curves, calibration, risk-grade statistics and metrics |
| Figure S9 | [`figureS9/`](figure/supplementary/figureS9/): Chongqing aggregate metrics, grade counts, capture and development thresholds |
| Tables S1–S3, S5–S8 | [`tables/`](figure/supplementary/tables/): tabular data, field definitions, cleaning rules and predictor/importance summaries |
| Table S4 | [`TableS4/`](figure/supplementary/tables/TableS4/): narrative-data scope note |

For Figure S8, the reported full-study values are **AUROC 0.762, AUPRC 0.071, Brier score 0.001106 and log loss 0.00805**. The released table retains underlying precision alongside reported values. AUPRC is calculated with `sklearn.metrics.average_precision_score`; the public metric label remains **AUPRC**.

## Replication notes

The runnable sample reproduces the model procedure on the fixed 10,000 public records. Exact retraining of the full study requires the complete research matrices, which are outside this release. Figure source tables support numerical inspection of the reported results; regenerating geographic maps requires the underlying third-party/location inputs. Scope notes distinguish conceptual panels and summarized validation evidence from released plotting values.

The Figure 3 observation window is 2016–2025. Effective building age is a separate predictor referenced to 2022. Supplied predictor values, scientific counts, denominators and figure statistics are preserved. Figure S9 evaluates 54,132 Chongqing observations with 71 positives using development-derived grade thresholds.

Run the automated tests and standalone checks:

```bash
python -m unittest discover -s tests -p test_public_release.py -v
python code/validation/validate_public_data.py
python code/validation/validate_figure_source_data.py
python code/validation/privacy_check.py
```

The privacy scanner inspects table headers, structured keys and all table rows for specified identifiers, location fields, record URLs, local paths and explicit coordinate values. Automated checks complement the deliberately limited release scope; they are not a claim that arbitrary inputs can be anonymized by this program.

## Status

This package provides the submission source data, a fixed executable demonstration and automated consistency checks. The validated environment is documented in [`metadata/tested_environment.json`](metadata/tested_environment.json).

## License

Software code is licensed under the [MIT License](LICENSE). This software license does not apply to datasets, figures or third-party material. A separate data reuse license has not been specified; see [`DATA_USAGE.md`](DATA_USAGE.md).

## Citation

Please cite **He, Z., Deng, M., Jing, R., Wang, M., Wang, C., Feron, S. & Cordero, R. R. Coupled socio-physical systems govern urban fire risk**, and identify this repository when using the code or source tables. Machine-readable citation information is provided in [`CITATION.cff`](CITATION.cff). No DOI is assigned in this package.
