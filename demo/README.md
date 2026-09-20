# Shanghai demonstration

The fixed sample contains 10,000 anonymous Shanghai buildings: 191 positive and 9,809 negative fire labels. Its checksum is recorded in [`demo_sample.json`](../metadata/demo_sample.json). Rows, labels and supplied predictor values are preserved.

This sample is case-enriched and not prevalence-representative. Its purpose is to demonstrate the analytical workflow; exact manuscript numerical results are provided separately as processed figure source data.

From the repository root, after installing the documented environment:

```bash
python demo/run_demo.py
```

The script calls the shared research implementation in `code/model/`. It runs stratified five-fold XGBoost, OOF diagnostics, a fit on all sample records and Tree SHAP for the same nine predictors. Outputs go to `demo/output/`; `--output-dir` selects another directory. `--stage check`, `--stage oof` and `--stage shap` run individual stages.

`Real_Total_Pop`, `Total_Built_Area` and `Usage` are retained descriptive columns, not additional predictors. The public building ID is an anonymous label with no released geographic crosswalk. This sample demonstrates the pipeline; its metrics are distinct from the manuscript's full-study estimates.
