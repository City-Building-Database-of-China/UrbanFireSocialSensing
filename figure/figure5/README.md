# Figure 5

Final manuscript figure: [Download Figure 5 (TIFF)](fig5.tif). This is the final manuscript artwork; the accompanying point CSVs are the sampled public data described below.

The model contains exactly nine predictors. Panel a supplies the full-data global importance summary and sampled feature distributions; panel b supplies the city-specific directional importance summary. Panels c–j supply sampled dependence points for Projected_Area, Mean_Age_Score, Height, Pop_Density, Price, Effective_Age, Plot_Ratio and Fee. EUI contributes to panel a only. The points in c–j are also used for the corresponding distributions in panel a.

The two importance summary tables are retained unchanged in this update; they have not been recalculated from the sampled points. SHAP values are on the raw model-margin scale. Predictor definitions are in [predictor_dictionary.csv](../../metadata/predictor_dictionary.csv).

## Public point samples

Each point CSV contains 10,000 rows sampled without replacement from its previously released plotting-point table. The source plotting tables retain feature values within the inclusive 1st–99th percentile bounds; they are not the complete building inventories. Sampling is stratified by city, with the allocation proportional to the city counts in each source table. Integer allocations use the largest-remainder method, with alphabetical city order to break ties.

Sampling uses Python random.Random with the feature-specific seeds below. Within each table, cities are processed alphabetically and records retain their source order before sampling. Each city is sampled using random.sample; the combined sample is then shuffled using the same generator. Each feature uses a separate generator and seed. Original columns, numerical values and precision are preserved. No building identifier, original row index or shared cross-feature key is added. Rows must not be joined across files by position.

| Panel | File | Source rows | Public rows | Seed |
|---|---|---:|---:|---:|
| a | [EUI](a/Fig5_points_EUI.csv) | 316,172 | 10,000 | 2026092401 |
| c | [Projected_Area](c/Fig5_points_Projected_Area.csv) | 312,384 | 10,000 | 2026092402 |
| d | [Mean_Age_Score](d/Fig5_points_Mean_Age_Score.csv) | 312,384 | 10,000 | 2026092403 |
| e | [Height](e/Fig5_points_Height.csv) | 316,140 | 10,000 | 2026092404 |
| f | [Pop_Density](f/Fig5_points_Pop_Density.csv) | 312,384 | 10,000 | 2026092405 |
| g | [Price](g/Fig5_points_Price.csv) | 312,482 | 10,000 | 2026092406 |
| h | [Effective_Age](h/Fig5_points_Effective_Age.csv) | 313,164 | 10,000 | 2026092407 |
| i | [Plot_Ratio](i/Fig5_points_Plot_Ratio.csv) | 312,791 | 10,000 | 2026092408 |
| j | [Fee](j/Fig5_points_Fee.csv) | 314,057 | 10,000 | 2026092409 |

City allocations are listed in Beijing / Guangzhou / Shanghai / Wuhan order:

| Feature | Source city counts | Public city counts |
|---|---|---|
| EUI | 42513 / 34932 / 180222 / 58505 | 1345 / 1105 / 5700 / 1850 |
| Projected_Area | 40989 / 35119 / 178515 / 57761 | 1312 / 1124 / 5715 / 1849 |
| Mean_Age_Score | 42513 / 36982 / 177032 / 55857 | 1361 / 1184 / 5667 / 1788 |
| Height | 42123 / 37270 / 179428 / 57319 | 1332 / 1179 / 5676 / 1813 |
| Pop_Density | 42258 / 37092 / 176632 / 56402 | 1353 / 1187 / 5654 / 1806 |
| Price | 42144 / 37413 / 177504 / 55421 | 1349 / 1197 / 5680 / 1774 |
| Effective_Age | 41270 / 36320 / 177665 / 57909 | 1318 / 1160 / 5673 / 1849 |
| Plot_Ratio | 42273 / 37117 / 176180 / 57221 | 1351 / 1187 / 5633 / 1829 |
| Fee | 41555 / 37322 / 177142 / 58038 | 1323 / 1188 / 5641 / 1848 |

## Interpretation and reproducibility

These samples support inspection and approximate visualization of the feature–SHAP relationships. They do not reproduce every point, exact density, tail observation, fitted curve or statistic of the full-data manuscript plots. The model was not retrained for this update. The full-data importance summaries remain the reference for reported importance; recomputing importance from these percentile-filtered samples need not recover those summaries. The seeds document the selection procedure, but rerunning the same selection also requires the original ordered source tables.

Sampling reduces the volume of released observations; it does not guarantee anonymity. Precise feature values may still be linkable to external information. Replacing these files also does not remove previously published versions from Git history or existing copies.
