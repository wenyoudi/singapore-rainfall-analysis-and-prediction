# Random Forest results — job 1459817

Completed on Vanda on 15 October 2026; scheduler Exit_status=0 and manifest status
`complete`. Pipeline runtime: about 55m29s. Spark 3.3.1, Python 3.10.4, Java 11.0.18.
All 14 RF tests passed before training.

Train 2017–2022, select on 2023, test 2024 using the existing shared prepared data.
Selected model: 50 trees, depth 8, seed 5208. Mean validation Brier: 0.0194223.
Each threshold below uses the same 6,335,645 test samples.

| Rainfall threshold | ROC-AUC | PR-AUC | Brier | Constant baseline Brier |
| --- | --- | --- | --- | --- |
| >0 mm | 0.8611 | 0.5942 | 0.043471 | 0.059575 |
| >1 mm | 0.8820 | 0.4587 | 0.019710 | 0.025050 |
| >5 mm | 0.8687 | 0.2197 | 0.007650 | 0.008310 |
| >10 mm | 0.8708 | 0.1200 | 0.004305 | 0.003813 |

The model improves Brier over the constant baseline at 0, 1 and 5 mm, but not
10 mm. At >10 mm its mean forecast is 2.0173%, versus 0.3828% observed, indicating
overprediction. Zero adjacent threshold-monotonicity violations were observed on
the test cohort; this is not a model-wide guarantee. No test-based tuning was done.

Files:

- `model/`: complete Spark PipelineModel with encoding and the selected forest.
- `manifest.json`: settings, data provenance, source hashes, counts and validation results.
- `test_metrics.json`: full per-threshold test metrics and constant baselines.
- `test_calibration.json`: ten-bin calibration summaries.
- `example_predictions.json`: first 20 test predictions, not a representative sample.

The model was saved, reloaded and checked on Vanda during the run. Use Spark 3.3.1
for loading. Inputs are station_id, hour, month, rainfall_previous_5m/30m/60m and
threshold_mm; no labels or future readings are needed for scoring.

The full original download remains in ignored `outputs/random_forest/1459817/`.
The standalone encoding copy and CRC sidecars are omitted here because `model/`
already contains the encoding. No other models or model comparisons are included.
