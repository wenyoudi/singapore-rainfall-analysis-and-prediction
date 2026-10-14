# Logistic Regression baseline: Vanda job 1458499

## Run and comparison contract

- Source cleaned Parquet: `/scratch/e0969984/singapore-rainfall-analysis-and-prediction/data/processed/`.
- Original completed run: `/scratch/e1496655/task2/final-1458499.stdct-mgmt-02/`.
- Shared prepared cohort: the `prepared/` subdirectory of that run; do not modify it.
- Training: 2017-2022; validation: 2023; final test: 2024.
- Thresholds: 0, 1, 5, 10 mm, using strict exceedance of six complete future slots.
- Feature history: t-5; t-30 through t-5; t-60 through t-5. Incomplete windows
  are excluded. Training/validation labels crossing into the next split are excluded.
- Additional inputs: station, hour, month and numeric rainfall threshold.
- Model selection: minimum equal-threshold mean validation Brier score.
- Selected regularization: 0; selected iterations: 21 (limit 80).
- Environment: Spark 3.3.1, Python 3.10.4, Java 11.0.18, one node, 8 CPUs,
  scheduler allocation 55 GB, Spark driver heap 24 GB.
- Scheduler exit status: 0; 28 synthetic tests passed before full processing.

## Saved results and trained model

The original `manifest.json`, `test_metrics.json`, `test_calibration.json`,
`example_predictions.json`, and complete six-stage `model/` directory were downloaded
from the completed Vanda run. Source SHA-256 hashes match the local preparation and
training scripts; calibration bin counts reconcile with test sample counts.
Model stage metadata and Parquet signatures were checked locally; actual Spark
save/reload verification was performed inside the successful Vanda run.

Start with [the compact summary](review/task2_summary.md). The
[results ZIP](review/task2_results.zip) contains the summary and four original JSON files.

Run `python3 src/task2_export_results.py --run-dir outputs/task2/logistic_regression/1458499`
from the repository root to generate a short Markdown review and ZIP package after
download. The ZIP excludes the model and datasets; the model folder is separate.
Do not commit the 48-million-row input, prepared dataset, or Spark spill directories.

## Initial interpretation

All four thresholds have 6,335,645 base test observations. Brier score improves over
the training-prevalence constant baseline at 0, 1 and 5 mm, but is approximately
6.7% worse at 10 mm. Report that limitation alongside AUC rather than claiming
uniform probability-quality improvement. Calibration remains to be inspected.

For model comparison, use the same prepared samples, threshold grid, feature
availability and metrics. Select parameters on 2023 only. The 2024 LR results have
already been viewed; document the timing of any subsequent model experiments.
The saved PipelineModel requires prepared historical features. A three-input
station/time/threshold history-lookup wrapper is still outstanding.

## Load the model on a compatible Spark environment

```python
from pyspark.ml import PipelineModel
model = PipelineModel.load("outputs/task2/logistic_regression/1458499/model")
# prepared_features needs station_id, hour, month, rainfall_previous_5m,
# rainfall_previous_30m, rainfall_previous_60m and threshold_mm.
predictions = model.transform(prepared_features)
```

The positive-class probability is element 1 of the `probability` vector. Do not
include future rainfall as a predictor. Keep the entire model directory structure.
