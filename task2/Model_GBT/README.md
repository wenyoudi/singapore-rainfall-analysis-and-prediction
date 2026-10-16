# Gradient Boosted Trees for Rainfall Prediction

This directory contains the GBT contribution to Task 2: estimating the probability that rainfall accumulated over the next 30 minutes strictly exceeds 0, 1, 5, or 10 mm.

## Files

| Path | Purpose |
| --- | --- |
| `src/train_gbt.py` | Spark training pipeline and training utilities |
| `notebooks/task2_gbt_training.ipynb` | Training and validation workflow with saved execution records |
| `notebooks/tune.ipynb` | Saved six-configuration tuning experiment |
| `notebooks/test_gbt_final.ipynb` | Self-contained evaluation of the frozen GBT-05 model |
| `outputs/` | Experiment artifacts stored locally or on Vanda; excluded from Git |

The tuning notebook references a remotely used `tune_gbt.py` that is not included in this checkout. Its saved outputs document the experiment; the checkout does not provide a complete executable tuning workflow.

## Prediction and evaluation design

The pipeline uses StringIndexer for station identifiers, VectorAssembler, and Spark MLlib GBTClassifier. Predictors are station identifier, hour, month, rainfall over the previous 5, 30, and 60 minutes, and rainfall threshold. Future rainfall and labels are excluded from predictors. Unseen stations use a reserved category. No class weighting or probability calibration is fitted.

The intended chronological split is 2017–2022 for training, 2023 for validation, and 2024 for testing. Prepared data must enforce complete historical and future windows and exclude target windows crossing split boundaries. See the shared data-preparation documentation for the contract. The remote prepared dataset used by the saved GBT run has not been independently verified locally.

The six recorded configurations use a deterministic 2% training sample and a 10% validation sample. All four thresholds for each sampled station–time remain together. Each threshold has 680,053 training examples and 668,341 validation examples; threshold rows are not independent station–time observations.

GBT-05 is the provisional candidate under the recorded mean Brier-skill ranking:

| Parameter | Value |
| --- | --- |
| Boosting iterations | 40 |
| Maximum tree depth | 5 |
| Step size | 0.10 |
| Seed | 5208 |
| Maximum bins | 128 |

Saved sampled-validation results for GBT-05:

| Threshold | Event rate | ROC-AUC | PR-AUC | Brier score |
| --- | ---: | ---: | ---: | ---: |
| > 0 mm | 6.690% | 0.8770 | 0.6453 | 0.036494 |
| > 1 mm | 2.643% | 0.8816 | 0.4897 | 0.017941 |
| > 5 mm | 0.890% | 0.8890 | 0.2927 | 0.007755 |
| > 10 mm | 0.423% | 0.8991 | 0.1884 | 0.004341 |

These are validation results, not final test results. At 10 mm, Brier score is worse than the constant-probability baseline using the training event rate. Good ranking does not guarantee accurate probabilities. Initial-model feature importance and calibration results must not be attributed to GBT-05.

## Run the final test

1. Sync `notebooks/test_gbt_final.ipynb` to Vanda and obtain a compute-node allocation.
2. Open it in the training environment, with compatible PySpark and Java. The recorded modelling experiment used Spark 3.5.6.
3. Run the cells in order. No additional Python helper files are required.

The notebook defaults to four cores, a 16 GB Spark driver heap, and a fixed classification probability threshold of 0.5. Allocate additional memory for Python and Spark overhead.

It searches `Model_GBT/outputs/tuning_stage1/` for a unique completed model matching GBT-05 parameters and reads the dataset path from that run's manifest. Set `MODEL_ROOT` if automatic project discovery fails. Set `RUN_ROOT` to the exact selected run directory if several matching models exist.

Required saved artifacts:

```text
selected_run/
├── model/
├── run_manifest.json
└── metrics.csv
```

The prepared dataset referenced by the manifest must also be accessible. Both shared prepared Parquet (`split=test/`) and data-preparation exports (`examples/split=test/`) are supported. The complete test split is evaluated without retraining or sampling. Missing or ambiguous models cause an error rather than automatic model selection.

Models and large datasets are not committed to Git; a fresh clone alone cannot execute the test.

## Outputs

Each evaluation creates `test_evaluation/<UTC timestamp>/` inside the selected run:

| Output | Contents |
| --- | --- |
| `test_metrics.csv` | Per-threshold ROC-AUC, PR-AUC, Brier, baseline Brier, Brier skill, precision, recall, F1 and confusion counts |
| `test_reliability_bins.csv` | Probability bins for reliability diagrams; no calibration model is fitted |
| `confusion_matrix.csv` | All actual/predicted class combinations, including zero counts |
| `validation_test_comparison.csv` | Saved validation sample versus complete test results |
| `prediction_preview.csv` | First 200 predictions ordered by time, station and rainfall threshold |
| `predictions/` | All scored observations as Parquet |
| `feature_importance.csv` | Importance recorded in the selected manifest, if available |
| `test_manifest.json` | Completion status, model provenance and metrics |

The baseline predicts each threshold's training-sample event rate. Positive Brier skill indicates improvement over that baseline. AUC uses Spark's 1,000-bin approximation and is empty for a single-class subset. Precision is empty when no positives are predicted; recall is empty when no positives occur.

Confirm both `TEST COMPLETE` in notebook output and `status: complete` in the test manifest. Failed runs may leave partial files.

## Reporting and verification

Select the model and decision rule using development data before testing. Do not tune parameters or decision thresholds using test outcomes. A model trained on 2% of training data remains a sampled-training model even when tested on the complete test split. Cross-model comparisons require identical evaluation samples and rainfall thresholds.

Notebook syntax and saved-model selection logic were checked locally. Full Spark testing has not been executed locally because the Java runtime, remote model artifacts and prepared dataset are unavailable. Final test results remain pending execution on Vanda.
