# Gradient-boosted trees for rainfall prediction

The GBT model estimates the probability that rainfall over the next 30 minutes exceeds 0, 1, 5 or 10 mm. A single Spark pipeline uses the rainfall threshold as an input and produces a probability for each station, prediction time and threshold.

GBT-05 is the selected model: 40 boosting iterations, maximum depth 5, step size 0.1, seed 5208 and 128 bins. It was trained on a deterministic 2% sample and selected using sampled-validation results. The saved final evaluation covers the complete test partition at a fixed probability cutoff of 0.5.

## Directory structure

| Path | Purpose |
| --- | --- |
| `src/train_gbt.py` | Dataset loading, deterministic sampling, pipeline training and validation metrics |
| `notebooks/task2_gbt_training.ipynb` | Initial training experiment and validation inspection, including saved outputs |
| `notebooks/tuning_record.md` | Original six-configuration tuning commands and saved execution output |
| `notebooks/test_gbt_final.ipynb` | Self-contained evaluation of the frozen GBT-05 model |
| `results/` | Small CSV summaries, report figures and captions |
| `results/make_report_figures.py` | Figure generation from completed test artifacts |
| `outputs/` | Selected GBT-05 model and small run records are committed; other experiment outputs remain ignored |

The initial notebook records a 20-iteration, depth-4 model. Its feature importance and reliability results describe that experiment. Use the selected run's artifacts for GBT-05 reporting. The tuning record preserves the historical experiment; its original remote helper script is unavailable.

## Data and model design

The split is 2017–2022 for training, 2023 for validation and 2024 for testing. The selected GBT-05 training and final-test manifests record the LR-generated shared dataset at `/scratch/e1496655/task2/final-1458499.stdct-mgmt-02/prepared`, also used by LR and RF. The authoritative preparation code is [src/task2_prepare.py](../../src/task2_prepare.py); see the [shared preparation guide](../../docs/task2_preparation.md) for its contract and reproduction commands. The separate `Task2DataPreparation` implementation is not used for these recorded results.

Require twelve complete historical readings from t−60 through t−5 and six future readings from t+5 through t+30. Future target windows must not cross split boundaries; historical observations may come from the preceding split. The observation at t is excluded from historical predictor values. The remote scratch path is provenance, not a permanent public download; reproduce the shared dataset from cleaned observations if it is unavailable.

The loader accepts either shared prepared Parquet under `split=train/`, `split=validation/` and `split=test/`, or data-preparation exports under `examples/split=.../`. Shared observations contain `station_id`, `prediction_time`, `hour`, `month`, `rainfall_previous_5m`, `rainfall_previous_30m`, `rainfall_previous_60m` and `future_rainfall_30m`. The loader expands each observation into four thresholds and defines the label as `future_rainfall_30m > threshold_mm`. Exported examples already contain `threshold_mm` and `label`; their timestamp and historical rainfall columns are renamed on loading.

Predictors are station identifier, hour, month, previous 5-, 30- and 60-minute rainfall, and rainfall threshold. Future rainfall and labels are excluded from features. StringIndexer learns station categories from training data and reserves a category for unseen stations. VectorAssembler feeds these features to Spark MLlib GBTClassifier. No class weighting or probability calibration is fitted. Predictions from the four threshold rows are not constrained to decrease as the threshold increases.

Sampling hashes station and prediction time with a fixed seed, keeping all four threshold rows together. The tuning experiment used 680,053 training and 668,341 validation examples per threshold. These are repeated threshold evaluations of station-time observations, not four independent samples.

## Training and validation

Use a compatible PySpark and Java environment on a Vanda compute node. The recorded experiment used Spark 3.5.6. The training notebook also requires pandas and Matplotlib.

1. Open `notebooks/task2_gbt_training.ipynb` and set `DATASET_ROOT` to the prepared dataset.
2. Set `RUN_TRAINING = True` to train, then choose sampling fractions and model parameters. The defaults reproduce the initial configuration; set 40 iterations and depth 5 for GBT-05 parameters.
3. Run the cells in order. Each training run saves a new timestamped directory beneath `outputs/`.
4. To inspect a saved run, set `RUN_TRAINING = False` and `PREFERRED_RUN` to its directory name. If omitted, the notebook loads the newest completed run; this is a convenience for inspection, not a model-selection rule.

Choose model parameters using validation data. Training exports `model/`, `run_manifest.json`, `metrics.csv` and `reliability_bins.csv`. A completed manifest records sampling fractions, parameters, training class counts and feature importance. Failed runs retain a failure status and error message.

The training helper's optional `evaluate_test=True` path requires full training and validation fractions. Use the final-test notebook to evaluate the existing sampled-training GBT-05 model without retraining.

## Final test

1. Open `notebooks/test_gbt_final.ipynb` in the training environment.
2. Set `MODEL_ROOT` if automatic discovery cannot locate `Model_GBT/outputs/tuning_stage1/`. Set `RUN_ROOT` to an exact saved run directory if multiple matching models exist.
3. Keep the decision cutoff fixed before testing and run the cells in order.

The notebook selects a unique completed model with 40 iterations, depth 5 and step size 0.1. It reads the dataset path from that run's manifest and requires `model/` and `metrics.csv`. The original dataset path must be accessible in the execution environment. The selected GBT-05 `model/`, training manifest, validation summaries and final-test manifest and CSV summaries are committed under `outputs/tuning_stage1/GBT-05/`. Other experiments and full prediction Parquet files remain local. The prepared dataset is not included, so a fresh clone still requires access to the dataset path recorded in the manifest to run evaluation.

The default Spark session uses four cores, a 16 GB driver heap and the Asia/Singapore timezone. Allocate additional memory for Python and Spark overhead. The complete test split is scored without sampling or retraining. Each execution creates `test_evaluation/<UTC timestamp>/` within the selected run.

| Output | Contents |
| --- | --- |
| `test_metrics.csv` | Per-threshold AUC, Brier score, baseline, Brier skill, precision, recall, F1 and confusion counts |
| `test_reliability_bins.csv` | Occupied equal-width probability bins, counts, mean probabilities and observed frequencies |
| `confusion_matrix.csv` | All actual/predicted class combinations, including zero counts |
| `validation_test_comparison.csv` | Saved sampled-validation metrics alongside complete-test metrics |
| `prediction_preview.csv` | First 200 predictions ordered by time, station and threshold |
| `predictions/` | Complete scored observations in Parquet format |
| `feature_importance.csv` | Selected model's importance values, when recorded in the training manifest |
| `test_manifest.json` | Completion status, model provenance, decision cutoff and metrics |

Confirm `TEST COMPLETE` in the notebook and `status: complete` in the manifest. Failed evaluations may leave partial files.

## Test performance

The recorded completed evaluation is `outputs/tuning_stage1/GBT-05/test_evaluation/20261016T103557030872Z`. Each threshold contains 6,335,645 test examples. Classification uses probability >= 0.5.

| Threshold | ROC-AUC | PR-AUC | Brier score | Precision | Recall | F1 | Brier skill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| > 0 mm | 0.8602 | 0.6270 | 0.036028 | 77.90% | 44.18% | 0.5638 | 39.52% |
| > 1 mm | 0.8515 | 0.4833 | 0.017659 | 69.97% | 34.44% | 0.4616 | 29.50% |
| > 5 mm | 0.8579 | 0.2681 | 0.007495 | 63.12% | 11.68% | 0.1971 | 9.81% |
| > 10 mm | 0.8635 | 0.1703 | 0.004049 | 51.89% | 4.02% | 0.0746 | -6.19% |

Brier score is the mean squared probability error. The baseline predicts each threshold's training-sample event frequency. Brier skill is `1 - model_brier / baseline_brier`; positive values indicate improvement over that baseline. ROC-AUC and PR-AUC use Spark's 1,000-bin approximation and are empty for single-class subsets. Precision is empty when no positives are predicted; recall is empty when no positives occur.

At 10 mm, recall is 4.02% and Brier skill is negative. The model provides useful ranking but misses most heavy-rainfall events at cutoff 0.5 and performs worse than the training-frequency probability baseline at this threshold. Lower test Brier scores than validation do not establish better calibration because event prevalence differs. Reliability diagrams show probability agreement; feature importance describes model usage, not causal effects.

Do not use test outcomes to select parameters or decision cutoffs. Compare models on the same observations and thresholds, and retain the 2% training qualification in reports.

## Report artifacts and verification

See [results](results/README.md) for figure generation and provenance. The three report CSV summaries match the corresponding local completed-test files. Confusion counts, derived classification metrics, Brier-skill arithmetic and reliability-bin totals can be checked from those artifacts.

Local checks cover Python and notebook syntax, file references and saved-result consistency. Spark training and inference require Java and the prepared dataset; they have not been rerun in this local environment. Saved execution output and completed manifests document the recorded remote runs.
