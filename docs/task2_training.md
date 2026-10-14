# Full-data Logistic Regression run

This is a formal full-data baseline run, not a station sample. Submit one PBS job
from the existing Vanda repository at
`/home/svu/e1496655/singapore-rainfall-analysis-and-prediction`.

```bash
qsub jobs/task2_train_vanda.pbs
```

The job runs tiny synthetic guards automatically in the same compute allocation,
then processes all stations and all eight annual cleaned Parquet directories.
A failed guard stops the job before full processing. There is no manual checkpoint
between stages. No Spark processing runs on the login node.

The input remains read-only under the teammate's scratch directory. Each job writes
a new root `/scratch/e1496655/task2/final-JOBID`; an existing root is rejected.
Set OUTPUT_PATH with qsub -v to choose another fresh directory under your scratch.
The job requests 8 CPUs, 32 GB and 12 hours, with a 24 GB driver. Vanda may adjust
resources; retain the scheduler resource record printed in the log. Runtime and
scratch space requirements have not been measured for this full training workload.

## Fixed experiment protocol

- Use every complete sample; no station or row sampling.
- Reuse the preparation convention: twelve exact past slots, six exact future
  slots, no missing-to-zero fill, Asia/Singapore time, cross-year history allowed,
  labels crossing into a later split excluded.
- Train on 2017-2022; select regularization using 2023 only; evaluate 2024 after
  selection. Do not retune after reading test scores and call them untouched results.
- Expand each base sample at thresholds 0, 1, 5, 10 mm, using strict exceedance.
  This quadruples the row count. All copies retain their base sample's split.
- Fit station, hour and month indexers/one-hot encoding on training only. Unknown
  categories use an explicit extra category. Numeric predictors are the three
  historical rainfall features and rainfall threshold. No target or calendar year
  is included in the feature vector.
- Fit binomial Logistic Regression with L2 regularization candidates 0, 0.01, 0.1,
  80 maximum iterations and tolerance 1e-6. No class weighting or resampling.
- Bound the threshold coefficient above by zero, with no threshold interactions,
  ensuring probabilities cannot increase with threshold for fixed other features.
- Select minimum mean validation Brier score with equal weight across thresholds;
  exact ties retain the earlier candidate. Retain the fitted 2017-2022 model.
- Report per-threshold test ROC-AUC, PR-AUC, Brier score, log loss, prevalence,
  and ten-bin calibration summaries. AUC uses Spark's 1000-bin approximation;
  single-class AUC is null. Constant baselines use training prevalence, not test
  prevalence. Calibration bins describe predictions; no calibrator is fitted.

Raw hour and month are one-hot encoded, rather than treating their integer codes
as ordinal. The base dataset retains the agreed feature definitions and identifiers
so other models can use the identical test cohort.

## Artifacts

- `prepared/`: reusable, threshold-independent ML dataset, partitioned by split/year.
- `preprocessing/`: train-fitted feature transformations.
- `model/`: selected complete Spark PipelineModel including Logistic Regression.
- `manifest.json`: status, versions, source hashes, settings, per-year input and
  retained counts, validation candidates, training prevalence, coefficients and
  iteration counts. These totals do not separately attribute window exclusion causes.
- `test_metrics.json`: compact per-threshold evaluation and constant baselines.
- `test_calibration.json`: calibration bin counts, predicted and observed rates.
- `example_predictions.json`: deterministic first 20 test predictions for inspection,
  not a representative sample of predictive performance.

The model artifact accepts prepared historical features, hour, month, station and
threshold. Loading it with `PipelineModel.load(path).transform(features)` does not
require labels or future observations. A separate station/time/history lookup CLI
is still needed for the final three-input user-facing interface; this run does not
implement that wrapper. Threshold inputs may be continuous, but evaluation here
covers only 0, 1, 5 and 10 mm. Do not claim validated accuracy between those values
or outside that range. This baseline has a restrictive shared linear threshold
slope and may not capture dry-weather mass or heavy tails accurately.

The expanded training and validation vectors and held-out predictions use disk-only
persistence, avoiding keeping all expanded rows in memory. Spark spill/cache uses
`/scratch/e1496655/task2/train-spill-JOBID`. Retained base Parquet allows later
recovery, but this runner does not yet implement automatic resume; do not rerun to
the same root. Partial artifacts do not establish successful completion.

## Completion and reporting

Require scheduler Exit_status=0, `TASK2 FINAL RUN COMPLETE` in the log, and
`status: complete` in the manifest. Check candidate iteration-limit flags and the
selected iteration count before claiming convergence; reaching the limit needs
investigation. Inspect class prevalence, AUC availability and performance against
constant baselines. Poor scores are legitimate results, not a reason to retune on
the test set. A runtime error or exhausted walltime may still require a corrected
run; submitting full data does not guarantee success or good predictive accuracy.

The existing 24 preparation tests passed on Vanda (job 1455698, 42.79 seconds).
The new training tests and full-data model have not yet been executed there.
Keep internal `task2_requirements_review.md` out of team uploads and commits.

## Spark 3.3.1 constant-column bound fix

Job 1458485 stopped in the preflight tests (26 passed, 1 failed); full-data
preparation and training did not start. The failed optimizer initialization was
traced to explicit infinite coefficient bounds on zero-variance columns. Spark
3.3.1 multiplies supplied bounds by the feature standard deviation in createBounds,
producing NaN for infinity times zero. Unknown-category one-hot columns are zero
throughout training and trigger this case.

The runner now calculates a training-only variance vector and supplies zero upper
bounds for constant columns and a zero upper bound for the threshold. The
initial infinite nonconstant bounds were subsequently replaced as described below. Constant-column indices are recorded in
the manifest. The regression test explicitly checks the zero-column case and
still exercises fitting, monotonicity, serialization, and unseen-station scoring.
The corrected tests require a Vanda run; syntax checks alone do not verify fitting.

Source: Apache Spark v3.3.1 LogisticRegression.scala, createBounds:
https://github.com/apache/spark/blob/v3.3.1/mllib/src/main/scala/org/apache/spark/ml/classification/LogisticRegression.scala#L735-L771

## Model reload compatibility fix

Job 1458491 progressed to the synthetic model reload test and failed with
`Do not know how to convert JString(Infinity) into double` (27 passed, 1 failed).
The full-data stage had not started. Explicit infinite bounds in a matrix parameter
cannot be reloaded by this Spark 3.3.1 JSON reader.

All explicit upper bounds are now finite: zero for constant columns and threshold,
and 1,000,000 for other columns. Lower bounds are left unset. This is a large
compatibility cap, not a claim of mathematical equivalence to an unbounded problem.
The runner stops if any non-threshold coefficient reaches 99% of the cap, and records
the cap in the manifest. Saved metadata is tested for Infinity, followed by actual
model reload and prediction equality checks. No changes were made to the dataset,
threshold grid, splits, metric definitions or threshold monotonicity requirement.
This correction has passed local syntax checks; Spark runtime verification remains
pending on Vanda.

## Compact downloadable results

Future jobs automatically run `src/task2_export_results.py` after successful
training and save `review/task2_summary.md` and `review/task2_results.zip` under
the run root. Upload the exporter alongside the updated PBS script before using
that script. The ZIP contains only the summary and four compact JSON result files;
it excludes the potentially large model and datasets. Full coefficients and Spark
configuration remain in the archived manifest but are omitted from the summary.

For an existing completed run, export without retraining or starting Spark:

```bash
python3 src/task2_export_results.py --run-dir /scratch/e1496655/task2/final-1458499.stdct-mgmt-02
```

The exporter uses only Python's standard library. Download the resulting ZIP to
the Mac and attach it for review. Missing optional calibration/example artifacts
are explicitly listed in the summary; a package reconstructed from pasted text
may therefore be incomplete compared with the original run directory.
