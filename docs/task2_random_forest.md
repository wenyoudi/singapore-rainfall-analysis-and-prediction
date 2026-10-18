# Task 2: Random Forest

This branch starts from `main` and contains the Random Forest model only.
Comparisons with other models will be done when all models are ready.

## Data and method

Read the existing shared dataset:
`/scratch/e1496655/task2/final-1458499.stdct-mgmt-02/prepared`.
The sibling `manifest.json` must also be readable. The script checks provenance,
counts, missing/invalid values, duplicate station/time keys and split boundaries.
It does not rebuild the dataset or change the shared files.

- Train: 2017–2022 (33,981,248 base samples).
- Validation: 2023 (6,683,962 samples).
- Test: 2024 (6,335,645 samples).
- Target: next 30-minute rainfall at t+5 through t+30 exceeds 0, 1, 5 or 10 mm.
- Features: station, hour, month, threshold, rainfall at t-5, and totals from
  t-30 through t-5 and t-60 through t-5. The reading at t is excluded.
- Fit categorical encoding on training only. Each base sample is used at all four
  thresholds; future rainfall, label, year and split are excluded from predictors.
- Fit one pooled forest: 50 trees, candidate depths 5/8, seed 5208,
  minInstancesPerNode=20, maxBins=32, sqrt feature sampling, bootstrap enabled.
- Select depth by mean validation Brier across thresholds, then evaluate 2024.
  No refit on validation, class weighting or test-based tuning.

Training uses Spark 3.3.1, Python 3.10 and Java 11 on one node with local[8].
The PBS script requests 32 GB, a 24 GB driver and 12 hours. Expanded training has
135,924,992 rows; vectors persist to disk. `src/task2_model_common.py` contains the
feature encoding and evaluation helpers used by the RF script.

## Upload from macOS

From your local project folder:

```bash
rsync -avh --relative \
  src/task2_random_forest.py src/task2_model_common.py \
  tests/test_task2_random_forest.py tests/conftest.py jobs/task2_rf_vanda.pbs \
  e0969984@vanda.nus.edu.sg:/scratch/e0969984/singapore-rainfall-analysis-and-prediction/
```

For Windows PowerShell, upload the same files with OpenSSH:

```powershell
$Remote = 'e0969984@vanda.nus.edu.sg'
$Root = '/scratch/e0969984/singapore-rainfall-analysis-and-prediction'
scp src/task2_random_forest.py src/task2_model_common.py "${Remote}:${Root}/src/"
scp tests/test_task2_random_forest.py tests/conftest.py "${Remote}:${Root}/tests/"
scp jobs/task2_rf_vanda.pbs "${Remote}:${Root}/jobs/"
```

The macOS command uses one connection; the Windows commands may prompt separately.
Training runs on Vanda in either case. Local Windows execution is unverified.

## Run on Vanda

```bash
ssh e0969984@vanda.nus.edu.sg
cd /scratch/e0969984/singapore-rainfall-analysis-and-prediction

# Submit once; check mode validates the dataset without fitting a model.
JOB_ID=$(qsub -v RF_MODE=check jobs/task2_rf_vanda.pbs)
echo "$JOB_ID"
qstat -xf "$JOB_ID" | grep -E 'job_state|Exit_status'
cat "rainfall_rf.o${JOB_ID%%.*}"
```

After validation finishes with Exit_status=0 and `RANDOM FOREST VALIDATED`, train:

```bash
JOB_ID=$(qsub -v RF_MODE=train jobs/task2_rf_vanda.pbs)
echo "$JOB_ID"

# Repeat these commands to monitor; do not repeat qsub.
qstat -xf "$JOB_ID" | grep -E 'job_state|resources_used\.|Exit_status'
tail -n 40 "rainfall_rf.o${JOB_ID%%.*}"
python3 -m json.tool "/scratch/e0969984/task2/rf-${JOB_ID}/manifest.json"
```

Training runs the synthetic tests first (pytest must be available to module
Python). Require scheduler Exit_status=0, `RANDOM FOREST COMPLETE` in the log and
manifest status `complete`. Outputs use new job-specific directories under your
account. Do not run full Spark jobs on the login node.

## Results

Run `1459817.stdct-mgmt-02` completed on 15 October 2026 in about 55m29s.
All 14 RF tests passed on Vanda. The selected model has 50 trees and depth 8.
See [the saved RF results](../outputs/task2/random_forest/1459817/README.md).

To download a completed run, use this on your Mac with the full job ID:

```bash
mkdir -p outputs/random_forest/JOB_ID
rsync -avh --progress \
  e0969984@vanda.nus.edu.sg:/scratch/e0969984/task2/rf-JOB_ID/ \
  outputs/random_forest/JOB_ID/
```

The full download stays ignored by Git. The versioned run keeps the complete
`model/` (including its encoding), manifest, metrics, calibration bins and example
predictions. It omits the duplicate standalone `preprocessing/` and Spark CRC files.

Metrics include ROC/PR AUC (Spark 1000-bin approximation), Brier, log loss and a
training-prevalence constant baseline. RF probabilities need not be calibrated
or monotonic in rainfall threshold; the manifest records observed monotonicity.
The model accepts prepared historical features, not just station and time. A
station/time history lookup interface is outside this implementation.

## Calibration figure for Section 6.3

Generate the RF-only reliability diagram from saved test bins (no Spark or retraining):

```bash
MPLCONFIGDIR=/tmp/rainfall-matplotlib .venv/bin/python src/plot_rf_calibration.py
```

The script checks bin totals and weighted averages against the saved metrics.
`outputs/task2/random_forest/1459817/rf_calibration.png` and `.pdf` contain four
threshold panels, with low-probability insets. Hollow markers identify bins with
fewer than 100 samples. The insets magnify existing bins, without rebinning.

## Feature importance for Section 6.5

```bash
MPLCONFIGDIR=/tmp/rainfall-matplotlib .venv/bin/python src/plot_rf_feature_importance.py
```

This produces an RF-only ranking in PNG/PDF and a seven-row CSV beside the saved
results. Importances are taken from the selected model's manifest. Categorical
columns are summed by feature; the saved encoder widths (89 station, 25 hour,
13 month) were checked directly in its Parquet metadata. The script checks the
saved feature order and normalized importance total. For a different fitted
encoder, supply its widths with `--category-sizes STATION HOUR MONTH`.
These are impurity-based training importances, not permutation effects or an
ablation study. Correlated historical windows make individual rankings dependent
on how the forest allocates splits.

## Section 6.2: classification at probability cutoff 0.5

Generate the supplementary classification table without Spark or model scoring:

```bash
python3 src/summarize_rf_classification.py
```

Outputs are `rf_classification.csv` (including all four confusion counts) and
`rf_classification.md` in `outputs/task2/random_forest/1459817/`. Original model,
manifest, probability metrics and calibration summaries are unchanged.

The original evaluator bins probabilities using `min(9, floor(p * 10))`, so bins
5–9 exactly correspond to positive predictions under the rule `p >= 0.5`.
The script recovers positive counts from each bin's sample count and unrounded
observed frequency, validates that they are integer counts within floating-point
tolerance, and reconciles samples/positives against `test_metrics.json`.
This supports the specified bin-aligned cutoff; arbitrary within-bin cutoffs
cannot be recovered from these summaries.

At >0 mm, RF predicts 7,364 positives: 7,009 true positives and 355 false positives.
Precision is 95.18%, recall 1.74%, and F1 0.0342. At >1, >5 and >10 mm, no positive
predictions are made at this cutoff. Precision is undefined, while recall and F1
are zero because actual exceedance events exist. Do not present undefined
precision as zero or infer good event detection from a low false-alarm count.
For consistency with the LR and GBT report figures, generate a grouped
precision/recall bar chart:

```bash
MPLCONFIGDIR=/tmp/rainfall-matplotlib .venv-interface/bin/python src/plot_rf_classification.py
```

This writes `rf_precision_recall.png` (300 dpi) and `.svg` (vector) alongside the
saved RF summaries. It uses the same blue/orange colours and 0–100% scale as the
GBT figure. Undefined precision has no bar and is explicitly labelled N/A;
zero recall is labelled 0.00%. The underlying classification table remains
available. Place the RF subsection between LR and GBT in Section 6.2. No original
model, manifest or numerical result is changed.
