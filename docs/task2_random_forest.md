# Random Forest on the shared modelling dataset

This branch starts directly from `main` (826c1a3), independently of the LR branch.
It adds Spark RandomForestClassifier as the third model alongside LR and GBT.
API reference: https://archive.apache.org/dist/spark/docs/3.3.1/api/scala/org/apache/spark/ml/classification/RandomForestClassifier.html

## Shared data and experiment

The input is read-only:
`/scratch/e1496655/task2/final-1458499.stdct-mgmt-02/prepared`.
No cleaned-data preparation is repeated. The sibling `manifest.json` must be
readable and describe the reviewed completed LR run: matching preparation source
hash, four thresholds and per-year retained counts. The actual dataset must match
47,000,855 base samples. Schema, finite/nonnegative values, duplicates, calendar
features, horizon and split boundaries are checked on the compute node. These
checks do not independently reconstruct historical or future windows from raw data.
Do not modify the shared dataset; retain its manifest with your results.

Training uses all 2017–2022 samples (33,981,248), validation 2023 (6,683,962),
test 2024 (6,335,645). Train-only one-hot encoding matches LR and supports unseen
categories. Historical inputs are rainfall at t-5, sums t-30 through t-5, sums
t-60 through t-5; current reading t is excluded. Station, hour and month are also
inputs. Labels use rainfall at t+5 through t+30, strict exceedance. Future total,
label, target end time, year and split never enter the feature vector.

Each base row is expanded for thresholds 0, 1, 5, 10 mm. One pooled RF uses threshold
as a feature, matching the LR experiment structure. Two candidates use 50 trees,
depths 5 and 8, seed 5208, minInstancesPerNode=20, maxBins=32, sqrt feature sampling,
bootstrap=True, subsamplingRate=1, no class weights or class resampling. Bootstrap
is the RF algorithm's own sampling mechanism. Select the minimum equal-threshold
mean validation Brier score; ties keep the first candidate. No 2023 refit. Test
is evaluated only after selection. Test labels participate in input integrity
checks, not hyperparameter selection.

Fourfold expansion means 135,924,992 training rows. Twelve hours/32 GB is an initial
allocation, not a measured runtime or capacity guarantee. Vectors persist to disk;
Spark may also spill to scratch. This is single-node Spark local[8], not a
multi-node cluster experiment. Keep the scheduler log and record resource usage.

Metrics match LR: per-threshold ROC/PR AUC (1000-bin approximation), Brier, log
loss, training-prevalence baseline and ten-bin calibration tables. Shared helper
`src/task2_model_common.py` preserves the reviewed LR feature and metric functions
from commit 2f1f6be, without depending on that branch's files.

Unlike constrained LR, Spark RF does not guarantee monotonic probabilities across
thresholds. The manifest reports test violations; no postprocessing is tuned on
test data. Only the four agreed thresholds are evaluated. Calibration plots and
a station/time inference interface are not included. RF probabilities are not
guaranteed calibrated; inspect the saved calibration table and Brier scores.

## Upload from your Mac

In Terminal:

```bash
cd ~/Desktop/"DSA5208 Scalable Distributed Computing"/singapore-rainfall-analysis-and-prediction
rsync -avh --relative \
  src/task2_random_forest.py src/task2_model_common.py \
  tests/test_task2_random_forest.py tests/conftest.py jobs/task2_rf_vanda.pbs \
  e0969984@vanda.nus.edu.sg:/scratch/e0969984/singapore-rainfall-analysis-and-prediction/
```

One connection transfers only the required files and preserves src/tests/jobs.
This does not switch the remote Git branch. If using Git instead, first publish
this branch, then fetch and check it out in a clean Vanda checkout; never overwrite
uncommitted remote edits. The RF scripts do not require the LR scripts.

Windows PowerShell with OpenSSH can use:

```powershell
# Run from your local repository folder.
$Remote = 'e0969984@vanda.nus.edu.sg'
$Root = '/scratch/e0969984/singapore-rainfall-analysis-and-prediction'
scp src/task2_random_forest.py src/task2_model_common.py "${Remote}:${Root}/src/"
scp tests/test_task2_random_forest.py tests/conftest.py "${Remote}:${Root}/tests/"
scp jobs/task2_rf_vanda.pbs "${Remote}:${Root}/jobs/"
```

PowerShell uploads use three connections and may prompt three times. The model
runs on Vanda with the same environment regardless of the upload computer. Windows
execution of the local tests has not been verified; PBS files use LF in Git.

## Validate on Vanda first

```bash
ssh e0969984@vanda.nus.edu.sg
cd /scratch/e0969984/singapore-rainfall-analysis-and-prediction
JOB_ID=$(qsub -v RF_MODE=check jobs/task2_rf_vanda.pbs)
echo "$JOB_ID"
qstat -xf "$JOB_ID" | grep -E 'job_state|Exit_status'
```

After completion require Exit_status=0, `RANDOM FOREST VALIDATED` in the PBS log
and `status: validated` in the new manifest. To inspect them in the same session:

```bash
cat "rainfall_rf.o${JOB_ID%%.*}"
python3 -m json.tool "/scratch/e0969984/task2/rf-${JOB_ID}/manifest.json"
```

Check mode scans all rows for integrity but does not fit a model. It needs compute
resources; never run spark-submit on the login node. The PBS script validates
Spark 3.3.1, Python 3.10 and Java 11. Its default mode is check. It accepts an
optional `PREPARED_PATH` environment variable, but expects the same reviewed cohort
and a compatible sibling manifest, not an arbitrary dataset.

## Train after validation passes

```bash
JOB_ID=$(qsub -v RF_MODE=train jobs/task2_rf_vanda.pbs)
echo "$JOB_ID"
qstat -xf "$JOB_ID" | grep -E 'job_state|Exit_status'
```

Training runs synthetic tests in the allocation before the full job. The module's
Python needs pytest (the repository's requirements-test.txt). Require all tests to
pass, scheduler Exit_status=0, `RANDOM FOREST COMPLETE`, and manifest status
`complete`. Do not retune on 2024 results. All writes use your account and a fresh
job-specific path; existing output directories are rejected. No automatic resume
is implemented. Never delete or alter her prepared dataset.

Artifacts under `/scratch/e0969984/task2/rf-JOB_ID/`:

- `manifest.json`: status, source/input-manifest hashes, counts, configuration,
  selected depth, validation results, feature importances and monotonicity checks.
- `preprocessing/` and `model/`: train-fitted encoding and complete saved pipeline.
- `test_metrics.json`, `test_calibration.json`, `example_predictions.json`:
  compact evaluation artifacts; examples are the first 20 rows, not representative.

The saved pipeline accepts prepared historical features, station, hour, month and
threshold; labels and future observations are not required for inference. Keep
metadata to map feature importances to the one-hot coordinates. This runner does
not duplicate the large prepared dataset or save all expanded predictions.

## Download results

On your Mac, substitute the complete job ID:

```bash
cd ~/Desktop/"DSA5208 Scalable Distributed Computing"/singapore-rainfall-analysis-and-prediction
mkdir -p outputs/random_forest/JOB_ID
rsync -avh --progress \
  e0969984@vanda.nus.edu.sg:/scratch/e0969984/task2/rf-JOB_ID/ \
  outputs/random_forest/JOB_ID/
```

This downloads the model too. For only compact JSON results:

```bash
rsync -avh --progress --include='/*.json' --exclude='*' \
  e0969984@vanda.nus.edu.sg:/scratch/e0969984/task2/rf-JOB_ID/ \
  outputs/random_forest/JOB_ID/
```

Generated run artifacts are ignored by Git. Archive model files separately before
scratch cleanup. Local tests use Spark 3.5.3/Python 3.11; actual Spark 3.3.1 and
full-data RF execution on Vanda remain to be verified. No remote job is submitted
by creating this branch.

## Verified run and saved comparison

Run `1459817.stdct-mgmt-02` completed on Vanda on 15 October 2026. The provided
scheduler output reports Exit_status=0; its manifest reports complete. All 14 RF
synthetic tests passed there. Full pipeline time from the manifest is about
55 minutes 29 seconds. The selected model has 50 trees and depth 8.

The downloaded original remains ignored at `outputs/random_forest/1459817/`.
A versioned archive, including the model and encoding but excluding Spark CRC
files, is at `outputs/task2/random_forest/1459817/`. The separate LR comparison
snapshot contains only compact JSON artifacts copied from its branch. Its
`source.json` records the source commit. No LR branch changes or merges were made.

Generate figures and comparison CSVs locally, with no Spark or retraining:

```bash
MPLCONFIGDIR=/tmp/rainfall-matplotlib .venv/bin/python src/task2_compare_results.py
```

On Windows after installing requirements-plots.txt into `.venv`:

```powershell
.\.venv\Scripts\python.exe src/task2_compare_results.py
```

Results are at `outputs/task2/comparison/README.md`, with full/zoomed calibration
and metric comparison figures in PNG/PDF, CSV tables and input hashes. GBT is
pending and is excluded from the calculations. Test findings must not guide
retuning of a model still described as evaluated on untouched 2024 data.
