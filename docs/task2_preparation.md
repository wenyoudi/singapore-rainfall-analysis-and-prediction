# Task 2: rainfall preparation

This stage prepares a Logistic Regression baseline dataset using Spark SQL window
operations. It does not fit a model, category indexer, encoder, or scaler.

## Semantics

All calendar calculations use Asia/Singapore. Prediction time `t` uses Task 1's
aligned `timestamp` when present; otherwise it uses `observation_time`. The existing
Task 1 cleaner preserves the original source time in `observation_time`, which can
have a one-second offset. `--time-column` can select either column explicitly.
The selected time must be exactly on a five-minute grid; this stage never rounds it.

- `future_rainfall_30m`: rainfall at t+5, +10, +15, +20, +25, +30 minutes.
- `rainfall_previous_5m`: rainfall at t-5 minutes.
- `rainfall_previous_30m`: six readings t-30 through t-5 minutes.
- `rainfall_previous_60m`: twelve readings t-60 through t-5 minutes.
- `hour`, `month`, `station_id`: features available at prediction time.
- `label`: double 1 if future rainfall is **strictly greater than** `threshold_mm`,
  otherwise 0. Amounts and thresholds are in mm.

The reading at t is not included in either sum. The historical convention is
strictly earlier observations, rather than a rolling interval ending at t.
Coordinates are not required features and are not carried into the output.

Invalid/null/negative/nonfinite rainfall, invalid times, and duplicate station/time
keys fail the run. All twelve historical and six future timestamps are checked
individually against their expected offsets. A sample with any missing slot is
excluded; missing values are never filled with zero. Station windows are isolated.

Full mode reads all eight annual Parquet directories together before constructing
windows. Training uses prediction years 2017–2022, validation 2023, testing 2024.
Training labels ending in 2023 and validation labels ending in 2024 are excluded
(the last six possible prediction slots of each preceding split). History from an
earlier split is allowed: those observations are already known at prediction time.
Internal training-year boundaries are retained when complete. Dataset endpoints
are naturally excluded when history or future observations are unavailable.

Validation mode reads one annual directory, filters to one station before window
computation, and excludes its incomplete year-edge windows. It intentionally does
not load adjacent years. Parquet may still scan other stations' row groups if the
source layout cannot prune them; only the selected station enters the windows.

## Output and scaling

Default input is the read-only directory:
`/scratch/e0969984/singapore-rainfall-analysis-and-prediction/data/processed`.
Files `rainfall_2017` through `rainfall_2024` must be readable by your account.

`--output` must resolve to a child of `/scratch/<your-current-account>/`.
The process account is detected automatically; another account's scratch directory
is rejected even when it is writable. Existing symlinks are resolved before checking. Writes use
`errorifexists`; there is no overwrite option. PBS defaults to a unique directory
`/scratch/<your-current-account>/task2/MODE-JOBID`. Output Parquet is partitioned by `split` and
`year` and includes prediction time, target end time, threshold, label, and the
features listed above. Read the output root to recover partition columns.
Do not use target, target end time, threshold, split, or year as model predictors.
Station encoding and any learned preprocessing should be fitted on training only
in the later model stage. Raw station IDs are retained for that step.

The pipeline stays in Spark; no full-data collection or pandas conversion occurs.
Station sorting and duplicate validation require shuffles; the default is 32
partitions. An individual station's full history is sorted together, so skew and
spill are possible. The one-node 8-CPU/32-GB PBS request with 24-GB driver memory
is an initial resource setting, not a full-scale performance guarantee. Adjust
resources and walltime after the smoke run. Temporary Spark spill is placed under
your scratch directory and can be removed after its job has finished.

## Vanda commands

Use your own Vanda account and copy of the repository. The example assumes you
cloned it under your home directory; if not, change `cd` to your actual checkout.
Keep all Task 2 Python modules together, including `src/task2_paths.py`.
All outputs and Spark temporary files default to your own scratch directory.

The default input is the shared cleaned dataset above and requires read access.
To use an equivalent cleaned dataset elsewhere, pass
`qsub -v "TASK2_MODE=full,THRESHOLD_MM=5,INPUT_ROOT=/path/to/processed" jobs/task2_vanda.pbs`.
That directory must contain `rainfall_2017` through `rainfall_2024`. Direct Python
invocations use `--input-root`; the PBS wrapper forwards `INPUT_ROOT`.
Do not change the input cohort when making direct model comparisons.

This job only prepares data; for Logistic Regression training see
[the training guide](task2_training.md). Then on Vanda:

```bash
cd ~/singapore-rainfall-analysis-and-prediction
# Tiny synthetic tests on a compute node; requires pytest in module Python.
qsub -v TASK2_MODE=tests jobs/task2_vanda.pbs

# Replace S50 with a station ID known to exist in the cleaned 2023 data.
qsub -v TASK2_MODE=validation,VALIDATION_STATION=S50,VALIDATION_YEAR=2023,THRESHOLD_MM=5 jobs/task2_vanda.pbs

# After tests and smoke output pass inspection:
qsub -v TASK2_MODE=full,THRESHOLD_MM=5 jobs/task2_vanda.pbs

# Optional explicit fresh output destination:
qsub -v "TASK2_MODE=full,THRESHOLD_MM=5,OUTPUT_PATH=/scratch/$(id -un)/task2/prepared_v1" jobs/task2_vanda.pbs
```

The script refuses execution without a PBS job ID. Submit from the repository
root; never launch large spark-submit runs on the login node. It loads the existing
repository's `Spark/3.3.1-foss-2022a` module and verifies Spark 3.3.1, Python 3.10,
and Java 11 before proceeding. This matches the recorded Task 1 Vanda environment;
if module dependencies change, the job fails rather than running a different stack.
If pytest is absent, install the repository's `requirements-test.txt` into an
appropriate Python 3.10 environment and make it available to the module interpreter.

Use `qstat -xf JOB_ID` and inspect `rainfall_task2.oJOB_NUMBER` for exit status 0,
`Saved Task 2 dataset`, and the split/year row counts. Test mode must report passing
tests, not skips. A station/year with no complete samples fails explicitly.
Existing destinations fail; choose a fresh path for each rerun.

On a compute node with the compatible environment, the direct test command is:

```bash
SPARK_LOCAL_IP=127.0.0.1 python3 -m pytest -q tests/test_task2_prepare.py
```

Tests cover all 18 required offsets, exact sums and threshold equality, station
isolation, future-value changes, chronological boundary purging, history across
annual Parquet files, one-year boundary exclusion, invalid input, timestamp
selection, output-path guards, and Parquet round trips without overwrite.
