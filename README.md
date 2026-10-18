# Singapore Rainfall Analysis and Prediction

DSA5208 group project using Singapore five-minute rainfall observations from
2017–2024. Task 1 explores rainfall patterns; Task 2 predicts the probability that
rainfall at a station over the next 30 minutes exceeds a requested threshold.

The repository includes preprocessing, analysis, three trained Spark pipelines,
individual model results and the final test comparison. The Streamlit interface
accepts station, historical time and rainfall threshold, retrieves historical
observations and scores the saved GBT model. No retraining is needed.

## Repository structure

| Location | Contents |
| --- | --- |
| `src/` | Task 1 processing/plots, shared Task 2 preparation, LR/RF training and comparison |
| `jobs/` | Vanda PBS scripts for Task 1, preparation, LR and RF |
| `task2/Model_GBT/` | GBT code, notebooks, trained pipeline and results |
| `outputs/task2/` | Saved LR/RF models, individual results and comparison table |
| `docs/` | Detailed execution instructions |
| `data/demo/` | Real cleaned observations for the interface demo; not an evaluation dataset |
| `tests/` | Small synthetic Spark tests and checks on saved Task 1 summaries/models |

## Data and common prediction definition

Download the eight annual CSVs from
[Historical Rainfall across Singapore](https://data.gov.sg/collections/2279/view).
Place them at `data/raw/HistoricalRainfallacrossSingapore2017.csv` through
`HistoricalRainfallacrossSingapore2024.csv`. Full raw data, full cleaned Parquet and
the large prepared modelling dataset are excluded from Git. A small real cleaned
subset is included in `data/demo/` solely to demonstrate the interface.

The authoritative preparation implementation is `src/task2_prepare.py`, originally
used for LR. All three reported models used its shared prepared dataset:

```text
/scratch/e1496655/task2/final-1458499.stdct-mgmt-02/prepared
```

The sibling `manifest.json` records the original run. Access requires permission
from its owner. This is a recorded scratch location, not a permanent public data
source; reproduce from the annual datasets when it is unavailable.

- Prediction time and calendar features use `Asia/Singapore`.
- Historical rainfall uses t−5; t−30 through t−5; and t−60 through t−5 minutes.
- The target sums the six exact future readings at t+5 through t+30.
- Require all twelve historical and six future slots; missing is not zero.
- Labels are 1 for strict exceedance of 0, 1, 5 or 10 mm, otherwise 0.
- Train on 2017–2022, validate on 2023, test on 2024. Future labels must not cross
  split boundaries; historical observations from earlier years are allowed.
- The shared cohort has 33,981,248 training, 6,683,962 validation and 6,335,645 test
  station–time samples. Each is expanded across four thresholds.

See [the preparation guide](docs/task2_preparation.md) for complete window definitions.
The other `Task2DataPreparation` branch is not used in this integrated workflow.

## Computing environments

| Workload | Recorded environment |
| --- | --- |
| Task 1 and LR/RF | Vanda, Spark 3.3.1, Python 3.10.4, Java 11.0.18, single-node `local[8]`, 24 GB driver heap |
| GBT | Spark 3.5.6; final-test notebook uses `local[4]` and 16 GB driver heap |
| Local plots and synthetic checks | macOS, Python 3.11; plotting dependencies in `requirements-plots.txt`; local Spark checks use 3.5.3 |

PBS templates specify requested resources; keep actual scheduler records separately.
The recorded Task 1 and LR scheduler allocations were 55 GB, despite 32 GB templates.
These are single-node multicore Spark runs, not evidence of multi-node scaling.
Use the original compatible Spark environment when reproducing each recorded run.

For plots from saved summaries on macOS:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-plots.txt
.venv/bin/python src/run_task1.py --summaries-only
```

Windows can use Python 3.11 and `.venv\Scripts\python.exe` with the same plotting
requirements. Full Spark jobs should run on Vanda; native Windows execution is
not verified. See [Task 1 instructions](docs/task1_workflow.md) for the complete
macOS/PowerShell setup, file transfers and data download checks.

## Task 1: cleaning and rainfall analysis

The six Spark stages are profiling, gap analysis, cleaning diagnostics, cleaning,
validation and EDA. Cleaned annual Parquet directories are saved under
`data/processed/rainfall_YEAR`. Figures are generated locally from compact summaries.

On Vanda, after uploading code and data, submit once from the repository root:

```bash
JOB_ID=$(qsub jobs/task1_vanda.pbs)
echo "$JOB_ID"
qstat -xf "$JOB_ID" | grep -E 'job_state|Exit_status'
```

The recorded completed run retained 48,538,627 observations, removed 86 redundant
records and passed structural validation. Gaps remain missing. The detailed
[Task 1 guide](docs/task1_workflow.md) contains the exact workflow, coverage
qualifications and commands to regenerate the 17 figure pairs.

## Task 2: preparation and training

Submit Spark work on compute nodes through PBS; do not train on the login node.
Existing trained models below can be loaded without rerunning these jobs.

### Shared preparation and Logistic Regression

In your Vanda checkout, `INPUT_ROOT` defaults to the original cleaned-data path.
Set it when using another cleaned copy. Outputs and spill paths use the executing
account's scratch directory. Each job needs a fresh output path.

```bash
# Tiny tests first; requires pytest available to the module Python.
qsub -v TASK2_MODE=tests jobs/task2_vanda.pbs

# Optional standalone preparation at threshold 0; training expands all thresholds.
qsub -v TASK2_MODE=full,THRESHOLD_MM=0 jobs/task2_vanda.pbs

# Full LR run: prepares all years, trains/selects, then evaluates the test set.
qsub jobs/task2_train_vanda.pbs
```

The LR training job creates `prepared/` and its parent run manifest. Standalone
preparation writes Parquet only; it does not supply RF's required provenance
manifest. Do not substitute a new dataset or edit the original provenance to
bypass checks. RF currently checks the reviewed preparation hash and cohort; a
fresh preparation version needs compatibility review before use with that runner.

Details: [preparation](docs/task2_preparation.md),
[LR training](docs/task2_training.md),
[saved LR run](outputs/task2/logistic_regression/1458499/README.md).

### Random Forest

The RF runner reads the reviewed shared prepared dataset and its manifest without
modifying them. Outputs go under `/scratch/e0969984/task2/`.

```bash
qsub -v RF_MODE=check jobs/task2_rf_vanda.pbs
# After validation completes successfully:
qsub -v RF_MODE=train jobs/task2_rf_vanda.pbs
```

See [RF instructions](docs/task2_random_forest.md) for upload, monitoring and
individual calibration/importance plots. This runner's output account is specific
to the recorded project owner; other accounts need the path restriction adjusted.

### Gradient-Boosted Trees

Use the separate Spark 3.5.6 environment and notebooks in
`task2/Model_GBT/notebooks/`. The training notebook defaults to the initial
experiment: use the documented GBT-05 parameters for the selected configuration.
The final-test notebook loads the frozen selected model without retraining.

See [GBT instructions](task2/Model_GBT/README.md) for the exact notebook setup,
model path and recorded tuning commands. The original tuning helper is not
available, so the tuning record is historical evidence, not an executable script.
The frozen GBT-05 run includes the compact inputs needed to regenerate its
report figures with `python task2/Model_GBT/results/make_report_figures.py`.
Full prediction Parquet and the shared prepared dataset remain outside Git.

## Load the committed models

| Model | PipelineModel path | Selected configuration |
| --- | --- | --- |
| LR | `outputs/task2/logistic_regression/1458499/model` | regParam=0; 21 iterations |
| RF | `outputs/task2/random_forest/1459817/model` | 50 trees; depth 8 |
| GBT | `task2/Model_GBT/outputs/tuning_stage1/GBT-05/model` | 40 boosting iterations; depth 5; step size 0.1 |

From the repository root in a compatible PySpark session:

```python
from pyspark.ml import PipelineModel
model = PipelineModel.load("task2/Model_GBT/outputs/tuning_stage1/GBT-05/model")
predictions = model.transform(prepared_features)
# Positive-class probability: predictions' probability vector, element 1.
```

`prepared_features` must contain station_id (string), hour/month (integer),
rainfall_previous_5m/30m/60m (numeric) and threshold_mm (numeric). Do not use future
rainfall to construct predictors. Keep the entire pipeline directory; it includes
fitted encoding. The Streamlit interface below provides the station/time/threshold history lookup.

## Results and comparison

Each model has four threshold-specific test results on the complete 2024 partition.
LR/RF used full training and validation data; GBT used deterministic 2% training
and 10% validation samples. Their encoding, tuning criteria and Spark versions
also differ. Describe these as comparisons of implemented pipelines.

- [LR results](outputs/task2/logistic_regression/1458499/README.md)
- [RF results](outputs/task2/random_forest/1459817/README.md)
- [GBT results](task2/Model_GBT/results/README.md)
- [Three-model Table 7](outputs/task2/model_comparison/table_7.md)

Regenerate Table 7 using `python src/task2_comparison_table.py`; it reads frozen
result commits preserved in Git history. No fitting or test-set tuning occurs.
All models lose to their training-frequency constant baseline at >10 mm.

## Verification and remaining submission work

For the tested local setup, activate a compatible Java runtime and use the
plotting environment above, then install the test dependencies:

```bash
.venv/bin/python -m pip install pyspark==3.5.3 -r requirements-test.txt
SPARK_LOCAL_IP=127.0.0.1 .venv/bin/python -m pytest -q tests
```

The full integrated suite requires Spark 3.5.x to load the GBT model. The Vanda
LR/RF PBS jobs run their scoped synthetic tests under Spark 3.3.1. During
integration, all **81 tests passed locally** under Spark 3.5.3/Python 3.11, including
loading and scoring all three committed pipelines with historical features only
and an unseen station. This is a local smoke check, not full-scale retraining.

Tests use synthetic data, compact saved summaries and committed model artifacts;
no full-data training is launched. GBT notebooks additionally contain recorded
training and test output, rather than a synthetic training test suite.

Individual model branches remain intact. This main integration preserves Task 1,
all three trained models, original run metadata and shared comparison outputs.
Still to complete: final report/AI declaration and the group submission package.
The interface has been checked locally; the user has also confirmed Vanda server
startup. Full remote inference has not been independently verified in this
revision. Do not claim submission readiness solely because branches were merged.

## Interactive prediction interface

Use the existing GBT model through a historical Streamlit dashboard. A fresh
checkout includes a demonstration dataset, so no Vanda account, raw-data download
or retraining is needed for this quick start. Install Python 3.11 and Java 11 or
17 first. Windows users should run these commands in WSL2 (native Windows is not
supported by the reused preparation module).

```bash
python3.11 -m venv .venv-interface
.venv-interface/bin/python -m pip install -r requirements-interface.txt
unset RAINFALL_DATA_ROOT
SPARK_LOCAL_IP=127.0.0.1 .venv-interface/bin/python -m streamlit run streamlit_app.py \
  --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://localhost:8501. The bundled **demonstration data** contains 1,728 real
cleaned five-minute observations, covering **1 January 2024 00:00 through
2 January 2024 23:55 SGT**, with 576 readings at each station:

| Station | Name |
| --- | --- |
| S24 | Upper Changi Road North |
| S33 | Jurong Pier Road |
| S50 | Clementi Road |

Choose a station using the searchable dropdown or by clicking its labelled point
on the Singapore map. Both controls stay synchronized; orange highlights the
selected station. The demo shows its three stations, while connecting full data
shows all available stations with recorded coordinates. The map uses a bundled
boundary and needs no map API key or additional package.

Predictions are available from 1 January 01:00 through 2 January 23:55; historical
verification is available through 2 January 23:25. Defaults follow the first
station and threshold options: S24, 1 January 2024 at 01:00 SGT (the first time
with a full preceding hour), and 0 mm.

For an optional wet-weather query, choose S24, 1 January 2024 at 14:10 SGT,
threshold 5 mm: the saved GBT displays 54.9%, and the subsequently observed total
is 19.60 mm. In contrast, S50 at 13:50 on the same day predicts 67.9% for >5 mm,
but only 0.40 mm occurs. A high probability does not guarantee an event. These
selected examples illustrate individual outcomes, not overall model performance.
Values are calculated from the saved model and observations, not hardcoded.

**The subset is for interface demonstration only. It is not the evaluation
dataset and does not replace the reported full experimental cohorts or metrics.**
It is not representative enough for performance assessment. Source and cleaning
provenance are recorded in [data/demo/README.md](data/demo/README.md).

For full station/date coverage, download the public annual CSVs, run our existing
cleaning code and set `RAINFALL_DATA_ROOT` to the resulting Parquet root. Follow
the [complete full-data workflow](docs/prediction_interface.md#connect-the-full-public-dataset)
for exact commands that preserve the committed experiment summaries. The default
is always `data/demo/`; `data/processed/` is used only when explicitly configured.
The Vanda PBS job already configures your full cleaned data explicitly.

The app never trains a model; supported thresholds are 0, 1, 5 and 10 mm.
Optional subsequent rainfall is labelled as historical verification and never
enters the forecast features. See [the interface guide](docs/prediction_interface.md)
for architecture, tests, Vanda launch and SSH tunnelling.
