# Singapore Rainfall Analysis and Prediction

DSA5208 group project using Singapore five-minute rainfall observations from
2017–2024. Task 1 explores rainfall patterns; Task 2 predicts the probability that
rainfall at a station over the next 30 minutes exceeds a requested threshold.

The repository includes preprocessing, analysis, three trained Spark pipelines,
individual model results and the final test comparison. The Streamlit interface
accepts station, historical time and rainfall threshold, retrieves historical
observations and scores the saved GBT model. No retraining is needed.

**Run the interface:** [demo quick start](#1-set-up-the-interface-environment) ·
[full-data workflow](#3-try-the-interface-with-full-data).

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

The Streamlit app loads the committed GBT-05 Spark model. Choose a monitoring
station, historical prediction time and rainfall threshold to estimate the
probability that the station accumulates **more than the threshold in the next
30 minutes**. No retraining is required.

### 1. Set up the interface environment

Use **Python 3.11** and **Java 11 or 17**. The commands below work in a macOS or
Linux terminal. Windows users should use **WSL2**, with Python and Java installed
inside WSL; native Windows is not supported by the reused preparation module.
The interface has been tested on macOS; WSL execution remains unverified.

The interface is currently on `task2/prediction-interface`. For a new checkout:

```bash
git clone --branch task2/prediction-interface \
  https://github.com/wenyoudi/singapore-rainfall-analysis-and-prediction.git
cd singapore-rainfall-analysis-and-prediction
```

If you already have this branch, open a terminal in its repository root instead.
Create a separate environment; the interface uses Spark 3.5.6 rather than the
Spark 3.3.1 environment used for the recorded Task 1 and LR/RF runs.

```bash
python3.11 --version
java -version
python3.11 -m venv .venv-interface
.venv-interface/bin/python -m pip install -r requirements-interface.txt
```

### 2. Run the included demonstration

No raw-data download, Vanda account or additional map dependency is needed.
From the repository root, run:

```bash
unset RAINFALL_DATA_ROOT
export RAINFALL_MODEL=GBT
SPARK_LOCAL_IP=127.0.0.1 .venv-interface/bin/python -m streamlit run streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open **http://localhost:8501**. Leave the terminal running; use **Ctrl+C** to stop
the app. If port 8501 is occupied, change `--server.port 8501` to
`--server.port 8503` and open http://localhost:8503 instead.

The default data root is `data/demo/`. It contains **1,728 real cleaned five-minute
observations**, with 576 readings per station, covering **1 January 2024 00:00
through 2 January 2024 23:55 SGT**:

| Station | Name |
| --- | --- |
| S24 | Upper Changi Road North |
| S33 | Jurong Pier Road |
| S50 | Clementi Road |

To try a prediction:

1. Select a station from the searchable dropdown or click its labelled point on
   the Singapore map. Both controls stay synchronized; orange marks the selection.
2. Choose a date and five-minute prediction time. The demonstration supports
   predictions from **1 January 01:00 through 2 January 23:55 SGT**. Earlier times
   lack the required preceding hour of observations.
3. Select a threshold: **0, 1, 5 or 10 mm**. Click **Predict Rainfall**.
4. Read the probability and preceding-hour rainfall chart. Optionally expand
   **Verify against the historical outcome** and click **Show observed next
   30 minutes**. Verification is available through **2 January 23:25 SGT**.

Defaults use the first station and threshold: S24, 1 January at 01:00 SGT, and
0 mm. The target uses readings at t+5 through t+30; equality to the threshold is
not an exceedance. All displayed times are Singapore time (UTC+8).

Two optional examples, both on 1 January 2024:

| Station | Prediction time (SGT) | Threshold | Predicted probability | Observed next 30-minute rainfall |
| --- | --- | --- | --- | --- |
| S24 | 14:10 | 5 mm | 54.9% | 19.60 mm |
| S50 | 13:50 | 5 mm | 67.9% | 0.40 mm |

The second event did not occur despite its higher predicted probability. These
selected examples illustrate individual outcomes, not overall model quality.
Values are calculated from the saved model and real observations, not hardcoded.

**This subset is for interface demonstration only. It is not the evaluation
dataset and does not replace the reported experimental cohorts or metrics.**
See [demo provenance and rebuilding instructions](data/demo/README.md).

### 3. Try the interface with full data

The full raw CSVs, cleaned Parquet and prepared modelling dataset are excluded
from Git. You can obtain the public data and clean it locally; a Vanda account is
not required. If you already have the cleaned annual Parquet, skip to **Connect
existing cleaned data** below.

#### Download the annual source CSVs

Open the NEA [Historical Rainfall across Singapore collection](https://data.gov.sg/collections/2279/view).
Download the **full CSV for each year 2017–2024**, excluding 2016. Extract any
archives, then place or rename the files to these exact paths:

```text
data/raw/HistoricalRainfallacrossSingapore2017.csv
data/raw/HistoricalRainfallacrossSingapore2018.csv
data/raw/HistoricalRainfallacrossSingapore2019.csv
data/raw/HistoricalRainfallacrossSingapore2020.csv
data/raw/HistoricalRainfallacrossSingapore2021.csv
data/raw/HistoricalRainfallacrossSingapore2022.csv
data/raw/HistoricalRainfallacrossSingapore2023.csv
data/raw/HistoricalRainfallacrossSingapore2024.csv
```

Use complete CSV downloads rather than a paginated API sample. The original
annual files used for this project total approximately 7.5 GB; allow additional
disk space for cleaned Parquet, Spark temporary files and the environment.

#### Clean and validate the observations

Stop any running interface with **Ctrl+C**. From the repository root, run the
existing cleaning and validation scripts in an isolated workspace. This keeps
new summaries separate from the committed experimental results.

```bash
for year in {2017..2024}; do
  ls -lh "data/raw/HistoricalRainfallacrossSingapore${year}.csv"
done

mkdir -p .local-cleaning/src .local-cleaning/data
cp src/data_cleaning.py src/cleaning_validation.py .local-cleaning/src/
test -e .local-cleaning/data/raw || ln -s "$PWD/data/raw" .local-cleaning/data/raw

export PYSPARK_PYTHON="$PWD/.venv-interface/bin/python"
export SPARK_LOCAL_IP=127.0.0.1

.venv-interface/bin/spark-submit --master 'local[4]' --driver-memory 8g \
  .local-cleaning/src/data_cleaning.py
.venv-interface/bin/spark-submit --master 'local[4]' --driver-memory 8g \
  .local-cleaning/src/cleaning_validation.py
```

These commands request four local threads and an 8 GB Java driver heap; additional
RAM is needed for Python and the operating system. Adjust resources to suit your
machine. Full cleaning is substantially heavier than running the demo.

Continue only after both commands succeed and validation prints
`VALIDATION: PASS structural checks`. Review `.local-cleaning/outputs/validation/`
if it reports issues. Cleaned annual directories will be written beneath
`.local-cleaning/data/processed/`, named `rainfall_2017` through `rainfall_2024`.
These commands clean observations; they do not retrain or reevaluate the models.

#### Connect existing cleaned data

For the data generated above:

```bash
export RAINFALL_DATA_ROOT="$PWD/.local-cleaning/data/processed"
export RAINFALL_MODEL=GBT
SPARK_LOCAL_IP=127.0.0.1 .venv-interface/bin/python -m streamlit run streamlit_app.py \
  --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

If your cleaned data is already elsewhere, replace the first line, for example:

```bash
export RAINFALL_DATA_ROOT="$PWD/data/processed"
```

The variable must point to the **parent of the annual directories**, not a single
Parquet file or the LR-generated `prepared/` modelling dataset:

```text
YOUR_DATA_ROOT/
├── rainfall_2017/
├── rainfall_2018/
├── ...
└── rainfall_2024/
```

Open http://localhost:8501. The first scan may take time. Station choices, map
points and date ranges now come from your connected data. Missing historical
windows produce an error rather than treating missing rainfall as zero. Stations
without valid coordinates remain selectable in the dropdown.

The app always defaults to the demo when `RAINFALL_DATA_ROOT` is unset, even if
full data exists in `data/processed/`. To return to the demo, stop the app, run
`unset RAINFALL_DATA_ROOT`, and repeat the demo launch command. Restart the app
whenever you change its data source or update inference code.

The recorded full cleaning run used Spark 3.3.1 on Vanda. Cleaner functions and
interface tests passed locally with Spark 3.5.6; the entire eight-year cleaning
run has not been repeated on a laptop in this revision. Connecting full data
allows broader historical replay; it does not change the saved model or reported
performance results.

For full data already on Vanda, see the [PBS and SSH-tunnel instructions](docs/prediction_interface.md#run-on-vanda-without-downloading-data).
See the [interface guide](docs/prediction_interface.md) for architecture, coordinate
handling, test commands and data-source attribution.
