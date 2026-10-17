# Task 1 workflow and verified results

Run all commands from the repository root.

## Data and environments

Download the eight annual CSVs from [Historical Rainfall across Singapore](https://data.gov.sg/collections/2279/view). Keep their names as:

```text
data/raw/HistoricalRainfallacrossSingapore2017.csv
...
data/raw/HistoricalRainfallacrossSingapore2024.csv
```

Raw CSVs and processed Parquet files are excluded from Git. The committed compact summaries under `outputs/` allow figures to be regenerated without downloading or processing the full dataset. The bundled Singapore boundary and its source metadata are in `data/geography/`; mapping uses these local files and makes no network requests.

The project's full-data processing environment is **NUS Vanda**, submitted through PBS. The provided template requests one node, 8 CPUs and 32 GB RAM. For the verified job `1449898.stdct-mgmt-02`, the scheduler record supplied by the project owner reports 55 GB memory; this differs from the template's 32 GB setting, so both are documented rather than treated as identical. Spark uses `local[8]` on that allocated node and 24 GB driver memory. This is single-node Spark execution on an HPC system. The local computer is used to edit code, transfer files, and generate plots from compact summaries. Development was performed on macOS; a native Windows PowerShell workflow is documented below.

Use two environments:

- **Spark processing on Vanda:** `module load Spark/3.3.1-foss-2022a`. Use the Python and Java supplied by the module; their actual versions are recorded in the job's environment log. No separate PySpark installation is needed when using this module. For a separate installation, see `requirements-spark.txt` and the [Spark 3.3.1 environment documentation](https://archive.apache.org/dist/spark/docs/3.3.1/).
- **Local analysis/plots:** Python 3.11 with `requirements-plots.txt`. These package versions were used to regenerate the figures. All large-data aggregation remains in Spark; pandas reads only compact summaries.

From the project root, create the plotting environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-plots.txt
```

All Spark scripts set the session timezone to `Asia/Singapore` and use 32 shuffle partitions. CPU and memory allocation are selected when submitting the job. Spark driver and worker Python versions must match; set `PYSPARK_PYTHON` to the Python interpreter belonging to the Spark environment when needed.

## Regenerate figures from existing summaries

```bash
.venv/bin/python src/run_task1.py --summaries-only
```

This regenerates four data-quality figures and thirteen Task 1 figures as PNG/PDF pairs, along with the sensitivity tables and selection audits. It records that saved summaries were used; it does not claim the raw-data pipeline was rerun.

To change thresholds or the spatial period, use the runner so both spatial scripts receive the same settings:

```bash
.venv/bin/python src/run_task1.py --summaries-only \
  --spatial-years 2021 2022 2023 2024 \
  --min-spatial-coverage 80 --min-monthly-coverage 90 \
  --coverage-thresholds 80 90
```

An impossible spatial selection fails explicitly. Empty sensitivity cohorts are retained in the CSVs with zero contributing stations and undefined rates; plots omit empty curves and preserve gaps for empty individual years.

## macOS workflow: run on Vanda and return results to the MacBook

Follow these steps in order: upload code from the MacBook, submit and verify the Vanda job, download the refreshed summaries, then generate plots locally. The paths below match the project owner's workflow; other users should substitute their own login and directories.

### 1. MacBook: upload the updated code and PBS job

Run in a local terminal:

```bash
cd ~/Desktop/"DSA5208 Scalable Distributed Computing"/singapore-rainfall-analysis-and-prediction

ssh e0969984@vanda.nus.edu.sg \
  'mkdir -p /scratch/e0969984/singapore-rainfall-analysis-and-prediction/src /scratch/e0969984/singapore-rainfall-analysis-and-prediction/jobs'

rsync -avh src jobs \
  e0969984@vanda.nus.edu.sg:/scratch/e0969984/singapore-rainfall-analysis-and-prediction/
```

Upload the complete `src/` directory so shared helpers are included. Keep the local plotting environment (`.venv`) on the MacBook.

### 2. Vanda: log in and check the input files

```bash
ssh e0969984@vanda.nus.edu.sg

cd /scratch/e0969984/singapore-rainfall-analysis-and-prediction

for year in {2017..2024}; do
  ls -lh "data/raw/HistoricalRainfallacrossSingapore${year}.csv"
done

ls -lh jobs/task1_vanda.pbs
```

All eight CSV files must exist before submission. If any are missing, open a separate **MacBook terminal** and upload them:

```bash
cd ~/Desktop/"DSA5208 Scalable Distributed Computing"/singapore-rainfall-analysis-and-prediction

rsync -avh --progress data/raw/ \
  e0969984@vanda.nus.edu.sg:/scratch/e0969984/singapore-rainfall-analysis-and-prediction/data/raw/
```

### 3. Vanda: submit the job

Run from the Vanda project root:

```bash
cd /scratch/e0969984/singapore-rainfall-analysis-and-prediction

JOB_ID=$(qsub jobs/task1_vanda.pbs)
echo "$JOB_ID"
```

Save the printed job ID. The job loads `Spark/3.3.1-foss-2022a` and runs these stages in order:

1. `data_profiling.py`
2. `gap_analysis.py`
3. `cleaning_diagnostics.py`
4. `data_cleaning.py`
5. `cleaning_validation.py`
6. `rainfall_eda.py`

It requests one node, 8 CPUs, 32 GB RAM and two hours, with Spark `local[8]` and 24 GB driver memory. The verified run on 11 October 2026 completed in 14 minutes 54 seconds of walltime; future runtimes can vary. Adjust the walltime before submission if needed, or split stages into separate jobs. For a later rerun limited to cleaning, validation and EDA, retain only the last three entries in the job's `stages` array.

The job stops if a stage fails and overwrites generated outputs; raw CSVs are read only. It uses the directory from which `qsub` was called and records the actual module, Python, Java and Spark versions in `outputs/environment/vanda_JOB_ID.txt`. Plotting runs separately on the MacBook. There is no `src/data_preprocessing.py` in this repository; use the stage names above.

### 4. Vanda: monitor progress

In the same terminal:

```bash
qstat -x "$JOB_ID"

JOB_NUMBER="${JOB_ID%%.*}"
LOG_FILE="rainfall_task1.o${JOB_NUMBER}"
```

Once the job starts and its output log exists:

```bash
ls -lh "$LOG_FILE"
tail -n 60 "$LOG_FILE"
```

To follow the log continuously:

```bash
tail -f "$LOG_FILE"
```

Press **Ctrl+C** to stop following the log; this does not cancel the job. If you reconnect later, restore the variables using the actual ID printed by `qsub`:

```bash
JOB_ID='YOUR_ACTUAL_JOB_ID'
JOB_NUMBER="${JOB_ID%%.*}"
LOG_FILE="rainfall_task1.o${JOB_NUMBER}"
```

### 5. Vanda: confirm successful completion

```bash
qstat -xf "$JOB_ID" | grep -E 'job_state|Exit_status'

tail -n 80 "$LOG_FILE"

cat outputs/validation/validation_issues.csv
```

Before downloading results, confirm all three:

- The job has finished with `Exit_status = 0`.
- The log ends with `Preprocessing and Task 1 summaries completed successfully.`
- `validation_issues.csv` contains only the header `issue`.

If the job fails, inspect the log:

```bash
grep -niE 'error|exception|traceback|outofmemory|walltime|killed' "$LOG_FILE"
tail -n 150 "$LOG_FILE"
```

Do not proceed with partial results. Searching for errors supplements checking the exit status; it does not replace it.

### 6. MacBook: download the refreshed summaries

Open a local MacBook terminal, or run `exit` to leave SSH first:

```bash
cd ~/Desktop/"DSA5208 Scalable Distributed Computing"/singapore-rainfall-analysis-and-prediction

mkdir -p outputs

rsync -avh --progress \
  --include='/profiling/***' \
  --include='/gap_analysis/***' \
  --include='/cleaning_diagnostics/***' \
  --include='/cleaning/***' \
  --include='/validation/***' \
  --include='/eda/***' \
  --include='/environment/***' \
  --exclude='*' \
  e0969984@vanda.nus.edu.sg:/scratch/e0969984/singapore-rainfall-analysis-and-prediction/outputs/ \
  outputs/
```

Confirm the transfer completed successfully before plotting. This single `rsync` command transfers all seven selected directories through one SSH connection. A folder-by-folder loop opens a separate connection for every folder and can request the password repeatedly. Authentication or MFA may still be required for each new SSH session; no password is saved in the repository. Downloading only `outputs/gap_analysis/` refreshes the raw-gap figures but does not refresh rainfall EDA or cleaning validation. The large processed Parquet directories can stay on Vanda for subsequent processing.

### 7. MacBook: generate the updated figures

The local `.venv` was created during development. On a fresh machine, first create it and install plotting dependencies:

```bash
cd ~/Desktop/"DSA5208 Scalable Distributed Computing"/singapore-rainfall-analysis-and-prediction

python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-plots.txt
```

With the plotting environment ready:

```bash
cd ~/Desktop/"DSA5208 Scalable Distributed Computing"/singapore-rainfall-analysis-and-prediction

.venv/bin/python src/run_task1.py --summaries-only
```

This generates all 17 PNG/PDF figure pairs, sensitivity tables, and analysis records from the downloaded summaries.

### 8. MacBook: open the results

```bash
open outputs/figures/eda
open outputs/figures/gap_analysis
```

These steps cover preprocessing and Task 1 only.

## Windows workflow: PowerShell locally, Linux commands on Vanda

The full-data Spark processing still runs on Vanda using the same PBS job. Windows users can transfer files with `ssh`/`scp` and run the plotting scripts with native Windows Python; WSL is optional. The local commands in the macOS section (`rsync`, `.venv/bin/python`, `open`, and Bash loops) are not PowerShell commands.

Use **64-bit Python 3.11** and the Windows **OpenSSH Client**. Check these in PowerShell:

```powershell
py -3.11 --version
Get-Command ssh, scp
```

If needed, install Python 3.11 and its launcher using the [Python Windows instructions](https://docs.python.org/3.11/using/windows.html). If `ssh`/`scp` are unavailable, enable the OpenSSH Client following [Microsoft's instructions](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse). Only the client is needed to connect to Vanda.

### 1. Windows PowerShell: set paths and upload code

Change `$Project` if the project is somewhere else, including a OneDrive-managed Desktop. Change both the login and scratch path if using a different Vanda account.

```powershell
$Project = Join-Path $HOME "Desktop\DSA5208 Scalable Distributed Computing\singapore-rainfall-analysis-and-prediction"
$Vanda = "e0969984@vanda.nus.edu.sg"
$RemoteRoot = "/scratch/e0969984/singapore-rainfall-analysis-and-prediction"
Set-Location $Project

ssh $Vanda "mkdir -p '$RemoteRoot/src' '$RemoteRoot/jobs' '$RemoteRoot/data/raw'"
if ($LASTEXITCODE -ne 0) { throw "Remote directory setup failed" }

scp -r .\src .\jobs "${Vanda}:${RemoteRoot}/"
if ($LASTEXITCODE -ne 0) { throw "Code upload failed" }
```

The repository's `.gitattributes` preserves LF line endings for Python and PBS files when checked out with Git. If editing the PBS file manually in a Windows editor, keep its line-ending setting at **LF**.

If the eight raw annual CSVs are not already on Vanda, upload them:

```powershell
scp -r .\data\raw "${Vanda}:${RemoteRoot}/data/"
if ($LASTEXITCODE -ne 0) { throw "Raw-data upload failed" }
```

### 2. Vanda: submit, monitor and validate

From PowerShell, connect:

```powershell
ssh $Vanda
```

Once connected, the terminal is running a Linux shell on Vanda. Use these **Bash** commands:

```bash
cd /scratch/e0969984/singapore-rainfall-analysis-and-prediction

for year in {2017..2024}; do
  ls -lh "data/raw/HistoricalRainfallacrossSingapore${year}.csv"
done
```

Confirm all eight CSVs exist, then submit:

```bash
JOB_ID=$(qsub jobs/task1_vanda.pbs)
echo "$JOB_ID"
qstat -x "$JOB_ID"

JOB_NUMBER="${JOB_ID%%.*}"
LOG_FILE="rainfall_task1.o${JOB_NUMBER}"
```

Once the log exists, monitor it with `tail -f "$LOG_FILE"`; Ctrl+C stops following the log without cancelling the job. After completion:

```bash
qstat -xf "$JOB_ID" | grep -E 'job_state|Exit_status'
tail -n 80 "$LOG_FILE"
cat outputs/validation/validation_issues.csv
```

Require `Exit_status = 0`, the final success message, and a validation issues file containing only its header, as in macOS workflow step 5. The same resource limits, runtime caveat, failure diagnostics, and job-ID restoration instructions apply.

Return to Windows PowerShell:

```bash
exit
```

### 3. Windows PowerShell: download all refreshed summaries

In the original PowerShell session, `$Project`, `$Vanda`, and `$RemoteRoot` remain set. If you opened a new terminal, first repeat the variable assignments in step 1.

```powershell
Set-Location $Project
New-Item -ItemType Directory -Force -Path .\outputs | Out-Null

scp -r "${Vanda}:${RemoteRoot}/outputs" .
if ($LASTEXITCODE -ne 0) { throw "Summary download failed; stop before plotting" }
```

This downloads the complete `outputs/` directory in one SCP transfer, avoiding one connection per summary folder. It can include older remote figures, so regenerate the figures in step 4. Authentication or MFA can still be required when opening a new SSH session. Keep the processed Parquet datasets on Vanda. Do not copy a Mac/Linux virtual environment to Windows; create a separate Windows environment.

### 4. Windows PowerShell: install dependencies and generate plots

On first use:

```powershell
Set-Location $Project
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed" }

.\.venv\Scripts\python.exe -m pip install -r requirements-plots.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed" }
```

On every plotting run:

```powershell
.\.venv\Scripts\python.exe src/run_task1.py --summaries-only
if ($LASTEXITCODE -ne 0) { throw "Plot generation failed" }

Invoke-Item .\outputs\figures\eda
Invoke-Item .\outputs\figures\gap_analysis
```

Calling the virtual environment's Python directly does not require activating PowerShell scripts or changing execution policy. Windows virtual environments use a `Scripts` directory; see the [Python virtual-environment documentation](https://docs.python.org/3.11/library/venv.html).

To check the compact-summary calculations without installing Spark locally:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q tests/test_task1_analysis.py tests/test_saved_task1_outputs.py
```

If only reproducing figures from committed summaries, steps 1-3 can be skipped: obtain the complete repository, set `$Project`, and run step 4. This does not constitute a new full-data processing run.

### Optional WSL workflow and verification limits

WSL users can use Linux `ssh`/`rsync` and `.venv/bin/python` commands with their actual Linux project path and an independently installed Python 3.11 environment. Use `explorer.exe .` to open the current project folder instead of macOS `open`. Run the same Vanda job and transfer the same output folders. Do not mix a native Windows virtual environment with a WSL environment.

The pinned plotting packages were successfully resolved and downloaded as Windows x64/Python 3.11-compatible wheels (or platform-independent wheels). This checks package availability, not installation or execution. Native Windows and WSL execution have not been tested during this revision. The Windows workflow documents the platform-specific commands, while the existing successful tests and full figure regeneration were performed on macOS. Numerical summaries should be compared within floating-point tolerance; rendered images can differ in fonts and PDF metadata across systems. A successful Windows installation, summary-test run and regeneration of all 17 figure pairs are the remaining verification steps on a Windows machine. This workflow does not claim support for running the full Spark pipeline natively on Windows.

## Spark stage reference

Run these stages in order in the Spark environment:

```bash
spark-submit src/data_profiling.py
spark-submit src/gap_analysis.py
spark-submit src/cleaning_diagnostics.py
spark-submit src/data_cleaning.py
spark-submit src/cleaning_validation.py
spark-submit src/rainfall_eda.py
```

Then regenerate the figures with the plotting environment as above. Cleaning overwrites annual Parquet directories; summary and figure scripts overwrite their named outputs. Raw CSVs are read only. Structural validation failures raise an error and stop a sequential run; reporting gaps are recorded without being treated as invalid rainfall.

For an environment where Spark and plotting are both configured, the same sequence can be launched through the runner:

```bash
.venv/bin/python src/run_task1.py \
  --spark-submit /absolute/path/to/spark-submit \
  --spark-options '--master local[8] --driver-memory 24g'
```

The resource options above are not automatic resource detection. For the Vanda/MacBook split, use the PBS workflow instead. The runner checks each stage's exit status and stops on the first failure.

## Methods and outputs

| Stage | Outputs | Method |
| --- | --- | --- |
| Raw profiling and gaps | `outputs/profiling/`, `outputs/gap_analysis/` | Raw timestamps; gap-based completeness within the first/last observed station-year span is approximate before alignment. A raw gap is attributed to its later observation's month. |
| Cleaning diagnostics | `outputs/cleaning_diagnostics/` | Timestamp offsets, alignment collisions, station metadata variants. |
| Cleaning | `data/processed/rainfall_YEAR/`, `outputs/cleaning/` | Trim station IDs, standardize names, reject invalid/non-finite rainfall and coordinates, align only offsets up to one second. Reject fractional seconds and aligned timestamps outside the file year. Exact-grid readings win duplicate ties; conflicting rainfall fails instead of being silently discarded. |
| Final validation | `outputs/validation/` | Check values, calendar year, five-minute alignment, uniqueness, row counts, gaps, and metadata. A failed check produces diagnostics and a nonzero exit. |
| Spark EDA | `outputs/eda/` | Annual, month, hour, station-year, station-month, location, positive histogram, and approximate percentile tables. Inputs are checked for duplicates and invalid timestamps before aggregation. |
| Data-quality figures | `outputs/figures/gap_analysis/` | Four figures generated by `plot_data_quality.py`. |
| Task 1 figures and audits | `outputs/figures/eda/` | Distributions, annual/monthly/hourly patterns, coverage sensitivity, accumulation, station scatter and two-panel IDW map. |

`timestamp` is the aligned analysis time. `original_timestamp` and `observation_time` preserve the original source time for traceability; subsequent analysis must use `timestamp`.

Rainfall is an amount in **mm per five-minute interval**. Occurrence uses strictly positive rainfall; exceedance uses strictly greater than the stated threshold. No missing observation is filled with zero. Summed rainfall across different stations is a network total, not a national rainfall depth. The extrapolated annual field is a diagnostic assuming missing slots have the observed mean; figures use recorded amounts/rates instead.

### Annual and monthly sensitivity

`annual_coverage_sensitivity.csv` includes both pooled rates (sum of rainy counts divided by sum of observed counts) and equal station-year averages for:

- All reporting stations and stations present in every study year.
- Stations meeting each coverage threshold separately in each year; station identity can change.
- A fixed station set meeting the threshold in **every** year from 2017 through 2024.

Calendar coverage uses every five-minute slot in the full year/month, including time outside a station's observed operating span. Leap years and month lengths are accounted for. This differs intentionally from the observed-span coverage used in gap diagnostics.

`annual_cohort_membership.csv` records each station/year selection, including wholly absent years. `monthly_coverage_sensitivity.csv` applies thresholds to each station-year-month, and compares pooled rates with equal station-month averages across the study years. Its station and station-period counts describe the changing contributors. `coverage_filtered_monthly_accumulation.csv` gives recorded station-month medians and quartiles with at least 90% monthly coverage by default.

For the committed summaries, there are 38 stations present in every year, 21 stations meeting 80% coverage in every year, and none meeting 90% in every year. No station meets 90% calendar-year coverage in 2018. These unavailable comparisons remain undefined rather than being filled or silently relaxing the threshold. The 2019 minimum is consistent across the available annual comparisons; the year with the largest mean rainfall changes between the complete network and some fixed station sets. Coverage filtering assesses sensitivity, but cannot establish that missingness is unrelated to rainfall.

### Spatial figures

Both spatial figures use stations with at least 80% calendar coverage in **each** year of the explicitly selected 2021-2024 period (56 stations for the saved summaries). Each station's occurrence and mean rainfall are equal averages of its annual rates. Coordinates are the median of its station-year coordinates within that period. Both panels use the same station-years; mismatched or stale spatial summary inputs are rejected.

`spatial_selected_station_years.csv` records the selected data; `spatial_balanced_station_summary.csv` records map values. IDW uses projected distances in EPSG:3414, power 2 and the eight nearest stations by default, clips to the boundary, and masks cells over 12 km from the nearest station. These surfaces are exploratory estimates. The cached boundary metadata declares the **Open Data Commons Open Database License 1.0**; see `data/geography/singapore_boundary_source.json` for its source and version.

### Run records

`outputs/task1_run_manifest.json` records mode, parameters, plotting package versions, source hashes, completed stages, and success/failure. `task1_analysis_manifest.json` records analysis settings, selected stations, and input-summary hashes. `spatial_map_manifest.json` records interpolation settings and boundary provenance/hash.

## Verification

```bash
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q tests/test_task1_analysis.py tests/test_saved_task1_outputs.py
```

Spark regression tests additionally require PySpark in the test interpreter's environment and a compatible Java runtime:

```bash
SPARK_LOCAL_IP=127.0.0.1 .venv/bin/python -m pytest -q tests
```

The Spark tests use small synthetic observations and do not read or overwrite the project datasets. They cover invalid rainfall, duplicate conflicts, exact-grid tie selection, unexpected offsets, year boundaries, and EDA safeguards. Summary tests cover pooled/equal weighting, missing years, empty cohorts, leap-year denominators, monthly selection, and the common spatial period.

During development, 24 synthetic/summary tests passed locally using PySpark 3.5.3 and Python 3.11. This included a tiny CSV-to-cleaning-to-validation-to-EDA pipeline and a deliberately corrupted temporary dataset to check that validation fails. The final local suite passes **33 tests**. Additional integration tests check the saved full-data outputs, including count reconciliation, rainfall metrics, threshold rates, histograms, percentiles, gaps and metadata.

The updated full-data Spark pipeline was run on **Vanda on 11 October 2026**, job `1449898.stdct-mgmt-02`, with a reported scheduler exit status of 0 and walltime of 14 minutes 54 seconds. The downloaded environment log records **Spark 3.3.1, Python 3.10.4 and Java 11.0.18**. The downloaded validation summary passes for all eight years: 48,538,713 input rows became 48,538,627 cleaned rows, with 86 redundant rows removed and no structural validation issues. Within observed station-year spans, 3,747,854 missing five-minute slots reconcile exactly with the gap summaries. Missing slots are retained as missing.

The Vanda summary CSVs were downloaded and checked locally, and all 17 figure pairs were regenerated from those refreshed summaries. The empty 90% all-year station cohort is expected and is represented with undefined rates; 21 stations qualify at 80%, and the fixed 2021-2024 spatial selection contains 56 stations. The known S113 coordinate variation remains reported rather than silently overwritten. Windows execution remains unverified as described above.
