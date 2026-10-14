# Shared Task 2 dataset

Status update (2026-10-14): saved full-data notebook outputs record 47,048,074
eligible station-time observations and 188,192,296 examples, exported to
`/nfs/home/svu/le.pan/singapore-rainfall-analysis-and-prediction/task2/data/task2/notebook_20261014T073533Z/`.
The recorded runtime is Spark 3.5.6 / Python 3.11.17, distinct from the original batch
template configuration below. Remote files have not been downloaded or directly
reverified locally. See the [Task 2 README](../README.md) for details.

Run on Vanda from **your own writable checkout**, reading the shared cleaned
Parquet without copying or changing it. Both models must use the same generated
examples. The baseline code is maintained separately; share the generated dataset with its author.

```bash
# First run small semantic checks on a compute node using the Spark module.
spark-submit tests/check_task2_dataset.py

# Run from task2/data_prep on a compute node.
spark-submit src/build_task2_dataset.py --input-root /path/to/processed --output-root /path/to/new/output
```

Default input:
`/scratch/e0969984/singapore-rainfall-analysis-and-prediction/data/processed/`.
Notebook output remains under `task2/data/task2/notebook_<timestamp>/`. The CLI requires an explicit new output directory:

```bash
spark-submit src/build_task2_dataset.py --input-root /path/to/processed --output-root /path/to/new/output
```

Existing output directories are rejected to avoid overwriting shared results.
Require successful execution and `manifest.json` status `complete` before training.
A failed run can leave partial output; select a new output directory on retry.

## Fixed contract

- SGT timestamps; station ID, hour, month, historical 5/30/60-minute rainfall,
  and threshold are the basic features.
- At prediction time t, the t observation is already known. History covers
  t-55 through t (12 slots). Future label covers t+5 through t+30 (6 slots).
- Unique five-minute timestamps and complete history/future windows are required.
  Missing observations are never filled with zero.
- Train: 2017-2022; validation: 2023; test: 2024. Labels crossing split boundaries
  are excluded. Historical observations from preceding splits are allowed.
- Thresholds: 0, 1, 5, 10 mm; strict greater-than labels. Each eligible station/time
  generates four examples. Keep all four in the same split.
- `future_30m_mm` is retained for auditing only: **never put it in model features**.
  Feature encoding must be fitted on training data, with an explicit policy for
  stations first appearing in validation/test.
- Do not adjust thresholds using validation/test distributions; any revision
  must be based on training data and applied to both models.

## Outputs

- `examples/`: Parquet partitioned by split and year.
- `quality_by_year.csv`: source counts, incomplete windows, boundary exclusions,
  eligible prediction times. History/future missing counts overlap.
- `class_balance.csv`: sample counts and positive rates per split/threshold.
- `manifest.json`: contract, environment, completion status and future rainfall
  percentiles by split (test figures are descriptive only).

```python
train = spark.read.parquet('/path/to/output/examples').where("split = 'train'")
validation = spark.read.parquet('/path/to/output/examples').where("split = 'validation'")
test = spark.read.parquet('/path/to/output/examples').where("split = 'test'")
```

Local checks cannot verify shared Vanda permissions, runtime or full-data counts.
