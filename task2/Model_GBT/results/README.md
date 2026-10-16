# GBT-05 report results

These CSV summaries match the completed test artifacts under `../outputs/tuning_stage1/GBT-05/test_evaluation/20261016T103557030872Z/`. The saved final-test notebook records the same run. GBT-05 used a 2% training sample and the complete test partition, with 6,335,645 examples per rainfall threshold and a classification cutoff of 0.5.

| File | Contents |
| --- | --- |
| `test_metrics.csv` | Per-threshold ranking, probability error and classification metrics |
| `validation_test_comparison.csv` | Sampled validation and complete test results |
| `feature_importance.csv` | Selected model's recorded feature importance |
| `figures/` | Four PNG/PDF figure pairs and English captions |
| `make_report_figures.py` | Recreates figures from a completed test directory |

## Generate figures

Install NumPy, pandas and Matplotlib, then run from the repository root:

```powershell
python task2/Model_GBT/results/make_report_figures.py
```

To use another completed evaluation:

```powershell
python task2/Model_GBT/results/make_report_figures.py --source /path/to/test_evaluation/run
```

The source must contain `test_manifest.json`, `test_metrics.csv`, `test_reliability_bins.csv`, `feature_importance.csv` and `validation_test_comparison.csv`. The script checks completion status and reconciles reliability-bin counts and event rates with test metrics. It writes 300 dpi PNG files, vector PDFs and `figures/captions_and_notes.md`. Existing figures with the same names are overwritten. Captions describe the recorded GBT-05 experiment; use an equivalent evaluation when supplying a different source.

## Interpretation

ROC-AUC and PR-AUC declined relative to sampled validation at every threshold. Brier scores were lower, but event prevalence was also lower. At 10 mm, negative Brier skill and 4.02% recall show limited heavy-rainfall performance at cutoff 0.5. Test outcomes must not be used to tune that cutoff.

Local artifact checks cover summary agreement, confusion-count totals, event counts, precision, recall, F1, Brier skill and reliability-bin totals. They do not rerun Spark inference or verify the remote dataset's year coverage. Full prediction Parquet files are retained under the local experiment outputs.
