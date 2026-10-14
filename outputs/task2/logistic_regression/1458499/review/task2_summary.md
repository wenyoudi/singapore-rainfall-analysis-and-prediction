# Task 2 run summary

- Job ID: 1458499.stdct-mgmt-02
- Pipeline status: complete
- Started (UTC): 2026-10-14T13:15:56.661877+00:00
- Finished (UTC): 2026-10-14T13:23:07.758408+00:00
- Spark / Python / Java: 3.3.1 / 3.10.4 / 11.0.18
- Selected regParam: 0
- Selected iterations: 21
- Validation mean Brier: 0.0214595
- Selection rule: minimum equal-threshold mean validation Brier score; ties keep first

## Test results (2024)

| Threshold mm | Samples | Positive rate | ROC-AUC | PR-AUC | Brier | Constant Brier | Brier skill |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 6335645 | 0.063608 | 0.801073 | 0.43457 | 0.0500805 | 0.0595747 | 0.159366 |
| 1 | 6335645 | 0.025705 | 0.859836 | 0.39229 | 0.0202675 | 0.0250499 | 0.190915 |
| 5 | 6335645 | 0.00837989 | 0.852063 | 0.237113 | 0.00782332 | 0.00830972 | 0.0585339 |
| 10 | 6335645 | 0.00382787 | 0.854573 | 0.152013 | 0.0040702 | 0.00381322 | -0.0673927 |

Positive Brier skill means improvement over the training-prevalence constant baseline.
AUC can be undefined for single-class data; these results use the recorded Spark AUC approximation.

## Validation candidates

| regParam | Mean Brier | Iterations | Iteration limit reached |
| --- | --- | --- | --- |
| 0 | 0.0214595 | 21 | False |
| 0.01 | 0.0223611 | 11 | False |
| 0.1 | 0.0235143 | 11 | False |

## Sample counts before threshold expansion

| Year | Split | Input | Retained | Excluded |
| --- | --- | --- | --- | --- |
| 2017 | train | 5256020 | 4803912 | 452108 |
| 2018 | train | 4594246 | 3973669 | 620577 |
| 2019 | train | 5026862 | 4931059 | 95803 |
| 2020 | train | 6334473 | 6229257 | 105216 |
| 2021 | train | 7044646 | 6973560 | 71086 |
| 2022 | train | 7133230 | 7069791 | 63439 |
| 2023 | validation | 6742890 | 6683962 | 58928 |
| 2024 | test | 6406260 | 6335645 | 70615 |

## Checks and limitations

- Inspect iteration counts above; being below the limit alone does not establish model quality.
- Scheduler exit status is not in these JSON files; retain the PBS log separately.
- The archive contains compact JSON artifacts, not the trained model or prepared dataset.
- Test scores must not be used to retune and still be described as untouched test results.
- Threshold evaluation is limited to the reported values.
