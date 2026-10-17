# Table 7. Model performance on the 2024 test set

| Rainfall threshold | Model | ROC-AUC ↑ | PR-AUC ↑ | Brier score ↓ |
| --- | --- | ---: | ---: | ---: |
| >0 mm | LR | 0.8011 | 0.4346 | 0.050081 |
| >0 mm | RF | **0.8611** | 0.5942 | 0.043471 |
| >0 mm | GBT | 0.8602 | **0.6270** | **0.036028** |
| >1 mm | LR | 0.8598 | 0.3923 | 0.020268 |
| >1 mm | RF | **0.8820** | 0.4587 | 0.019710 |
| >1 mm | GBT | 0.8515 | **0.4833** | **0.017659** |
| >5 mm | LR | 0.8521 | 0.2371 | 0.007823 |
| >5 mm | RF | **0.8687** | 0.2197 | 0.007650 |
| >5 mm | GBT | 0.8579 | **0.2681** | **0.007495** |
| >10 mm | LR | 0.8546 | 0.1520 | 0.004070 |
| >10 mm | RF | **0.8708** | 0.1200 | 0.004305 |
| >10 mm | GBT | 0.8635 | **0.1703** | **0.004049** |

**Note.** Each threshold uses 6,335,645 test observations. LR = Logistic Regression; RF = Random Forest; GBT = Gradient-Boosted Trees. Bold identifies the best value within each threshold and metric. Higher AUC and lower Brier indicate better performance.

LR and RF used the full training and validation datasets. GBT used 2% of training and 10% of validation data, but the complete test partition. LR/RF selected settings using mean validation Brier score; GBT used mean validation Brier skill. Encoding, tuning budgets and Spark versions also differ, so results compare the implemented pipelines rather than isolating algorithm effects. Classification metrics and model-specific Brier skill are omitted because they are not available under a common policy/baseline for all three models.

## Reproduce and sources

Run `python src/task2_comparison_table.py` from a clone containing the source commits below (use `git fetch origin` if necessary). No training is performed and no model branch is modified.

Test sample and positive counts match at every threshold. Exact station/time keys were not independently compared by this script.

| Model | Source commit | Metric file |
| --- | --- | --- |
| LR | `cb5210c` | `outputs/task2/logistic_regression/1458499/test_metrics.json` |
| RF | `a6ea65d` | `outputs/task2/random_forest/1459817/test_metrics.json` |
| GBT | `d77126c` | `task2/Model_GBT/results/test_metrics.csv` |
