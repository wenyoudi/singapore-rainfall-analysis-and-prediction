# Random Forest classification performance at probability cutoff 0.5

| Rainfall threshold | Predicted positives | Precision | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| >0 mm | 7,364 | 95.18% | 1.74% | 0.0342 |
| >1 mm | 0 | Undefined | 0.00% | 0.0000 |
| >5 mm | 0 | Undefined | 0.00% | 0.0000 |
| >10 mm | 0 | Undefined | 0.00% | 0.0000 |

Each rainfall threshold uses 6,335,645 observations from the 2024 test set. A positive prediction means p >= 0.5. Undefined precision means no positive predictions; recall and F1 are zero because actual positive events exist.

Derived from the original test_metrics.json and test_calibration.json, not a new model run. The saved evaluator in src/task2_model_common.py uses min(9, floor(p * 10)); bins 5-9 exactly represent p >= 0.5. Bin positives are recovered from n * observed_frequency, rounded only within numerical tolerance to integer counts. Both total samples and positives are checked against the saved metrics. This method is exact for this bin-aligned cutoff; it cannot recover arbitrary within-bin cutoffs.

Regenerate from the repository root:

```bash
python3 src/summarize_rf_classification.py
```
