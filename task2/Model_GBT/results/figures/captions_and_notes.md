# GBT report figures and captions

Source: completed GBT-05 test run `20261016T103557030872Z`. Each threshold uses 6,335,645 test samples. The frozen model used a 2% training sample. Figures are exported as 300 dpi PNG and vector PDF. CSV metrics and reliability-bin totals were reconciled before plotting. No model was retrained.

## Section 6.3 — gbt_test_calibration.png

Figure X. GBT-05 reliability diagrams for future 30-minute rainfall exceedance on the complete test partition. Points compare mean predicted probabilities with observed event frequencies within occupied equal-width probability bins. The dashed diagonal indicates perfect calibration; points below it indicate overprediction and points above it indicate underprediction. Insets magnify the 0–0.1 probability region without changing bins. Hollow markers identify bins with fewer than 100 samples. No post-training calibration was applied.

Calibration checks:

> 0 mm: overall predicted 7.779%, observed 6.361%; lowest bin predicted 3.916%, observed 2.364%, share 91.32%.
> 1 mm: overall predicted 4.543%, observed 2.571%; lowest bin predicted 3.188%, observed 1.091%, share 96.04%.
> 5 mm: overall predicted 3.151%, observed 0.838%; lowest bin predicted 2.841%, observed 0.491%, share 98.83%.
> 10 mm: overall predicted 2.898%, observed 0.383%; lowest bin predicted 2.755%, observed 0.238%, share 99.27%.

These overall means do not describe all within-bin errors. The lowest probability bin overpredicts event frequency at every threshold; the pattern in higher bins must be read from the individual diagrams. Event prevalence and bin population should accompany any calibration claim.

## Section 6.5 — gbt_feature_importance.png

Figure X. Recorded tree feature importance for the selected GBT-05 model. Recent rainfall over the previous 5, 30 and 60 minutes accounts for 53.63% of total importance. Importance indicates model usage rather than causal influence; overlapping historical windows are correlated.

## Section 6.1 or 6.4 — gbt_validation_test_comparison.png

Figure X. GBT-05 discrimination and probability error on the sampled validation set and complete test partition. Each threshold has 668,341 validation samples and 6,335,645 test samples. ROC-AUC and PR-AUC are lower on test data at every threshold. Lower test Brier scores do not independently establish better calibration because event prevalence also differs. Higher AUC and lower Brier scores indicate better performance.

## Section 6.2 — gbt_precision_recall.png

Figure X. Test precision and recall for GBT-05 at a fixed classification probability cutoff of 0.5. Recall falls to 4.02% for rainfall exceeding 10 mm, indicating that most heavy-rainfall exceedances are missed at this cutoff. The cutoff was not optimized using test outcomes.
