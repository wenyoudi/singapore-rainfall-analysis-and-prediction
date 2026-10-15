# Completed Random Forest run 1459817

- Vanda job: `1459817.stdct-mgmt-02`; scheduler Exit_status=0 as supplied in the terminal transcript.
- Manifest status: complete; 15 October 2026, 15:58:05 to 16:53:34 Asia/Singapore (about 55m29s).
- Runtime: Spark 3.3.1, Python 3.10.4, Java 11.0.18, local[8], 24 GB driver heap.
- Shared prepared dataset: `/scratch/e1496655/task2/final-1458499.stdct-mgmt-02/prepared`.
- Train 2017–2022, select on 2023, test 2024. All 47,000,855 base samples retained from the shared cohort.
- Four strict-exceedance thresholds: 0, 1, 5, 10 mm. Threshold is a feature in one pooled forest.
- Selected 50 trees, depth 8, seed 5208 by mean validation Brier (0.0194223).
- All 14 RF synthetic tests passed on Vanda before full training.
- Zero adjacent threshold-monotonicity violations observed on the test cohort; no general guarantee.

This archive preserves the downloaded JSON results, complete Spark PipelineModel,
and preprocessing pipeline. Spark dot/CRC sidecars are omitted. Training code hashes
are in manifest.json; inventory.json lists SHA256 for the archived run files.
The model was reloaded and checked against validation predictions on Vanda during
the run. No additional cross-version reload claim is made for the local copy.

Use the same Spark version for loading the model. It expects station_id, hour,
month, rainfall_previous_5m/30m/60m and threshold_mm as prepared features; labels
and future totals are not needed for model scoring. Building those features from
a station and prediction time still requires a history lookup and is not an
implemented user-facing interface.

See [frozen RF/LR comparison](../../comparison/README.md). RF loses to the constant
baseline at >10 mm and overpredicts heavy-rain probabilities. No retraining or
calibration fitting was performed after viewing test results. GBT is pending.

Original downloaded files remain under ignored outputs/random_forest/1459817/.
