"""Recover fixed-cutoff RF classification counts from saved calibration bins.

The original evaluator uses min(9, floor(p * 10)); therefore bins 5-9 are
exactly p >= 0.5. No individual probabilities, fitting or new scoring are needed.
"""
import argparse
import csv
import json
import math
from pathlib import Path

DEFAULT_RUN = Path(__file__).resolve().parents[1]/'outputs/task2/random_forest/1459817'


def summarize(metrics, calibration):
    """Reconcile integer class counts before calculating cutoff-0.5 metrics."""
    thresholds = {row['threshold_mm'] for row in metrics}
    if len(thresholds) != len(metrics):
        raise ValueError('Duplicate metric thresholds')
    grouped = {threshold: {} for threshold in thresholds}
    for row in calibration:
        threshold, index, n = row['threshold_mm'], row['bin'], row['n']
        if (threshold not in grouped or index not in range(10) or index != int(index)
                or index in grouped[threshold] or n <= 0 or n != int(n)):
            raise ValueError('Invalid or duplicate calibration bin')
        frequency = row['observed_frequency']
        if not math.isfinite(frequency) or not 0 <= frequency <= 1:
            raise ValueError('Invalid observed frequency')
        raw_positive = n * frequency
        positive = round(raw_positive)
        if not math.isclose(raw_positive, positive, rel_tol=0, abs_tol=1e-6):
            raise ValueError('Cannot recover integer positive count from saved frequency')
        grouped[threshold][index] = (int(n), positive)
    results = []
    for metric in sorted(metrics, key=lambda row: row['threshold_mm']):
        threshold, n, positives = metric['threshold_mm'], metric['n'], metric['positives']
        bins = grouped[threshold]
        if (n != int(n) or positives != int(positives) or not 0 < positives < n
                or sum(count for count, _ in bins.values()) != n
                or sum(positive for _, positive in bins.values()) != positives):
            raise ValueError('Calibration counts do not reconcile with saved test metrics')
        predicted_positive = sum(count for index, (count, _) in bins.items() if index >= 5)
        tp = sum(positive for index, (_, positive) in bins.items() if index >= 5)
        fp, fn = predicted_positive-tp, int(positives)-tp
        results.append({
            'threshold_mm': threshold, 'decision_cutoff': .5, 'samples': int(n),
            'actual_positives': int(positives), 'predicted_positives': predicted_positive,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
            'true_negative': int(n)-tp-fp-fn,
            'precision': tp/predicted_positive if predicted_positive else None,
            'recall': tp/positives, 'f1': 2*tp/(2*tp+fp+fn)
        })
    return results


def write_table(run):
    if json.loads((run/'manifest.json').read_text())['status'] != 'complete':
        raise ValueError('The source RF run is not complete')
    metrics = json.loads((run/'test_metrics.json').read_text())
    bins = json.loads((run/'test_calibration.json').read_text())
    if {row['threshold_mm'] for row in metrics} != {0., 1., 5., 10.}:
        raise ValueError('Expected four reported rainfall thresholds')
    rows = summarize(metrics, bins)
    if len({row['samples'] for row in rows}) != 1:
        raise ValueError('Expected the same test cohort size at every threshold')
    with (run/'rf_classification.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Random Forest classification performance at probability cutoff 0.5', '',
             '| Rainfall threshold | Predicted positives | Precision | Recall | F1 score |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        precision = f"{row['precision']:.2%}" if row['precision'] is not None else 'Undefined'
        lines.append(f"| >{row['threshold_mm']:g} mm | {row['predicted_positives']:,} | {precision} | {row['recall']:.2%} | {row['f1']:.4f} |")
    lines += ['', f"Each rainfall threshold uses {rows[0]['samples']:,} observations from the 2024 test set. "
              'A positive prediction means p >= 0.5. Undefined precision means no positive '
              'predictions; recall and F1 are zero because actual positive events exist.', '',
              'Derived from the original test_metrics.json and test_calibration.json, '
              'not a new model run. The saved evaluator in src/task2_model_common.py uses '
              'min(9, floor(p * 10)); bins 5-9 exactly represent p >= 0.5. Bin positives '
              'are recovered from n * observed_frequency, rounded only within numerical '
              'tolerance to integer counts. Both total samples and positives are checked '
              'against the saved metrics. This method is exact for this bin-aligned cutoff; '
              'it cannot recover arbitrary within-bin cutoffs.', '',
              'Regenerate from the repository root:', '',
              '```bash', 'python3 src/summarize_rf_classification.py', '```', '']
    (run/'rf_classification.md').write_text('\n'.join(lines))
    for name in ['rf_classification.csv', 'rf_classification.md']:
        print(run/name)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    write_table(parser.parse_args().run_dir)
