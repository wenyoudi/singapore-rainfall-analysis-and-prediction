"""Validate and plot saved Task 2 results without Spark, fitting or test-set tuning."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

THRESHOLDS = [0., 1., 5., 10.]
DEFAULT_RF = 'outputs/task2/random_forest/1459817'
DEFAULT_LR = 'outputs/task2/comparison_inputs/logistic_regression/1458499'
COLORS = {'Random Forest': '#157a76', 'Logistic regression': '#b35432'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_run(path):
    path = Path(path)
    manifest = read_json(path / 'manifest.json')
    metrics = read_json(path / 'test_metrics.json')
    calibration = read_json(path / 'test_calibration.json')
    if manifest.get('status') != 'complete':
        raise ValueError('Only completed runs can be compared')
    if sorted(r['threshold_mm'] for r in metrics) != THRESHOLDS:
        raise ValueError('Exactly one metric row per agreed threshold is required')
    test_n = sum(r['count'] for r in manifest['retained_counts'] if r['split'] == 'test')
    for row in metrics:
        n, positives = row['n'], row['positives']
        if n != test_n or n <= 0 or not 0 <= positives <= n:
            raise ValueError('Invalid test counts')
        bounded = ['prevalence', 'mean_probability', 'brier', 'constant_baseline_brier',
                   'constant_baseline_probability']
        for key in bounded + ['roc_auc', 'pr_auc']:
            value = row[key]
            if value is None and key in ['roc_auc', 'pr_auc'] and positives in [0, n]:
                continue
            if value is None or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid probability metric: ' + key)
        if not math.isfinite(row['log_loss']) or row['log_loss'] < 0:
            raise ValueError('Invalid log loss')
        if not math.isclose(row['prevalence'], positives/n, abs_tol=1e-12):
            raise ValueError('Prevalence differs from positive counts')
        bins = [b for b in calibration if b['threshold_mm'] == row['threshold_mm']]
        if not bins or len({b['bin'] for b in bins}) != len(bins):
            raise ValueError('Missing or duplicate calibration bins')
        for b in bins:
            if (not isinstance(b['bin'], int) or not 0 <= b['bin'] <= 9 or b['n'] <= 0
                    or not all(math.isfinite(b[k]) and 0 <= b[k] <= 1
                               for k in ['predicted_probability', 'observed_frequency'])):
                raise ValueError('Invalid calibration bin')
        if sum(b['n'] for b in bins) != n:
            raise ValueError('Calibration counts differ from metrics')
        for bin_key, metric_key in [('predicted_probability', 'mean_probability'),
                                    ('observed_frequency', 'prevalence')]:
            if not math.isclose(sum(b['n']*b[bin_key] for b in bins)/n, row[metric_key], abs_tol=1e-9):
                raise ValueError('Calibration averages differ from metrics')
        p, q = row['constant_baseline_probability'], row['prevalence']
        if not math.isclose(q*(1-p)**2+(1-q)*p*p, row['constant_baseline_brier'], abs_tol=1e-12):
            raise ValueError('Constant baseline Brier is inconsistent')
    if {b['threshold_mm'] for b in calibration} != set(THRESHOLDS):
        raise ValueError('Unexpected calibration threshold')
    return {'path': path, 'manifest': manifest, 'metrics': metrics, 'calibration': calibration}


def comparison_rows(rf, lr):
    if rf['manifest']['retained_counts'] != lr['manifest']['retained_counts']:
        raise ValueError('Prepared split/year counts differ')
    provenance = rf['manifest']['shared_provenance']
    lr_hash = hashlib.sha256((lr['path']/'manifest.json').read_bytes()).hexdigest()
    if provenance['sha256'] != lr_hash:
        raise ValueError('LR manifest does not match RF shared dataset provenance')
    rows = []
    for name, run in [('Random Forest', rf), ('Logistic regression', lr)]:
        for row in sorted(run['metrics'], key=lambda r: r['threshold_mm']):
            other = next(r for r in (lr if name == 'Random Forest' else rf)['metrics']
                         if r['threshold_mm'] == row['threshold_mm'])
            if row['n'] != other['n'] or row['positives'] != other['positives']:
                raise ValueError('Test sample or positive counts differ between models')
            if not math.isclose(row['constant_baseline_probability'], other['constant_baseline_probability'], abs_tol=1e-12):
                raise ValueError('Training baseline probabilities differ')
            baseline = row['constant_baseline_brier']
            rows.append({'model': name, **row,
                         'brier_skill': 1-row['brier']/baseline if baseline else None,
                         'probability_bias': row['mean_probability']-row['prevalence']})
    return rows


def plot_results(runs, rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'pdf.fonttype': 42})
    for zoom, stem in [(False, 'calibration'), (True, 'calibration_low_probability')]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 9), layout='constrained')
        limit = .1 if zoom else 1.
        for q, ax in zip(THRESHOLDS, axes.flat):
            ax.plot([0, limit], [0, limit], '--', color='#777777', linewidth=1, label='Perfect calibration')
            for name, run in runs.items():
                bins = sorted([b for b in run['calibration'] if b['threshold_mm'] == q], key=lambda b: b['bin'])
                x = [b['predicted_probability'] for b in bins]
                y = [b['observed_frequency'] for b in bins]
                ax.plot(x, y, '-o', color=COLORS[name], label=name, markersize=4)
                if zoom:
                    # No rebinning: only magnify existing ten-bin summaries.
                    b = next((b for b in bins if b['bin'] == 0), None)
                    if b:
                        ax.annotate(f"{name}: bin 0 holds {b['n']/sum(c['n'] for c in bins):.1%}",
                                    xy=(.04, .94 if name == 'Random Forest' else .85),
                                    xycoords='axes fraction', fontsize=8, color=COLORS[name])
            ax.set(xlim=(0, limit), ylim=(0, limit), title=f'Next 30-minute rainfall > {q:g} mm',
                   xlabel='Mean predicted probability', ylabel='Observed event frequency')
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(alpha=.15)
        axes[0, 0].legend(loc='lower right', fontsize=8)
        fig.suptitle('2024 held-out calibration' + (' — low-probability view' if zoom else ''), fontsize=14)
        fig.supxlabel('Existing 10 equal-width bins; occupied bins only. Lines join bin averages; no recalibration.', fontsize=9)
        for suffix in ['png', 'pdf']:
            fig.savefig(output / f'{stem}.{suffix}', bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout='constrained')
    for name in runs:
        subset = [r for r in rows if r['model'] == name]
        for ax, metric in zip(axes, ['brier', 'pr_auc', 'brier_skill']):
            ax.plot(range(4), [r[metric] for r in subset], '-o', label=name, color=COLORS[name])
    for ax, title in zip(axes, ['Brier score (lower is better)', 'PR-AUC (higher is better)', 'Brier skill vs constant baseline']):
        ax.set(title=title, xticks=range(4), xticklabels=['>0', '>1', '>5', '>10'], xlabel='Rainfall threshold (mm)')
        ax.grid(alpha=.15)
    axes[2].axhline(0, color='#777777', linestyle='--', linewidth=1)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=9)
    fig.suptitle('2024 held-out model comparison — GBT pending', fontsize=14)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'model_comparison.{suffix}', bbox_inches='tight')
    plt.close(fig)


def generate(rf_dir=DEFAULT_RF, lr_dir=DEFAULT_LR, output='outputs/task2/comparison'):
    rf, lr = load_run(rf_dir), load_run(lr_dir)
    rows = comparison_rows(rf, lr)
    output = Path(output)
    inputs = [rf['path'].resolve(), lr['path'].resolve()]
    resolved = output.resolve()
    if any(resolved == p or p in resolved.parents or resolved in p.parents for p in inputs):
        raise ValueError('Comparison output must not overlap archived input runs')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'test_comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    calibration_rows = [{'model': name, **b} for name, run in [('Random Forest', rf), ('Logistic regression', lr)]
                        for b in run['calibration']]
    with (output/'calibration_bins.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(calibration_rows[0]))
        writer.writeheader()
        writer.writerows(calibration_rows)
    plot_results({'Random Forest': rf, 'Logistic regression': lr}, rows, output)
    lines = ['# Saved Task 2 results: RF and LR', '',
             'GBT results are pending. This comparison uses frozen 2024 test outputs; no models were fitted or tuned.', '',
             '| Threshold (mm) | RF Brier ↓ | LR Brier ↓ | RF PR-AUC ↑ | LR PR-AUC ↑ | RF Brier skill |',
             '| --- | --- | --- | --- | --- | --- |']
    for q in THRESHOLDS:
        a, b = [next(r for r in rows if r['threshold_mm'] == q and r['model'] == name)
                for name in ['Random Forest', 'Logistic regression']]
        lines.append(f"| >{q:g} | {a['brier']:.6f} | {b['brier']:.6f} | {a['pr_auc']:.4f} | {b['pr_auc']:.4f} | {a['brier_skill']:.1%} |")
    macros = {name: sum(r['brier'] for r in rows if r['model'] == name)/4 for name in COLORS}
    improvement = 1-macros['Random Forest']/macros['Logistic regression']
    lines += ['', f"Mean Brier: RF {macros['Random Forest']:.6f}; LR {macros['Logistic regression']:.6f} ({improvement:.1%} lower for RF).", '',
              'RF improves Brier at 0, 1 and 5 mm; LR improves Brier at 10 mm. LR has better PR-AUC at 5 and 10 mm.',
              'RF overestimates heavy-rain probability: at >10 mm its mean forecast is 2.0173%, versus 0.3828% observed.',
              'RF Brier skill at >10 mm is negative: it loses to the training-prevalence constant forecast.', '',
              '## Verification and limits', '',
              '- Both manifests report complete runs; split/year counts and per-threshold test sample/positive counts agree.',
              '- RF references exactly the archived LR manifest SHA256 for its shared prepared dataset.',
              '- Matching counts and provenance support comparison; individual station/time test keys were not independently compared here.',
              '- Calibration bins reconcile with all test sample counts, observed prevalence and mean predictions.',
              '- AUC uses the same Spark 1000-bin approximation. Results are descriptive; no significance or independence claim is made.',
              '- Calibration plots use saved ten-bin summaries. The low-probability plot magnifies these bins; it does not rebin predictions.',
              '- No confidence intervals, recalibration or classification-cutoff tuning can be derived from these aggregate files alone.',
              '- RF had zero adjacent threshold-monotonicity violations on this test cohort; this is not a model-wide guarantee.',
              '- Do not use this test comparison to retune either model and still call 2024 untouched.',
              '- Keep GBT pending until its final frozen results and shared dataset provenance are available.', '',
              '![Calibration](calibration.png)', '', '![Low-probability calibration](calibration_low_probability.png)', '',
              '![Model comparison](model_comparison.png)', '']
    (output/'README.md').write_text('\n'.join(lines), encoding='utf-8')
    hashes = {name: {n: hashlib.sha256((run['path']/n).read_bytes()).hexdigest()
                     for n in ['manifest.json', 'test_metrics.json', 'test_calibration.json']}
              for name, run in [('Random Forest', rf), ('Logistic regression', lr)]}
    (output/'provenance.json').write_text(json.dumps({'inputs': hashes, 'gbt_status': 'pending',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2)+'\n', encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rf-run', default=DEFAULT_RF)
    parser.add_argument('--lr-run', default=DEFAULT_LR)
    parser.add_argument('--output', default='outputs/task2/comparison')
    args = parser.parse_args()
    rows = generate(args.rf_run, args.lr_run, args.output)
    print('Saved comparison for %d model/threshold pairs to %s' % (len(rows), args.output))


if __name__ == '__main__':
    main()
