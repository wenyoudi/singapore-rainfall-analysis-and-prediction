"""Plot Random Forest reliability from saved test bins; no fitting or Spark needed."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'outputs/task2/random_forest/1459817'


def plot(run_dir):
    run = Path(run_dir)
    bins = json.loads((run / 'test_calibration.json').read_text())
    metrics = json.loads((run / 'test_metrics.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('A completed run is required')
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))
    fig.subplots_adjust(top=.86, bottom=.14, left=.09, right=.98, hspace=.35, wspace=.28)
    color = '#126f78'
    for q, ax in zip([0., 1., 5., 10.], axes.flat):
        rows = sorted([b for b in bins if b['threshold_mm'] == q], key=lambda b: b['bin'])
        metric = next(m for m in metrics if m['threshold_mm'] == q)
        total = sum(b['n'] for b in rows)
        if total != metric['n'] or len({b['bin'] for b in rows}) != len(rows):
            raise ValueError('Calibration bin counts do not reconcile')
        for key, expected in [('predicted_probability', metric['mean_probability']),
                              ('observed_frequency', metric['prevalence'])]:
            if not math.isclose(sum(b['n']*b[key] for b in rows)/total, expected, abs_tol=1e-9):
                raise ValueError('Calibration averages do not reconcile')
        # Identical axes permit direct reading across rainfall thresholds.
        inset = ax.inset_axes([.56, .13, .40, .36])
        for target, limit in [(ax, 1.), (inset, .05)]:
            target.plot([0, limit], [0, limit], '--', color='#777777', linewidth=1)
            target.plot([b['predicted_probability'] for b in rows],
                        [b['observed_frequency'] for b in rows], color=color, linewidth=1.4)
            for b in rows:
                target.plot(b['predicted_probability'], b['observed_frequency'], 'o',
                            markersize=5, markeredgecolor=color,
                            markerfacecolor=color if b['n'] >= 100 else 'white')
            target.set(xlim=(0, limit), ylim=(0, limit))
            target.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            target.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            target.grid(alpha=.15)
        inset.set(xticks=[0, .05], yticks=[0, .05], title='Low-probability detail')
        inset.title.set_fontsize(8)
        inset.tick_params(labelsize=7, pad=1)
        lowest = next(b for b in rows if b['bin'] == 0)
        ax.set(xlabel='Mean predicted probability', ylabel='Observed event frequency')
        ax.set_title(f'Next 30-minute rainfall > {q:g} mm\n'
                     f"Lowest bin: {lowest['n']/total:.1%} of samples", fontsize=10)
    fig.suptitle('Random Forest probability calibration — 2024 test set', fontsize=14, y=.985)
    fig.legend(handles=[Line2D([], [], color=color, marker='o', label='Random Forest'),
                        Line2D([], [], color='#777777', linestyle='--', label='Perfect calibration'),
                        Line2D([], [], color=color, marker='o', markerfacecolor='white',
                               linestyle='none', label='Bin with fewer than 100 samples')],
               loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, frameon=False, fontsize=9)
    fig.text(.5, .025, 'Ten equal-width bins; occupied bins only. Insets magnify existing bins (no rebinning).\n'
             '6,335,645 samples per threshold. Lines join bin averages; no recalibration applied.',
             ha='center', fontsize=9)
    for suffix in ['png', 'pdf']:
        path = run / f'rf_calibration.{suffix}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    plot(parser.parse_args().run_dir)
