"""Plot RF test precision and recall from frozen summaries; no fitting or scoring."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from summarize_rf_classification import DEFAULT_RUN, summarize


def plot(run_dir):
    run = Path(run_dir)
    if json.loads((run/'manifest.json').read_text())['status'] != 'complete':
        raise ValueError('A completed RF run is required')
    rows = summarize(json.loads((run/'test_metrics.json').read_text()),
                     json.loads((run/'test_calibration.json').read_text()))
    if [r['threshold_mm'] for r in rows] != [0., 1., 5., 10.]:
        raise ValueError('Expected the four reported rainfall thresholds')
    if len({r['samples'] for r in rows}) != 1:
        raise ValueError('Expected a common test cohort')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    # Match the grouped precision/recall layout and colours used by the GBT figure.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for offset, metric, color in [(-.18, 'precision', '#25678D'),
                                   (.18, 'recall', '#C77730')]:
        valid = [(i+offset, row[metric]) for i, row in enumerate(rows) if row[metric] is not None]
        bars = ax.bar([x for x, _ in valid], [value for _, value in valid],
                      .36, label=metric.title(), color=color)
        ax.bar_label(bars, labels=[f'{value:.2%}' for _, value in valid], padding=3, fontsize=9)
        for i, row in enumerate(rows):
            if row[metric] is None:
                # Text is an absence indicator, not a zero-height precision bar.
                ax.text(i+offset, .025, 'N/A', ha='center', color=color, fontsize=9)
    ax.set_xticks(range(4), ['>0', '>1', '>5', '>10'])
    ax.set(ylim=(0, 1), xlabel='Rainfall threshold (mm)', ylabel='Proportion',
           title='Random Forest test precision and recall at probability cutoff 0.5')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.15)
    ax.legend()
    fig.text(.5, .025,
             f"{rows[0]['samples']:,} test samples per threshold. N/A: precision undefined because no positive predictions were made.",
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, 1))
    for suffix in ['png', 'svg']:
        output = run/f'rf_precision_recall.{suffix}'
        fig.savefig(output, dpi=300, bbox_inches='tight')
        print(output)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    plot(parser.parse_args().run_dir)
