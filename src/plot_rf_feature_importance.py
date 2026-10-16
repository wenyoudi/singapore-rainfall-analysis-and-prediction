"""Plot grouped impurity importances from the completed RF run; no model fitting."""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

DEFAULT_RUN = Path(__file__).resolve().parents[1] / 'outputs/task2/random_forest/1459817'
# Verified from this saved model's OneHotEncoder data/categorySizes: 89, 25, 13.
# Each includes the reserved unknown category. Numerical inputs follow these groups.
CATEGORY_SIZES = [89, 25, 13]
LABELS = ['Station ID', 'Hour', 'Month', 'Previous 5-minute rainfall',
          'Previous 30-minute rainfall', 'Previous 60-minute rainfall', 'Rainfall threshold']


def plot(run_dir, category_sizes=CATEGORY_SIZES):
    run = Path(run_dir)
    manifest = json.loads((run/'manifest.json').read_text())
    values = manifest['feature_importances']
    sizes = list(category_sizes) + [1, 1, 1, 1]
    if (manifest['status'] != 'complete' or sum(sizes) != len(values)
            or any(n < 1 for n in sizes)
            or any(not math.isfinite(v) or v < 0 for v in values)
            or not math.isclose(sum(values), 1., abs_tol=1e-9)):
        raise ValueError('Invalid feature importances or category dimensions')
    stages = run/'model/stages'
    assembler = json.loads(next(stages.glob('*VectorAssembler*/metadata/part-*')).read_text())
    expected = ['station_id_vector', 'hour_vector', 'month_vector', 'rainfall_previous_5m',
                'rainfall_previous_30m', 'rainfall_previous_60m', 'threshold_mm']
    if assembler['paramMap']['inputCols'] != expected:
        raise ValueError('Unexpected saved feature order')
    rows, offset = [], 0
    for name, width in zip(LABELS, sizes):
        rows.append({'feature': name, 'encoded_columns': width,
                     'importance': sum(values[offset:offset+width])})
        offset += width
    with (run/'rf_feature_importance.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['feature', 'encoded_columns', 'importance'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: -r['importance']))
    ordered = sorted(rows, key=lambda r: r['importance'])
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(9, 5.3), layout='constrained')
    ax.barh([r['feature'] for r in ordered], [r['importance'] for r in ordered], color='#126f78', height=.65)
    for y, row in enumerate(ordered):
        ax.text(row['importance'] + .004, y, f"{row['importance']:.3%}", va='center', fontsize=10)
    ax.set(xlim=(0, max(r['importance'] for r in rows)*1.23),
           xlabel='Normalized impurity-based importance',
           title='Random Forest feature importance (50 trees, depth 8)')
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    fig.supxlabel('Categorical importances are summed across their one-hot columns.\n'
                  'These training-based scores do not establish causal effects or independent contributions.', fontsize=9)
    for suffix in ['png', 'pdf']:
        target = run/f'rf_feature_importance.{suffix}'
        fig.savefig(target, dpi=220, bbox_inches='tight')
        print(target)
    plt.close(fig)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--category-sizes', nargs=3, type=int, default=CATEGORY_SIZES,
                        help='Saved OneHotEncoder widths for station, hour and month')
    args = parser.parse_args()
    plot(args.run_dir, args.category_sizes)
