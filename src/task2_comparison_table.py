"""Build report Table 7 from frozen branch results; no Spark or model fitting."""
import csv
import io
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
# Frozen result commits preserve the exact inputs used in this table.
SOURCES = [
    ('LR', 'cb5210c', 'outputs/task2/logistic_regression/1458499/test_metrics.json'),
    ('RF', 'a6ea65d', 'outputs/task2/random_forest/1459817/test_metrics.json'),
    ('GBT', 'd77126c', 'task2/Model_GBT/results/test_metrics.csv'),
]


def generate():
    rows = []
    for model, commit, path in SOURCES:
        content = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT, text=True)
        records = list(csv.DictReader(io.StringIO(content))) if path.endswith('.csv') else json.loads(content)
        if sorted(float(r['threshold_mm']) for r in records) != [0., 1., 5., 10.]:
            raise ValueError('Expected exactly the four agreed thresholds for ' + model)
        for r in records:
            row = {'threshold_mm': float(r['threshold_mm']), 'model': model,
                   'samples': int(r['samples'] if model == 'GBT' else r['n']),
                   'positives': int(float(r['positives'])),
                   'roc_auc': float(r['areaUnderROC'] if model == 'GBT' else r['roc_auc']),
                   'pr_auc': float(r['areaUnderPR'] if model == 'GBT' else r['pr_auc']),
                   'brier': float(r['brier'])}
            if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1 for k in ['roc_auc', 'pr_auc', 'brier']):
                raise ValueError('Invalid metric for ' + model)
            rows.append(row)
    rows.sort(key=lambda r: (r['threshold_mm'], ['LR', 'RF', 'GBT'].index(r['model'])))
    for q in [0., 1., 5., 10.]:
        group = [r for r in rows if r['threshold_mm'] == q]
        if len({(r['samples'], r['positives']) for r in group}) != 1:
            raise ValueError('Test counts differ between models')
        if any(r['samples'] != 6335645 or not 0 <= r['positives'] <= r['samples'] for r in group):
            raise ValueError('Unexpected test cohort size or positive count')
    output = ROOT/'outputs/task2/model_comparison'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'table_7.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Table 7. Model performance on the 2024 test set', '',
             '| Rainfall threshold | Model | ROC-AUC ↑ | PR-AUC ↑ | Brier score ↓ |',
             '| --- | --- | ---: | ---: | ---: |']
    for r in rows:
        group = [a for a in rows if a['threshold_mm'] == r['threshold_mm']]
        values = []
        for key, digits in [('roc_auc', 4), ('pr_auc', 4), ('brier', 6)]:
            best = (min if key == 'brier' else max)(a[key] for a in group)
            value = f"{r[key]:.{digits}f}"
            values.append(f'**{value}**' if r[key] == best else value)
        lines.append(f"| >{r['threshold_mm']:g} mm | {r['model']} | " + ' | '.join(values) + ' |')
    lines += ['', '**Note.** Each threshold uses 6,335,645 test observations. LR = Logistic Regression; '
              'RF = Random Forest; GBT = Gradient-Boosted Trees. Bold identifies the best value within each '
              'threshold and metric. Higher AUC and lower Brier indicate better performance.', '',
              'LR and RF used the full training and validation datasets. GBT used 2% of training and 10% '
              'of validation data, but the complete test partition. LR/RF selected settings using mean '
              'validation Brier score; GBT used mean validation Brier skill. Encoding, tuning budgets and '
              'Spark versions also differ, so results compare the implemented pipelines rather than '
              'isolating algorithm effects. Classification metrics and model-specific Brier skill are '
              'omitted because they are not available under a common policy/baseline for all three models.', '',
              '## Reproduce and sources', '',
              'Run `python src/task2_comparison_table.py` from a clone containing the source commits below '
              '(use `git fetch origin` if necessary). No training is performed and no model branch is modified.', '',
              'Test sample and positive counts match at every threshold. Exact station/time keys were not '
              'independently compared by this script.', '',
              '| Model | Source commit | Metric file |', '| --- | --- | --- |']
    for model, commit, path in SOURCES:
        lines.append(f'| {model} | `{commit}` | `{path}` |')
    (output/'table_7.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines[:18]))
    return rows


if __name__ == '__main__':
    generate()
