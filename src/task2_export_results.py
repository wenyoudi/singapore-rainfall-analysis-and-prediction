"""Export compact Task 2 results with the Python standard library; no Spark jobs."""
import argparse
import json
from pathlib import Path
import zipfile

FILES = ['manifest.json', 'test_metrics.json', 'test_calibration.json', 'example_predictions.json']


def number(value):
    return 'N/A' if value is None else format(value, '.6g')


def export_results(run_dir, output_dir=None):
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / 'review'
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    metrics = json.loads((run_dir / 'test_metrics.json').read_text())
    lines = ['# Task 2 run summary', '',
             '- Job ID: %s' % manifest.get('pbs_job_id', 'unknown'),
             '- Pipeline status: %s' % manifest.get('status', 'unknown'),
             '- Started (UTC): %s' % manifest.get('started_utc', 'unknown'),
             '- Finished (UTC): %s' % manifest.get('finished_utc', 'unknown'),
             '- Spark / Python / Java: %s / %s / %s' % tuple(
                 manifest.get(k, 'unknown') for k in ['spark', 'python', 'java']),
             '- Selected regParam: %s' % number(manifest.get('selected_reg_param')),
             '- Selected iterations: %s' % manifest.get('selected_iterations', 'unknown'),
             '- Validation mean Brier: %s' % number(manifest.get('validation_macro_brier')),
             '- Selection rule: %s' % manifest.get('selection', 'unknown'), '',
             '## Test results (2024)', '',
             '| Threshold mm | Samples | Positive rate | ROC-AUC | PR-AUC | Brier | Constant Brier | Brier skill |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in sorted(metrics, key=lambda r: r['threshold_mm']):
        baseline = row.get('constant_baseline_brier')
        skill = 1 - row['brier'] / baseline if baseline and baseline > 0 else None
        values = [row.get(k) for k in ['threshold_mm', 'n', 'prevalence', 'roc_auc',
                                      'pr_auc', 'brier', 'constant_baseline_brier']] + [skill]
        lines.append('| ' + ' | '.join(str(v) if i == 1 else number(v)
                                      for i, v in enumerate(values)) + ' |')
    lines += ['', 'Positive Brier skill means improvement over the training-prevalence constant baseline.',
              'AUC can be undefined for single-class data; these results use the recorded Spark AUC approximation.',
              '', '## Validation candidates', '', '| regParam | Mean Brier | Iterations | Iteration limit reached |',
              '| --- | --- | --- | --- |']
    for c in manifest.get('validation_candidates', []):
        lines.append('| %s | %s | %s | %s |' % (number(c.get('reg_param')), number(c.get('macro_brier')),
                     c.get('iterations', 'unknown'), c.get('iteration_limit_reached', 'unknown')))
    lines += ['', '## Sample counts before threshold expansion', '',
              '| Year | Split | Input | Retained | Excluded |', '| --- | --- | --- | --- | --- |']
    inputs = {r['year']: r['count'] for r in manifest.get('input_counts', [])}
    for r in manifest.get('retained_counts', []):
        n = inputs.get(r['year'])
        lines.append('| %s | %s | %s | %s | %s |' % (
            r['year'], r['split'], n if n is not None else 'N/A', r['count'],
            n-r['count'] if n is not None else 'N/A'))
    limit = manifest.get('parameters', {}).get('max_iter')
    iterations = manifest.get('selected_iterations')
    lines += ['', '## Checks and limitations', '']
    if manifest.get('status') != 'complete':
        lines.append('- WARNING: pipeline is not marked complete; do not report as a completed run.')
    if limit is not None and iterations is not None and iterations >= limit:
        lines.append('- WARNING: selected model reached the iteration limit; investigate convergence.')
    else:
        lines.append('- Inspect iteration counts above; being below the limit alone does not establish model quality.')
    lines += ['- Scheduler exit status is not in these JSON files; retain the PBS log separately.',
              '- The archive contains compact JSON artifacts, not the trained model or prepared dataset.',
              '- Test scores must not be used to retune and still be described as untouched test results.',
              '- Threshold evaluation is limited to the reported values.', '']
    missing = [name for name in FILES if not (run_dir / name).is_file()]
    if missing:
        lines.append('- Missing optional artifacts: ' + ', '.join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / 'task2_summary.md'
    summary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    archive = output_dir / 'task2_results.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(summary, summary.name)
        for name in FILES:
            if (run_dir / name).is_file():
                bundle.write(run_dir / name, name)
    return summary, archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    for path in export_results(args.run_dir, args.output_dir):
        print(path)


if __name__ == '__main__':
    main()
