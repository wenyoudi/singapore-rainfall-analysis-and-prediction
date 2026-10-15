"""Check archived evaluation integrity and refusal of incompatible comparisons."""
import json
from pathlib import Path
import shutil

import pytest
from task2_compare_results import DEFAULT_LR, DEFAULT_RF, load_run, comparison_rows

ROOT = Path(__file__).resolve().parents[1]


def test_archived_runs_reconcile():
    rows = comparison_rows(load_run(ROOT/DEFAULT_RF), load_run(ROOT/DEFAULT_LR))
    assert len(rows) == 8
    heavy = next(r for r in rows if r['model'] == 'Random Forest' and r['threshold_mm'] == 10)
    assert heavy['brier_skill'] < 0
    assert heavy['mean_probability'] > heavy['prevalence'] * 5


def copied_run(tmp_path):
    dest = tmp_path/'run'
    dest.mkdir()
    for name in ['manifest.json', 'test_metrics.json', 'test_calibration.json']:
        shutil.copyfile(ROOT/DEFAULT_RF/name, dest/name)
    return dest


def test_incomplete_run_rejected(tmp_path):
    path = copied_run(tmp_path)
    m = json.loads((path/'manifest.json').read_text())
    m['status'] = 'running'
    (path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='completed'):
        load_run(path)


def test_missing_bin_population_rejected(tmp_path):
    path = copied_run(tmp_path)
    bins = json.loads((path/'test_calibration.json').read_text())
    bins.pop()
    (path/'test_calibration.json').write_text(json.dumps(bins))
    with pytest.raises(ValueError, match='Calibration counts'):
        load_run(path)


def test_changed_bin_probabilities_rejected(tmp_path):
    path = copied_run(tmp_path)
    bins = json.loads((path/'test_calibration.json').read_text())
    bins[0]['predicted_probability'] = .9
    (path/'test_calibration.json').write_text(json.dumps(bins))
    with pytest.raises(ValueError, match='Calibration averages'):
        load_run(path)


def test_wrong_source_manifest_rejected():
    rf, lr = load_run(ROOT/DEFAULT_RF), load_run(ROOT/DEFAULT_LR)
    rf['manifest']['shared_provenance']['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='provenance'):
        comparison_rows(rf, lr)


def test_different_test_labels_rejected():
    rf, lr = load_run(ROOT/DEFAULT_RF), load_run(ROOT/DEFAULT_LR)
    rf['metrics'][0]['positives'] += 1
    with pytest.raises(ValueError, match='positive counts'):
        comparison_rows(rf, lr)
