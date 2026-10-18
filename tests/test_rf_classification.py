"""Check fixed-cutoff recovery, undefined precision and saved-count reconciliation."""
import json
from pathlib import Path
import pytest
from summarize_rf_classification import summarize

RUN = Path(__file__).resolve().parents[1]/'outputs/task2/random_forest/1459817'


def test_cutoff_includes_bin_five_but_excludes_bin_four():
    metrics = [{'threshold_mm': 0., 'n': 8, 'positives': 3}]
    bins = [{'threshold_mm': 0., 'bin': index, 'n': n, 'observed_frequency': frequency}
            for index, n, frequency in [(0, 4, .25), (4, 1, 1.), (5, 2, .5), (9, 1, 0.)]]
    row = summarize(metrics, bins)[0]
    assert (row['true_positive'], row['false_positive'], row['false_negative'], row['true_negative']) == (1, 2, 2, 3)
    assert row['precision'] == row['recall'] == row['f1'] == pytest.approx(1/3)


def test_no_positive_predictions_and_inconsistent_bins():
    metrics = [{'threshold_mm': 1., 'n': 10, 'positives': 2}]
    bins = [{'threshold_mm': 1., 'bin': 0, 'n': 10, 'observed_frequency': .2}]
    row = summarize(metrics, bins)[0]
    assert row['precision'] is None
    assert row['recall'] == row['f1'] == 0
    with pytest.raises(ValueError, match='reconcile'):
        summarize([{**metrics[0], 'positives': 3}], bins)
    with pytest.raises(ValueError, match='Duplicate|duplicate'):
        summarize(metrics, bins+bins)
    with pytest.raises(ValueError, match='integer positive'):
        summarize(metrics, [{**bins[0], 'observed_frequency': .234}])


def test_original_saved_rf_classification_counts():
    rows = summarize(json.loads((RUN/'test_metrics.json').read_text()),
                     json.loads((RUN/'test_calibration.json').read_text()))
    assert [row['samples'] for row in rows] == [6335645]*4
    assert [row['true_positive'] for row in rows] == [7009, 0, 0, 0]
    assert [row['false_positive'] for row in rows] == [355, 0, 0, 0]
    assert [row['false_negative'] for row in rows] == [395989, 162858, 53092, 24252]
    assert rows[0]['precision'] == pytest.approx(.951792504073873)
    assert rows[0]['recall'] == pytest.approx(.017392145866728867)
    assert all(row['precision'] is None for row in rows[1:])
