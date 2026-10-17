"""Synthetic shared-data validation and RF serialization checks."""
import os
import sys
from datetime import datetime, timedelta

import pytest

pytest.importorskip("pyspark")
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from task2_random_forest import forest, validate_prepared
from task2_model_common import expand_thresholds, preprocessing, probabilities


@pytest.fixture(scope='module')
def spark():
    previous = os.environ.get('PYSPARK_PYTHON')
    os.environ['PYSPARK_PYTHON'] = sys.executable
    session = (SparkSession.builder.master('local[2]').appName('RFTests')
               .config('spark.sql.session.timeZone', 'Asia/Singapore')
               .config('spark.sql.shuffle.partitions', '2')
               .config('spark.ui.enabled', 'false').getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    yield session
    session.stop()
    if previous is None:
        os.environ.pop('PYSPARK_PYTHON', None)
    else:
        os.environ['PYSPARK_PYTHON'] = previous


def base_rows(spark, year=2023):
    t = datetime.fromisoformat('%d-06-01T12:00:00+08:00' % year)
    split = 'train' if year <= 2022 else 'validation' if year == 2023 else 'test'
    return spark.createDataFrame([
        ('S%d' % i, t, t + timedelta(minutes=30), 12, 6, year, split,
         r, r*6, r*12, r*6)
        for i, r in enumerate([0., .1, 1., 3.])],
        'station_id string, prediction_time timestamp, target_end_time timestamp, '
        'hour int, month int, year int, split string, '
        'rainfall_previous_5m double, rainfall_previous_30m double, '
        'rainfall_previous_60m double, future_rainfall_30m double')


def tiny(spark):
    # Enough rows for the production minInstancesPerNode=20 to grow real splits.
    return (base_rows(spark, 2020).crossJoin(spark.range(16))
            .withColumn('prediction_time', F.expr('prediction_time + id * INTERVAL 5 MINUTES'))
            .withColumn('target_end_time', F.expr('target_end_time + id * INTERVAL 5 MINUTES'))
            .drop('id'))


def test_shared_dataset_checks(spark):
    base = base_rows(spark)
    assert validate_prepared(base, {2023: 4})[0]['count'] == 4
    with pytest.raises(ValueError, match='counts differ'):
        validate_prepared(base, {2023: 2})
    with pytest.raises(ValueError, match='Duplicate'):
        validate_prepared(base.unionByName(base))
    with pytest.raises(ValueError, match='Invalid'):
        validate_prepared(base.withColumn('split', F.lit('train')))
    with pytest.raises(ValueError, match='Invalid'):
        validate_prepared(base.withColumn('future_rainfall_30m', F.lit(float('nan'))))
    with pytest.raises(ValueError, match='Missing'):
        validate_prepared(base.drop('target_end_time'))


def test_rf_reload_and_unseen_station(spark, tmp_path):
    df = expand_thresholds(tiny(spark))
    encoder = preprocessing().fit(df)
    fitted = forest(3, 2, 5208).fit(encoder.transform(df))
    model = PipelineModel(stages=encoder.stages + [fitted])
    destination = str(tmp_path / 'rf')
    model.write().save(destination)
    restored = PipelineModel.load(destination)
    query = df.withColumn('station_id', F.lit('UNSEEN')).drop('label', 'future_rainfall_30m')
    a = model.transform(query).select('probability').collect()
    b = restored.transform(query).select('probability').collect()
    assert len(a) == len(b) == 256
    assert [r.probability[1] for r in a] == pytest.approx([r.probability[1] for r in b])
    assert all(0 <= r.probability[1] <= 1 for r in b)
    predictions = probabilities(restored, df)
    assert predictions.count() == 256
    assert predictions.select('p').distinct().count() > 1


def test_complete_runner_on_small_shared_parquet(spark, tmp_path, monkeypatch):
    import json
    from functools import reduce
    from types import SimpleNamespace
    import task2_random_forest as rf
    frames = [base_rows(spark, year) for year in [2020, 2023, 2024]]
    base = reduce(lambda a, b: a.unionByName(b), frames)
    path = str(tmp_path / 'shared')
    base.write.partitionBy('split', 'year').parquet(path)
    monkeypatch.setattr(rf, 'EXPECTED_COUNTS', {2020: 4, 2023: 4, 2024: 4})
    output = tmp_path / 'output'
    output.mkdir()
    args = SimpleNamespace(prepared=path, check_only=False, trees=2, depths=[2], seed=5208)
    manifest = {}
    rf.run(spark, args, output, manifest)
    assert manifest['status'] == 'complete'
    assert manifest['selected_depth'] == 2
    rows = json.loads((output / 'test_metrics.json').read_text())
    assert len(rows) == 4 and all(r['n'] == 4 for r in rows)
    assert all(0 <= r['brier'] <= 1 for r in rows)
    assert (output / 'model').is_dir()


@pytest.mark.parametrize('column,value', [
    ('hour', 11), ('month', 5), ('year', 2022), ('rainfall_previous_5m', -1.),
    ('rainfall_previous_30m', float('inf')), ('rainfall_previous_60m', None),
    ('rainfall_previous_5m', 999.), ('station_id', '')])
def test_corrupt_shared_rows_rejected(spark, column, value):
    with pytest.raises(ValueError, match='Invalid'):
        validate_prepared(base_rows(spark).withColumn(column, F.lit(value)))


def test_horizon_rejected_and_check_only(spark, tmp_path, monkeypatch):
    import task2_random_forest as rf
    from types import SimpleNamespace
    base = base_rows(spark)
    with pytest.raises(ValueError, match='Invalid'):
        validate_prepared(base.withColumn('target_end_time',
                          F.expr('target_end_time + INTERVAL 5 MINUTES')))
    path = str(tmp_path / 'shared')
    base.write.partitionBy('split', 'year').parquet(path)
    monkeypatch.setattr(rf, 'EXPECTED_COUNTS', {2023: 4})
    def never_fit():
        raise AssertionError('Check mode must not fit an encoder')
    monkeypatch.setattr(rf, 'preprocessing', never_fit)
    manifest = {}
    rf.run(spark, SimpleNamespace(prepared=path, check_only=True), tmp_path, manifest)
    assert manifest['status'] == 'validated'
    assert not (tmp_path / 'model').exists()


def test_provenance_and_output_guards(tmp_path):
    import json
    import task2_random_forest as rf
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    source = {'status': 'complete', 'thresholds_mm': rf.THRESHOLDS,
              'source_sha256': {'task2_prepare.py': rf.PREPARATION_HASH},
              'retained_counts': [{'year': y, 'count': n} for y, n in rf.EXPECTED_COUNTS.items()]}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(source))
    assert len(rf.validate_provenance(str(prepared))['sha256']) == 64
    source['status'] = 'failed'
    manifest.write_text(json.dumps(source))
    with pytest.raises(ValueError, match='manifest differs'):
        rf.validate_provenance(str(prepared))
    for output in ['/scratch/e1496655/new', '/scratch/e0969984',
                   '/scratch/e0969984/../other']:
        with pytest.raises(ValueError, match='Output'):
            rf.destination(output, rf.PREPARED)
    assert str(rf.destination('/scratch/e0969984/task2/new', rf.PREPARED)).endswith('/task2/new')


def test_monotonicity_diagnostic_and_label_equality(spark):
    from task2_random_forest import monotonicity
    df = spark.createDataFrame([('S1', datetime(2024, 1, 1), q, p)
                                for q, p in [(0., .6), (1., .7), (5., .2), (10., .1)]],
                               'station_id string, prediction_time timestamp, threshold_mm double, p double')
    result = monotonicity(df)
    assert result['adjacent_pairs'] == 3 and result['violating_pairs'] == 1
    expanded = expand_thresholds(tiny(spark))
    assert expanded.where('future_rainfall_30m = threshold_mm AND label != 0').count() == 0
