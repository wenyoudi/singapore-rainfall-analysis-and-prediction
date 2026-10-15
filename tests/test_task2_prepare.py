"""Tiny synthetic tests only; no project data or scratch outputs are touched."""
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip('pyspark')
from pyspark.sql import SparkSession, functions as F
from task2_prepare import normalize_input, prepare_features, validate_input


@pytest.fixture(scope='module')
def spark():
    previous = os.environ.get('PYSPARK_PYTHON')
    os.environ['PYSPARK_PYTHON'] = sys.executable
    session = (SparkSession.builder.master('local[2]').appName('Task2Tests')
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


def series(spark, center='2023-06-01T12:00:00+08:00', missing=(), station='S1'):
    t = datetime.fromisoformat(center)
    rows = [(station, t + timedelta(minutes=5 * i), float(i + 20))
            for i in range(-12, 7) if i not in missing]
    return spark.createDataFrame(rows, 'station_id string, prediction_time timestamp, rainfall double')


def test_exact_features_and_strict_threshold(spark):
    df = series(spark)
    validate_input(df)
    row = prepare_features(df, 141).first()
    assert row.rainfall_previous_5m == 19
    assert row.rainfall_previous_30m == sum(range(14, 20))
    assert row.rainfall_previous_60m == sum(range(8, 20))
    assert row.future_rainfall_30m == sum(range(21, 27)) == 141
    assert (row.hour, row.month, row.split, row.label) == (12, 6, 'validation', 0.)
    assert prepare_features(df, 140).first().label == 1.


@pytest.mark.parametrize('missing', list(range(-12, 0)) + list(range(1, 7)))
def test_every_required_slot(spark, missing):
    # Add a more distant row: a row-count-only window would incorrectly accept it.
    df = series(spark, missing=(missing,))
    t = datetime.fromisoformat('2023-06-01T12:00:00+08:00')
    extra = -13 if missing < 0 else 7
    df = df.unionByName(spark.createDataFrame(
        [('S1', t + timedelta(minutes=5 * extra), 99.)], df.schema))
    target = prepare_features(df, 1).where(F.hour('prediction_time') == 12)
    target = target.where(F.minute('prediction_time') == 0)
    assert target.count() == 0


def test_year_boundaries_and_split_purge(spark):
    # December history is available at January prediction time across every split.
    for year, split in [(2022, 'train'), (2023, 'validation'), (2024, 'test')]:
        df = series(spark, '%d-01-01T00:00:00+08:00' % year)
        assert prepare_features(df, 1).first().split == split
    assert prepare_features(series(spark, '2021-12-31T23:45:00+08:00'), 1).count() == 1
    for year in [2022, 2023]:
        assert prepare_features(series(spark, '%d-12-31T23:45:00+08:00' % year), 1).count() == 0


def test_station_isolation_and_future_not_used_in_features(spark):
    a = series(spark)
    b = series(spark, station='S2').withColumn('rainfall', F.lit(100.))
    rows = {r.station_id: r for r in prepare_features(a.unionByName(b), 1).collect()}
    assert rows['S1'].future_rainfall_30m == 141
    assert rows['S2'].rainfall_previous_60m == 1200
    changed = a.withColumn('rainfall', F.when(F.col('prediction_time') >=
                           F.to_timestamp(F.lit('2023-06-01 12:00:00')), 999.).otherwise(F.col('rainfall')))
    assert prepare_features(changed, 1).first().rainfall_previous_60m == rows['S1'].rainfall_previous_60m


def test_invalid_data_and_duplicates(spark):
    df = series(spark)
    with pytest.raises(ValueError, match='Duplicate'):
        validate_input(df.unionByName(df.limit(1)))
    for value in [None, float('nan'), float('inf'), -1.]:
        with pytest.raises(ValueError, match='Invalid'):
            validate_input(df.withColumn('rainfall', F.lit(value).cast('double')))
    with pytest.raises(ValueError, match='Invalid'):
        validate_input(df.withColumn('prediction_time', F.expr('prediction_time + INTERVAL 1 SECOND')))


def test_time_column_precedence(spark):
    raw = (series(spark).withColumnRenamed('prediction_time', 'timestamp')
           .withColumn('observation_time', F.expr('timestamp + INTERVAL 1 SECOND')))
    validate_input(normalize_input(raw))
    with pytest.raises(ValueError, match='Invalid'):
        validate_input(normalize_input(raw, 'observation_time'))
    validate_input(normalize_input(raw.drop('timestamp').withColumn(
        'observation_time', F.expr('observation_time - INTERVAL 1 SECOND'))))


def test_parquet_roundtrip_and_no_overwrite(spark, tmp_path):
    # Write annual pieces, then read together to exercise cross-file windows.
    df = series(spark, '2023-01-01T00:00:00+08:00')
    paths = [str(tmp_path / str(y)) for y in [2022, 2023]]
    for year, path in zip([2022, 2023], paths):
        df.where(F.year('prediction_time') == year).write.parquet(path)
    result = prepare_features(spark.read.parquet(*paths), 1)
    assert result.count() == 1
    # A one-year smoke run deliberately excludes the incomplete boundary.
    assert prepare_features(spark.read.parquet(paths[1]), 1).count() == 0
    dest = str(tmp_path / 'prepared')
    result.write.mode('errorifexists').partitionBy('split', 'year').parquet(dest)
    assert spark.read.parquet(dest).first().label == 1.
    with pytest.raises(Exception, match='(?i)already exists'):
        result.write.mode('errorifexists').parquet(dest)
