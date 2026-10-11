"""Small Spark regression tests; no project data are read or overwritten."""
import sys
import os
import pytest

pytest.importorskip('pyspark')
from pyspark.sql import SparkSession
from data_cleaning import prepare_records, deduplicate_records
from rainfall_eda import validate_input, enrich_coverage


@pytest.fixture(scope='module')
def spark():
    previous = os.environ.get('PYSPARK_PYTHON')
    os.environ['PYSPARK_PYTHON'] = sys.executable
    session = (SparkSession.builder.master('local[2]').appName('Task1RegressionTests')
               .config('spark.sql.session.timeZone', 'Asia/Singapore')
               .config('spark.sql.shuffle.partitions', '2').config('spark.ui.enabled', 'false')
               .config('spark.pyspark.python', sys.executable).getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    yield session
    session.stop()
    if previous is None:
        os.environ.pop('PYSPARK_PYTHON', None)
    else:
        os.environ['PYSPARK_PYTHON'] = previous


def records(spark, times=('2017-06-01T12:00:00+08:00',), rain='.2', latitude='1.3'):
    return spark.createDataFrame([(' S1 ', 'Station One', t, rain, latitude, '103.8') for t in times],
                                 'station_id string, station_name string, timestamp string, '
                                 'reading_value string, location_latitude string, location_longitude string')


def test_alignment_and_exact_grid_wins_on_both_sides(spark):
    raw = records(spark, ('2017-06-01T11:59:59+08:00', '2017-06-01T12:00:00+08:00',
                          '2017-06-01T12:00:01+08:00'))
    cleaned, stats = deduplicate_records(prepare_records(raw, 2017), raw.columns)
    row = cleaned.first()
    assert stats['extras'] == 2 and stats['conflicts'] == 0
    assert row.original_timestamp == '2017-06-01T12:00:00+08:00'
    assert row.station_id == 'S1'
    assert row.timestamp.hour == row.observation_time.hour  # both collected in same host zone
    assert cleaned.selectExpr('hour(timestamp) as h').first().h == 12


@pytest.mark.parametrize('rain', ['Infinity', '-Infinity', 'NaN', '-.2', None, 'bad'])
def test_invalid_rainfall_rejected_before_writing(spark, rain):
    with pytest.raises(ValueError, match='Invalid'):
        prepare_records(records(spark, rain=rain), 2017)


@pytest.mark.parametrize('time', ['2017-06-01T12:00:02+08:00', '2017-06-01T12:00:00.5+08:00',
                                 '2017-12-31T23:59:59+08:00'])
def test_unexpected_offsets_and_aligned_year_crossing_rejected(spark, time):
    with pytest.raises(ValueError):
        prepare_records(records(spark, (time,)), 2017)


def test_invalid_coordinates_and_conflicting_duplicates_rejected(spark):
    with pytest.raises(ValueError):
        prepare_records(records(spark, latitude='Infinity'), 2017)
    raw = records(spark).unionByName(records(spark, rain='.4'))
    with pytest.raises(ValueError, match='conflicting'):
        deduplicate_records(prepare_records(raw, 2017), raw.columns)


def test_eda_cannot_count_duplicates_as_coverage(spark):
    valid = prepare_records(records(spark), 2017)
    validate_input(valid, 2017)
    with pytest.raises(ValueError, match='Duplicate'):
        validate_input(valid.unionByName(valid), 2017)
    with pytest.raises(ValueError, match='Invalid'):
        validate_input(valid, 2018)


def test_leap_year_denominator_and_no_measured_extrapolation():
    rows = enrich_coverage([dict(year=2020, observed_intervals=100, recorded_rainfall_mm=5)])
    assert rows[0]['calendar_expected_intervals'] == 366 * 288
    assert rows[0]['recorded_rainfall_mm'] == 5
