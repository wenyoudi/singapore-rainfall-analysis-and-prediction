"""Check window semantics, leakage prevention and saved-model inference."""
from datetime import datetime, timedelta
import math
import os
import sys

import pytest
pytest.importorskip("pyspark")
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.functions import vector_to_array
from rainfall_inference import SG, InputError, RainfallPredictor, features_from_history
from task2_prepare import prepare_features

WHEN = datetime(2024, 1, 1, 12, tzinfo=SG)


def history():
    return {int((WHEN-timedelta(minutes=5*i)).timestamp()): float(i) for i in range(13)}


def test_exact_feature_windows_and_future_exclusion():
    values = history()
    row = features_from_history(values, 'S50', WHEN, 5.)
    assert row['rainfall_previous_5m'] == 1.
    assert row['rainfall_previous_30m'] == 21.
    assert row['rainfall_previous_60m'] == 78.
    values[int(WHEN.timestamp())] = 999.
    values[int((WHEN+timedelta(minutes=5)).timestamp())] = 99999.
    assert features_from_history(values, 'S50', WHEN, 5.) == row


@pytest.mark.parametrize('threshold', [-1., .5, 2., float('nan'), float('inf')])
def test_unsupported_thresholds(threshold):
    with pytest.raises(InputError, match='thresholds'):
        features_from_history(history(), 'S50', WHEN, threshold)


def test_missing_invalid_and_off_grid_inputs():
    values = history()
    values.pop(int((WHEN-timedelta(minutes=10)).timestamp()))
    with pytest.raises(InputError, match='Insufficient history'):
        features_from_history(values, 'S50', WHEN, 5.)
    for value in [-1., float('nan'), float('inf')]:
        values = history()
        values[int(WHEN.timestamp())] = value
        with pytest.raises(InputError, match='invalid'):
            features_from_history(values, 'S50', WHEN, 5.)
    with pytest.raises(InputError, match='five-minute'):
        features_from_history(history(), 'S50', WHEN+timedelta(minutes=1), 5.)
    with pytest.raises(InputError, match='2017'):
        features_from_history(history(), 'S50', WHEN.replace(year=2025), 5.)


@pytest.fixture(scope='module')
def spark():
    os.environ['PYSPARK_PYTHON'] = sys.executable
    session = (SparkSession.builder.master('local[2]').appName('InterfaceChecks')
               .config('spark.sql.session.timeZone', 'Asia/Singapore')
               .config('spark.sql.shuffle.partitions', '2')
               .config('spark.ui.enabled', 'false').getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    yield session
    session.stop()


def test_saved_gbt_matches_existing_pipeline(spark, tmp_path):
    rows = [('S50', WHEN+timedelta(minutes=5*i), float(i+12)/10) for i in range(-12, 7)]
    frame = spark.createDataFrame(rows, 'station_id string, prediction_time timestamp, rainfall double')
    (frame.withColumnRenamed('prediction_time', 'timestamp')
     .write.parquet(str(tmp_path/'rainfall_2024')))
    predictor = RainfallPredictor(tmp_path)
    assert predictor.stations()[0]['id'] == 'S50'
    for q in [0., 1., 5., 10.]:
        prepared = prepare_features(frame, q).where(F.col('prediction_time') == F.lit(WHEN))
        expected = predictor.model.transform(prepared).select(vector_to_array('probability')[1]).first()[0]
        result = predictor.predict('S50', WHEN, q)
        actual_features = prepared.first().asDict()
        assert all(actual_features[k] == pytest.approx(v) if isinstance(v, float)
                   else actual_features[k] == v for k, v in result['features'].items())
        assert result['probability'] == pytest.approx(expected, abs=1e-12)
        assert len(result['history']) == 12
    actual = predictor.verify('S50', WHEN, sum(float(i+12)/10 for i in range(1, 7)))
    assert not actual['exceeded']  # Strict exceedance, equality is negative.
    # Future observations can be absent without preventing a forecast.
    past_path = tmp_path/'past_only'
    (frame.where(F.col('prediction_time') <= F.lit(WHEN)).withColumnRenamed('prediction_time','timestamp')
     .write.parquet(str(past_path/'rainfall_2024')))
    past = RainfallPredictor(past_path)
    assert past.predict('S50', WHEN, 5.)['probability'] == pytest.approx(predictor.predict('S50', WHEN, 5.)['probability'])
    assert past.verify('S50', WHEN, 5.) is None
    with pytest.raises(InputError, match='No observation'):
        predictor.predict('MISSING', WHEN, 5.)


def test_year_boundary_and_duplicate_history(spark, tmp_path):
    when = datetime(2024, 1, 1, tzinfo=SG)
    frame = spark.createDataFrame(
        [('S50', when+timedelta(minutes=5*i), .2) for i in range(-12, 1)],
        'station_id string, timestamp timestamp, rainfall double')
    for year in [2023, 2024]:
        frame.where(F.year('timestamp') == year).write.parquet(str(tmp_path/f'rainfall_{year}'))
    predictor = RainfallPredictor(tmp_path)
    assert predictor.predict('S50', when, 0.)['features']['rainfall_previous_60m'] == pytest.approx(2.4)
    duplicate_path = tmp_path/'duplicate'
    frame.unionByName(frame.limit(1)).write.parquet(str(duplicate_path/'rainfall_2024'))
    with pytest.raises(InputError, match='duplicate'):
        RainfallPredictor(duplicate_path).predict('S50', when, 0.)
