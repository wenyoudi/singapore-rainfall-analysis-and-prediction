"""Smoke-test committed pipelines with historical features only, without fitting."""
import math
from pathlib import Path
import os
import sys

import pytest
pytest.importorskip('pyspark')
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['outputs/task2/logistic_regression/1458499/model',
          'outputs/task2/random_forest/1459817/model',
          'task2/Model_GBT/outputs/tuning_stage1/GBT-05/model']


@pytest.fixture(scope='module')
def model_spark():
    old = os.environ.get('PYSPARK_PYTHON')
    os.environ['PYSPARK_PYTHON'] = sys.executable
    session = (SparkSession.builder.master('local[2]').appName('SavedModelChecks')
               .config('spark.ui.enabled', 'false').getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    yield session
    session.stop()
    if old is None:
        os.environ.pop('PYSPARK_PYTHON', None)
    else:
        os.environ['PYSPARK_PYTHON'] = old


@pytest.mark.parametrize('path', MODELS, ids=['LR', 'RF', 'GBT'])
def test_committed_pipeline_scores_without_future_or_label(model_spark, path):
    model = PipelineModel.load(str(ROOT/path))
    data = model_spark.createDataFrame(
        [(station, 12, 6, .2, 1.2, 2.4, threshold)
         for station in ['S50', 'UNSEEN_STATION'] for threshold in [0., 1., 5., 10.]],
        'station_id string, hour int, month int, rainfall_previous_5m double, '
        'rainfall_previous_30m double, rainfall_previous_60m double, threshold_mm double')
    probabilities = model.transform(data).select('probability').collect()
    assert len(probabilities) == 8
    for row in probabilities:
        assert len(row.probability) == 2
        assert all(math.isfinite(float(p)) and 0 <= p <= 1 for p in row.probability)
        assert sum(row.probability) == pytest.approx(1.)
