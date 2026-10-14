"""Small runtime guards executed before the full job in the same PBS allocation."""
from datetime import datetime, timedelta

import pytest
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from test_task2_prepare import spark
from task2_train import (THRESHOLDS, classifier, evaluate, expand_thresholds,
                         preprocessing, probabilities, probability_metrics, feature_variances, COEFFICIENT_UPPER_BOUND)


def tiny(spark):
    start = datetime(2020, 1, 1)
    rows = [('S%d' % (i % 2), start + timedelta(minutes=5*i),
             float(i % 4), float(i % 4)*3, float(i % 4)*6,
             i % 24, i % 12 + 1, float([0, 1, 5, 20][i % 4]), 'train')
            for i in range(32)]
    return spark.createDataFrame(rows, 'station_id string, prediction_time timestamp, '
        'rainfall_previous_5m double, rainfall_previous_30m double, rainfall_previous_60m double, '
        'hour int, month int, future_rainfall_30m double, split string')


def test_threshold_labels_and_split_preservation(spark):
    expanded = expand_thresholds(tiny(spark))
    assert expanded.count() == 128
    assert expanded.where("split != 'train'").count() == 0
    equal = expanded.where(F.col('future_rainfall_30m') == F.col('threshold_mm'))
    assert equal.count() > 0 and equal.where('label != 0').count() == 0
    assert {r.threshold_mm for r in expanded.select('threshold_mm').distinct().collect()} == set(THRESHOLDS)


def test_training_monotonicity_serialization_and_unseen_station(spark, tmp_path):
    expanded = expand_thresholds(tiny(spark))
    encoder = preprocessing().fit(expanded)
    features = encoder.transform(expanded)
    variances = feature_variances(features)
    constant_indices = [i for i, v in enumerate(variances) if v == 0.]
    assert constant_indices  # Reserved unseen-category columns are all zero.
    estimator = classifier(variances, 0.01, max_iter=15)
    bounds = estimator.getUpperBoundsOnCoefficients().toArray()[0]
    assert all(bounds[i] == 0. for i in constant_indices)
    lr = estimator.fit(features)
    assert all(abs(lr.coefficients[i]) < 1e-12 for i in constant_indices)
    assert lr.coefficients[-1] <= 1e-10
    combined = PipelineModel(stages=encoder.stages + [lr])
    path = str(tmp_path / 'model')
    combined.write().save(path)
    for metadata in (tmp_path / 'model').rglob('part-*'):
        if metadata.parent.name == 'metadata':
            assert 'Infinity' not in metadata.read_text()
    restored = PipelineModel.load(path)
    # Neither future accumulation nor a label is required for model scoring.
    query = (expand_thresholds(tiny(spark).limit(1)).drop('label', 'future_rainfall_30m')
             .withColumn('station_id', F.lit('UNSEEN')))
    a = combined.transform(query).orderBy('threshold_mm').select('probability').collect()
    b = restored.transform(query).orderBy('threshold_mm').select('probability').collect()
    values = [float(r.probability[1]) for r in b]
    assert len(values) == 4
    assert values == pytest.approx([float(r.probability[1]) for r in a])
    assert all(0 <= p <= 1 for p in values)
    assert all(x >= y - 1e-12 for x, y in zip(values, values[1:]))
    predicted = probabilities(restored, expanded)
    assert len(probability_metrics(predicted).collect()) == 4


def test_metrics_and_single_class_auc(spark):
    df = spark.createDataFrame([(0., 0., .2), (0., 1., .8), (10., 0., .1)],
                               'threshold_mm double, label double, p double')
    metrics, bins = evaluate(df)
    assert metrics[0]['brier'] == pytest.approx(.04)
    assert metrics[0]['roc_auc'] == pytest.approx(1.)
    assert metrics[1]['roc_auc'] is None and metrics[1]['pr_auc'] is None
    assert sum(row['n'] for row in bins) == 3


def test_bound_scaling_for_constant_columns(spark):
    import math
    estimator = classifier([0., 2., 0., 4.], 0.01)
    bounds = estimator.getUpperBoundsOnCoefficients().toArray()[0]
    assert list(bounds) == [0., COEFFICIENT_UPPER_BOUND, 0., 0.]
    assert all(math.isfinite(v) for v in bounds)
    scaled = [bound * math.sqrt(v) for bound, v in zip(bounds, [0., 2., 0., 4.])]
    assert not any(math.isnan(v) for v in scaled)
    assert all(0. <= v for v in scaled)
    with pytest.raises(ValueError, match='Threshold must vary'):
        classifier([1., 0.], 0.01)
