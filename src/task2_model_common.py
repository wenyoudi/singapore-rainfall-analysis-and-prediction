"""Shared feature and metric definitions, matching the reviewed LR experiment.

Copied from LR commit 2f1f6be so the RF branch can start directly from main.
Only the listed historical/calendar/station features and threshold enter models.
"""
import json
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

THRESHOLDS = [0., 1., 5., 10.]
RAINFALL_FEATURES = ['rainfall_previous_5m', 'rainfall_previous_30m', 'rainfall_previous_60m']
CATEGORIES = ['station_id', 'hour', 'month']

def expand_thresholds(df):
    """Equal weight for each threshold; copies retain the original split and key."""
    return (df.drop('threshold_mm', 'label')
            .withColumn('threshold_mm', F.explode(F.array(*[F.lit(r) for r in THRESHOLDS])))
            .withColumn('label', (F.col('future_rainfall_30m') > F.col('threshold_mm')).cast('double')))

def preprocessing():
    stages = [StringIndexer(inputCol=c, outputCol=c + '_index', handleInvalid='keep')
              for c in CATEGORIES]
    stages.append(OneHotEncoder(inputCols=[c + '_index' for c in CATEGORIES],
                                outputCols=[c + '_vector' for c in CATEGORIES], dropLast=False))
    # Preserve LR's feature order, including threshold as the final coordinate.
    stages.append(VectorAssembler(inputCols=[c + '_vector' for c in CATEGORIES] +
                                  RAINFALL_FEATURES + ['threshold_mm'], outputCol='features'))
    return Pipeline(stages=stages)

def probabilities(model, df):
    return (model.transform(df)
            .withColumn('p', vector_to_array('probability')[1])
            .select('station_id', 'prediction_time', 'threshold_mm', 'label', 'p'))

def probability_metrics(predictions):
    p = F.greatest(F.lit(1e-15), F.least(F.lit(1. - 1e-15), F.col('p')))
    return (predictions.groupBy('threshold_mm').agg(
        F.count('*').alias('n'), F.sum('label').alias('positives'),
        F.avg('label').alias('prevalence'), F.avg('p').alias('mean_probability'),
        F.avg(F.pow(F.col('p') - F.col('label'), 2)).alias('brier'),
        F.avg(-F.col('label') * F.log(p) - (1 - F.col('label')) * F.log(1-p)).alias('log_loss'))
        .orderBy('threshold_mm'))

def evaluate(predictions):
    """Collect only four metric rows and at most 40 calibration bins."""
    rows = [r.asDict() for r in probability_metrics(predictions).collect()]
    for row in rows:
        subset = predictions.where(F.col('threshold_mm') == row['threshold_mm'])
        for name, metric in [('roc_auc', 'areaUnderROC'), ('pr_auc', 'areaUnderPR')]:
            row[name] = (BinaryClassificationEvaluator(
                labelCol='label', rawPredictionCol='p', metricName=metric, numBins=1000
            ).evaluate(subset) if 0 < row['positives'] < row['n'] else None)
    calibration = (predictions.withColumn('bin', F.least(F.lit(9), F.floor(F.col('p') * 10)))
                   .groupBy('threshold_mm', 'bin').agg(
                       F.count('*').alias('n'), F.avg('p').alias('predicted_probability'),
                       F.avg('label').alias('observed_frequency'))
                   .orderBy('threshold_mm', 'bin'))
    return rows, [r.asDict() for r in calibration.collect()]

def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
