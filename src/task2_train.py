"""Full chronological, multi-threshold Logistic Regression baseline for Spark 3.3.1."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform

from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Matrices
from pyspark.ml.stat import Summarizer
from pyspark.sql import SparkSession, functions as F

from task2_prepare import INPUT, normalize_input, prepare_features, validate_input, output_path

THRESHOLDS = [0., 1., 5., 10.]
COEFFICIENT_UPPER_BOUND = 1e6
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
    # Threshold must remain the final coordinate for its monotonic bound.
    stages.append(VectorAssembler(inputCols=[c + '_vector' for c in CATEGORIES] +
                                  RAINFALL_FEATURES + ['threshold_mm'], outputCol='features'))
    return Pipeline(stages=stages)


def feature_variances(df):
    """Collect one small vector of training-only feature variances."""
    return df.select(Summarizer.variance(F.col('features')).alias('variance')).first().variance


def classifier(variances, reg_param, max_iter=80):
    # Spark 3.3.1 scales supplied bounds by feature standard deviation, even
    # when standardization=False. Infinity * zero becomes NaN for constant
    # columns, including the reserved unknown-category one-hot coordinates.
    # A zero upper bound is feasible for those coefficients, which Spark
    # returns as zero. Use a large finite bound elsewhere: Spark 3.3.1
    # cannot reload explicit Infinity values from matrix-parameter JSON.
    dimension = len(variances)
    if dimension < 1 or any(not math.isfinite(v) or v < 0 for v in variances):
        raise ValueError('Feature variances must be finite and nonnegative')
    if variances[-1] == 0:
        raise ValueError('Threshold must vary in the training data')
    bounds = [0. if v == 0. else COEFFICIENT_UPPER_BOUND for v in variances]
    bounds[-1] = 0.
    return LogisticRegression(
        featuresCol='features', labelCol='label', family='binomial',
        regParam=reg_param, elasticNetParam=0., maxIter=max_iter, tol=1e-6,
        upperBoundsOnCoefficients=Matrices.dense(1, dimension, bounds))


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


def run(spark, args, root, manifest):
    print('STAGE: prepare all stations, 2017-2024', flush=True)
    raw = spark.read.parquet(*[args.input_root.rstrip('/') + '/rainfall_%d' % y
                               for y in range(2017, 2025)])
    clean = normalize_input(raw)
    validate_input(clean)
    manifest['input_counts'] = [r.asDict() for r in clean.groupBy(
        F.year('prediction_time').alias('year')).count().orderBy('year').collect()]
    prepared_path = str(root / 'prepared')
    (prepare_features(clean, 0.).drop('label', 'threshold_mm').write.mode('errorifexists')
     .partitionBy('split', 'year').parquet(prepared_path))
    base = spark.read.parquet(prepared_path)
    counts = [r.asDict() for r in base.groupBy('split', 'year').count().orderBy('year').collect()]
    if {r['year'] for r in counts} != set(range(2017, 2025)):
        raise ValueError('At least one study year has no complete samples')
    manifest['retained_counts'] = counts
    save_json(root / 'manifest.json', manifest)

    print('STAGE: fit encoding using training only', flush=True)
    # Fit indexers once on unique base observations; expansion is deferred.
    fitting = base.where("split = 'train'").withColumn('threshold_mm', F.lit(0.))
    encoder = preprocessing().fit(fitting)
    encoder.write().save(str(root / 'preprocessing'))
    train = (encoder.transform(expand_thresholds(base.where("split = 'train'")))
             .select('features', 'label', 'threshold_mm').persist(StorageLevel.DISK_ONLY))
    validation = (encoder.transform(expand_thresholds(base.where("split = 'validation'")))
                  .select('station_id', 'prediction_time', 'features', 'label', 'threshold_mm')
                  .persist(StorageLevel.DISK_ONLY))
    try:
        rates = [r.asDict() for r in train.groupBy('threshold_mm').agg(
            F.count('*').alias('n'), F.avg('label').alias('probability')).collect()]
        if len(rates) != 4 or any(not 0 < r['probability'] < 1 for r in rates):
            raise ValueError('Training requires both classes at each agreed threshold')
        manifest['training_rates'] = rates
        variances = feature_variances(train)
        manifest['constant_feature_indices'] = [i for i, v in enumerate(variances) if v == 0.]
        results = []
        best_model, best_score, best_reg = None, float('inf'), None
        for reg in args.reg_params:
            print('STAGE: fit regParam=%s' % reg, flush=True)
            fitted = classifier(variances, reg, args.max_iter).fit(train)
            if any(not math.isfinite(float(c)) for c in fitted.coefficients):
                raise ValueError('Nonfinite fitted coefficient')
            if any(c >= COEFFICIENT_UPPER_BOUND * 0.99 for c in fitted.coefficients[:-1]):
                raise ValueError('Finite compatibility bound is active; inspect training before reporting')
            metrics = [r.asDict() for r in probability_metrics(
                probabilities(fitted, validation)).collect()]
            if len(metrics) != 4:
                raise ValueError('Validation must contain all thresholds')
            score = sum(r['brier'] for r in metrics) / len(metrics)
            iterations = fitted.summary.totalIterations
            results.append({'reg_param': reg, 'macro_brier': score, 'metrics': metrics,
                            'iterations': iterations, 'iteration_limit_reached': iterations >= args.max_iter})
            manifest['validation_candidates'] = results
            if not math.isfinite(score):
                raise ValueError('Nonfinite validation score')
            if score < best_score:
                best_model, best_score, best_reg = fitted, score, reg
            save_json(root / 'manifest.json', manifest)
        if best_model.coefficients[-1] > 1e-10:
            raise ValueError('Threshold coefficient violates monotonicity')
        # No retraining on 2023: final model retains the agreed training years.
        model = PipelineModel(stages=encoder.stages + [best_model])
        model.write().save(str(root / 'model'))
        # Confirm serialization reproduces a deterministic validation example.
        example = (expand_thresholds(base.where("split = 'validation'"))
                   .orderBy('station_id', 'prediction_time', 'threshold_mm').limit(4))
        original = [r.p for r in probabilities(model, example).orderBy('threshold_mm').collect()]
        restored = PipelineModel.load(str(root / 'model'))
        reloaded = [r.p for r in probabilities(restored, example).orderBy('threshold_mm').collect()]
        if any(abs(a-b) > 1e-12 for a, b in zip(original, reloaded)) or len(original) != len(reloaded):
            raise ValueError('Saved model roundtrip differs')
        manifest.update(selected_reg_param=best_reg, validation_macro_brier=best_score,
                        coefficients=best_model.coefficients.toArray().tolist(),
                        intercept=float(best_model.intercept),
                        selected_iterations=best_model.summary.totalIterations)
        save_json(root / 'manifest.json', manifest)
    finally:
        train.unpersist()
        validation.unpersist()

    print('STAGE: final held-out 2024 evaluation; model selection is complete', flush=True)
    test = expand_thresholds(base.where("split = 'test'"))
    predictions = probabilities(model, test).persist(StorageLevel.DISK_ONLY)
    try:
        metrics, calibration = evaluate(predictions)
        rates_by_threshold = {r['threshold_mm']: r['probability'] for r in rates}
        for row in metrics:
            p = rates_by_threshold[row['threshold_mm']]
            q = row['prevalence']
            row['constant_baseline_probability'] = p
            row['constant_baseline_brier'] = q*(1-p)**2 + (1-q)*p**2
        save_json(root / 'test_metrics.json', metrics)
        save_json(root / 'test_calibration.json', calibration)
        examples = predictions.orderBy('station_id', 'prediction_time', 'threshold_mm').limit(20)
        save_json(root / 'example_predictions.json', [
            {**r.asDict(), 'prediction_time': r.prediction_time.isoformat()} for r in examples.collect()])
    finally:
        predictions.unpersist()
    manifest['status'] = 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=INPUT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reg-params', nargs='+', type=float, default=[0., 0.01, 0.1])
    parser.add_argument('--max-iter', type=int, default=80)
    args = parser.parse_args()
    if args.max_iter < 1 or any(not math.isfinite(r) or r < 0 for r in args.reg_params):
        parser.error('Invalid iteration limit or regularization')
    root = Path(output_path(args.output))
    root.mkdir(parents=True, exist_ok=False)
    spark = (SparkSession.builder.appName('Task2FinalLogisticRegression')
             .config('spark.sql.session.timeZone', 'Asia/Singapore')
             .config('spark.sql.shuffle.partitions', '32').getOrCreate())
    manifest = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'parameters': vars(args), 'thresholds_mm': THRESHOLDS,
                'python': platform.python_version(), 'spark': spark.version,
                'java': spark.sparkContext._jvm.java.lang.System.getProperty('java.version'),
                'pbs_job_id': os.environ.get('PBS_JOBID'), 'spark_conf': dict(spark.sparkContext.getConf().getAll()),
                'selection': 'minimum equal-threshold mean validation Brier score; ties keep first',
                'auc_num_bins': 1000, 'coefficient_upper_bound': COEFFICIENT_UPPER_BOUND,
                'source_sha256': {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ['task2_train.py', 'task2_prepare.py']}}
    try:
        save_json(root / 'manifest.json', manifest)
        run(spark, args, root, manifest)
        print('TASK2 FINAL RUN COMPLETE:', root, flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(root / 'manifest.json', manifest)
        spark.stop()


if __name__ == '__main__':
    main()
