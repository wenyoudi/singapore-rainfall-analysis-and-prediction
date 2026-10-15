"""Train or validate Random Forest on the immutable shared Task 2 dataset."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from functools import reduce

from pyspark import StorageLevel
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession, Window, functions as F

from task2_model_common import (THRESHOLDS, RAINFALL_FEATURES, expand_thresholds,
                               preprocessing, probabilities, probability_metrics,
                               evaluate, save_json)

PREPARED = '/scratch/e1496655/task2/final-1458499.stdct-mgmt-02/prepared'
EXPECTED_COUNTS = {2017: 4803912, 2018: 3973669, 2019: 4931059, 2020: 6229257,
                   2021: 6973560, 2022: 7069791, 2023: 6683962, 2024: 6335645}
PREPARATION_HASH = 'f5936d95318cf30f20a0263867b80ea4fe59cc93d33436373d9faab80bbfa2be'


def validate_prepared(df, expected_counts=None):
    """Validate base rows without changing the cohort or reconstructing windows."""
    required = {'station_id', 'prediction_time', 'target_end_time', 'hour', 'month',
                'year', 'split', 'future_rainfall_30m', *RAINFALL_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError('Missing prepared columns: %s' % sorted(missing))
    epoch = F.col('prediction_time').cast('long')
    expected_split = (F.when(F.col('year') <= 2022, 'train')
                      .when(F.col('year') == 2023, 'validation').otherwise('test'))
    invalid = [F.col(c).isNull() for c in required]
    invalid += [(F.trim('station_id') == ''), ~F.col('year').between(2017, 2024),
                F.col('split') != expected_split,
                F.year('prediction_time') != F.col('year'),
                F.hour('prediction_time') != F.col('hour'),
                F.month('prediction_time') != F.col('month'), epoch % 300 != 0,
                F.col('prediction_time').cast('double') != epoch,
                F.col('target_end_time').cast('double') != epoch + 1800,
                ((F.col('split') == 'train') & (F.year('target_end_time') > 2022)),
                ((F.col('split') == 'validation') & (F.year('target_end_time') != 2023)),
                ((F.col('split') == 'test') & (F.year('target_end_time') != 2024))]
    for col in RAINFALL_FEATURES + ['future_rainfall_30m']:
        invalid += [F.isnan(col), F.abs(F.col(col)) == float('inf'), F.col(col) < 0]
    # Nonnegative rainfall implies nested historical sums, allowing float error.
    invalid += [F.col(RAINFALL_FEATURES[0]) > F.col(RAINFALL_FEATURES[1]) + 1e-8,
                F.col(RAINFALL_FEATURES[1]) > F.col(RAINFALL_FEATURES[2]) + 1e-8]
    if df.where(reduce(lambda a, b: a | b, invalid)).limit(1).count():
        raise ValueError('Invalid prepared values, horizon, calendar or split')
    if (df.groupBy('station_id', 'prediction_time').count()
            .where(F.col('count') > 1).limit(1).count()):
        raise ValueError('Duplicate prepared station/time keys')
    counts = [r.asDict() for r in df.groupBy('split', 'year').count().orderBy('year').collect()]
    if not counts:
        raise ValueError('Prepared dataset is empty')
    if expected_counts is not None and {r['year']: r['count'] for r in counts} != expected_counts:
        raise ValueError('Prepared counts differ from the agreed shared dataset')
    return counts


def validate_provenance(prepared):
    """Read sibling metadata; never write into or alter the shared run."""
    path = Path(prepared).resolve().parent / 'manifest.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    counts = {r['year']: r['count'] for r in record.get('retained_counts', [])}
    if (record.get('status') != 'complete' or counts != EXPECTED_COUNTS or
            record.get('source_sha256', {}).get('task2_prepare.py') != PREPARATION_HASH or
            record.get('thresholds_mm') != THRESHOLDS):
        raise ValueError('Shared manifest differs from the reviewed completed preparation')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'pbs_job_id': record.get('pbs_job_id'), 'preparation_sha256': PREPARATION_HASH}


def forest(trees, depth, seed):
    return RandomForestClassifier(numTrees=trees, maxDepth=depth, seed=seed,
                                  featureSubsetStrategy='sqrt', maxBins=32,
                                  minInstancesPerNode=20, subsamplingRate=1.0,
                                  bootstrap=True, impurity='gini')


def monotonicity(predictions):
    """Report, rather than tune away, violations across the four thresholds."""
    window = Window.partitionBy('station_id', 'prediction_time').orderBy('threshold_mm')
    compared = predictions.withColumn('_previous_p', F.lag('p').over(window))
    row = compared.agg(
        F.sum(F.when(F.col('_previous_p').isNotNull(), 1).otherwise(0)).alias('adjacent_pairs'),
        F.sum(F.when(F.col('p') > F.col('_previous_p') + 1e-12, 1).otherwise(0))
        .alias('violating_pairs')).first().asDict()
    row['violation_fraction'] = row['violating_pairs'] / row['adjacent_pairs'] if row['adjacent_pairs'] else None
    return row


def run(spark, args, root, manifest):
    print('STAGE: validate shared prepared dataset (read only)', flush=True)
    base = spark.read.parquet(args.prepared)
    manifest['retained_counts'] = validate_prepared(base, EXPECTED_COUNTS)
    save_json(root / 'manifest.json', manifest)
    if args.check_only:
        base.printSchema()
        manifest['status'] = 'validated'
        return

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
            raise ValueError('Both training classes required at all thresholds')
        manifest['training_rates'] = rates
        results = []
        best, best_depth, best_score = None, None, float('inf')
        for depth in args.depths:
            print('STAGE: fit RF depth=%s trees=%s' % (depth, args.trees), flush=True)
            fitted = forest(args.trees, depth, args.seed).fit(train)
            metrics = [r.asDict() for r in probability_metrics(probabilities(fitted, validation)).collect()]
            score = sum(r['brier'] for r in metrics) / 4
            if len(metrics) != 4 or not math.isfinite(score):
                raise ValueError('Invalid validation metrics')
            results.append({'max_depth': depth, 'num_trees': args.trees,
                            'macro_brier': score, 'metrics': metrics})
            if score < best_score:
                best, best_depth, best_score = fitted, depth, score
            manifest['validation_candidates'] = results
            save_json(root / 'manifest.json', manifest)
        model = PipelineModel(stages=encoder.stages + [best])
        model.write().save(str(root / 'model'))
        example = (expand_thresholds(base.where("split = 'validation'"))
                   .orderBy('station_id', 'prediction_time', 'threshold_mm').limit(4))
        original = [r.p for r in probabilities(model, example).orderBy('threshold_mm').collect()]
        restored = PipelineModel.load(str(root / 'model'))
        reloaded = [r.p for r in probabilities(restored, example).orderBy('threshold_mm').collect()]
        if len(original) != len(reloaded) or any(abs(a-b) > 1e-12 for a, b in zip(original, reloaded)):
            raise ValueError('Saved RF model prediction roundtrip differs')
        manifest.update(selected_depth=best_depth, num_trees=args.trees,
                        validation_macro_brier=best_score,
                        feature_importances=best.featureImportances.toArray().tolist())
        save_json(root / 'manifest.json', manifest)
    finally:
        train.unpersist()
        validation.unpersist()

    print('STAGE: held-out 2024 evaluation (selection frozen)', flush=True)
    predictions = probabilities(model, expand_thresholds(base.where("split = 'test'")))
    predictions.persist(StorageLevel.DISK_ONLY)
    try:
        metrics, calibration = evaluate(predictions)
        baseline = {r['threshold_mm']: r['probability'] for r in rates}
        for row in metrics:
            p, q = baseline[row['threshold_mm']], row['prevalence']
            row['constant_baseline_probability'] = p
            row['constant_baseline_brier'] = q*(1-p)**2 + (1-q)*p**2
        save_json(root / 'test_metrics.json', metrics)
        save_json(root / 'test_calibration.json', calibration)
        manifest['test_monotonicity'] = monotonicity(predictions)
        examples = predictions.orderBy('station_id', 'prediction_time', 'threshold_mm').limit(20)
        save_json(root / 'example_predictions.json', [
            {**r.asDict(), 'prediction_time': r.prediction_time.isoformat()} for r in examples.collect()])
    finally:
        predictions.unpersist()
    manifest['status'] = 'complete'


def destination(output, prepared):
    path, source = Path(output).resolve(), Path(prepared).resolve()
    allowed = Path('/scratch/e0969984').resolve()
    if allowed not in path.parents:
        raise ValueError('Output must be a new child of /scratch/e0969984/')
    if path == source or source in path.parents or path in source.parents:
        raise ValueError('Output must not overlap the shared dataset')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', default=PREPARED)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--trees', type=int, default=50)
    parser.add_argument('--depths', type=int, nargs='+', default=[5, 8])
    parser.add_argument('--seed', type=int, default=5208)
    args = parser.parse_args()
    if args.trees < 1 or any(not 1 <= d <= 30 for d in args.depths):
        parser.error('Trees must be positive; depths must be between 1 and 30')
    provenance = validate_provenance(args.prepared)
    root = destination(args.output, args.prepared)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'model': 'RandomForestClassifier',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'parameters': vars(args), 'thresholds_mm': THRESHOLDS,
                'shared_provenance': provenance, 'python': platform.python_version(),
                'pbs_job_id': os.environ.get('PBS_JOBID'), 'auc_num_bins': 1000,
                'selection': 'minimum equal-threshold mean validation Brier; ties keep first',
                'source_sha256': {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                  for n in ['task2_random_forest.py', 'task2_model_common.py']}}
    spark = None
    try:
        save_json(root / 'manifest.json', manifest)
        spark = (SparkSession.builder.appName('Task2RandomForest')
                 .config('spark.sql.session.timeZone', 'Asia/Singapore')
                 .config('spark.sql.shuffle.partitions', '32').getOrCreate())
        manifest.update(spark=spark.version,
                        java=spark.sparkContext._jvm.java.lang.System.getProperty('java.version'),
                        spark_conf=dict(spark.sparkContext.getConf().getAll()))
        run(spark, args, root, manifest)
        print('RANDOM FOREST %s: %s' % ('VALIDATED' if args.check_only else 'COMPLETE', root), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(root / 'manifest.json', manifest)
        if spark is not None:
            spark.stop()


if __name__ == '__main__':
    main()
