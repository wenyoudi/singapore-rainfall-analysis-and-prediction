"""Train a threshold-conditioned Spark GBT using the shared prepared rainfall dataset."""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

FEATURES = [
    'station_index', 'hour', 'month', 'rainfall_previous_5m',
    'rainfall_previous_30m', 'rainfall_previous_60m', 'threshold_mm'
]
THRESHOLDS = [0.0, 1.0, 5.0, 10.0]
RAINFALL_COLUMNS = [
    'rainfall_previous_5m', 'rainfall_previous_30m',
    'rainfall_previous_60m', 'future_rainfall_30m'
]
REQUIRED_COLUMNS = [
    'station_id', 'prediction_time', 'hour', 'month', *RAINFALL_COLUMNS
]


def select_fraction(frame, fraction, seed):
    """Deterministically sample station/times, keeping all thresholds together."""
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Sampling fraction must be in (0, 1]')
    if fraction == 1:
        return frame
    bucket = F.pmod(
        F.xxhash64('station_id', 'prediction_time', F.lit(seed)),
        F.lit(1_000_000)
    )
    return frame.where(bucket < int(fraction * 1_000_000))


def evaluate(predictions, split):
    scored = predictions.withColumn(
        'p', vector_to_array('probability')[1]
    ).persist(StorageLevel.DISK_ONLY)
    rows = []
    try:
        for threshold in THRESHOLDS:
            part = scored.where(F.col('threshold_mm') == threshold)
            summary = part.agg(
                F.count('*').alias('samples'),
                F.sum('label').alias('positives'),
                F.avg(F.pow(F.col('p') - F.col('label'), 2)).alias('brier')
            ).first()
            if not summary.samples:
                raise ValueError(f'Empty evaluation threshold: {threshold}')
            result = dict(
                summary.asDict(), split=split, threshold_mm=threshold,
                positive_rate=summary.positives / summary.samples
            )
            for metric in ('areaUnderROC', 'areaUnderPR'):
                result[metric] = (
                    BinaryClassificationEvaluator(
                        labelCol='label', rawPredictionCol='p',
                        metricName=metric, numBins=1000
                    ).evaluate(part)
                    if 0 < summary.positives < summary.samples else None
                )
            rows.append(result)
            print(result, flush=True)

        reliability = (
            scored.withColumn('bin', F.least(F.floor(F.col('p') * 10), F.lit(9)))
            .groupBy('threshold_mm', 'bin')
            .agg(
                F.count('*').alias('samples'),
                F.avg('p').alias('mean_probability'),
                F.avg('label').alias('observed_rate')
            )
            .orderBy('threshold_mm', 'bin')
            .collect()
        )
        return rows, [dict(r.asDict(), split=split) for r in reliability]
    finally:
        scored.unpersist()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def read_dataset_split(spark, dataset_root, name):
    """Read shared prepared observations or data_prep threshold examples."""
    if name not in ('train', 'validation', 'test'):
        raise ValueError(f'Unknown split: {name}')
    dataset = Path(dataset_root).resolve()
    examples = (dataset / 'examples').is_dir()
    base = dataset / 'examples' if examples else dataset
    frame = spark.read.option('basePath', str(base)).parquet(str(base / f'split={name}'))
    if examples:
        aliases = {'timestamp': 'prediction_time', 'past_5m_mm': 'rainfall_previous_5m',
                   'past_30m_mm': 'rainfall_previous_30m', 'past_60m_mm': 'rainfall_previous_60m'}
        for old, new in aliases.items():
            if old in frame.columns:
                frame = frame.withColumnRenamed(old, new)
        required = [c for c in REQUIRED_COLUMNS if c != 'future_rainfall_30m'] + ['threshold_mm', 'label']
    else:
        required = REQUIRED_COLUMNS
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns in {name}: {sorted(missing)}')
    frame = frame.dropna(subset=required)
    for column in required:
        if column in RAINFALL_COLUMNS or column in ('label', 'threshold_mm'):
            frame = frame.where((~F.isnan(F.col(column))) & (F.abs(F.col(column)) != float('inf')))
    if examples:
        if frame.where((~F.col('label').isin(0., 1.)) | (~F.col('threshold_mm').isin(THRESHOLDS))).limit(1).count():
            raise ValueError('Invalid labels or thresholds')
    else:
        frame = frame.withColumn('threshold_mm', F.explode(F.array(*[F.lit(t) for t in THRESHOLDS])))
        frame = frame.withColumn('label', (F.col('future_rainfall_30m') > F.col('threshold_mm')).cast('double'))
    return frame.select('station_id', 'prediction_time', 'hour', 'month',
                        'rainfall_previous_5m', 'rainfall_previous_30m',
                        'rainfall_previous_60m', 'threshold_mm', 'label')


def run_training(spark, dataset_root, output_root, train_fraction=.01,
                 validation_fraction=.1, max_iter=20, max_depth=4,
                 step_size=.1, seed=42, evaluate_test=False):
    """Train on prepared data; reserve test for explicit final evaluation."""
    dataset = Path(dataset_root).resolve()
    output = Path(output_root).resolve()
    if dataset == output or dataset in output.parents or output in dataset.parents:
        raise ValueError('Input and output must not overlap')
    if not dataset.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {dataset}')
    if evaluate_test and (train_fraction != 1 or validation_fraction != 1):
        raise ValueError('Final test evaluation requires full train and validation fractions')
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError('max_iter must be a positive integer')
    if not isinstance(max_depth, int) or not 1 <= max_depth <= 30:
        raise ValueError('max_depth must be an integer in [1, 30]')
    if not math.isfinite(step_size) or not 0 < step_size <= 1:
        raise ValueError('step_size must be in (0, 1]')
    for fraction in (train_fraction, validation_fraction):
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError('Sampling fractions must be in (0, 1]')

    def read_split(name):
        return read_dataset_split(spark, dataset, name)

    full_train = read_split('train')
    train = select_fraction(full_train, train_fraction, seed).persist(StorageLevel.DISK_ONLY)
    validation = select_fraction(
        read_split('validation'), validation_fraction, seed
    ).persist(StorageLevel.DISK_ONLY)

    output.mkdir(parents=True, exist_ok=False)
    record = {
        'status': 'running', 'dataset_root': str(dataset),
        'dataset_format': 'shared_prepared_parquet', 'thresholds_mm': THRESHOLDS,
        'spark_version': spark.version,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'train_fraction': train_fraction, 'validation_fraction': validation_fraction,
        'seed': seed, 'max_iter': max_iter, 'max_depth': max_depth,
        'step_size': step_size, 'features': FEATURES,
        'evaluate_test': evaluate_test, 'class_weighting': 'none',
        'sampling': 'station/prediction_time hash, natural class distribution',
        'auc_num_bins': 1000, 'calibration': 'not fitted; reliability bins exported'
    }

    def save_record():
        (output / 'run_manifest.json').write_text(
            json.dumps(record, indent=2), encoding='utf-8'
        )

    save_record()
    try:
        balances = train.groupBy('threshold_mm').agg(
            F.count('*').alias('samples'), F.sum('label').alias('positives')
        ).collect()
        if {r.threshold_mm for r in balances} != set(THRESHOLDS) or any(
            not 0 < r.positives < r.samples for r in balances
        ):
            raise ValueError('Training sample needs both classes at every threshold; increase fraction')
        record['training_balance'] = [r.asDict() for r in balances]
        print('Training counts:', record['training_balance'], flush=True)

        indexer = StringIndexer(
            inputCol='station_id', outputCol='station_index', handleInvalid='keep'
        ).fit(full_train.select('station_id').distinct())
        max_bins = max(128, len(indexer.labels) + 1)
        record['max_bins'] = max_bins
        assembler = VectorAssembler(
            inputCols=FEATURES, outputCol='features', handleInvalid='error'
        )
        gbt = GBTClassifier(
            labelCol='label', featuresCol='features', maxIter=max_iter,
            maxDepth=max_depth, stepSize=step_size, maxBins=max_bins, seed=seed
        )
        model = Pipeline(stages=[indexer, assembler, gbt]).fit(train)
        model.write().save(str(output / 'model'))

        metrics, reliability = evaluate(model.transform(validation), 'validation')
        write_csv(output / 'metrics.csv', metrics)
        write_csv(output / 'reliability_bins.csv', reliability)
        if evaluate_test:
            test_metrics, test_reliability = evaluate(
                model.transform(read_split('test')), 'test'
            )
            write_csv(output / 'metrics.csv', metrics + test_metrics)
            write_csv(output / 'reliability_bins.csv', reliability + test_reliability)
        record['feature_importance'] = dict(zip(
            FEATURES, model.stages[-1].featureImportances.toArray().tolist()
        ))
        record['status'] = 'complete'
        save_record()
        return model, metrics
    except Exception as exc:
        record.update(status='failed', error=str(exc))
        save_record()
        raise
    finally:
        train.unpersist()
        validation.unpersist()
