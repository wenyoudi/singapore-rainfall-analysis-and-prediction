"""Build shared Task 2 examples from cleaned five-minute observations."""
import argparse
import csv
import json
from pathlib import Path
from pyspark.sql import SparkSession, Window, functions as F
from pyspark import StorageLevel


def build_examples(observations, thresholds):
    """Requires unique station/timestamps on a five-minute grid (validated by CLI)."""
    df = observations.select('station_id', 'timestamp', 'rainfall')
    df = df.withColumn('_epoch', F.col('timestamp').cast('long'))
    order = Window.partitionBy('station_id').orderBy('_epoch')
    history = order.rangeBetween(-3300, 0)  # t-55 through t: 12 slots
    future = order.rangeBetween(300, 1800)  # t+5 through t+30: 6 slots
    df = (df.withColumn('history_count', F.count('*').over(history))
          .withColumn('future_count', F.count('*').over(future))
          .withColumn('past_5m_mm', F.col('rainfall'))
          .withColumn('past_30m_mm', F.sum('rainfall').over(order.rangeBetween(-1500, 0)))
          .withColumn('past_60m_mm', F.sum('rainfall').over(history))
          .withColumn('future_30m_mm', F.sum('rainfall').over(future))
          .withColumn('_future_end', F.max('timestamp').over(future))
          .withColumn('year', F.year('timestamp'))
          .withColumn('hour', F.hour('timestamp'))
          .withColumn('month', F.month('timestamp')))
    # Uniqueness and grid validation make these counts proof of continuity.
    df = (df.withColumn('history_complete', F.col('history_count') == 12)
          .withColumn('future_complete', F.col('future_count') == 6)
          .withColumn('split_safe',
                      F.when(F.col('year') <= 2022, F.year('_future_end') <= 2022)
                      .otherwise(F.year('_future_end') == F.col('year'))))
    valid = df.where(F.col('history_complete') & F.col('future_complete') & F.col('split_safe'))
    valid = valid.withColumn('split', F.when(F.col('year') <= 2022, 'train')
                             .when(F.col('year') == 2023, 'validation').otherwise('test'))
    examples = (valid.withColumn('threshold_mm', F.explode(F.array(*[F.lit(t) for t in thresholds])))
                .withColumn('label', (F.col('future_30m_mm') > F.col('threshold_mm')).cast('double'))
                .select('station_id', 'timestamp', 'year', 'split', 'hour', 'month',
                        'past_5m_mm', 'past_30m_mm', 'past_60m_mm', 'threshold_mm',
                        'label', 'future_30m_mm'))
    return df, examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0., 1., 5., 10.])
    parser.add_argument('--shuffle-partitions', type=int, default=128)
    args = parser.parse_args()
    import math
    if (not args.thresholds or any(not math.isfinite(t) or t < 0 for t in args.thresholds)
            or len(set(args.thresholds)) != len(args.thresholds) or args.shuffle_partitions < 1):
        parser.error('Use distinct finite nonnegative thresholds and positive shuffle partitions')
    source, destination = Path(args.input_root).resolve(), Path(args.output_root).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error('Input and output directories must not overlap')
    if destination.exists():
        parser.error('Output already exists; use a new output directory')
    paths = [source / ('rainfall_%d' % y) for y in range(2017, 2025)]
    for path in paths:
        if not path.is_dir():
            parser.error('Missing input: %s' % path)
    spark = (SparkSession.builder.appName('RainfallTask2Dataset')
             .config('spark.sql.session.timeZone', 'Asia/Singapore')
             .config('spark.sql.shuffle.partitions', str(args.shuffle_partitions)).getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    try:
        frames = []
        for year, path in zip(range(2017, 2025), paths):
            frame = spark.read.parquet(str(path)).select('station_id', 'timestamp', 'rainfall')
            invalid = (F.col('station_id').isNull() | (F.trim('station_id') == '') |
                       F.col('timestamp').isNull() | (F.year('timestamp') != year) |
                       (F.col('timestamp').cast('long') % 300 != 0) |
                       (F.col('timestamp').cast('double') != F.col('timestamp').cast('long')) |
                       F.col('rainfall').isNull() | F.isnan('rainfall') |
                       (F.col('rainfall') < 0) | (F.abs('rainfall') == float('inf')))
            if frame.where(invalid).limit(1).count():
                raise ValueError('Invalid cleaned input in %d' % year)
            frames.append(frame)
        observations = frames[0]
        for frame in frames[1:]:
            observations = observations.unionByName(frame)
        if observations.groupBy('station_id', 'timestamp').count().where('count > 1').limit(1).count():
            raise ValueError('Duplicate station/timestamp input')
        audit, examples = build_examples(observations, args.thresholds)
        audit = audit.persist(StorageLevel.DISK_ONLY)
        quality = audit.groupBy('year').agg(
            F.count('*').alias('observations'),
            F.sum((~F.col('history_complete')).cast('long')).alias('incomplete_history'),
            F.sum((~F.col('future_complete')).cast('long')).alias('incomplete_future'),
            F.sum((F.col('history_complete') & F.col('future_complete') &
                   ~F.col('split_safe')).cast('long')).alias('split_boundary_excluded'),
            F.sum((F.col('history_complete') & F.col('future_complete') &
                   F.col('split_safe')).cast('long')).alias('eligible_times')).orderBy('year').collect()
        examples = examples.persist(StorageLevel.DISK_ONLY)
        stats = examples.groupBy('split', 'threshold_mm').agg(
            F.count('*').alias('samples'), F.sum('label').alias('positive_samples'),
            F.avg('label').alias('positive_rate')).orderBy('split', 'threshold_mm').collect()
        if {r['split'] for r in stats} != {'train', 'validation', 'test'}:
            raise ValueError('One or more splits are empty')
        distribution = examples.where(F.col('threshold_mm') == args.thresholds[0]).groupBy('split').agg(
            F.expr('percentile_approx(future_30m_mm, array(0.5,0.9,0.95,0.99,0.999),10000)').alias('percentiles_mm'),
            F.max('future_30m_mm').alias('max_mm')).collect()
        destination.mkdir(parents=True)
        manifest = {'status': 'running', 'input_root': str(source), 'thresholds_mm': args.thresholds,
                    'timezone': 'Asia/Singapore', 'spark_version': spark.version,
                    'features': ['station_id', 'hour', 'month', 'past_5m_mm', 'past_30m_mm',
                                 'past_60m_mm', 'threshold_mm'],
                    'forbidden_features': ['label', 'future_30m_mm', 'split'],
                    'history': '12 continuous slots t-55 through t; t observation must be available',
                    'target': 'sum of 6 continuous slots t+5 through t+30 > threshold',
                    'split': {'train': '2017-2022', 'validation': '2023', 'test': '2024'},
                    'missingness': 'exclude incomplete history or future; no zero imputation'}
        manifest_path = destination / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        examples.write.mode('errorifexists').partitionBy('split', 'year').parquet(str(destination / 'examples'))
        for name, rows in [('quality_by_year.csv', quality), ('class_balance.csv', stats)]:
            with (destination / name).open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].asDict().keys())
                writer.writeheader()
                writer.writerows(r.asDict() for r in rows)
        manifest['future_distribution'] = [r.asDict() for r in distribution]
        manifest['status'] = 'complete'
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print('Task 2 dataset complete:', destination, flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
