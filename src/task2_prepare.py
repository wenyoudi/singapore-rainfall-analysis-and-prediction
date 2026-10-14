"""Prepare leakage-safe rainfall features and threshold labels; no model fitting."""
import argparse
from functools import reduce
import math
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F

INPUT = '/scratch/e0969984/singapore-rainfall-analysis-and-prediction/data/processed'
OUTPUT = '/scratch/e1496655/task2/prepared'


def output_path(value):
    """Restrict all writes to a new child directory of the user's scratch root."""
    root = Path('/scratch/e1496655').resolve()
    path = Path(value).resolve()
    if root not in path.parents:
        raise ValueError('Output must be a child directory of /scratch/e1496655/')
    return str(path)


def normalize_input(raw, time_column='auto'):
    column = ('timestamp' if 'timestamp' in raw.columns else 'observation_time') if time_column == 'auto' else time_column
    required = {'station_id', 'rainfall', column}
    if not required.issubset(raw.columns):
        raise ValueError('Missing columns: %s' % sorted(required - set(raw.columns)))
    print('Using prediction time column:', column, flush=True)
    return raw.select(F.col('station_id').cast('string'),
                      F.col(column).cast('timestamp').alias('prediction_time'),
                      F.col('rainfall').cast('double'))


def validate_input(df):
    """Fail on invalid values or duplicates; missing slots remain missing."""
    epoch = F.col('prediction_time').cast('long')
    invalid = (F.col('station_id').isNull() | (F.trim('station_id') == '') |
               F.col('prediction_time').isNull() | F.col('rainfall').isNull() |
               F.isnan('rainfall') | (F.abs('rainfall') == float('inf')) |
               (F.col('rainfall') < 0) | (epoch % 300 != 0) |
               (F.col('prediction_time').cast('double') != epoch) |
               ~F.year('prediction_time').between(2017, 2024))
    if df.where(invalid).limit(1).count():
        raise ValueError('Invalid station, timestamp or rainfall')
    if (df.groupBy('station_id', 'prediction_time').count()
            .where(F.col('count') > 1).limit(1).count()):
        raise ValueError('Duplicate station/timestamp observations')


def prepare_features(df, threshold_mm):
    """Require all 12 past and six future slots; caller must validate input first."""
    if not math.isfinite(threshold_mm) or threshold_mm < 0:
        raise ValueError('Threshold must be finite and nonnegative')
    base = df.withColumn('_epoch', F.col('prediction_time').cast('long'))
    window = Window.partitionBy('station_id').orderBy('_epoch')
    expressions = []
    checks = []
    for prefix, direction, size, shift in [('past', -1, 12, F.lag), ('future', 1, 6, F.lead)]:
        for i in range(1, size + 1):
            expressions.extend([shift('rainfall', i).over(window).alias('_%s_r%d' % (prefix, i)),
                                shift('_epoch', i).over(window).alias('_%s_t%d' % (prefix, i))])
            checks.append(F.col('_%s_t%d' % (prefix, i)) == F.col('_epoch') + direction * i * 300)
    features = base.select('*', *expressions).where(reduce(lambda a, b: a & b, checks))
    def total(prefix, size):
        return reduce(lambda a, b: a + b, [F.col('_%s_r%d' % (prefix, i)) for i in range(1, size + 1)])
    features = features.select(
        'station_id', 'prediction_time',
        F.col('_past_r1').alias('rainfall_previous_5m'),
        total('past', 6).alias('rainfall_previous_30m'),
        total('past', 12).alias('rainfall_previous_60m'),
        F.hour('prediction_time').alias('hour'), F.month('prediction_time').alias('month'),
        total('future', 6).alias('future_rainfall_30m'),
        F.col('_future_t6').cast('timestamp').alias('target_end_time'))
    features = (features.withColumn('year', F.year('prediction_time'))
                .withColumn('split', F.when(F.col('year') <= 2022, 'train')
                            .when(F.col('year') == 2023, 'validation').otherwise('test')))
    # Labels must not consume observations in a later evaluation split.
    features = features.where(
        ((F.col('split') == 'train') & (F.year('target_end_time') <= 2022)) |
        ((F.col('split') == 'validation') & (F.year('target_end_time') == 2023)) |
        ((F.col('split') == 'test') & (F.year('target_end_time') == 2024)))
    return (features.withColumn('threshold_mm', F.lit(float(threshold_mm)))
            .withColumn('label', (F.col('future_rainfall_30m') > threshold_mm).cast('double')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', default=INPUT)
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--threshold-mm', type=float, required=True)
    parser.add_argument('--time-column', choices=['auto', 'timestamp', 'observation_time'], default='auto')
    parser.add_argument('--validation-station')
    parser.add_argument('--validation-year', type=int, choices=range(2017, 2025))
    parser.add_argument('--shuffle-partitions', type=int, default=32)
    args = parser.parse_args()
    if bool(args.validation_station) != (args.validation_year is not None):
        parser.error('Supply both --validation-station and --validation-year')
    if not math.isfinite(args.threshold_mm) or args.threshold_mm < 0 or args.shuffle_partitions < 1:
        parser.error('Threshold must be finite/nonnegative and shuffle partitions positive')
    destination = output_path(args.output)
    spark = (SparkSession.builder.appName('Task2RainfallPreparation')
             .config('spark.sql.session.timeZone', 'Asia/Singapore')
             .config('spark.sql.shuffle.partitions', str(args.shuffle_partitions)).getOrCreate())
    try:
        years = [args.validation_year] if args.validation_year is not None else range(2017, 2025)
        raw = spark.read.parquet(*['%s/rainfall_%d' % (args.input_root.rstrip('/'), y) for y in years])
        if args.validation_station:
            raw = raw.where(F.col('station_id') == args.validation_station)
        df = normalize_input(raw, args.time_column)
        if args.validation_year is not None:
            df = df.where(F.year('prediction_time') == args.validation_year)
        validate_input(df)
        prepared = prepare_features(df, args.threshold_mm)
        if not prepared.take(1):
            raise ValueError('No complete samples remain; check station, year and coverage')
        # No overwrite option: existing destinations cause a failure.
        prepared.write.mode('errorifexists').partitionBy('split', 'year').parquet(destination)
        print('Saved Task 2 dataset:', destination, flush=True)
        spark.read.parquet(destination).groupBy('split', 'year').count().orderBy('year').show(20, False)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
