"""Coverage-aware rainfall EDA for cleaned Singapore 5-minute Parquet data.

Run: spark-submit src/rainfall_eda.py
Requires PySpark 3.3.1; writes compact CSV tables under outputs/eda/.
Rainfall is a 5-minute amount (mm), NOT an hourly intensity.
"""
from pathlib import Path
import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data' / 'processed'
OUTPUT = ROOT / 'outputs' / 'eda'
YEARS = range(2017, 2025)
THRESHOLDS_MM = (0.0, 0.2, 1.0, 5.0, 10.0)
# Positive-rainfall histogram bins; final bin includes all amounts >= 20 mm.
BINS_MM = (0.0, 0.2, 0.4, 0.6, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
PERCENTILES = (0.5, 0.9, 0.95, 0.99, 0.999)
REQUIRED = {'timestamp', 'station_id', 'station_name', 'rainfall',
            'location_latitude', 'location_longitude'}


def write_csv(filename, rows, fields):
    path = OUTPUT / filename
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print('Saved %s (%d rows)' % (path, len(rows)), flush=True)


def metrics(df, group_cols):
    """Metrics use observed 5-minute slots as denominator; no imputation."""
    expressions = [
        F.count('*').alias('observed_intervals'),
        F.sum('rainfall').alias('recorded_rainfall_mm'),
        F.avg('rainfall').alias('mean_mm_per_observed_interval'),
        F.sum(F.when(F.col('rainfall') > 0, 1).otherwise(0)).alias('rainy_intervals'),
        F.avg(F.when(F.col('rainfall') > 0, F.col('rainfall'))).alias('mean_positive_rainfall_mm'),
        F.max('rainfall').alias('max_five_min_rainfall_mm'),
    ]
    for t in THRESHOLDS_MM:
        label = str(t).replace('.', 'p')
        expressions.append(F.sum(F.when(F.col('rainfall') > F.lit(t), 1).otherwise(0))
                           .alias('exceed_%s_mm_intervals' % label))
    grouped = df.groupBy(*group_cols).agg(*expressions) if group_cols else df.agg(*expressions)
    grouped = grouped.withColumn('rainy_fraction', F.col('rainy_intervals') / F.col('observed_intervals'))
    for t in THRESHOLDS_MM:
        label = str(t).replace('.', 'p')
        grouped = grouped.withColumn('exceed_%s_mm_fraction' % label,
                                     F.col('exceed_%s_mm_intervals' % label) / F.col('observed_intervals'))
    return grouped


def collect_dicts(df):
    return [r.asDict(recursive=True) for r in df.collect()]


def bin_expr():
    result = F.lit(len(BINS_MM) - 1)
    for i in reversed(range(len(BINS_MM) - 1)):
        result = F.when((F.col('rainfall') >= BINS_MM[i]) &
                        (F.col('rainfall') < BINS_MM[i + 1]), F.lit(i)).otherwise(result)
    return result


def enrich_coverage(rows):
    """Full-calendar denominator distinguishes short operating spans from gaps."""
    for row in rows:
        year = int(row['year'])
        expected = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days * 288
        observed = int(row['observed_intervals'])
        row['calendar_expected_intervals'] = expected
        row['calendar_coverage_pct'] = 100.0 * observed / expected
        # Extrapolation assumes missing slots have the same mean as observed slots.
        # Diagnostic only: not a measured annual total.
        row['calendar_extrapolated_rainfall_mm'] = (
            float(row['recorded_rainfall_mm']) * expected / observed if observed else None)
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.appName('SingaporeRainfallEDA')
             .config('spark.sql.shuffle.partitions', '32')
             .config('spark.sql.session.timeZone', 'Asia/Singapore').getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    tables = defaultdict(list)
    try:
        for year in YEARS:
            path = INPUT / ('rainfall_%d' % year)
            if not path.exists():
                raise FileNotFoundError('Missing cleaned Parquet directory: %s' % path)
            print('EDA year %d' % year, flush=True)
            df = spark.read.parquet(str(path))
            absent = REQUIRED - set(df.columns)
            if absent:
                raise ValueError('Year %d missing columns: %s' % (year, sorted(absent)))
            df = df.select('timestamp', 'station_id', 'station_name', 'rainfall',
                           'location_latitude', 'location_longitude')
            invalid = (F.col('timestamp').isNull() | F.col('station_id').isNull() |
                       F.col('rainfall').isNull() | F.isnan('rainfall') |
                       (F.col('rainfall') < 0) | (F.abs('rainfall') == float('inf')))
            if df.where(invalid).limit(1).count():
                raise ValueError('Invalid input found in %d: rerun validation' % year)
            df = (df.withColumn('year', F.lit(year))
                  .withColumn('month', F.month('timestamp'))
                  .withColumn('hour', F.hour('timestamp'))
                  .persist())
            try:
                annual = collect_dicts(metrics(df, ['year']))
                tables['annual'].extend(annual)
                tables['monthly'].extend(collect_dicts(metrics(df, ['year', 'month'])))
                tables['hourly'].extend(collect_dicts(metrics(df, ['year', 'hour'])))
                tables['station_year'].extend(collect_dicts(metrics(df, ['year', 'station_id'])))
                tables['station_month'].extend(collect_dicts(metrics(df, ['year', 'month', 'station_id'])))
                # Spatial coordinates: median of recorded values per station-year.
                # Handles the known small S113 coordinate variation without rewriting source.
                loc = df.groupBy('station_id').agg(
                    F.first('station_name', ignorenulls=True).alias('station_name'),
                    F.expr('percentile_approx(location_latitude, 0.5)').alias('latitude'),
                    F.expr('percentile_approx(location_longitude, 0.5)').alias('longitude'))
                for r in collect_dicts(loc):
                    r['year'] = year
                    tables['station_locations'].append(r)
                positive = df.where(F.col('rainfall') > 0)
                percentiles = positive.agg(
                    F.expr('percentile_approx(rainfall, array(0.5,0.9,0.95,0.99,0.999), 10000)').alias('p'),
                    F.count('*').alias('positive_intervals')).first()
                tables['positive_percentiles'].append(dict(
                    year=year, positive_intervals=percentiles['positive_intervals'],
                    **{'p%s_mm' % str(p).replace('.', 'p'): value
                       for p, value in zip(PERCENTILES, percentiles['p'] or [None] * len(PERCENTILES))}))
                hist = collect_dicts(positive.withColumn('bin_id', bin_expr())
                                     .groupBy('bin_id').count())
                counts = {int(r['bin_id']): int(r['count']) for r in hist}
                for i in range(len(BINS_MM)):
                    tables['positive_histogram'].append(dict(
                        year=year, bin_id=i, lower_mm=BINS_MM[i],
                        upper_mm=BINS_MM[i + 1] if i + 1 < len(BINS_MM) else '',
                        count=counts.get(i, 0)))
                print('Year %d: %d observations' % (year, annual[0]['observed_intervals']), flush=True)
            finally:
                df.unpersist()

        enrich_coverage(tables['station_year'])
        # Merge location metadata into station-year rows for spatial plotting.
        location_map = {(r['year'], r['station_id']): r for r in tables['station_locations']}
        for row in tables['station_year']:
            loc = location_map[(row['year'], row['station_id'])]
            row.update(station_name=loc['station_name'], latitude=loc['latitude'], longitude=loc['longitude'])
        # Derived monthly calendar coverage for station-month (includes inactive days).
        import calendar
        for row in tables['station_month']:
            slots = calendar.monthrange(int(row['year']), int(row['month']))[1] * 288
            row['calendar_expected_intervals'] = slots
            row['calendar_coverage_pct'] = 100 * row['observed_intervals'] / slots

        for key, filename in [
            ('annual', 'annual_rainfall_stats.csv'),
            ('monthly', 'monthly_rainfall_stats.csv'),
            ('hourly', 'hourly_rainfall_stats.csv'),
            ('station_year', 'station_year_rainfall_stats.csv'),
            ('station_month', 'station_month_rainfall_stats.csv'),
            ('station_locations', 'station_locations.csv'),
            ('positive_percentiles', 'positive_rainfall_percentiles.csv'),
            ('positive_histogram', 'positive_rainfall_histogram.csv')]:
            rows = tables[key]
            if rows:
                write_csv(filename, rows, list(rows[0].keys()))
        print('EDA aggregation complete. Annual sums are network totals, NOT Singapore annual rainfall.', flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
