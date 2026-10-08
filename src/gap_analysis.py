"""Station-level rainfall reporting gap diagnostics, 2017-2024 (Spark 3.3.1)."""
from pathlib import Path
import csv
from pyspark.sql import SparkSession, functions as F, Window

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data' / 'raw'
OUTPUT = ROOT / 'outputs' / 'gap_analysis'
OUTPUT.mkdir(parents=True, exist_ok=True)
YEARS = range(2017, 2025)

spark = (SparkSession.builder.appName('RainfallGapAnalysis')
         .config('spark.sql.shuffle.partitions', '32').getOrCreate())
spark.sparkContext.setLogLevel('WARN')


def write_rows(name, rows, fields):
    path = OUTPUT / name
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {path}', flush=True)


def main():
    annual = []
    station_rows = []
    month_rows = []
    longest_rows = []
    short_rows = []
    for year in YEARS:
        print(f'Analyzing {year}...', flush=True)
        source = INPUT / f'HistoricalRainfallacrossSingapore{year}.csv'
        raw = spark.read.option('header', True).option('inferSchema', False).csv(str(source))
        valid = (raw.select('station_id', F.to_timestamp('timestamp').alias('ts'))
                 .where(F.col('station_id').isNotNull() & F.col('ts').isNotNull())
                 .dropDuplicates(['station_id', 'ts']))
        # Work within each annual file; do not bridge gaps between years.
        w = Window.partitionBy('station_id').orderBy('ts')
        intervals = (valid.withColumn('prev_ts', F.lag('ts').over(w))
                     .withColumn('gap_seconds', F.col('ts').cast('long') - F.col('prev_ts').cast('long'))
                     .where(F.col('prev_ts').isNotNull()))
        # Approximate missing 5-minute slots for long gaps. If timestamps are
        # off-grid, count complete 300-second intervals strictly between endpoints.
        intervals = intervals.withColumn(
            'estimated_missing_slots',
            F.when(F.col('gap_seconds') > 300,
                   F.greatest(F.lit(0), F.ceil(F.col('gap_seconds') / F.lit(300)).cast('long') - 1))
             .otherwise(F.lit(0)))
        intervals = intervals.withColumn(
            'gap_class',
            F.when(F.col('gap_seconds') < 300, 'under_5min')
             .when(F.col('gap_seconds') == 300, 'exact_5min')
             .when(F.col('gap_seconds') <= 600, '5_to_10min')
             .when(F.col('gap_seconds') <= 1800, '10_to_30min')
             .when(F.col('gap_seconds') <= 3600, '30_to_60min')
             .when(F.col('gap_seconds') <= 86400, '1_to_24h')
             .when(F.col('gap_seconds') <= 604800, '1_to_7d')
             .otherwise('over_7d'))

        counts = intervals.groupBy('gap_class').agg(
            F.count('*').alias('intervals'),
            F.sum('estimated_missing_slots').alias('estimated_missing_slots')).collect()
        buckets = {r['gap_class']: r for r in counts}
        for label in ['under_5min', 'exact_5min', '5_to_10min', '10_to_30min',
                      '30_to_60min', '1_to_24h', '1_to_7d', 'over_7d']:
            row = buckets.get(label)
            annual.append(dict(year=year, gap_class=label,
                               intervals=row['intervals'] if row else 0,
                               estimated_missing_slots=row['estimated_missing_slots'] if row else 0))

        per_station = valid.groupBy('station_id').agg(
            F.count('*').alias('observed_unique'),
            F.min('ts').alias('first_ts'), F.max('ts').alias('last_ts'))
        gap_station = intervals.groupBy('station_id').agg(
            F.sum(F.when(F.col('gap_seconds') > 300, 1).otherwise(0)).alias('gaps_over_5min'),
            F.sum('estimated_missing_slots').alias('estimated_missing_slots'),
            F.max('gap_seconds').alias('max_gap_seconds'))
        station_stats = (per_station.join(gap_station, 'station_id', 'left')
                         .withColumn('expected_slots_span',
                             F.floor((F.col('last_ts').cast('long') - F.col('first_ts').cast('long')) / 300) + 1)
                         .withColumn('coverage_span_approx',
                             F.col('observed_unique') / F.col('expected_slots_span')))
        for r in station_stats.collect():
            d = r.asDict()
            d['year'] = year
            # A ratio above 1 indicates off-grid readings; this is not a strict completeness percentage.
            station_rows.append(d)

        # Attribute each gap to the month when the *later* observation appears.
        monthly = (intervals.where(F.col('gap_seconds') > 300)
                   .withColumn('month', F.month('ts'))
                   .groupBy('month').agg(F.count('*').alias('gaps_over_5min'),
                       F.sum('estimated_missing_slots').alias('estimated_missing_slots')))
        for r in monthly.collect():
            month_rows.append(dict(year=year, **r.asDict()))

        top = (intervals.where(F.col('gap_seconds') > 300)
               .orderBy(F.desc('gap_seconds')).limit(20)
               .select('station_id', 'prev_ts', 'ts', 'gap_seconds', 'estimated_missing_slots'))
        for r in top.collect():
            longest_rows.append(dict(year=year, **r.asDict()))
        if year == 2017:
            for r in (intervals.where(F.col('gap_seconds') < 300)
                      .orderBy('gap_seconds').limit(150)
                      .select('station_id', 'prev_ts', 'ts', 'gap_seconds').collect()):
                short_rows.append(dict(year=year, **r.asDict()))
        print(f'Finished {year}', flush=True)

    write_rows('annual_gap_distribution.csv', annual,
               ['year', 'gap_class', 'intervals', 'estimated_missing_slots'])
    write_rows('station_coverage.csv', station_rows,
               ['year', 'station_id', 'observed_unique', 'first_ts', 'last_ts',
                'gaps_over_5min', 'estimated_missing_slots', 'max_gap_seconds',
                'expected_slots_span', 'coverage_span_approx'])
    write_rows('monthly_gap_counts.csv', month_rows,
               ['year', 'month', 'gaps_over_5min', 'estimated_missing_slots'])
    write_rows('longest_gaps.csv', longest_rows,
               ['year', 'station_id', 'prev_ts', 'ts', 'gap_seconds', 'estimated_missing_slots'])
    write_rows('short_intervals_2017.csv', short_rows,
               ['year', 'station_id', 'prev_ts', 'ts', 'gap_seconds'])
    spark.stop()


if __name__ == '__main__':
    main()
