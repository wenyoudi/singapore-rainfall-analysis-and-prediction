from pathlib import Path
import csv
from pyspark.sql import SparkSession, Window, functions as F

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / 'data' / 'processed'
CLEANING = ROOT / 'outputs' / 'cleaning' / 'cleaning_summary.csv'
REPORT = ROOT / 'outputs' / 'validation'
YEARS = range(2017, 2025)
REQUIRED = {'station_id', 'station_name', 'timestamp', 'rainfall',
            'location_latitude', 'location_longitude', 'original_timestamp'}


def save(name, rows, fields):
    path = REPORT / name
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', path, flush=True)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.appName('RainfallFinalValidation')
             .config('spark.sql.shuffle.partitions', '32')
             .config('spark.sql.session.timeZone', 'Asia/Singapore')
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    expected = {}
    if CLEANING.exists():
        with CLEANING.open(newline='', encoding='utf-8') as fh:
            expected = {int(r['year']): int(r['output_rows']) for r in csv.DictReader(fh)}

    annual, coverage, gaps, metadata, failures = [], [], [], [], []
    try:
        for year in YEARS:
            print('Validating:', year, flush=True)
            path = PROCESSED / ('rainfall_%d' % year)
            if not path.exists():
                raise FileNotFoundError(str(path))
            df = spark.read.parquet(str(path))
            missing = REQUIRED - set(df.columns)
            if missing:
                raise ValueError('%d missing columns: %s' % (year, sorted(missing)))
            # The cleaning script retains both timestamp (aligned) and
            # observation_time (original parsed time). Validate timestamp.
            df = df.withColumn('_epoch', F.col('timestamp').cast('long'))
            invalid_rain = (F.col('rainfall').isNull() | F.isnan('rainfall') |
                            (F.abs('rainfall') == float('inf')) | (F.col('rainfall') < 0))
            invalid_station = F.col('station_id').isNull() | (F.trim('station_id') == '')
            invalid_name = F.col('station_name').isNull() | (F.trim('station_name') == '')
            invalid_coord = (F.col('location_latitude').isNull() |
                             F.col('location_longitude').isNull() |
                             F.isnan('location_latitude') | F.isnan('location_longitude') |
                             (F.abs('location_latitude') == float('inf')) |
                             (F.abs('location_longitude') == float('inf')) |
                             ~F.col('location_latitude').between(-90, 90) |
                             ~F.col('location_longitude').between(-180, 180))
            aggregate = df.agg(
                F.count('*').alias('rows'),
                F.countDistinct('station_id').alias('stations'),
                F.sum(F.when(invalid_station, 1).otherwise(0)).alias('invalid_station_ids'),
                F.sum(F.when(invalid_name, 1).otherwise(0)).alias('invalid_station_names'),
                F.sum(F.when(F.col('timestamp').isNull(), 1).otherwise(0)).alias('null_timestamps'),
                F.sum(F.when(F.col('_epoch').isNotNull() &
                            ((F.col('_epoch') % F.lit(300)) != 0), 1).otherwise(0)).alias('off_grid'),
                F.sum(F.when(invalid_rain, 1).otherwise(0)).alias('invalid_rainfall'),
                F.sum(F.when(invalid_coord, 1).otherwise(0)).alias('invalid_coordinates'),
                F.min('timestamp').alias('first_timestamp'),
                F.max('timestamp').alias('last_timestamp')).first().asDict()
            # Each station-year must have unique timestamps.
            duplicates = (df.groupBy('station_id', 'timestamp').count()
                          .where(F.col('count') > 1)
                          .agg(F.coalesce(F.sum(F.col('count') - 1), F.lit(0)).alias('extra'))
                          .first()['extra'])
            aggregate.update(year=year, duplicate_extra_rows=duplicates,
                             expected_output_rows=expected.get(year, ''),
                             row_count_matches_summary=(aggregate['rows'] == expected[year])
                             if year in expected else 'not_checked')
            annual.append(aggregate)
            for key in ('invalid_station_ids', 'invalid_station_names', 'null_timestamps',
                        'off_grid', 'invalid_rainfall', 'invalid_coordinates', 'duplicate_extra_rows'):
                if aggregate[key]:
                    failures.append('%d: %s = %s' % (year, key, aggregate[key]))
            if year in expected and aggregate['rows'] != expected[year]:
                failures.append('%d: output row count differs from cleaning summary' % year)

            # Coverage is calculated BETWEEN a station's first and last
            # observation in each year. This does NOT treat time before a
            # station starts or after it stops as a missing measurement.
            station_stats = (df.groupBy('station_id').agg(
                F.first('station_name', ignorenulls=True).alias('station_name'),
                F.count('*').alias('observed'),
                F.min('_epoch').alias('first_epoch'),
                F.max('_epoch').alias('last_epoch')))
            for r in station_stats.collect():
                if r['first_epoch'] is None or r['last_epoch'] is None:
                    continue
                expected_slots = (r['last_epoch'] - r['first_epoch']) // 300 + 1
                coverage.append(dict(year=year, station_id=r['station_id'],
                                     station_name=r['station_name'],
                                     observed_intervals=r['observed'],
                                     expected_between_first_last=expected_slots,
                                     missing_intervals=max(0, expected_slots - r['observed']),
                                     coverage_pct=round(100 * r['observed'] / expected_slots, 4),
                                     first_timestamp=str(__import__('datetime').datetime.fromtimestamp(
                                         r['first_epoch'], __import__('datetime').timezone(
                                             __import__('datetime').timedelta(hours=8)))),
                                     last_timestamp=str(__import__('datetime').datetime.fromtimestamp(
                                         r['last_epoch'], __import__('datetime').timezone(
                                             __import__('datetime').timedelta(hours=8))))))

            # Consecutive-observation gaps; no dense five-minute grid generated.
            w = Window.partitionBy('station_id').orderBy('_epoch')
            deltas = (df.select('station_id', '_epoch')
                      .withColumn('_prev', F.lag('_epoch').over(w))
                      .withColumn('gap_seconds', F.col('_epoch') - F.col('_prev'))
                      .where(F.col('gap_seconds') > 300)
                      .withColumn('missing_intervals', F.col('gap_seconds') / 300 - 1))
            gap_stats = (deltas.groupBy('station_id').agg(
                F.count('*').alias('gap_events'),
                F.sum('missing_intervals').cast('long').alias('missing_intervals'),
                F.max('gap_seconds').alias('longest_gap_seconds'),
                F.sum(F.when(F.col('missing_intervals') == 1, 1).otherwise(0)).alias('one_missing'),
                F.sum(F.when(F.col('missing_intervals').between(2, 5), 1).otherwise(0)).alias('two_to_five_missing'),
                F.sum(F.when(F.col('missing_intervals') >= 6, 1).otherwise(0)).alias('six_plus_missing')))
            gaps.extend([dict(year=year, **r.asDict()) for r in gap_stats.collect()])

            # Report remaining metadata variants, rather than automatically
            # overwriting historical coordinates or station names.
            variants = (df.groupBy('station_id').agg(
                F.countDistinct('station_name').alias('distinct_names'),
                F.countDistinct(F.struct('location_latitude', 'location_longitude')).alias('coordinate_pairs'))
                .where((F.col('distinct_names') > 1) | (F.col('coordinate_pairs') > 1)))
            metadata.extend([dict(year=year, **r.asDict()) for r in variants.collect()])
            if variants.where(F.col('distinct_names') > 1).limit(1).count():
                failures.append('%d: station ID has multiple standardized names' % year)
            print('%d: %d rows, %d stations, %d duplicate extras' %
                  (year, aggregate['rows'], aggregate['stations'], duplicates), flush=True)

        save('final_validation_summary.csv', annual,
             ['year', 'rows', 'stations', 'first_timestamp', 'last_timestamp',
              'invalid_station_ids', 'invalid_station_names', 'null_timestamps',
              'off_grid', 'invalid_rainfall', 'invalid_coordinates',
              'duplicate_extra_rows', 'expected_output_rows', 'row_count_matches_summary'])
        save('station_coverage_summary.csv', coverage,
             ['year', 'station_id', 'station_name', 'observed_intervals',
              'expected_between_first_last', 'missing_intervals', 'coverage_pct',
              'first_timestamp', 'last_timestamp'])
        save('gap_distribution_summary.csv', gaps,
             ['year', 'station_id', 'gap_events', 'missing_intervals',
              'longest_gap_seconds', 'one_missing', 'two_to_five_missing',
              'six_plus_missing'])
        save('station_metadata_exceptions.csv', metadata,
             ['year', 'station_id', 'distinct_names', 'coordinate_pairs'])
        save('validation_issues.csv', [{'issue': x} for x in failures], ['issue'])
        print('VALIDATION:', 'PASS structural checks' if not failures else 'REVIEW REQUIRED', flush=True)
        for issue in failures:
            print(' -', issue, flush=True)
        print('Coverage gaps and coordinate variants are reported, not automatically treated as failures.')
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
