"""Clean Singapore five-minute rainfall records, 2017-2024 (PySpark 3.3.1).

Run from a project with data/raw/ and src/data_cleaning.py. Does not edit raw CSVs.
Outputs annual Parquet datasets and auditable CSV summaries.
"""
from pathlib import Path
import csv
from collections import defaultdict
from pyspark.sql import SparkSession, Window, functions as F

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
REPORT = ROOT / 'outputs' / 'cleaning'
YEARS = range(2017, 2025)
REQUIRED = ['station_id', 'station_name', 'timestamp', 'reading_value',
            'location_latitude', 'location_longitude']


def csv_write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('Saved', path, flush=True)


def read_year(spark, year):
    path = RAW / ('HistoricalRainfallacrossSingapore%d.csv' % year)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    df = spark.read.option('header', True).option('inferSchema', False).csv(str(path))
    absent = sorted(set(REQUIRED) - set(df.columns))
    if absent:
        raise ValueError('%s lacks columns %s' % (path, absent))
    return df


def main():
    spark = (SparkSession.builder.appName('RainfallDataCleaning')
             .config('spark.sql.shuffle.partitions', '32')
             .config('spark.sql.session.timeZone', 'Asia/Singapore')
             .getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        # Build a small global station-name lookup from actual observations.
        # Prefer descriptive names over bare IDs, then the most frequent variant;
        # ties are broken lexicographically to ensure reproducibility.
        name_counts = defaultdict(int)
        for year in YEARS:
            print('Reading station-name variants:', year, flush=True)
            raw = read_year(spark, year)
            grouped = (raw.groupBy('station_id', 'station_name')
                       .agg(F.count('*').alias('n')).collect())
            for r in grouped:
                if r['station_id'] is not None and r['station_name'] is not None:
                    sid = r['station_id'].strip()
                    name = ' '.join(r['station_name'].strip().split())
                    if sid and name:
                        name_counts[(sid, name)] += r['n']

        variants = defaultdict(list)
        for (sid, name), count in name_counts.items():
            descriptive = (name.casefold() != sid.casefold())
            variants[sid].append((name, count, descriptive))
        chosen = {}
        mapping_rows = []
        for sid in sorted(variants):
            options = variants[sid]
            # Most common descriptive spelling wins; capitalization tie is stable.
            winner = sorted(options, key=lambda x: (-int(x[2]), -x[1], x[0].casefold(), x[0]))[0][0]
            chosen[sid] = winner
            for name, count, descriptive in sorted(options):
                mapping_rows.append({'station_id': sid, 'original_name': name,
                                     'standard_name': winner, 'observations': count,
                                     'name_changed': int(name != winner)})
        csv_write(REPORT / 'station_name_mapping.csv', mapping_rows,
                  ['station_id', 'original_name', 'standard_name', 'observations', 'name_changed'])
        lookup = spark.createDataFrame([(sid, name) for sid, name in chosen.items()],
                                       ['lookup_station_id', 'standard_station_name'])

        summaries = []
        for year in YEARS:
            print('Cleaning', year, flush=True)
            raw = read_year(spark, year)
            before = raw.count()
            df = (raw.withColumn('original_timestamp', F.col('timestamp'))
                  .withColumn('observation_time', F.to_timestamp('timestamp'))
                  .withColumn('rainfall', F.col('reading_value').cast('double'))
                  .withColumn('location_latitude', F.col('location_latitude').cast('double'))
                  .withColumn('location_longitude', F.col('location_longitude').cast('double')))
            invalid = (F.col('station_id').isNull() | (F.trim('station_id') == '') |
                       F.col('observation_time').isNull() | F.col('rainfall').isNull() |
                       F.isnan('rainfall') | (F.col('rainfall') < 0))
            bad = df.where(invalid).count()
            if bad:
                raise ValueError('%d invalid records in %d: review before cleaning' % (bad, year))
            df = (df.withColumn('original_epoch', F.col('observation_time').cast('long'))
                  .withColumn('aligned_epoch',
                              (F.floor((F.col('original_epoch') + F.lit(150)) / F.lit(300))
                               * F.lit(300)).cast('long'))
                  .withColumn('shift_seconds', F.col('aligned_epoch') - F.col('original_epoch')))
            # Diagnostics established that every 2017 offset is one second and all
            # other years are aligned. Fail loudly if new/unexpected offsets occur.
            unexpected = df.where(F.abs('shift_seconds') > 1).limit(1).count()
            if unexpected:
                raise ValueError('Unexpected timestamp shift >1 second in %d' % year)
            shifts = df.where(F.col('shift_seconds') != 0).count()
            df = df.withColumn('timestamp', F.from_unixtime('aligned_epoch').cast('timestamp'))
            df = df.join(F.broadcast(lookup), df.station_id == lookup.lookup_station_id, 'left')
            missing_lookup = df.where(F.col('standard_station_name').isNull()).limit(1).count()
            if missing_lookup:
                raise ValueError('Missing station name mapping in %d' % year)
            name_changes = df.where(F.col('station_name') != F.col('standard_station_name')).count()
            df = df.withColumn('station_name', F.col('standard_station_name'))

            # Detect *all* conflicts before discarding collision records.
            collisions = (df.groupBy('station_id', 'aligned_epoch')
                          .agg(F.count('*').alias('n'),
                               F.countDistinct('rainfall').alias('distinct_rainfall'))
                          .where(F.col('n') > 1))
            stats = collisions.agg(F.count('*').alias('groups'),
                                   F.coalesce(F.sum(F.col('n') - 1), F.lit(0)).alias('extras'),
                                   F.coalesce(F.sum(F.when(F.col('distinct_rainfall') > 1, 1)
                                                    .otherwise(0)), F.lit(0)).alias('conflicts')).first()
            if stats['conflicts']:
                raise ValueError('%d conflicting rainfall collision groups in %d' % (stats['conflicts'], year))
            # Exact-grid original wins; remaining ties are ordered by original
            # timestamp and original metadata, never by Spark partition order.
            tie_columns = [F.col('shift_seconds').asc(), F.col('original_timestamp').asc()]
            for col in sorted(raw.columns):
                if col not in ('timestamp', 'station_name', 'location_latitude', 'location_longitude'):
                    tie_columns.append(F.col(col).asc_nulls_last())
            tie_columns += [F.col('location_latitude').asc_nulls_last(),
                            F.col('location_longitude').asc_nulls_last()]
            window = Window.partitionBy('station_id', 'aligned_epoch').orderBy(*tie_columns)
            cleaned = (df.withColumn('_rank', F.row_number().over(window))
                       .where(F.col('_rank') == 1)
                       .drop('_rank', 'lookup_station_id', 'standard_station_name',
                             'original_epoch', 'aligned_epoch', 'shift_seconds'))
            after = cleaned.count()
            if before - after != stats['extras']:
                raise RuntimeError('Row-count validation failed for %d' % year)
            # Keep original timestamp as audit trail; preserve all other source
            # columns, plus typed rainfall and observation_time.
            output = PROCESSED / ('rainfall_%d' % year)
            cleaned.write.mode('overwrite').parquet(str(output))
            summaries.append({'year': year, 'input_rows': before,
                              'shifted_timestamps': shifts,
                              'alignment_collision_groups': stats['groups'],
                              'collision_rows_removed': stats['extras'],
                              'station_name_rows_standardized': name_changes,
                              'output_rows': after, 'output_path': str(output)})
            csv_write(REPORT / 'cleaning_summary.csv', summaries,
                      ['year', 'input_rows', 'shifted_timestamps',
                       'alignment_collision_groups', 'collision_rows_removed',
                       'station_name_rows_standardized', 'output_rows', 'output_path'])
            print('Finished %d: %d -> %d rows' % (year, before, after), flush=True)
        print('Cleaning completed successfully.', flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
