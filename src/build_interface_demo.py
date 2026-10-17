"""Rebuild the small interface demo from real 2024 source observations.

Run from the repository root with the interface Python environment. This writes
only data/demo/ and reuses the existing cleaner and committed station-name map.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from pyspark.sql import SparkSession, functions as F
from data_cleaning import prepare_records, deduplicate_records

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ('S24', 'S33', 'S50')
START, END = '2024-01-01', '2024-01-03'  # End exclusive, Singapore time.


def main():
    source = ROOT/'data/raw/HistoricalRainfallacrossSingapore2024.csv'
    rows = []
    digest = hashlib.sha256()
    with source.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        for row in reader:
            if row['station_id'].strip() in STATIONS and START <= row['timestamp'][:10] < END:
                rows.append(row)
                digest.update((json.dumps(row, sort_keys=True) + '\n').encode())
    if not rows:
        raise ValueError('No observations found for the demonstration coverage.')
    names = {}
    with (ROOT/'outputs/cleaning/station_name_mapping.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            names[row['station_id']] = row['standard_name']
    os.environ['PYSPARK_PYTHON'] = sys.executable
    spark = (SparkSession.builder.master('local[2]').appName('BuildRainfallInterfaceDemo')
             .config('spark.sql.session.timeZone', 'Asia/Singapore')
             .config('spark.sql.shuffle.partitions', '2').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    try:
        raw = spark.createDataFrame(rows)
        cleaned, stats = deduplicate_records(prepare_records(raw, 2024), columns)
        lookup = spark.createDataFrame([(sid, names[sid]) for sid in STATIONS],
                                       ['lookup_station_id', 'standard_station_name'])
        cleaned = (cleaned.join(F.broadcast(lookup), cleaned.station_id == lookup.lookup_station_id)
                   .withColumn('station_name', F.col('standard_station_name'))
                   .drop('lookup_station_id', 'standard_station_name'))
        coverage = cleaned.groupBy('station_id', 'station_name').agg(
            F.count('*').alias('rows'), F.min('timestamp').cast('string').alias('first_sgt'),
            F.max('timestamp').cast('string').alias('last_sgt'),
            F.sum(F.when(F.col('rainfall') > 0, 1).otherwise(0)).alias('wet_readings')
        ).orderBy('station_id').collect()
        assert {r.station_id for r in coverage} == set(STATIONS)
        # Two complete days per station; never fill missing observations.
        assert all(r.rows == 576 for r in coverage), coverage
        output = ROOT/'data/demo'
        cleaned.orderBy('station_id', 'timestamp').coalesce(1).write.mode('overwrite').parquet(str(output/'rainfall_2024'))
        manifest = {
            'purpose': 'Interface demonstration only; not a training or evaluation dataset.',
            'source': 'NEA Historical Rainfall across Singapore (2024)',
            'source_url': 'https://data.gov.sg/collections/2279/view',
            'licence_url': 'https://data.gov.sg/open-data-licence',
            'source_file': source.name,
            'selected_source_rows_sha256': digest.hexdigest(),
            'selection': {'station_ids': list(STATIONS), 'start_sgt': START, 'end_exclusive_sgt': END},
            'cleaning': 'data_cleaning.prepare_records + deduplicate_records; existing global station-name mapping',
            'source_rows': len(rows), 'duplicate_rows_removed': stats['extras'],
            'rows': sum(r.rows for r in coverage), 'coverage': [r.asDict() for r in coverage]
        }
        manifest['example'] = {
            'station_id': 'S50', 'prediction_time': '2024-01-01T13:50:00+08:00',
            'threshold_mm': 5.,
            'selection_note': 'Deliberately selected wet-weather illustration; not representative of model performance.'
        }
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest, indent=2))
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
