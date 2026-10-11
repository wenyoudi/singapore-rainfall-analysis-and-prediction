"""Exercise CSV -> cleaning -> validation -> EDA using a temporary tiny dataset."""
import csv
import sys
import pytest

pytest.importorskip('pyspark')
from pyspark.sql import SparkSession
import data_cleaning as cleaning
import cleaning_validation as validation
import rainfall_eda as eda


def test_small_pipeline_and_failed_validation(tmp_path, monkeypatch):
    monkeypatch.setenv('PYSPARK_PYTHON', sys.executable)
    raw, processed = tmp_path / 'raw', tmp_path / 'processed'
    raw.mkdir()
    fields = cleaning.REQUIRED
    source = raw / 'HistoricalRainfallacrossSingapore2017.csv'
    with source.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for time, rain in [('00:00:00', '.2'), ('00:00:01', '.2'), ('00:10:00', '.4')]:
            writer.writerow(dict(station_id='S1', station_name='Station One',
                                 timestamp='2017-01-01T%s+08:00' % time, reading_value=rain,
                                 location_latitude='1.3', location_longitude='103.8'))
    monkeypatch.setattr(cleaning, 'RAW', raw)
    monkeypatch.setattr(cleaning, 'PROCESSED', processed)
    monkeypatch.setattr(cleaning, 'REPORT', tmp_path / 'cleaning')
    monkeypatch.setattr(validation, 'PROCESSED', processed)
    monkeypatch.setattr(validation, 'CLEANING', tmp_path / 'cleaning/cleaning_summary.csv')
    monkeypatch.setattr(validation, 'REPORT', tmp_path / 'validation')
    monkeypatch.setattr(eda, 'INPUT', processed)
    monkeypatch.setattr(eda, 'OUTPUT', tmp_path / 'eda')
    for module in (cleaning, validation, eda):
        monkeypatch.setattr(module, 'YEARS', [2017])

    def start():
        return (SparkSession.builder.master('local[2]').appName('Task1PipelineTest')
                .config('spark.ui.enabled', 'false')
                .config('spark.sql.session.timeZone', 'Asia/Singapore').getOrCreate())

    for stage in (cleaning.main, validation.main, eda.main):
        start().sparkContext.setLogLevel('ERROR')
        stage()  # Each stage stops its Spark session, as in a submitted job.
    with (tmp_path / 'cleaning/cleaning_summary.csv').open() as handle:
        summary = next(csv.DictReader(handle))
    assert summary['input_rows'] == '3' and summary['output_rows'] == '2'
    with (tmp_path / 'eda/annual_rainfall_stats.csv').open() as handle:
        summary = next(csv.DictReader(handle))
    assert summary['observed_intervals'] == '2'
    assert float(summary['recorded_rainfall_mm']) == pytest.approx(.6)
    with (tmp_path / 'validation/station_coverage_summary.csv').open() as handle:
        summary = next(csv.DictReader(handle))
    assert summary['missing_intervals'] == '1'

    # Corrupt a separate copy; validation must fail rather than merely log issues.
    spark = start()
    df = spark.read.parquet(str(processed / 'rainfall_2017'))
    corrupt = tmp_path / 'corrupt'
    df.unionByName(df).write.parquet(str(corrupt / 'rainfall_2017'))
    spark.stop()
    monkeypatch.setattr(validation, 'PROCESSED', corrupt)
    start().sparkContext.setLogLevel('ERROR')
    with pytest.raises(ValueError, match='validation failed'):
        validation.main()
    with (tmp_path / 'validation/validation_issues.csv').open() as handle:
        assert any('duplicate_extra_rows' in r['issue'] for r in csv.DictReader(handle))
