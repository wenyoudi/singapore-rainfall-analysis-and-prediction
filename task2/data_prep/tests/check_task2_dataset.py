"""Small Spark checks; run directly without pytest, including on Vanda."""
import sys
from pathlib import Path
from datetime import datetime, timedelta
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pyspark.sql import SparkSession
from build_task2_dataset import build_examples

spark = (SparkSession.builder.master('local[2]').appName('Task2Checks')
         .config('spark.sql.session.timeZone', 'Asia/Singapore')
         .config('spark.sql.shuffle.partitions', '2').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
try:
    start = datetime(2023, 6, 1)
    rows = [('A', start + timedelta(minutes=5*i), 1.) for i in range(25)]
    rows += [('B', start + timedelta(minutes=5*i), 100.) for i in range(25)]
    df = spark.createDataFrame(rows, ['station_id', 'timestamp', 'rainfall'])
    _, examples = build_examples(df, [0., 1., 5., 6., 10.])
    result = examples.collect()
    assert len(result) == 2 * 8 * 5
    a = [r for r in result if r.station_id == 'A']
    assert all(r.past_5m_mm == 1 and r.past_30m_mm == 6 and r.past_60m_mm == 12
               and r.future_30m_mm == 6 and r.split == 'validation' for r in a)
    assert all(r.label == float(r.threshold_mm < 6) for r in a)
    missing = spark.createDataFrame([r for r in rows if r[0] == 'A' and r[1] != start + timedelta(minutes=75)], df.schema)
    _, ex = build_examples(missing, [0.])
    assert ex.count() == 0, 'A gap must invalidate windows instead of shifting row offsets'
    boundary_start = datetime(2022, 12, 31, 22)
    boundary = spark.createDataFrame([('A', boundary_start + timedelta(minutes=5*i), 1.) for i in range(50)], df.schema)
    _, ex = build_examples(boundary, [0.])
    result = ex.collect()
    assert all(r.timestamp <= datetime(2022,12,31,23,25) for r in result if r.split == 'train')
    assert any(r.timestamp == datetime(2023,1,1) and r.past_60m_mm == 12 for r in result)
    print('PASS: exact windows, station isolation, strict thresholds, gaps, split boundaries, historical carryover')
finally:
    spark.stop()
