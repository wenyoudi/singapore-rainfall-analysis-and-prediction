"""Read-only historical inference using the project's saved Spark pipelines.

Historical features exactly match task2_prepare.prepare_features. Unlike that
training builder, prediction does not require future observations or labels.
"""
import csv
from datetime import datetime, timedelta
import math
import os
import sys
from pathlib import Path
from threading import RLock
from zoneinfo import ZoneInfo

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F
from task2_prepare import normalize_input

ROOT = Path(__file__).resolve().parents[1]
SG = ZoneInfo('Asia/Singapore')
THRESHOLDS = (0., 1., 5., 10.)
MODEL_PATHS = {
    'GBT': ROOT/'task2/Model_GBT/outputs/tuning_stage1/GBT-05/model',
    'RF': ROOT/'outputs/task2/random_forest/1459817/model',
    'LR': ROOT/'outputs/task2/logistic_regression/1458499/model',
}


class InputError(ValueError):
    """A user-correctable input, coverage or data-quality issue."""


def prediction_time(value):
    if value.tzinfo is None:
        value = value.replace(tzinfo=SG)
    value = value.astimezone(SG)
    if not 2017 <= value.year <= 2024:
        raise InputError('Choose a date from 2017 through 2024.')
    if value.second or value.microsecond or value.minute % 5:
        raise InputError('Choose a time on the five-minute grid, such as 14:00 or 14:05.')
    return value


def features_from_history(observations, station, when, threshold):
    """Construct one feature row from epoch-keyed observations at/before t only."""
    when = prediction_time(when)
    if not math.isfinite(threshold) or threshold not in THRESHOLDS:
        raise InputError('Supported rainfall thresholds are 0, 1, 5 and 10 mm.')
    end = int(when.timestamp())
    required = [end - i*300 for i in range(12, 0, -1)]
    if end not in observations:
        raise InputError('No observation exists at the selected station and time. Choose another time.')
    if any(t not in observations for t in required):
        missing = sum(t not in observations for t in required)
        raise InputError(f'Insufficient history: {missing} of the 12 required five-minute readings are missing. Choose another time.')
    if any(not math.isfinite(observations[t]) or observations[t] < 0 for t in [*required, end]):
        raise InputError('The selected window contains an invalid rainfall reading. Choose another time.')
    # Sum in the same order as the existing lag-based training builder.
    previous = [observations[end-i*300] for i in range(1, 13)]
    return {'station_id': station, 'hour': when.hour, 'month': when.month,
            'rainfall_previous_5m': previous[0],
            'rainfall_previous_30m': sum(previous[:6]),
            'rainfall_previous_60m': sum(previous), 'threshold_mm': float(threshold)}


class RainfallPredictor:
    """One cached Spark session/model, with serialized access across UI sessions."""
    def __init__(self, data_root, model_name='GBT', master='local[2]'):
        if model_name not in MODEL_PATHS:
            raise InputError('Configured model must be GBT, RF or LR.')
        self.root = Path(data_root).expanduser().resolve()
        paths = [str(self.root/f'rainfall_{y}') for y in range(2017, 2025)
                 if (self.root/f'rainfall_{y}').is_dir()]
        if not paths:
            raise InputError('Cleaned historical data is not connected. Point RAINFALL_DATA_ROOT to the directory containing rainfall_2017 … rainfall_2024.')
        self.lock = RLock()
        self.name = model_name
        os.environ['PYSPARK_PYTHON'] = sys.executable
        self.spark = (SparkSession.builder.master(master).appName('RainfallPredictionInterface')
                      .config('spark.sql.session.timeZone', 'Asia/Singapore')
                      .config('spark.sql.shuffle.partitions', '8')
                      .config('spark.driver.memory', os.environ.get('RAINFALL_DRIVER_MEMORY', '4g'))
                      .config('spark.ui.enabled', 'false').getOrCreate())
        self.spark.sparkContext.setLogLevel('ERROR')
        self.model = PipelineModel.load(str(MODEL_PATHS[model_name]))
        raw = self.spark.read.parquet(*paths)
        self.observations = normalize_input(raw).withColumn('_epoch', F.col('prediction_time').cast('long'))
        # Retain subsecond information to reject malformed input rather than rounding.
        self.observations = self.observations.withColumn('_precise', F.col('prediction_time').cast('double'))

    def stations(self):
        """Small catalog of stations actually present in the connected Parquet."""
        with self.lock:
            rows = (self.observations.where(F.year('prediction_time').between(2017, 2024))
                    .groupBy('station_id').agg(F.min('_epoch').alias('first'),
                                               F.max('_epoch').alias('last')).collect())
        names = {}
        with (ROOT/'outputs/cleaning/station_name_mapping.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                names[r['station_id']] = r['standard_name']
        return sorted([{'id': r.station_id, 'name': names.get(r.station_id, r.station_id),
                        'first': datetime.fromtimestamp(r.first, SG),
                        'last': datetime.fromtimestamp(r.last, SG)}
                       for r in rows if r.station_id],
                      key=lambda r: (0, int(r['id'][1:]), r['id'])
                      if r['id'].startswith('S') and r['id'][1:].isdigit()
                      else (1, 0, r['id']))

    def window(self, station, start, end):
        with self.lock:
            # Collect only the requested station's bounded window (at most 13/6 rows).
            rows = (self.observations.where((F.col('station_id') == station)
                    & F.col('_epoch').between(int(start.timestamp()), int(end.timestamp())))
                    .select('_epoch', '_precise', 'rainfall').limit(100).collect())
        result = {}
        for r in rows:
            if (r._epoch in result or r._epoch % 300 or r._precise != r._epoch
                    or r.rainfall is None or not math.isfinite(r.rainfall) or r.rainfall < 0):
                raise InputError('The selected window contains duplicate, misaligned or invalid observations.')
            result[r._epoch] = r.rainfall
        return result

    def predict(self, station, when, threshold):
        when = prediction_time(when)
        history = self.window(station, when-timedelta(minutes=60), when)
        features = features_from_history(history, station, when, threshold)
        schema = ('station_id string, hour int, month int, rainfall_previous_5m double, '
                  'rainfall_previous_30m double, rainfall_previous_60m double, threshold_mm double')
        with self.lock:
            df = self.spark.createDataFrame([tuple(features.values())], schema)
            probability = self.model.transform(df).select(vector_to_array('probability')[1].alias('p')).first().p
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError('The saved model returned an invalid probability.')
        return {'probability': probability, 'features': features,
                'history': [{'minutes': -i*5, 'time': (when-timedelta(minutes=i*5)).strftime('%Y-%m-%d %H:%M'),
                             'rainfall': history[int(when.timestamp())-i*300]} for i in range(12, 0, -1)]}

    def verify(self, station, when, threshold):
        """Separate optional lookup; never called by predict or used as a feature."""
        when = prediction_time(when)
        future = self.window(station, when+timedelta(minutes=5), when+timedelta(minutes=30))
        expected = [int(when.timestamp())+i*300 for i in range(1, 7)]
        if any(t not in future for t in expected):
            return None
        total = sum(future[t] for t in expected)
        return {'total_mm': total, 'exceeded': total > threshold}
