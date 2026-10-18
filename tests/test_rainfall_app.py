"""Exercise Streamlit using a real saved model, not mocked probabilities."""
from datetime import timedelta
from pathlib import Path
import pytest
pytest.importorskip("streamlit")
from pyspark.sql import functions as F
from streamlit.testing.v1 import AppTest
from test_rainfall_interface import spark, WHEN
from rainfall_inference import RainfallPredictor

ROOT = Path(__file__).resolve().parents[1]


def test_unconnected_data_has_helpful_message(tmp_path, monkeypatch):
    monkeypatch.setenv('RAINFALL_DATA_ROOT', str(tmp_path))
    at = AppTest.from_file(str(ROOT/'streamlit_app.py'), default_timeout=90).run()
    assert not at.exception
    assert any('not connected' in x.value for x in at.warning)
    assert not at.metric


def test_ui_probability_history_and_validation(spark, tmp_path, monkeypatch):
    rows = [('S50', WHEN+timedelta(minutes=5*i), float(i+12)/10) for i in range(-12, 7)]
    df = spark.createDataFrame(rows, 'station_id string, timestamp timestamp, rainfall double')
    df.write.parquet(str(tmp_path/'rainfall_2024'))
    monkeypatch.setenv('RAINFALL_DATA_ROOT', str(tmp_path))
    expected = RainfallPredictor(tmp_path).predict('S50', WHEN, 5.)['probability']
    at = AppTest.from_file(str(ROOT/'streamlit_app.py'), default_timeout=90).run()
    assert not at.exception
    assert any('coordinates are unavailable' in x.value for x in at.info)
    at.time_input[0].set_value(WHEN.time().replace(tzinfo=None)).run()
    at.selectbox[1].select(5.).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == f'{expected:.1%}'
    assert any('S50' in x.value and 'more than 5 mm' in x.value for x in at.markdown)
    at.button[1].click().run()
    assert not at.exception and len(at.metric) == 2
    assert at.selectbox[1].options == ['0 mm', '1 mm', '5 mm', '10 mm']
    at.selectbox[1].select(10.).run()
    at.button[0].click().run()
    assert not at.exception
    expected_ten = RainfallPredictor(tmp_path).predict('S50', WHEN, 10.)['probability']
    assert at.metric[0].value == f'{expected_ten:.1%}'
    assert any('more than 10 mm' in x.value for x in at.markdown)
    assert not at.error


def test_bundled_demo_default_real_model_and_coverage(spark, monkeypatch):
    """A fresh grader launch needs neither raw data nor a Vanda environment."""
    import json
    from datetime import datetime
    from rainfall_inference import SG
    from task2_prepare import prepare_features
    from pyspark.ml.functions import vector_to_array

    monkeypatch.delenv('RAINFALL_DATA_ROOT', raising=False)
    predictor = RainfallPredictor(ROOT/'data/demo')
    manifest = json.loads((ROOT/'data/demo/manifest.json').read_text())
    assert predictor.observations.count() == manifest['rows'] == 4608
    assert [s['id'] for s in predictor.stations()] == ['S24', 'S33', 'S50', 'S104', 'S109', 'S115', 'S116', 'S117']
    assert predictor.observations.select('station_id', 'prediction_time').distinct().count() == 4608
    when = datetime(2024, 1, 1, 1, tzinfo=SG)
    for station in predictor.stations():
        result = predictor.predict(station['id'], when, 0.)
        assert 0 <= result['probability'] <= 1
        assert len(result['history']) == 12
        assert predictor.verify(station['id'], when, 0.) is not None
    for threshold in [0., 1., 5., 10.]:
        reference = prepare_features(predictor.observations, threshold).where(
            (F.col('station_id') == 'S24') & (F.col('prediction_time') == F.lit(when)))
        expected = predictor.model.transform(reference).select(vector_to_array('probability')[1]).first()[0]
        actual = predictor.predict('S24', when, threshold)['probability']
        assert actual == pytest.approx(expected, abs=1e-12)
        if threshold == 0.:
            default_probability = expected
    assert predictor.verify('S50', when.replace(hour=13, minute=50), 5.)['total_mm'] == pytest.approx(.4)
    wet_time = when.replace(hour=14, minute=10)
    assert predictor.verify('S24', wet_time, 5.)['total_mm'] == pytest.approx(19.6)
    assert predictor.predict('S24', wet_time, 5.)['probability'] == pytest.approx(.5488267034837925)
    at = AppTest.from_file(str(ROOT/'streamlit_app.py'), default_timeout=90).run()
    assert not at.exception
    assert any('Demonstration data' in x.value and 'not the dataset used to evaluate' in x.value for x in at.info)
    assert len(at.selectbox[0].options) == 8
    assert at.selectbox[0].value == 'S24'
    assert at.selectbox[1].value == 0.
    assert at.time_input[0].value.hour == 0 and at.time_input[0].value.minute == 0
    import pyarrow as pa
    map_element = next(c for c in at.get('vega_lite_chart')
                       if json.loads(c.proto.spec).get('description', '').startswith('Singapore map'))
    map_rows = [row for dataset in map_element.proto.datasets
                for row in pa.ipc.open_stream(dataset.data.data).read_all().to_pylist()]
    points = {row['id']: row for row in map_rows if 'id' in row}
    assert (points['S24']['latitude'], points['S24']['longitude']) == (1.3678, 103.9826)
    # The mainland exterior must face the right way for Vega, rather than fill the world.
    feature = json.loads(map_element.proto.spec)['layer'][0]['data']['values'][0]
    polygons = feature['geometry']['coordinates']
    largest = max((polygon[0] for polygon in polygons), key=len)
    assert sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(largest, largest[1:])) < 0
    assert at.date_input[0].min.isoformat() == '2024-01-01'
    assert at.date_input[0].max.isoformat() == '2024-01-02'
    at.button[0].click().run()
    assert not at.exception
    assert any('Insufficient history' in x.value for x in at.error)
    assert not at.metric
    at.time_input[0].set_value(when.time().replace(tzinfo=None)).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == f'{default_probability:.1%}'
    chart = next(json.loads(c.proto.spec) for c in at.get('vega_lite_chart')
                 if json.loads(c.proto.spec).get('encoding', {}).get('y', {}).get('field') == 'rainfall')
    assert chart['encoding']['y']['scale']['domain'][0] == 0
    assert chart['encoding']['y']['scale']['domain'][1] > 0
    at.selectbox[0].select('S33').run()
    assert not at.metric  # Changing station clears the old station's estimate.
    at.date_input[0].set_value(datetime(2024, 1, 2).date()).run()
    at.button[0].click().run()
    assert not at.exception
    expected = predictor.predict('S33', datetime(2024, 1, 2, 1, tzinfo=SG), 0.)['probability']
    assert at.metric[0].value == f'{expected:.1%}'
    assert any('S33' in x.value for x in at.markdown)
    # A dry real-data window must still have a non-degenerate, non-negative axis.
    at.selectbox[0].select('S24').run()
    at.date_input[0].set_value(when.date()).run()
    at.time_input[0].set_value(when.replace(hour=1, minute=0).time()).run()
    at.button[0].click().run()
    assert not at.exception
    chart = next(json.loads(c.proto.spec) for c in at.get('vega_lite_chart')
                 if json.loads(c.proto.spec).get('encoding', {}).get('y', {}).get('field') == 'rainfall')
    assert chart['encoding']['y']['scale']['domain'] == [0, 1.]
    at.date_input[0].set_value(when.date()).run()
    at.time_input[0].set_value(when.replace(hour=0).time()).run()
    at.button[0].click().run()
    assert not at.exception
    assert any('Insufficient history' in x.value for x in at.error)
    assert not at.metric
