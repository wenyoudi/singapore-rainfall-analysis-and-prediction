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
    assert predictor.observations.count() == manifest['rows'] == 1728
    assert [s['id'] for s in predictor.stations()] == ['S24', 'S33', 'S50']
    assert predictor.observations.select('station_id', 'prediction_time').distinct().count() == 1728
    when = datetime(2024, 1, 1, 1, tzinfo=SG)
    for threshold in [0., 1., 5., 10.]:
        reference = prepare_features(predictor.observations, threshold).where(
            (F.col('station_id') == 'S24') & (F.col('prediction_time') == F.lit(when)))
        expected = predictor.model.transform(reference).select(vector_to_array('probability')[1]).first()[0]
        actual = predictor.predict('S24', when, threshold)['probability']
        assert actual == pytest.approx(expected, abs=1e-12)
        if threshold == 5.:
            default_probability = expected
    at = AppTest.from_file(str(ROOT/'streamlit_app.py'), default_timeout=90).run()
    assert not at.exception
    assert any('Demonstration data' in x.value and 'not the dataset used to evaluate' in x.value for x in at.info)
    assert len(at.selectbox[0].options) == 3
    assert at.date_input[0].min.isoformat() == '2024-01-01'
    assert at.date_input[0].max.isoformat() == '2024-01-02'
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value == f'{default_probability:.1%}'
    at.selectbox[0].select(predictor.stations()[1]).run()
    assert not at.metric  # Changing station clears the old station's estimate.
    at.date_input[0].set_value(datetime(2024, 1, 2).date()).run()
    at.button[0].click().run()
    assert not at.exception
    expected = predictor.predict('S33', datetime(2024, 1, 2, 1, tzinfo=SG), 5.)['probability']
    assert at.metric[0].value == f'{expected:.1%}'
    assert any('S33' in x.value for x in at.markdown)
    at.date_input[0].set_value(when.date()).run()
    at.time_input[0].set_value(when.replace(hour=0).time()).run()
    at.button[0].click().run()
    assert not at.exception
    assert any('Insufficient history' in x.value for x in at.error)
    assert not at.metric
