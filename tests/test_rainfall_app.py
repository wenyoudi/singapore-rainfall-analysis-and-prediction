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
