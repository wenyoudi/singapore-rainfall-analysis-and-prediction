"""Historical Singapore rainfall prediction dashboard. Run with Streamlit."""
from datetime import datetime, time, timedelta
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
import altair as alt
import pandas as pd
import streamlit as st
from rainfall_inference import InputError, RainfallPredictor, SG

st.set_page_config(page_title='Singapore Rainfall | Prediction', page_icon='🌧️', layout='wide')
st.markdown('''<style>
.block-container {max-width:1120px; padding-top:2.3rem;}
h1 {letter-spacing:-.045em;}
[data-testid="stMetricValue"] {color:#147981;}
[data-testid="stForm"] {border-radius:14px;}
</style>''', unsafe_allow_html=True)


@st.cache_resource(show_spinner='Connecting historical observations and loading the saved model…')
def engine(data_root, model_name, master):
    return RainfallPredictor(data_root, model_name, master)


@st.cache_data(show_spinner='Finding available stations…', ttl=600)
def catalog(data_root, model_name, master):
    return engine(data_root, model_name, master).stations()


def main():
    st.caption('SINGAPORE RAINFALL · DSA5208')
    st.title('A closer look at the next 30 minutes')
    st.write('Select a station and historical time to estimate the chance of rainfall exceeding your threshold.')
    data_root = os.environ.get('RAINFALL_DATA_ROOT', str(ROOT/'data/processed'))
    model_name = os.environ.get('RAINFALL_MODEL', 'GBT').upper()
    master = os.environ.get('RAINFALL_SPARK_MASTER', 'local[2]')
    try:
        predictor = engine(data_root, model_name, master)
        stations = catalog(data_root, model_name, master)
        if not stations:
            st.warning('No monitoring stations were found in the connected historical data.')
            return
    except InputError as exc:
        st.warning(str(exc))
        st.info('Connect the cleaned historical Parquet to begin. See docs/prediction_interface.md for local and Vanda launch instructions.')
        return
    except Exception:
        st.error('The prediction service could not start. Check the data/model paths and the Python/Java environment in the launch guide.')
        with st.expander('Technical details'):
            st.exception(sys.exc_info()[1])
        return

    st.caption(f'{model_name} saved model · historical replay · all times in Singapore time (UTC+8)')
    left, right = st.columns([1, 1.5], gap='large')
    with left:
        st.subheader('Prediction inputs')
        # Outside the form so changing station updates its date range immediately.
        selected = st.selectbox('Monitoring station', stations,
                                format_func=lambda s: f"{s['name']} · {s['id']}")
        st.caption(f"Records: {selected['first']:%d %b %Y} – {selected['last']:%d %b %Y}. Gaps may occur within this range.")
        earliest = max(selected['first'].date(), datetime(2017, 1, 1).date())
        latest = min(selected['last'].date(), datetime(2024, 12, 31).date())
        default = min(selected['first'] + timedelta(minutes=60), selected['last'])
        with st.form('rainfall_prediction'):
            date = st.date_input('Prediction date', value=default.date(), min_value=earliest,
                                 max_value=latest, key=f"date_{selected['id']}")
            clock = st.time_input('Prediction time (SGT)', value=time(default.hour, default.minute), step=300,
                                  help='Use a five-minute timestamp with twelve complete preceding readings.')
            threshold = st.number_input('Rainfall threshold (mm)', min_value=0., max_value=10., value=5., step=1.)
            st.caption('Supported thresholds: 0, 1, 5 and 10 mm. Other thresholds were not evaluated.')
            submitted = st.form_submit_button('Predict Rainfall', type='primary', width='stretch')
        # Clear stale results whenever station changes, even before form submission.
        if st.session_state.get('station_id') != selected['id']:
            st.session_state.pop('prediction_result', None)
            st.session_state['station_id'] = selected['id']
        if submitted:
            st.session_state.pop('prediction_result', None)
            when = datetime.combine(date, clock).replace(tzinfo=SG)
            try:
                with st.spinner('Reading the historical window and estimating probability…'):
                    result = predictor.predict(selected['id'], when, threshold)
                st.session_state['prediction_result'] = {'result': result, 'when': when,
                                                         'threshold': threshold, 'station': selected}
            except InputError as exc:
                st.error(str(exc))
            except Exception:
                st.error('Prediction could not be completed. Check the data connection and try again.')
                with st.expander('Technical details'):
                    st.exception(sys.exc_info()[1])
    with right:
        st.subheader('Rainfall outlook')
        saved = st.session_state.get('prediction_result')
        if not saved:
            st.info('Your estimate will appear here. Choose your inputs and select Predict Rainfall.')
        else:
            result, when, threshold = saved['result'], saved['when'], saved['threshold']
            station = saved['station']
            end = when + timedelta(minutes=30)
            st.metric('Probability of exceeding the rainfall threshold', f"{result['probability']:.1%}")
            st.write(f"There is a **{result['probability']:.1%} probability** that **{station['name']} ({station['id']})** "
                     f"will accumulate **more than {threshold:g} mm** of rainfall over the next 30 minutes.")
            st.caption(f"Issued: {when:%d %b %Y, %H:%M} SGT · End: {end:%d %b %Y, %H:%M} SGT")
            st.caption(f"Target readings: {(when+timedelta(minutes=5)):%H:%M} through {end:%H:%M} at five-minute intervals. Equality to the threshold does not count as exceedance.")
            st.subheader('Recent rainfall history')
            chart = (alt.Chart(pd.DataFrame(result['history'])).mark_line(point=True, color='#147981')
                     .encode(x=alt.X('minutes:Q', title='Minutes before prediction', axis=alt.Axis(values=[-60,-45,-30,-15,-5])),
                             y=alt.Y('rainfall:Q', title='Five-minute rainfall (mm)', scale=alt.Scale(zero=True)),
                             tooltip=[alt.Tooltip('time:N', title='Time (SGT)'), alt.Tooltip('rainfall:Q', title='Rainfall (mm)')])
                     .properties(height=235))
            st.altair_chart(chart, width='stretch')
            st.caption('Features use only the 12 readings from t−60 through t−5. The reading at t is excluded, matching training.')
            with st.expander('Verify against the historical outcome'):
                st.caption('Optional retrospective check. Subsequent observations are never used to generate the prediction.')
                if st.button('Show observed next 30 minutes'):
                    try:
                        actual = predictor.verify(station['id'], when, threshold)
                        if actual is None:
                            st.info('Verification unavailable: all six subsequent readings are required. The prediction is still valid as a model output.')
                        else:
                            st.metric('Observed next 30-minute rainfall', f"{actual['total_mm']:.2f} mm")
                            st.write('The threshold was exceeded.' if actual['exceeded'] else 'The threshold was not exceeded.')
                    except InputError as exc:
                        st.warning(str(exc))
    st.divider()
    st.caption('Academic historical demonstration, not a live weather forecast. Probability estimates can be poorly calibrated, especially for rare heavy rainfall. Earlier years include model training/validation data; only 2024 was held out for testing.')


if __name__ == '__main__':
    main()
