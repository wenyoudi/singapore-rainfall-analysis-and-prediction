"""Historical Singapore rainfall prediction dashboard. Run with Streamlit."""
from datetime import datetime, time, timedelta
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
import altair as alt
import pandas as pd
import streamlit as st
from rainfall_inference import InputError, RainfallPredictor, SG, THRESHOLDS

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


def choose_map_station():
    """Apply a map click before the dropdown is recreated on the next rerun."""
    points = st.session_state.get('station_map', {}).get('selection', {}).get('station_pick', [])
    if points and points[0].get('id') in st.session_state.get('station_ids', []):
        st.session_state['station_choice'] = points[0]['id']


@st.cache_data
def singapore_boundary():
    features = json.loads((ROOT/'data/geography/singapore_adm0.geojson').read_text())['features']
    # Vega expects clockwise exteriors, unlike the bundled RFC 7946 GeoJSON.
    # Reverse ring order only in memory; leave the source geometry untouched.
    for feature in features:
        geometry = feature['geometry']
        polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
        for polygon in polygons:
            for index, ring in enumerate(polygon):
                area = sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(ring, ring[1:]))
                if (area > 0) == (index == 0):
                    ring.reverse()
    return features


def station_map(stations, selected_id):
    mapped = [s for s in stations if s['latitude'] is not None and s['longitude'] is not None]
    if not mapped:
        st.info('Recorded station coordinates are unavailable in this dataset. Use the dropdown to select a station.')
        return
    land = alt.Chart(alt.Data(values=singapore_boundary())).mark_geoshape(
        fill='#e8f0ed', stroke='#a2b9b1', strokeWidth=.8)
    locations = alt.Chart(pd.DataFrame(mapped)).encode(longitude='longitude:Q', latitude='latitude:Q')
    pick = alt.selection_point(name='station_pick', fields=['id'], on='click',
                               toggle=False, clear=False, empty=False)
    dots = locations.mark_circle(size=135, stroke='white', strokeWidth=1.5).encode(
        color=alt.condition(alt.datum.id == selected_id, alt.value('#d95e35'), alt.value('#147981')),
        tooltip=[alt.Tooltip('id:N', title='Station'), alt.Tooltip('name:N', title='Name'),
                 alt.Tooltip('latitude:Q', title='Latitude', format='.4f'),
                 alt.Tooltip('longitude:Q', title='Longitude', format='.4f')]
    ).add_params(pick)
    labels = locations.mark_text(dy=-13, fontSize=11, color='#29463e').encode(text='id:N')
    chart = (alt.layer(land, dots, labels).project(type='mercator')
             .properties(height=270, description='Singapore map with labelled monitoring stations. Click a station point to select it.')
             .configure_view(stroke=None))
    st.altair_chart(chart, key='station_map', on_select=choose_map_station,
                    selection_mode='station_pick', width='stretch')
    st.caption('Click a station point or use the dropdown. Orange marks your selection. '
               'Locations use the latest valid coordinates recorded in the connected data; historical locations may differ.')
    if len(mapped) < len(stations):
        st.caption(f'{len(stations)-len(mapped)} stations have no valid coordinates and remain available in the dropdown.')


def main():
    st.caption('SINGAPORE RAINFALL · DSA5208')
    st.title('A closer look at the next 30 minutes')
    st.write('Select a station and historical time to estimate the chance of rainfall exceeding your threshold.')
    data_root = os.environ.get('RAINFALL_DATA_ROOT', str(ROOT/'data/demo'))
    demo = Path(data_root).expanduser().resolve() == (ROOT/'data/demo').resolve()
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

    if demo:
        first = min(s['first'] for s in stations)
        last = max(s['last'] for s in stations)
        st.info(f'Demonstration data · {len(stations)} stations ({", ".join(s["id"] for s in stations)}) · '
                f'{first:%d %b %Y} – {last:%d %b %Y} (SGT). '
                'Real cleaned observations for trying the interface, not the dataset used to evaluate the models. '
                'See the README for full 2017–2024 data and connection instructions.')
    st.caption(f'{model_name} saved model · historical replay · all times in Singapore time (UTC+8)')
    st.subheader('Choose a monitoring station')
    by_id = {s['id']: s for s in stations}
    st.session_state['station_ids'] = list(by_id)
    if st.session_state.get('station_choice') not in by_id:
        st.session_state['station_choice'] = stations[0]['id']
    station_inputs, map_panel = st.columns([1, 2], gap='large')
    with station_inputs:
        station_id = st.selectbox('Monitoring station', list(by_id), key='station_choice',
                                  format_func=lambda sid: f"{by_id[sid]['name']} · {sid}")
        selected = by_id[station_id]
        st.caption(f"Records: {selected['first']:%d %b %Y, %H:%M} – {selected['last']:%d %b %Y, %H:%M} SGT. Gaps may occur within this range.")
        if demo:
            first_prediction = selected['first'] + timedelta(minutes=60)
            last_verified = selected['last'] - timedelta(minutes=30)
            st.caption(f'Demo predictions start at {first_prediction:%d %b, %H:%M}. '
                       f'Historical verification is available through {last_verified:%d %b, %H:%M} SGT.')
    with map_panel:
        station_map(stations, selected['id'])
    left, right = st.columns([1, 1.5], gap='large')
    with left:
        st.subheader('Prediction inputs')
        earliest = max(selected['first'].date(), datetime(2017, 1, 1).date())
        latest = min(selected['last'].date(), datetime(2024, 12, 31).date())
        with st.form('rainfall_prediction'):
            date = st.date_input('Prediction date', value=earliest, min_value=earliest,
                                 max_value=latest, key=f"date_{selected['id']}")
            clock = st.time_input('Prediction time (SGT)', value=time(0, 0), step=300,
                                  help='Use a five-minute timestamp with twelve complete preceding readings.')
            threshold = st.selectbox('Rainfall threshold (mm)', THRESHOLDS, index=0,
                                     format_func=lambda value: f'{value:g} mm')
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
            # Explicit bounds also handle the all-zero case without a negative auto-domain.
            rainfall_max = max(point['rainfall'] for point in result['history'])
            rainfall_upper = rainfall_max * 1.1 if rainfall_max > 0 else 1.
            chart = (alt.Chart(pd.DataFrame(result['history'])).mark_line(point=True, color='#147981')
                     .encode(x=alt.X('minutes:Q', title='Minutes before prediction', axis=alt.Axis(values=[-60,-45,-30,-15,-5])),
                             y=alt.Y('rainfall:Q', title='Five-minute rainfall (mm)',
                                     scale=alt.Scale(domain=[0, rainfall_upper], nice=False)),
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
    st.caption('Source: NEA [Historical Rainfall across Singapore](https://data.gov.sg/collections/2279/view), '
               'made available under the [Singapore Open Data Licence](https://data.gov.sg/open-data-licence).')
    st.caption('Map boundary: [geoBoundaries](https://www.geoboundaries.org/), derived from URA boundaries · '
               '[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/).')


if __name__ == '__main__':
    main()
