"""Task 1 selection and sensitivity calculations on compact Spark summaries.

No raw observations are collected here. Coverage means full-calendar coverage,
including time before/after a station's observed operating span.
"""
import calendar
import numpy as np
import pandas as pd

YEARS = tuple(range(2017, 2025))
SPATIAL_YEARS = (2021, 2022, 2023, 2024)
COUNTS = ['observed_intervals', 'rainy_intervals', 'recorded_rainfall_mm']


def validate_summary(frame, keys, years=YEARS):
    """Reject duplicates, invalid totals and inconsistent calendar denominators."""
    required = set(keys + COUNTS + ['calendar_expected_intervals', 'calendar_coverage_pct'])
    if required - set(frame.columns):
        raise ValueError('Missing summary columns: %s' % sorted(required - set(frame.columns)))
    if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Summary must have non-null, unique keys: %s' % keys)
    if set(frame['year']) != set(years):
        raise ValueError('Expected all study years %s' % (tuple(years),))
    values = frame[COUNTS + ['calendar_expected_intervals', 'calendar_coverage_pct']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Summary counts, rainfall and coverage must be finite and nonnegative')
    if (frame['observed_intervals'] <= 0).any():
        raise ValueError('Each summary row must contain observations')
    if (frame['rainy_intervals'] > frame['observed_intervals']).any():
        raise ValueError('Rainy intervals exceed observed intervals')
    for col in ('observed_intervals', 'rainy_intervals'):
        if (frame[col] != np.floor(frame[col])).any():
            raise ValueError('Interval counts must be integers')
    if 'month' in keys:
        if not frame['month'].isin(range(1, 13)).all():
            raise ValueError('Months must be integers in 1..12')
        expected = np.array([calendar.monthrange(int(r.year), int(r.month))[1] * 288
                             for r in frame.itertuples()])
    else:
        expected = np.array([(366 if calendar.isleap(int(y)) else 365) * 288
                             for y in frame['year']])
    if not np.array_equal(frame['calendar_expected_intervals'].to_numpy(), expected):
        raise ValueError('Incorrect calendar denominator (including leap years)')
    if (frame['observed_intervals'].to_numpy() > expected).any():
        raise ValueError('Coverage exceeds 100%; check duplicates or timestamps')
    if not np.allclose(frame['calendar_coverage_pct'],
                       100 * frame['observed_intervals'] / expected, atol=1e-7, rtol=0):
        raise ValueError('Coverage does not agree with observed counts')


def balanced_ids(station_year, years=YEARS, min_coverage=0):
    """Stations meeting the threshold in EVERY requested year, not just on average."""
    if not years or len(set(years)) != len(years) or not 0 <= min_coverage <= 100:
        raise ValueError('Unique years and coverage in 0..100 are required')
    selected = station_year.loc[station_year['year'].isin(years) &
                                (station_year['calendar_coverage_pct'] >= min_coverage)]
    counts = selected.groupby('station_id')['year'].nunique()
    return sorted(counts[counts == len(years)].index)


def summarize(frame, group, periods, cohort, threshold, weighting):
    """Pooled rates weight by observations; equal rates weight each row equally.

For annual inputs, a row is a station-year; for monthly inputs it is a
station-year-month. Empty periods are explicit, with undefined (NaN) rates.
"""
    rows = []
    for period in periods:
        part = frame.loc[frame[group] == period]
        n = int(part['observed_intervals'].sum())
        rainy = int(part['rainy_intervals'].sum())
        rain = float(part['recorded_rainfall_mm'].sum())
        if weighting == 'pooled':
            rate = 100 * rainy / n if n else np.nan
            mean = rain / n if n else np.nan
        elif weighting == 'equal_station_period':
            rate = (100 * part['rainy_intervals'] / part['observed_intervals']).mean()
            mean = (part['recorded_rainfall_mm'] / part['observed_intervals']).mean()
        else:
            raise ValueError('Unknown weighting: %s' % weighting)
        rows.append({group: period, 'cohort': cohort, 'min_coverage_pct': threshold,
                     'weighting': weighting, 'stations': part['station_id'].nunique(),
                     'station_periods': len(part), 'observed_intervals': n,
                     'rainy_intervals': rainy, 'recorded_rainfall_mm': rain,
                     'rainy_pct': rate, 'mean_mm_per_interval': mean})
    return pd.DataFrame(rows)


def coverage_sensitivity(station_year, station_month, thresholds=(80., 90.), years=YEARS):
    """Annual cohorts and seasonal checks, with auditable station-year membership."""
    validate_summary(station_year, ['year', 'station_id'], years)
    validate_summary(station_month, ['year', 'month', 'station_id'], years)
    annual, monthly, audits = [], [], []
    cohorts = [('all_reporting', 0., station_year)]
    ids = balanced_ids(station_year, years)
    cohorts.append(('common_all_years', 0., station_year[station_year['station_id'].isin(ids)]))
    for threshold in thresholds:
        eligible = station_year[station_year['calendar_coverage_pct'] >= threshold]
        cohorts.append(('eligible_each_year', threshold, eligible))
        ids = balanced_ids(station_year, years, threshold)
        cohorts.append(('balanced_all_years', threshold,
                        station_year[station_year['station_id'].isin(ids)]))
    for cohort, threshold, frame in cohorts:
        for weighting in ('pooled', 'equal_station_period'):
            annual.append(summarize(frame, 'year', years, cohort, threshold, weighting))
        # Reindex the full station-year grid so wholly absent years are visible.
        audit = station_year.set_index(['station_id', 'year']).reindex(
            pd.MultiIndex.from_product([sorted(station_year['station_id'].unique()), years],
                                       names=['station_id', 'year']))
        audit = audit.reset_index()[['station_id', 'year', 'calendar_coverage_pct']]
        membership = set(zip(frame['station_id'], frame['year']))
        audit['selected'] = [(s, y) in membership for s, y in zip(audit.station_id, audit.year)]
        audit['cohort'], audit['min_coverage_pct'] = cohort, threshold
        audits.append(audit)
    # Seasonal selection applies the threshold to station-MONTH coverage. A
    # complete year can still contain an incomplete month, and vice versa.
    for threshold in (0.,) + tuple(thresholds):
        eligible = station_month[station_month['calendar_coverage_pct'] >= threshold]
        for weighting in ('pooled', 'equal_station_period'):
            monthly.append(summarize(eligible, 'month', range(1, 13),
                                     'eligible_station_months', threshold, weighting))
    return pd.concat(annual, ignore_index=True), pd.concat(monthly, ignore_index=True), pd.concat(audits, ignore_index=True)


def spatial_summary(station_year, years=SPATIAL_YEARS, min_coverage=80.):
    """Fixed common period; both map panels use exactly the same station-years."""
    validate_summary(station_year, ['year', 'station_id'])
    ids = balanced_ids(station_year, years, min_coverage)
    selected = station_year[station_year['year'].isin(years) & station_year['station_id'].isin(ids)].copy()
    selected['annual_rainy_pct'] = 100 * selected['rainy_intervals'] / selected['observed_intervals']
    selected['annual_mean_mm'] = selected['recorded_rainfall_mm'] / selected['observed_intervals']
    points = selected.groupby('station_id', as_index=False).agg(
        rainy_pct=('annual_rainy_pct', 'mean'), mean_mm_per_interval=('annual_mean_mm', 'mean'),
        eligible_years=('year', 'nunique'), min_calendar_coverage_pct=('calendar_coverage_pct', 'min'),
        latitude=('latitude', 'median'), longitude=('longitude', 'median'))
    points['analysis_years'] = ','.join(map(str, years))
    if not np.isfinite(points[['rainy_pct', 'mean_mm_per_interval', 'latitude', 'longitude']].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite spatial values or coordinates')
    return points.sort_values('station_id'), selected.sort_values(['station_id', 'year'])
