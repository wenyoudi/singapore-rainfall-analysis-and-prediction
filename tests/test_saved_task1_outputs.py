"""Integration checks on the committed/downloaded Task 1 summary CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

OUTPUTS = Path(__file__).resolve().parents[1] / 'outputs'
YEARS = set(range(2017, 2025))
COUNTS = ['observed_intervals', 'rainy_intervals', 'recorded_rainfall_mm']
THRESHOLDS = ['0p0', '0p2', '1p0', '5p0', '10p0']


def load(folder, name):
    return pd.read_csv(OUTPUTS / folder / name)


def test_cleaning_and_validation_agree_with_eda():
    cleaning = load('cleaning', 'cleaning_summary.csv').set_index('year').sort_index()
    validation = load('validation', 'final_validation_summary.csv').set_index('year').sort_index()
    annual = load('eda', 'annual_rainfall_stats.csv').set_index('year').sort_index()
    assert set(cleaning.index) == set(validation.index) == set(annual.index) == YEARS
    assert len(cleaning) == len(validation) == len(annual) == 8
    np.testing.assert_array_equal(cleaning.output_rows, validation.rows)
    np.testing.assert_array_equal(validation.rows, annual.observed_intervals)
    np.testing.assert_array_equal(cleaning.input_rows - cleaning.output_rows, cleaning.collision_rows_removed)
    assert cleaning.input_rows.sum() == 48_538_713
    assert cleaning.output_rows.sum() == 48_538_627
    assert cleaning.collision_rows_removed.sum() == 86
    checks = ['invalid_station_ids', 'invalid_station_names', 'null_timestamps', 'wrong_year',
              'off_grid', 'invalid_rainfall', 'invalid_coordinates', 'duplicate_extra_rows']
    assert validation[checks].eq(0).all().all()
    assert validation.row_count_matches_summary.eq(True).all()
    assert load('validation', 'validation_issues.csv').empty


@pytest.mark.parametrize('filename', ['annual_rainfall_stats.csv', 'monthly_rainfall_stats.csv',
                                    'hourly_rainfall_stats.csv', 'station_year_rainfall_stats.csv',
                                    'station_month_rainfall_stats.csv'])
def test_saved_metrics_are_finite_and_consistent(filename):
    frame = load('eda', filename)
    assert set(frame.year) == YEARS
    assert np.isfinite(frame[COUNTS].to_numpy(dtype=float)).all()
    assert frame[COUNTS].ge(0).all().all()
    assert frame.observed_intervals.gt(0).all()
    np.testing.assert_allclose(frame.rainy_fraction, frame.rainy_intervals / frame.observed_intervals)
    np.testing.assert_allclose(frame.mean_mm_per_observed_interval,
                               frame.recorded_rainfall_mm / frame.observed_intervals)
    positive = frame.rainy_intervals > 0
    np.testing.assert_allclose(frame.loc[positive, 'mean_positive_rainfall_mm'],
                               frame.loc[positive, 'recorded_rainfall_mm'] / frame.loc[positive, 'rainy_intervals'])
    assert frame.loc[~positive, 'mean_positive_rainfall_mm'].isna().all()
    counts = frame[['exceed_%s_mm_intervals' % t for t in THRESHOLDS]].to_numpy()
    assert (np.diff(counts, axis=1) <= 0).all()
    np.testing.assert_array_equal(counts[:, 0], frame.rainy_intervals)
    assert (counts <= frame.observed_intervals.to_numpy()[:, None]).all()
    for t in THRESHOLDS:
        np.testing.assert_allclose(frame['exceed_%s_mm_fraction' % t],
                                   frame['exceed_%s_mm_intervals' % t] / frame.observed_intervals)


def test_all_aggregation_levels_reconcile():
    annual = load('eda', 'annual_rainfall_stats.csv').set_index('year').sort_index()
    for filename in ['monthly_rainfall_stats.csv', 'hourly_rainfall_stats.csv',
                     'station_year_rainfall_stats.csv', 'station_month_rainfall_stats.csv']:
        grouped = load('eda', filename).groupby('year')[COUNTS].sum().sort_index()
        np.testing.assert_allclose(grouped, annual[COUNTS], rtol=1e-10, atol=1e-5)
    station_month = load('eda', 'station_month_rainfall_stats.csv')
    for keys, filename in [(['year', 'station_id'], 'station_year_rainfall_stats.csv'),
                           (['year', 'month'], 'monthly_rainfall_stats.csv')]:
        actual = station_month.groupby(keys)[COUNTS].sum().sort_index()
        reference = load('eda', filename).set_index(keys)[COUNTS].sort_index()
        assert actual.index.equals(reference.index)
        np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-5)


def test_positive_histogram_and_percentiles_reconcile():
    annual = load('eda', 'annual_rainfall_stats.csv').set_index('year').sort_index()
    histogram = load('eda', 'positive_rainfall_histogram.csv')
    assert not histogram.duplicated(['year', 'bin_id']).any()
    assert len(histogram) == 80
    np.testing.assert_array_equal(histogram.groupby('year')['count'].sum().sort_index(), annual.rainy_intervals)
    percentiles = load('eda', 'positive_rainfall_percentiles.csv').set_index('year').sort_index()
    np.testing.assert_array_equal(percentiles.positive_intervals, annual.rainy_intervals)
    values = percentiles[['p0p5_mm', 'p0p9_mm', 'p0p95_mm', 'p0p99_mm', 'p0p999_mm']].to_numpy()
    assert np.isfinite(values).all() and (values > 0).all() and (np.diff(values, axis=1) >= 0).all()
    assert (values <= annual.max_five_min_rainfall_mm.to_numpy()[:, None]).all()


def test_gap_counts_and_station_metadata_reconcile():
    coverage = load('validation', 'station_coverage_summary.csv').set_index(['year', 'station_id']).sort_index()
    gaps = load('validation', 'gap_distribution_summary.csv').set_index(['year', 'station_id'])
    gap_missing = gaps.missing_intervals.reindex(coverage.index, fill_value=0)
    np.testing.assert_array_equal(coverage.expected_between_first_last - coverage.observed_intervals,
                                  coverage.missing_intervals)
    np.testing.assert_array_equal(gap_missing, coverage.missing_intervals)
    assert coverage.missing_intervals.sum() == 3_747_854
    np.testing.assert_allclose(coverage.coverage_pct,
                               100 * coverage.observed_intervals / coverage.expected_between_first_last,
                               atol=5.1e-5, rtol=0)
    locations = load('eda', 'station_locations.csv').set_index(['year', 'station_id']).sort_index()
    station_year = load('eda', 'station_year_rainfall_stats.csv').set_index(['year', 'station_id']).sort_index()
    assert locations.index.equals(station_year.index)
    np.testing.assert_allclose(locations[['latitude', 'longitude']], station_year[['latitude', 'longitude']])
    assert locations.latitude.between(-90, 90).all()
    assert locations.longitude.between(-180, 180).all()
