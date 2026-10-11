import calendar
import numpy as np
import pandas as pd
import pytest

from task1_analysis import (YEARS, balanced_ids, coverage_sensitivity, spatial_summary,
                            summarize, validate_summary)


@pytest.fixture
def summaries():
    annual, monthly = [], []
    for year in YEARS:
        for sid, coverage in [('A', .95), ('B', .85), ('C', .99)]:
            if sid == 'C' and year == 2017:
                continue
            months = []
            for month in range(1, 13):
                expected = calendar.monthrange(year, month)[1] * 288
                observed = int(expected * coverage)
                row = dict(year=year, month=month, station_id=sid, observed_intervals=observed,
                           rainy_intervals=observed // 20, recorded_rainfall_mm=observed / 10,
                           calendar_expected_intervals=expected,
                           calendar_coverage_pct=100 * observed / expected)
                months.append(row)
                monthly.append(row)
            expected = sum(r['calendar_expected_intervals'] for r in months)
            observed = sum(r['observed_intervals'] for r in months)
            annual.append(dict(year=year, station_id=sid, observed_intervals=observed,
                               rainy_intervals=sum(r['rainy_intervals'] for r in months),
                               recorded_rainfall_mm=sum(r['recorded_rainfall_mm'] for r in months),
                               calendar_expected_intervals=expected,
                               calendar_coverage_pct=100 * observed / expected,
                               latitude=1.3 + .001 * (year - 2017), longitude=103.8))
    return pd.DataFrame(annual), pd.DataFrame(monthly)


def test_pooled_and_equal_weighting_answer_different_questions():
    frame = pd.DataFrame([dict(year=2017, station_id='A', observed_intervals=10000,
                               rainy_intervals=500, recorded_rainfall_mm=1000),
                          dict(year=2017, station_id='B', observed_intervals=1000,
                               rainy_intervals=10, recorded_rainfall_mm=10)])
    pooled = summarize(frame, 'year', [2017], 'example', 0, 'pooled').iloc[0]
    equal = summarize(frame, 'year', [2017], 'example', 0, 'equal_station_period').iloc[0]
    assert pooled.rainy_pct == pytest.approx(100 * 510 / 11000)
    assert equal.rainy_pct == pytest.approx(3.)
    assert equal.mean_mm_per_interval == pytest.approx(.055)


def test_fixed_cohorts_require_every_year(summaries):
    annual, _ = summaries
    assert balanced_ids(annual, min_coverage=80) == ['A', 'B']
    assert balanced_ids(annual, min_coverage=90) == ['A']
    # 2021-2024 has a different common set; absent 2017 cannot be overlooked.
    assert balanced_ids(annual, (2021, 2022, 2023, 2024), 90) == ['A', 'C']


def test_empty_cohorts_have_undefined_rates_and_missing_year_audit(summaries):
    annual, monthly = summaries
    result, _, audit = coverage_sensitivity(annual, monthly, thresholds=(100.,))
    empty = result[(result.cohort == 'balanced_all_years') & (result.min_coverage_pct == 100)]
    assert len(empty) == 16
    assert empty.stations.eq(0).all() and empty.rainy_pct.isna().all()
    missing = audit[(audit.station_id == 'C') & (audit.year == 2017)]
    assert missing.calendar_coverage_pct.isna().all() and not missing.selected.any()


def test_monthly_filter_uses_month_not_year_coverage(summaries):
    annual, monthly = summaries
    target = (monthly.station_id == 'A') & (monthly.year == 2020) & (monthly.month == 2)
    monthly.loc[target, 'observed_intervals'] = 1000
    monthly.loc[target, 'rainy_intervals'] = 100
    monthly.loc[target, 'calendar_coverage_pct'] = 100 * 1000 / (29 * 288)
    _, seasonal, _ = coverage_sensitivity(annual, monthly)
    feb = seasonal[(seasonal.month == 2) & (seasonal.min_coverage_pct == 90)]
    assert feb.station_periods.eq(14).all()  # 8 A + 7 C, minus incomplete A month.


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_year', 'over_coverage', 'bad_denominator', 'nan'])
def test_invalid_summaries_fail(summaries, mutation):
    annual, _ = summaries
    if mutation == 'duplicate':
        annual = pd.concat([annual, annual.iloc[[0]]])
    elif mutation == 'missing_year':
        annual = annual[annual.year != 2017]
    elif mutation == 'over_coverage':
        annual.loc[0, 'observed_intervals'] = 200000
    elif mutation == 'bad_denominator':
        annual.loc[annual.year == 2020, 'calendar_expected_intervals'] = 365 * 288
    else:
        annual.loc[0, 'calendar_coverage_pct'] = np.nan
    with pytest.raises(ValueError):
        validate_summary(annual, ['year', 'station_id'])


def test_spatial_period_coordinates_and_equal_year_weights(summaries):
    annual, _ = summaries
    # Exclude earlier coordinates and do not weight annual rates by row counts.
    annual.loc[annual.year < 2021, 'latitude'] = 2.
    points, selected = spatial_summary(annual)
    assert set(points.station_id) == {'A', 'B', 'C'}
    assert set(selected.year) == {2021, 2022, 2023, 2024}
    assert points.eligible_years.eq(4).all()
    assert points.latitude.to_numpy() == pytest.approx([1.3055] * 3)
    assert points.mean_mm_per_interval.to_numpy() == pytest.approx([.1, .1, .1])
