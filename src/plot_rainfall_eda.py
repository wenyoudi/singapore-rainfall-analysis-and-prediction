#!/usr/bin/env python3
"""Plot rainfall EDA from small Spark-produced CSV summaries (no Spark needed).

Run from anywhere: python src/plot_rainfall_eda.py
Dependencies: pandas, numpy, matplotlib. All plots are saved as PNG and PDF.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


def load(folder, filename):
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Copy the EDA CSV outputs into {folder}.')
    return pd.read_csv(path)


def save(fig, folder, name):
    fig.savefig(folder / (name + '.png'), dpi=220, bbox_inches='tight')
    fig.savefig(folder / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('Saved', name, flush=True)


def weighted_rate(df, group, numerator='rainy_intervals'):
    a = df.groupby(group, as_index=False)[[numerator,'observed_intervals']].sum()
    a['rate_pct'] = 100 * a[numerator] / a['observed_intervals']
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'outputs' / 'eda')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'figures' / 'eda')
    parser.add_argument('--min-spatial-coverage', type=float, default=80.0,
                        help='Minimum calendar-year station coverage for spatial summary (percent)')
    parser.add_argument('--min-spatial-years', type=int, default=4,
                        help='Minimum number of eligible years per station for spatial map')
    parser.add_argument('--min-monthly-coverage', type=float, default=90.0,
                        help='Minimum calendar-month coverage for accumulation comparison (percent)')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    annual = load(args.input_dir, 'annual_rainfall_stats.csv').sort_values('year')
    monthly = load(args.input_dir, 'monthly_rainfall_stats.csv')
    hourly = load(args.input_dir, 'hourly_rainfall_stats.csv')
    histogram = load(args.input_dir, 'positive_rainfall_histogram.csv')
    percentiles = load(args.input_dir, 'positive_rainfall_percentiles.csv')
    station_year = load(args.input_dir, 'station_year_rainfall_stats.csv')
    station_month = load(args.input_dir, 'station_month_rainfall_stats.csv')
    locations = load(args.input_dir, 'station_locations.csv')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 120})

    # 4.1: Positive rainfall histogram; bin widths differ, so plot shares per bin.
    hist = (histogram.groupby(['bin_id','lower_mm','upper_mm'], dropna=False, as_index=False)['count']
            .sum().sort_values('bin_id'))
    hist = hist[hist['count'] > 0].copy()
    labels = [f'{r.lower_mm:g}–{r.upper_mm:g}' if pd.notna(r.upper_mm)
              else f'≥{r.lower_mm:g}' for r in hist.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.bar(labels, hist['count'] / hist['count'].sum() * 100)
    ax.set(xlabel='Five-minute rainfall bin (mm)', ylabel='Share of positive readings (%)',
           title='Binned frequency of positive five-minute rainfall measurements')
    ax.tick_params(axis='x', rotation=35)
    ax.text(.99, .96, 'Unequal-width bins: bar height shows share, not density',
            ha='right', va='top', transform=ax.transAxes, fontsize=8)
    save(fig, out, 'fig_4_1_positive_rainfall_distribution')

    # 4.1: Threshold exceedance over all observed five-minute readings.
    thresholds = [('>0', 'exceed_0p0_mm_intervals'), ('>0.2','exceed_0p2_mm_intervals'),
                  ('>1','exceed_1p0_mm_intervals'), ('>5','exceed_5p0_mm_intervals'),
                  ('>10','exceed_10p0_mm_intervals')]
    rates = [100 * annual[col].sum() / annual['observed_intervals'].sum() for _, col in thresholds]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    threshold_values = [0, 0.2, 1, 5, 10]
    ax.plot(threshold_values, rates, marker='o')
    ax.set_xticks(threshold_values, [x for x,_ in thresholds])
    ax.set_xlim(-0.3, 10.5)
    ax.set_yscale('log')
    ax.set(xlabel='Rainfall threshold (mm per five minutes)',
           ylabel='Observed exceedance rate (%) — log scale',
           title='Frequency of five-minute rainfall exceedances')
    ax.grid(axis='y', alpha=.25)
    save(fig, out, 'fig_4_1_threshold_exceedance')

    # 4.2: Observed-network annual rates, not national annual totals.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(annual['year'], 100 * annual['rainy_intervals'] / annual['observed_intervals'], marker='o')
    axes[0].set(ylabel='Rainy intervals (%)', xlabel='Year', title='A. Rainfall occurrence')
    axes[1].plot(annual['year'], annual['recorded_rainfall_mm'] / annual['observed_intervals'], marker='o')
    axes[1].set(ylabel='Mean rainfall per observed interval (mm)', xlabel='Year',
                title='B. Mean observed rainfall')
    for ax in axes:
        ax.set_xticks(annual['year'])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Annual observed-network rainfall characteristics (station mix varies)')
    fig.tight_layout()
    save(fig, out, 'fig_4_2_annual_observed_network')

    # 4.2: Sensitivity analysis holding station identity constant across all eight years.
    # Rates remain conditional on observed intervals; missingness can still bias estimates.
    n_years = station_year['year'].nunique()
    station_counts = station_year.groupby('station_id')['year'].nunique()
    common_ids = station_counts[station_counts == n_years].index
    common = station_year[station_year['station_id'].isin(common_ids)].copy()
    if not common.empty:
        comp = common.groupby('year', as_index=False)[
            ['rainy_intervals', 'observed_intervals', 'recorded_rainfall_mm']].sum()
        comp['rainy_pct'] = 100 * comp['rainy_intervals'] / comp['observed_intervals']
        comp['mean_mm'] = comp['recorded_rainfall_mm'] / comp['observed_intervals']
        comp['stations'] = len(common_ids)
        comp.to_csv(out / 'common_station_annual_summary.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        axes[0].plot(annual['year'], 100 * annual['rainy_intervals'] /
                     annual['observed_intervals'], marker='o', label='All reporting stations')
        axes[0].plot(comp['year'], comp['rainy_pct'], marker='s',
                     label=f'Common stations (n={len(common_ids)})')
        axes[0].set(ylabel='Rainy observed intervals (%)', title='A. Rainfall occurrence')
        axes[1].plot(annual['year'], annual['recorded_rainfall_mm'] /
                     annual['observed_intervals'], marker='o', label='All reporting stations')
        axes[1].plot(comp['year'], comp['mean_mm'], marker='s',
                     label=f'Common stations (n={len(common_ids)})')
        axes[1].set(ylabel='Mean rainfall per observed interval (mm)',
                    title='B. Mean rainfall')
        for ax in axes:
            ax.set(xlabel='Year')
            ax.set_xticks(annual['year'])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle('Annual sensitivity check: fixed station set (missingness remains)')
        fig.tight_layout()
        save(fig, out, 'fig_4_2_common_station_comparison')
        common[['year', 'station_id', 'calendar_coverage_pct',
                'observed_intervals']].to_csv(out / 'common_station_coverage.csv', index=False)

    # 4.2: Full-calendar station-year coverage distribution (quality context).
    fig, ax = plt.subplots(figsize=(9, 4.5))
    groups = [station_year.loc[station_year['year'] == y, 'calendar_coverage_pct'].dropna()
              for y in sorted(station_year['year'].unique())]
    ax.boxplot(groups, tick_labels=[str(y) for y in sorted(station_year['year'].unique())],
               showfliers=False)
    ax.axhline(90, linestyle='--', linewidth=1, label='90% calendar coverage')
    ax.set(xlabel='Year', ylabel='Station-year calendar coverage (%)',
           title='Station reporting coverage varies substantially across years')
    ax.legend(loc='lower right')
    save(fig, out, 'fig_4_2_station_coverage')

    # 4.3: Aggregate monthly rates using summed numerators and denominators.
    month = weighted_rate(monthly, 'month').set_index('month').reindex(range(1,13))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(range(1,13), month['rate_pct'], marker='o')
    ax.set_xticks(range(1,13), MONTHS)
    ax.set(xlabel='Month', ylabel='Rainy intervals (%)',
           title='Monthly rainfall occurrence across 2017–2024 (observed network)')
    ax.grid(axis='y', alpha=.25)
    save(fig, out, 'fig_4_3_monthly_occurrence')

    # 4.3: Year-month occurrence heatmap.
    mm = monthly.copy()
    mm['rainy_pct'] = 100 * mm['rainy_intervals'] / mm['observed_intervals']
    heat = mm.pivot(index='year', columns='month', values='rainy_pct').reindex(columns=range(1,13))
    fig, ax = plt.subplots(figsize=(11, 4.7))
    im = ax.imshow(heat.values, aspect='auto', cmap='Blues')
    ax.set_xticks(range(12), MONTHS)
    ax.set_yticks(range(len(heat)), heat.index.astype(str))
    ax.set(xlabel='Month', ylabel='Year', title='Rainfall occurrence by year and month')
    fig.colorbar(im, ax=ax, label='Rainy intervals (%)')
    save(fig, out, 'fig_4_3_year_month_heatmap')

    # 4.3: Coverage-aware recorded monthly accumulation, not extrapolated totals.
    # Summarize station-month totals only when >= threshold of expected slots exist.
    good_months = station_month.loc[
        station_month['calendar_coverage_pct'] >= args.min_monthly_coverage].copy()
    if not good_months.empty:
        month_summary = (good_months.groupby('month')['recorded_rainfall_mm']
                         .agg(median_mm='median', q25_mm=lambda x: x.quantile(.25),
                              q75_mm=lambda x: x.quantile(.75), n_station_months='size')
                         .reindex(range(1, 13)))
        month_summary.to_csv(out / 'coverage_filtered_monthly_accumulation.csv',
                             index_label='month')
        fig, ax = plt.subplots(figsize=(9, 4.7))
        x = np.arange(1, 13)
        ax.plot(x, month_summary['median_mm'], marker='o', label='Median station-month')
        ax.fill_between(x, month_summary['q25_mm'].to_numpy(dtype=float),
                        month_summary['q75_mm'].to_numpy(dtype=float), alpha=.2,
                        label='25th–75th percentile')
        ax.set_xticks(x, MONTHS)
        ax.set(xlabel='Month', ylabel='Recorded station-month rainfall (mm)',
               title=f'Monthly rainfall accumulation (station-month coverage ≥{args.min_monthly_coverage:g}%)')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=.2)
        save(fig, out, 'fig_4_3_monthly_accumulation')

    # 4.4: Diurnal profiles.
    hour = hourly.groupby('hour', as_index=False)[['observed_intervals', 'rainy_intervals',
                                                   'recorded_rainfall_mm']].sum().sort_values('hour')
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(hour['hour'], 100*hour['rainy_intervals']/hour['observed_intervals'], marker='o')
    axes[0].set(title='A. Rainfall occurrence', ylabel='Rainy intervals (%)')
    axes[1].plot(hour['hour'], hour['recorded_rainfall_mm']/hour['observed_intervals'], marker='o')
    axes[1].set(title='B. Mean observed rainfall', ylabel='Mean rainfall per interval (mm)')
    for ax in axes:
        ax.set(xlabel='Hour of day (Singapore time)', xlim=(0,23))
        ax.set_xticks(range(0,24,3))
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Diurnal rainfall characteristics across 2017–2024')
    fig.tight_layout()
    save(fig, out, 'fig_4_4_diurnal_patterns')

    # 4.5: Compare station-level annual occurrence on a common eligible-year set.
    # A year must be represented at every plotted station, so differences in years
    # contributing to the map cannot explain spatial variation.
    eligible = station_year.loc[
        station_year['calendar_coverage_pct'] >= args.min_spatial_coverage].copy()
    year_counts = eligible.groupby('station_id')['year'].nunique()
    keep = year_counts[year_counts >= args.min_spatial_years].index
    eligible = eligible[eligible['station_id'].isin(keep)].copy()
    if eligible.empty:
        print('WARNING: No stations meet spatial criteria; skipping spatial map.')
    else:
        # Select the longest common period retaining at least 40 stations.
        # If no multi-year period qualifies, use the best available pair.
        from itertools import combinations
        candidate_years = sorted(int(y) for y in eligible['year'].unique())
        year_sets = {y: set(eligible.loc[eligible['year'] == y, 'station_id'])
                     for y in candidate_years}
        options = []
        for k in range(2, len(candidate_years) + 1):
            for years in combinations(candidate_years, k):
                ids = set.intersection(*(year_sets[y] for y in years))
                if ids:
                    options.append((years, ids))
        feasible = [(years, ids) for years, ids in options if len(ids) >= 40]
        pool = feasible if feasible else options
        if not pool:
            raise ValueError('No stations have two eligible years for spatial comparison')
        best_years, common_ids = max(pool, key=lambda item: (len(item[0]), len(item[1]))) \
            if feasible else max(pool, key=lambda item: (len(item[1]), len(item[0])))
        best_stations = sorted(common_ids)
        balanced = eligible.loc[eligible['year'].isin(best_years) &
                                eligible['station_id'].isin(best_stations)].copy()
        balanced['annual_rainy_pct'] = 100 * balanced['rainy_intervals'] / balanced['observed_intervals']
        by_station = balanced.groupby('station_id', as_index=False).agg(
            rainy_pct=('annual_rainy_pct', 'mean'),
            eligible_years=('year', 'nunique'))
        coords = (locations.sort_values('year').drop_duplicates('station_id', keep='last')
                  [['station_id', 'latitude', 'longitude']])
        points = by_station.merge(coords, on='station_id', how='inner')
        points.to_csv(out / 'spatial_balanced_station_summary.csv', index=False)
        fig, ax = plt.subplots(figsize=(9, 6))
        scatter = ax.scatter(points['longitude'], points['latitude'], c=points['rainy_pct'],
                             s=55, cmap='viridis', edgecolors='white', linewidths=.4)
        fig.colorbar(scatter, ax=ax, label='Mean annual rainy intervals (%)')
        ax.set(xlabel='Longitude (°E)', ylabel='Latitude (°N)',
               title=f'Station rainfall occurrence: common eligible years {sorted(best_years)}')
        ax.text(.01, .01,
                f'{len(points)} stations; ≥{args.min_spatial_coverage:g}% annual coverage; no basemap',
                transform=ax.transAxes, fontsize=8, va='bottom')
        save(fig, out, 'fig_4_5_station_spatial_occurrence')
        print(f'Spatial balanced comparison: {len(points)} stations, years {sorted(best_years)}')

    # Diagnostics: count eligible station-years and months, useful for interpreting plots.
    counts = (station_year.assign(eligible=station_year['calendar_coverage_pct'] >= args.min_spatial_coverage)
              .groupby('year', as_index=False).agg(stations=('station_id','nunique'),
                                                   eligible_station_years=('eligible','sum')))
    counts.to_csv(out / 'spatial_coverage_selection.csv', index=False)
    print('\nStation-year coverage selection:')
    print(counts.to_string(index=False))
    print(f'Loaded {len(station_month)} station-month rows, {len(percentiles)} percentile rows.')
    print(f'All figures saved to {out}')


if __name__ == '__main__':
    main()
