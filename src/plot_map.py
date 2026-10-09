#!/usr/bin/env python3
"""Two-panel Singapore rainfall map: occurrence and mean five-minute rainfall.

Local plotting only; no Spark required.
Inputs:
  outputs/figures/eda/spatial_balanced_station_summary.csv
  outputs/eda/station_year_rainfall_stats.csv
Uses the SAME 56 stations and 2021-2024 eligible station-years in both panels.

Install: pip install pandas numpy matplotlib scipy geopandas shapely pyproj requests
Run: python src/plot_singapore_rainfall_map.py
Optional: --boundary data/geography/singapore_adm0.geojson
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.prepared import prep

API = 'https://www.geoboundaries.org/api/current/gbOpen/SGP/ADM0/'
CRS = 'EPSG:3414'
YEARS = (2021, 2022, 2023, 2024)


def download_boundary(path: Path) -> Path:
    if path.is_file():
        return path
    print('Downloading Singapore ADM0 boundary from geoBoundaries...', flush=True)
    try:
        response = requests.get(API, timeout=35)
        response.raise_for_status()
        metadata = response.json()
        url = metadata.get('simplifiedGeometryGeoJSON') or metadata.get('gjDownloadURL')
        if not url:
            raise RuntimeError('No GeoJSON download URL returned')
        geo = requests.get(url, timeout=90)
        geo.raise_for_status()
        payload = geo.json()
        if payload.get('type') not in ('FeatureCollection', 'Feature', 'Polygon', 'MultiPolygon'):
            raise ValueError('Unexpected GeoJSON format')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding='utf-8')
        (path.parent / 'singapore_boundary_source.json').write_text(
            json.dumps({k: metadata.get(k) for k in ('boundaryName', 'boundarySource',
                'boundaryLicense', 'boundaryYearRepresented', 'boundaryID',
                'gjDownloadURL', 'simplifiedGeometryGeoJSON')}, indent=2), encoding='utf-8')
        return path
    except Exception as exc:
        raise RuntimeError('Could not download boundary; use --boundary PATH to a Singapore GeoJSON. '
                           f'Original error: {exc}') from exc


def idw(tree, values, targets, power, neighbors):
    distances, indices = tree.query(targets, k=min(neighbors, len(values)))
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    exact = distances < 1e-8
    weights = np.maximum(distances, 1e-8) ** (-power)
    estimate = np.sum(weights * values[indices], axis=1) / np.sum(weights, axis=1)
    exact_rows = np.flatnonzero(exact.any(axis=1))
    if len(exact_rows):
        estimate[exact_rows] = values[indices[exact_rows, exact[exact_rows].argmax(axis=1)]]
    return estimate


def load_station_metrics(points_path, station_year_path, min_coverage):
    points = pd.read_csv(points_path)
    yearly = pd.read_csv(station_year_path)
    required_points = {'station_id', 'rainy_pct', 'latitude', 'longitude'}
    required_yearly = {'station_id', 'year', 'calendar_coverage_pct',
                       'mean_mm_per_observed_interval', 'rainy_intervals', 'observed_intervals'}
    if required_points - set(points.columns) or required_yearly - set(yearly.columns):
        raise ValueError('Input CSV columns do not match the EDA pipeline. '
                         f'Missing station columns: {sorted(required_points - set(points.columns))}; '
                         f'missing annual columns: {sorted(required_yearly - set(yearly.columns))}')
    if points['station_id'].duplicated().any() or len(points) < 3:
        raise ValueError('Spatial summary must contain at least 3 distinct stations')
    yearly = yearly[yearly['year'].isin(YEARS) &
                    yearly['station_id'].isin(points['station_id'])].copy()
    if yearly.duplicated(['station_id', 'year']).any():
        raise ValueError('Duplicate station-year records')
    coverage = yearly.groupby('station_id')['year'].nunique()
    complete = coverage[coverage == len(YEARS)].index
    yearly = yearly[yearly['station_id'].isin(complete) &
                    (yearly['calendar_coverage_pct'] >= min_coverage)].copy()
    valid = yearly.groupby('station_id')['year'].nunique()
    valid = valid[valid == len(YEARS)].index
    points = points[points['station_id'].isin(valid)].copy()
    yearly = yearly[yearly['station_id'].isin(valid)].copy()
    if len(points) < 3:
        raise ValueError('Too few stations satisfy the shared four-year coverage criterion')
    yearly['annual_rainy_pct'] = 100 * yearly['rainy_intervals'] / yearly['observed_intervals']
    metrics = yearly.groupby('station_id', as_index=False).agg(
        rainy_pct_check=('annual_rainy_pct', 'mean'),
        mean_mm_per_interval=('mean_mm_per_observed_interval', 'mean'),
        eligible_years=('year', 'nunique'))
    points = points.merge(metrics, on='station_id', validate='one_to_one')
    max_diff = (points['rainy_pct'] - points['rainy_pct_check']).abs().max()
    if max_diff > 1e-5:
        raise ValueError(f'Occurrence summary differs from annual data (max difference {max_diff:.4g}); '
                         'regenerate spatial_balanced_station_summary.csv with the current plotting pipeline')
    for col in ('rainy_pct', 'mean_mm_per_interval', 'latitude', 'longitude'):
        if not np.isfinite(points[col].to_numpy(dtype=float)).all():
            raise ValueError(f'Non-finite values in {col}')
    return points


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root / 'outputs/figures/eda/spatial_balanced_station_summary.csv')
    parser.add_argument('--station-year', type=Path, default=root / 'outputs/eda/station_year_rainfall_stats.csv')
    parser.add_argument('--boundary', type=Path, default=None)
    parser.add_argument('--cache', type=Path, default=root / 'data/geography/singapore_adm0.geojson')
    parser.add_argument('--output', type=Path, default=root / 'outputs/figures/eda')
    parser.add_argument('--min-coverage', type=float, default=80.0)
    parser.add_argument('--resolution', type=int, default=420)
    parser.add_argument('--power', type=float, default=2.0)
    parser.add_argument('--neighbors', type=int, default=8)
    parser.add_argument('--max-distance-km', type=float, default=12.0,
                        help='Mask cells farther than this from any station; 0 disables')
    args = parser.parse_args()
    if args.power <= 0 or args.neighbors < 1 or args.resolution < 100 or args.max_distance_km < 0:
        parser.error('power > 0, neighbors >= 1, resolution >= 100, max-distance-km >= 0 required')
    for path in (args.input, args.station_year):
        if not path.is_file():
            parser.error(f'Missing {path}')
    points = load_station_metrics(args.input, args.station_year, args.min_coverage)
    boundary_path = args.boundary or download_boundary(args.cache)
    if not boundary_path.is_file():
        parser.error(f'Boundary not found: {boundary_path}')
    border = gpd.read_file(boundary_path)
    if border.crs is None:
        border = border.set_crs('EPSG:4326')
    border = border.to_crs(CRS)
    geometry = border.geometry.union_all() if hasattr(border.geometry, 'union_all') else border.geometry.unary_union
    stations = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points.longitude, points.latitude),
                                crs='EPSG:4326').to_crs(CRS)
    xy = np.column_stack((stations.geometry.x, stations.geometry.y))
    tree = cKDTree(xy)
    minx, miny, maxx, maxy = border.total_bounds
    padding = 3000
    minx -= padding; maxx += padding; miny -= padding; maxy += padding
    nx = args.resolution
    ny = max(90, int(nx * (maxy-miny)/(maxx-minx)))
    xx, yy = np.meshgrid(np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    prepared = prep(geometry)
    inside = np.fromiter((prepared.covers(Point(x, y)) for x, y in grid), dtype=bool, count=len(grid))
    targets = grid[inside]
    if not len(targets):
        raise ValueError('Boundary contains no grid cells: verify its coordinate system')
    near = tree.query(targets, k=1)[0]
    valid = np.ones(len(targets), dtype=bool)
    if args.max_distance_km:
        valid = near <= args.max_distance_km * 1000

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), layout='constrained')
    fig.patch.set_facecolor('white')
    cmap = plt.get_cmap('YlGnBu').copy()
    cmap.set_bad(alpha=0)
    specifications = [
        ('rainy_pct', 'A. Rainfall occurrence', 'Rainy five-minute intervals (%)'),
        ('mean_mm_per_interval', 'B. Mean rainfall', 'Mean rainfall per five-minute interval (mm)'),
    ]
    for ax, (column, title, colorbar_label) in zip(axes, specifications):
        values = stations[column].to_numpy(dtype=float)
        result = np.full(len(grid), np.nan)
        interp = idw(tree, values, targets, args.power, args.neighbors)
        interp[~valid] = np.nan
        result[inside] = interp
        surface = np.ma.masked_invalid(result.reshape(ny, nx))
        ax.set_facecolor('#edf4f7')
        im = ax.imshow(surface, origin='lower', extent=(minx, maxx, miny, maxy),
                       cmap=cmap, vmin=values.min(), vmax=values.max(), interpolation='bilinear')
        border.boundary.plot(ax=ax, color='#283746', linewidth=0.7, zorder=3)
        ax.scatter(xy[:, 0], xy[:, 1], s=12, facecolors='none', edgecolors='white',
                   linewidths=0.65, zorder=4)
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.015, fraction=0.045)
        cb.set_label(colorbar_label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_aspect('equal')
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        ax.set_axis_off()  # removes Easting/Northing labels and unnecessary ticks
    fig.suptitle('Spatial rainfall patterns across Singapore (2021–2024)', fontsize=15, weight='bold')
    args.output.mkdir(parents=True, exist_ok=True)
    name = 'fig_4_5_singapore_rainfall_two_panel'
    for ext in ('png', 'pdf'):
        path = args.output / f'{name}.{ext}'
        fig.savefig(path, dpi=260, bbox_inches='tight', facecolor='white')
        print('Saved:', path)
    plt.close(fig)
    print(f'Included {len(points)} stations; common years {YEARS}; '
          f'coverage >= {args.min_coverage:g}%; IDW p={args.power:g}, k={args.neighbors}.')
    print('Note: map colours are interpolated estimates, not direct observations; '
          'boundary from geoBoundaries gbOpen (CC BY 4.0).')


if __name__ == '__main__':
    main()
