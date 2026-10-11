#!/usr/bin/env python3
"""Two-panel Singapore rainfall map: occurrence and mean five-minute rainfall.

Local plotting only; no Spark required.
Inputs:
  outputs/figures/eda/spatial_balanced_station_summary.csv
  outputs/eda/station_year_rainfall_stats.csv
Uses a shared, explicit period and coverage rule for both panels.
Station counts are derived from the input summaries, never hard-coded.

Install: pip install -r requirements-plots.txt
Run: python src/plot_map.py
Optional: --boundary data/geography/singapore_adm0.geojson
"""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from task1_analysis import SPATIAL_YEARS, spatial_summary
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.prepared import prep

CRS = 'EPSG:3414'


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


def load_station_metrics(points_path, station_year_path, min_coverage, years=SPATIAL_YEARS):
    points = pd.read_csv(points_path)
    expected, _ = spatial_summary(pd.read_csv(station_year_path), years, min_coverage)
    if points['station_id'].duplicated().any() or len(expected) < 3:
        raise ValueError('Spatial summary must contain at least 3 distinct eligible stations')
    if set(points.station_id) != set(expected.station_id):
        raise ValueError('Spatial station set differs; regenerate plots with matching years and coverage')
    if 'analysis_years' not in points or set(points.analysis_years) != {','.join(map(str, years))}:
        raise ValueError('Spatial analysis period differs; regenerate the EDA plots')
    columns = ['rainy_pct', 'mean_mm_per_interval', 'latitude', 'longitude']
    actual = points.set_index('station_id').sort_index()[columns].to_numpy(dtype=float)
    reference = expected.set_index('station_id').sort_index()[columns].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.allclose(actual, reference, atol=1e-8, rtol=0):
        raise ValueError('Spatial summary differs from annual data; regenerate the EDA plots')
    return expected


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root / 'outputs/figures/eda/spatial_balanced_station_summary.csv')
    parser.add_argument('--station-year', type=Path, default=root / 'outputs/eda/station_year_rainfall_stats.csv')
    parser.add_argument('--boundary', type=Path, default=root / 'data/geography/singapore_adm0.geojson')
    parser.add_argument('--spatial-years', type=int, nargs='+', default=SPATIAL_YEARS)
    parser.add_argument('--output', type=Path, default=root / 'outputs/figures/eda')
    parser.add_argument('--min-coverage', type=float, default=80.0)
    parser.add_argument('--resolution', type=int, default=420)
    parser.add_argument('--power', type=float, default=2.0)
    parser.add_argument('--neighbors', type=int, default=8)
    parser.add_argument('--max-distance-km', type=float, default=12.0,
                        help='Mask cells farther than this from any station; 0 disables')
    args = parser.parse_args()
    if (not np.isfinite([args.power, args.max_distance_km, args.min_coverage]).all() or
            not 0 <= args.min_coverage <= 100 or args.power <= 0 or args.neighbors < 1 or
            args.resolution < 100 or args.max_distance_km < 0):
        parser.error('power > 0, neighbors >= 1, resolution >= 100, max-distance-km >= 0 required')
    for path in (args.input, args.station_year):
        if not path.is_file():
            parser.error(f'Missing {path}')
    args.spatial_years = sorted(args.spatial_years)
    points = load_station_metrics(args.input, args.station_year, args.min_coverage, args.spatial_years)
    boundary_path = args.boundary
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
    period = ', '.join(map(str, args.spatial_years))
    fig.suptitle(f'Spatial rainfall patterns across Singapore ({period})\n'
                 f'{len(points)} stations; ≥{args.min_coverage:g}% coverage in every year; '
                 f'IDW power {args.power:g}, {args.neighbors} neighbors', fontsize=13, weight='bold')
    args.output.mkdir(parents=True, exist_ok=True)
    name = 'fig_4_5_singapore_rainfall_two_panel'
    for ext in ('png', 'pdf'):
        path = args.output / f'{name}.{ext}'
        fig.savefig(path, dpi=260, bbox_inches='tight', facecolor='white')
        print('Saved:', path)
    plt.close(fig)
    print(f'Included {len(points)} stations; common years {args.spatial_years}; '
          f'coverage >= {args.min_coverage:g}%; IDW p={args.power:g}, k={args.neighbors}.')
    source_path = boundary_path.parent / 'singapore_boundary_source.json'
    source = json.loads(source_path.read_text()) if source_path.is_file() else {}
    manifest = {'spatial_years': args.spatial_years, 'min_coverage_pct': args.min_coverage,
                'station_ids': points.station_id.tolist(), 'idw_power': args.power,
                'idw_neighbors': min(args.neighbors, len(points)), 'resolution': args.resolution,
                'max_distance_km': args.max_distance_km, 'crs': CRS,
                'boundary_path': str(boundary_path.resolve()), 'boundary_metadata': source,
                'boundary_sha256': hashlib.sha256(boundary_path.read_bytes()).hexdigest()}
    (args.output / 'spatial_map_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Note: map colours are interpolated estimates, not direct observations.')
    print('Boundary license:', source.get('boundaryLicense', 'See boundary source license'))


if __name__ == '__main__':
    main()
