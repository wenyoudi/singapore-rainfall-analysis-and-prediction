#!/usr/bin/env python3
"""Reproduce preprocessing and Task 1, or regenerate figures from saved summaries."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from importlib.metadata import version
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summaries-only', action='store_true',
                        help='Use existing summary CSVs; skip all Spark processing')
    parser.add_argument('--spark-submit', default='spark-submit')
    parser.add_argument('--spark-options', default='',
                        help='Options passed to spark-submit, e.g. "--master local[8] --driver-memory 24g"')
    parser.add_argument('--spatial-years', type=int, nargs='+', default=[2021, 2022, 2023, 2024])
    parser.add_argument('--min-spatial-coverage', type=float, default=80.)
    parser.add_argument('--min-monthly-coverage', type=float, default=90.)
    parser.add_argument('--coverage-thresholds', type=float, nargs='+', default=[80., 90.])
    args = parser.parse_args()
    if (len(set(args.spatial_years)) != len(args.spatial_years) or
            not set(args.spatial_years).issubset(range(2017, 2025))):
        parser.error('Spatial years must be unique study years in 2017..2024')
    if any(not 0 <= t <= 100 for t in [args.min_spatial_coverage, args.min_monthly_coverage,
                                      *args.coverage_thresholds]):
        parser.error('Coverage thresholds must be in 0..100')
    if not args.summaries_only and not shutil.which(args.spark_submit):
        parser.error('spark-submit not found; activate the Spark environment or use --summaries-only')
    commands = []
    if not args.summaries_only:
        for script in ['data_profiling.py', 'gap_analysis.py', 'cleaning_diagnostics.py',
                       'data_cleaning.py', 'cleaning_validation.py', 'rainfall_eda.py']:
            commands.append([args.spark_submit, *shlex.split(args.spark_options), str(ROOT / 'src' / script)])
    common_years = list(map(str, args.spatial_years))
    commands.extend([
        [sys.executable, str(ROOT / 'src/plot_data_quality.py')],
        [sys.executable, str(ROOT / 'src/plot_rainfall_eda.py'),
         '--spatial-years', *common_years, '--min-spatial-coverage', str(args.min_spatial_coverage),
         '--min-monthly-coverage', str(args.min_monthly_coverage),
         '--coverage-thresholds', *map(str, args.coverage_thresholds)],
        [sys.executable, str(ROOT / 'src/plot_map.py'), '--spatial-years', *common_years,
         '--min-coverage', str(args.min_spatial_coverage)],
    ])
    path = ROOT / 'outputs/task1_run_manifest.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'started_utc': datetime.now(timezone.utc).isoformat(),
                'mode': 'saved_summaries' if args.summaries_only else 'full_pipeline',
                'python': sys.version, 'parameters': vars(args), 'completed_commands': [],
                'plot_packages': {name: version(name) for name in
                                  ['pandas', 'numpy', 'matplotlib', 'scipy', 'geopandas', 'shapely', 'pyproj']},
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / 'src').glob('*.py'))},
                'status': 'running'}
    def save():
        path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    save()
    try:
        for command in commands:
            print('Running:', shlex.join(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            manifest['completed_commands'].append(command)
            save()
        manifest['status'] = 'complete'
    except (subprocess.CalledProcessError, OSError) as exc:
        manifest['status'] = 'failed'
        manifest['error'] = str(exc)
        raise
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main()
