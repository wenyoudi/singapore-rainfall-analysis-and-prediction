# Real rainfall observations for the interface demonstration

**Demonstration only — not a training or evaluation dataset.** The reported
models and metrics retain their original experimental datasets. This small,
convenient selection is not representative of rainfall event frequency or model
performance. No reported metrics were recalculated from it.

Contains information from NEA's **Historical Rainfall across Singapore (2024)**,
obtained from [data.gov.sg](https://data.gov.sg/collections/2279/view), made available
under the [Singapore Open Data Licence version 1.0](https://data.gov.sg/open-data-licence).
The subset was derived on 18 October 2026 from the project's existing local annual
CSV. No readings were simulated or missing values filled.

| Station | Name | Readings | Readings with rainfall >0 mm |
| --- | --- | ---: | ---: |
| S24 | Upper Changi Road North | 576 | 67 |
| S33 | Jurong Pier Road | 576 | 14 |
| S50 | Clementi Road | 576 | 27 |
| S104 | Woodlands Avenue 9 | 576 | 5 |
| S109 | Ang Mo Kio Avenue 5 | 576 | 40 |
| S115 | Tuas South Avenue 3 | 576 | 5 |
| S116 | West Coast Highway | 576 | 11 |
| S117 | Banyan Road | 576 | 18 |

Each station covers **1 January 2024 00:00 through 2 January 2024 23:55 SGT**,
with complete five-minute observations at all eight stations: **4,608 rows total**.
There is one small Snappy Parquet file under `rainfall_2024/` (approximately 28 KiB).

- Prediction timestamps: 1 January 01:00 through 2 January 23:55, every five minutes.
- Optional subsequent-rainfall verification: through 2 January 23:25.
- Earlier times lack the required preceding hour; later verification windows
  extend beyond the supplied observations. The interface handles these explicitly.
- Defaults: first station S24, 1 January 2024 at 00:00 SGT, and first threshold
  0 mm. This first midnight lacks history; choose 01:00 or later on 1 January,
  or midnight on 2 January, to make a prediction.
- Optional wet-weather query: S24, 1 January 2024 at 14:10 SGT, threshold 5 mm.
  The saved GBT displays 54.9%; observed subsequent rainfall is 19.60 mm.
- Counterexample: S50 at 13:50 on the same day, threshold 5 mm, predicts 67.9%,
  but subsequent rainfall is only 0.40 mm. The exceedance event does not occur.
- Dry-weather comparison: S33, 1 January 2024 at 01:00 SGT, threshold 5 mm.
  The saved GBT displays 3.4%; subsequent observed rainfall totals 0 mm.

These selected cases illustrate individual outcomes, not model performance.

## Provenance and rebuilding

`src/build_interface_demo.py` streams the local 2024 source CSV, selects these
stations/dates, calls the existing `data_cleaning.prepare_records` and
`deduplicate_records` functions, then applies the committed full-data station-name
mapping. This reproduces the observation-cleaning rules without modifying any
full-data outputs or experiment summaries. The selection removed no duplicates.
`manifest.json` records coverage, source, cleaning functions and a SHA-256 digest
of the selected raw rows. The digest is of canonical JSON rows in source order,
not the entire annual CSV. Parquet filenames/bytes can vary across rebuilds.

With the original CSV at `data/raw/HistoricalRainfallacrossSingapore2024.csv`:

```bash
SPARK_LOCAL_IP=127.0.0.1 .venv-interface/bin/python src/build_interface_demo.py
```

Rebuilding is optional; the included Parquet works without the raw file. Use the
[full-data instructions](../../docs/prediction_interface.md#connect-the-full-public-dataset)
to obtain and clean 2017–2024 data. Set `RAINFALL_DATA_ROOT` to connect it; the
saved model and inference method are identical for demo and full data.
