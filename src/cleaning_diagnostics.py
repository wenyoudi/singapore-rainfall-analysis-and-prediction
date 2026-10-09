"""Read-only cleaning diagnostics for Singapore rainfall data (Spark 3.3.1)."""
from pathlib import Path
import csv
from pyspark.sql import SparkSession, functions as F

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "raw"
OUTPUT = ROOT / "outputs" / "cleaning_diagnostics"
OUTPUT.mkdir(parents=True, exist_ok=True)
YEARS = range(2017, 2025)
SAMPLE_LIMIT = 100


def save_csv(name, rows, columns):
    path = OUTPUT / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    print("Saved", path, flush=True)


def main():
    spark = (SparkSession.builder.appName("RainfallCleaningDiagnostics")
             .config("spark.sql.shuffle.partitions", "32")
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    yearly = []
    offsets = []
    collision_examples = []
    station_variants = []
    try:
        for year in YEARS:
            print("Analyzing", year, flush=True)
            path = INPUT / ("HistoricalRainfallacrossSingapore%d.csv" % year)
            if not path.is_file():
                raise FileNotFoundError(str(path))
            raw = (spark.read.option("header", True)
                   .option("inferSchema", False).csv(str(path)))
            required = ["station_id", "station_name", "timestamp", "reading_value",
                        "location_latitude", "location_longitude"]
            missing = set(required) - set(raw.columns)
            if missing:
                raise ValueError("Missing columns in %s: %s" % (path, sorted(missing)))

            df = (raw.select(*required)
                  .withColumn("ts", F.to_timestamp("timestamp"))
                  .withColumn("rainfall", F.col("reading_value").cast("double"))
                  .where(F.col("ts").isNotNull() & F.col("station_id").isNotNull())
                  .withColumn("epoch", F.col("ts").cast("long"))
                  .withColumn("offset_seconds", F.expr("pmod(epoch, 300)"))
                  # Nearest 5-minute boundary; exactly halfway rounds upward.
                  .withColumn("aligned_epoch", (F.floor((F.col("epoch") + 150) / 300) * 300).cast("long"))
                  .withColumn("shift_seconds", F.col("aligned_epoch") - F.col("epoch")))

            summary = df.agg(
                F.count("*").alias("valid_rows"),
                F.sum(F.when(F.col("offset_seconds") != 0, 1).otherwise(0)).alias("off_grid_rows"),
                F.sum(F.when(F.col("shift_seconds") != 0, 1).otherwise(0)).alias("shifted_rows"),
                F.max(F.abs(F.col("shift_seconds"))).alias("max_abs_shift_seconds")
            ).first().asDict()

            offset_rows = (df.groupBy("offset_seconds")
                           .agg(F.count("*").alias("rows"))
                           .orderBy("offset_seconds").collect())
            offsets.extend(dict(year=year, **r.asDict()) for r in offset_rows)

            # Check collisions *after* hypothetical alignment; never modify source.
            groups = (df.groupBy("station_id", "aligned_epoch")
                      .agg(F.count("*").alias("records"),
                           F.countDistinct("rainfall").alias("distinct_rainfall_values"),
                           F.min("rainfall").alias("min_rainfall"),
                           F.max("rainfall").alias("max_rainfall"),
                           F.min("ts").alias("first_original_ts"),
                           F.max("ts").alias("last_original_ts"))
                      .where(F.col("records") > 1))
            collision_stats = groups.agg(
                F.count("*").alias("collision_groups"),
                F.coalesce(F.sum(F.col("records") - 1), F.lit(0)).alias("extra_rows_if_aligned"),
                F.coalesce(F.sum(F.when(F.col("distinct_rainfall_values") > 1, 1).otherwise(0)), F.lit(0)).alias("conflicting_groups")
            ).first().asDict()
            yearly.append(dict(year=year, **summary, **collision_stats))
            examples = groups.orderBy(F.desc("records"), "station_id", "aligned_epoch").limit(SAMPLE_LIMIT).collect()
            for r in examples:
                d = r.asDict()
                d["year"] = year
                d["aligned_timestamp"] = str(__import__("datetime").datetime.fromtimestamp(
                    d.pop("aligned_epoch"), __import__("datetime").timezone.utc).astimezone(
                    __import__("datetime").timezone(__import__("datetime").timedelta(hours=8))))
                collision_examples.append(d)

            # Small per-year distinct metadata table, rather than collecting raw records.
            variants = (raw.groupBy("station_id", "station_name",
                                    "location_latitude", "location_longitude")
                        .agg(F.count("*").alias("observations"))
                        .collect())
            station_variants.extend(dict(year=year, **r.asDict()) for r in variants)
            print("Finished %d: %d off-grid rows; %d collision groups" % (
                year, summary["off_grid_rows"], collision_stats["collision_groups"]), flush=True)

        save_csv("timestamp_annual_summary.csv", yearly,
                 ["year", "valid_rows", "off_grid_rows", "shifted_rows", "max_abs_shift_seconds",
                  "collision_groups", "extra_rows_if_aligned", "conflicting_groups"])
        save_csv("timestamp_offset_distribution.csv", offsets,
                 ["year", "offset_seconds", "rows"])
        save_csv("alignment_collision_examples.csv", collision_examples,
                 ["year", "station_id", "aligned_timestamp", "records", "distinct_rainfall_values",
                  "min_rainfall", "max_rainfall", "first_original_ts", "last_original_ts"])
        save_csv("station_metadata_variants.csv", station_variants,
                 ["year", "station_id", "station_name", "location_latitude",
                  "location_longitude", "observations"])

        # Cross-year consistency of observed metadata, preserving exact original strings.
        by_station = {}
        for r in station_variants:
            sid = r["station_id"]
            item = by_station.setdefault(sid, {"years": set(), "names": set(), "coordinates": set()})
            item["years"].add(r["year"])
            item["names"].add(r["station_name"])
            item["coordinates"].add((r["location_latitude"], r["location_longitude"]))
        consistency = []
        for sid, info in sorted(by_station.items()):
            consistency.append({
                "station_id": sid,
                "years_present": ",".join(map(str, sorted(info["years"]))),
                "distinct_names": len(info["names"]),
                "distinct_coordinate_pairs": len(info["coordinates"]),
                "names": " | ".join(sorted(str(v) for v in info["names"])),
                "coordinates": " | ".join(sorted(str(v) for v in info["coordinates"]))
            })
        save_csv("station_consistency_summary.csv", consistency,
                 ["station_id", "years_present", "distinct_names", "distinct_coordinate_pairs",
                  "names", "coordinates"])
        print("Diagnostics complete. Raw data were not modified.", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
