from pathlib import Path
import csv

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "profiling"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2017, 2025)


# ============================================================
# 2. Initialize Spark
# ============================================================

spark = (
    SparkSession.builder
    .appName("RainfallProfiling")
    .config("spark.sql.shuffle.partitions", "32")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# 3. Helper: Load annual dataset
# ============================================================

def load_data(year):

    file_path = (
        DATA_DIR /
        f"HistoricalRainfallacrossSingapore{year}.csv"
    )

    print(f"\nLoading {year} dataset...")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(str(file_path))
    )

    return df


# ============================================================
# 4. Profile one year
# ============================================================

def profile_year(df, year):

    # Preserve original data; casts are for analysis only.
    df = (
        df
        .withColumn(
            "observation_time",
            F.to_timestamp("timestamp")
        )
        .withColumn(
            "rainfall",
            F.col("reading_value").cast("double")
        )
    )

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    basic = df.agg(
        F.count("*").alias("total_rows"),

        F.countDistinct("station_id").alias("stations"),

        F.min("observation_time").alias("first_timestamp"),

        F.max("observation_time").alias("last_timestamp"),

        F.min("rainfall").alias("min_rainfall"),

        F.max("rainfall").alias("max_rainfall"),

        F.avg("rainfall").alias("mean_rainfall"),

        F.sum(
            F.when(F.col("rainfall") < 0, 1).otherwise(0)
        ).alias("negative_rainfall"),

        F.sum(
            F.when(
                F.col("reading_value").isNotNull()
                & F.col("rainfall").isNull(),
                1
            ).otherwise(0)
        ).alias("invalid_rainfall"),

        F.sum(
            F.when(
                F.col("timestamp").isNotNull()
                & F.col("observation_time").isNull(),
                1
            ).otherwise(0)
        ).alias("invalid_timestamps")
    ).first().asDict()

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_expressions = []

    for column in df.columns:

        if column in ["observation_time", "rainfall"]:
            continue

        missing_expressions.append(
            F.sum(
                F.when(
                    F.col(column).isNull()
                    | (F.trim(F.col(column)) == ""),
                    1
                ).otherwise(0)
            ).alias(column)
        )

    missing = (
        df.agg(*missing_expressions)
        .first()
        .asDict()
    )

    # --------------------------------------------------------
    # Duplicate station-timestamp pairs
    # --------------------------------------------------------

    # Retain only the columns needed for duplicate detection
    duplicate_input = df.select("station_id", "timestamp")

    # Distribute records across partitions before aggregation
    duplicate_input = duplicate_input.repartition(
        32, "station_id", "timestamp"
    )

    # Count occurrences of each station-timestamp pair
    duplicate_counts = (
        duplicate_input
        .groupBy("station_id", "timestamp")
        .count()
    )

    # Calculate duplicate statistics
    duplicate_stats = (
        duplicate_counts
        .agg(
            F.sum(
                F.when(F.col("count") > 1, 1).otherwise(0)
            ).alias("duplicate_groups"),

            F.sum(
                F.when(
                    F.col("count") > 1,
                    F.col("count") - 1
                ).otherwise(0)
            ).alias("duplicate_extra_rows")
        )
        .first()
    )

    # --------------------------------------------------------
    # Timestamp gaps
    # --------------------------------------------------------

    window = (
        Window
        .partitionBy("station_id")
        .orderBy("observation_time")
    )

    gaps = (
        df
        .filter(
            F.col("station_id").isNotNull()
            & F.col("observation_time").isNotNull()
        )
        .select("station_id", "observation_time")
        .distinct()
        .withColumn(
            "previous_timestamp",
            F.lag("observation_time").over(window)
        )
        .withColumn(
            "gap_seconds",
            F.col("observation_time").cast("long")
            - F.col("previous_timestamp").cast("long")
        )
    )

    gap_stats = gaps.agg(
        F.sum(
            F.when(F.col("gap_seconds") > 300, 1).otherwise(0)
        ).alias("gaps_over_5min"),

        F.sum(
            F.when(
                (F.col("gap_seconds") > 0)
                & (F.col("gap_seconds") < 300),
                1
            ).otherwise(0)
        ).alias("gaps_under_5min"),

        F.max("gap_seconds").alias("max_gap_seconds")
    ).first().asDict()

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    return {
        "year": year,
        **basic,
        "duplicate_groups": duplicate_stats["duplicate_groups"],
        "duplicate_extra_rows": duplicate_stats["duplicate_extra_rows"],
        **gap_stats,
        **{
            f"missing_{key}": value
            for key, value in missing.items()
        }
    }


# ============================================================
# 5. Save results
# ============================================================

def save_summary(results):

    output_path = OUTPUT_DIR / "annual_summary.csv"

    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved profiling summary to {output_path}")


# ============================================================
# 6. Main
# ============================================================

def main():

    results = []

    for year in YEARS:

        df = load_data(year)

        print(f"Profiling {year}...")

        result = profile_year(df, year)

        results.append(result)

        # Save completed years immediately
        save_summary(results)

        print(
            f"{year}: "
            f"{result['total_rows']:,} rows, "
            f"{result['stations']} stations"
        )

    spark.stop()


if __name__ == "__main__":
    main()