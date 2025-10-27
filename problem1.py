from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, count, rand
import os
import shutil
import tempfile
import glob

def write_single_csv(df, output_path):
    """
    Writes a DataFrame to a single flat CSV file (not a directory).
    """
    tmp_dir = tempfile.mkdtemp()
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(tmp_dir)
    # Move the generated CSV to the desired flat path
    for file in os.listdir(tmp_dir):
        if file.endswith(".csv"):
            shutil.move(os.path.join(tmp_dir, file), output_path)
    shutil.rmtree(tmp_dir)

def main():
    # Output directory and cleanup
    output_dir = "data/output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Spark session
    spark = SparkSession.builder.appName("Problem1_LogAnalysis").getOrCreate()

    # Read all log files as a DataFrame (each line = one log entry)
    log_path = "s3a://ahs131-assignment-spark-cluster-logs/data/*/*"
    logs_df = spark.read.text(log_path).withColumnRenamed("value", "log_entry")

    # Regex pattern to extract log level (INFO, WARN, ERROR, DEBUG)
    pattern = r"^\d{2}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2}\s+(INFO|WARN|ERROR|DEBUG)"

    logs_df = logs_df.withColumn(
        "log_level",
        regexp_extract(col("log_entry"), pattern, 1)
    )

    # Filter out rows without valid log levels
    logs_with_levels_df = logs_df.filter(col("log_level") != "")

    logs_with_levels_df.show(10, truncate=False)
    print(f"Matched {logs_with_levels_df.count()} lines with log levels")

    # --- 1. COUNT EACH LOG LEVEL ---
    counts_df = (
        logs_with_levels_df.groupBy("log_level")
        .agg(count("*").alias("count"))
        .orderBy("log_level")
    )
    counts_output = os.path.join(output_dir, "problem1_counts.csv")
    write_single_csv(counts_df, counts_output)

    # --- 2. RANDOM SAMPLE OF 10 LOG ENTRIES ---
    sample_df = logs_with_levels_df.orderBy(rand()).limit(10)
    sample_output = os.path.join(output_dir, "problem1_sample.csv")
    write_single_csv(sample_df.select("log_entry", "log_level"), sample_output)

    # --- 3. SUMMARY STATISTICS ---
    total_lines = logs_df.count()
    total_with_levels = logs_with_levels_df.count()
    unique_levels = counts_df.count()
    counts = {row['log_level']: row['count'] for row in counts_df.collect()}
    total_count = sum(counts.values())

    # Prepare summary text
    summary_lines = [
        f"Total log lines processed: {total_lines:,}",
        f"Total lines with log levels: {total_with_levels:,}",
        f"Unique log levels found: {unique_levels}\n",
        "Log level distribution:"
    ]
    for level, cnt in counts.items():
        pct = (cnt / total_count) * 100 if total_count else 0
        summary_lines.append(f"  {level:<6}: {cnt:>10,} ({pct:6.2f}%)")

    summary_text = "\n".join(summary_lines)
    summary_output = os.path.join(output_dir, "problem1_summary.txt")
    with open(summary_output, "w") as f:
        f.write(summary_text + "\n")

    spark.stop()


if __name__ == "__main__":
    main()
