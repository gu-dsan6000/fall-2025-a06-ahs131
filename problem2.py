#!/usr/bin/env python3
"""
problem2.py
Cluster Usage Analysis

Outputs (all written to data/output/ as flat files):
 - problem2_timeline.csv
 - problem2_cluster_summary.csv
 - problem2_stats.txt
 - problem2_bar_chart.png
 - problem2_density_plot.png

Usage:
  python problem2.py [--skip-spark]
"""
import os
import sys
import argparse
import glob
import shutil
import tempfile

# plotting libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# default plotting style (no explicit colors)
sns.set(style="whitegrid")

# helper to write single flat CSV from Spark DF (if used)
def write_single_csv_from_spark(df, out_path):
    """
    df: pyspark.sql.DataFrame
    out_path: final csv path (string)
    """
    tmp_dir = tempfile.mkdtemp()
    df.coalesce(1).write.option("header", True).mode("overwrite").csv(tmp_dir)
    # locate CSV and move
    moved = False
    for f in os.listdir(tmp_dir):
        if f.endswith(".csv"):
            shutil.move(os.path.join(tmp_dir, f), out_path)
            moved = True
            break
    shutil.rmtree(tmp_dir)
    if not moved:
        raise RuntimeError(f"No CSV produced in temp dir {tmp_dir}")

def write_single_csv_from_pandas(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

def ensure_output_clean(output_dir="data/output"):
    if os.path.exists(output_dir):
        for f in glob.glob(os.path.join(output_dir, "*")):
            try:
                os.remove(f)
            except IsADirectoryError:
                shutil.rmtree(f)
    else:
        os.makedirs(output_dir, exist_ok=True)

def generate_visuals_and_stats(timeline_csv, cluster_summary_csv, stats_txt, bar_png, density_png):
    # Read timeline CSV
    timeline = pd.read_csv(timeline_csv, parse_dates=["start_time", "end_time"])
    # cluster summary
    cluster_summary = pd.read_csv(cluster_summary_csv, parse_dates=["cluster_first_app", "cluster_last_app"])

    # Basic stats
    total_unique_clusters = cluster_summary.shape[0]
    total_applications = timeline.shape[0]
    avg_apps_per_cluster = cluster_summary['num_applications'].mean()

    # Top clusters
    top_clusters = cluster_summary.sort_values("num_applications", ascending=False).head(10)

    # Write textual stats
    with open(stats_txt, "w") as f:
        f.write(f"Total unique clusters: {total_unique_clusters}\n")
        f.write(f"Total applications: {total_applications}\n")
        f.write(f"Average applications per cluster: {avg_apps_per_cluster:.2f}\n\n")
        f.write("Most heavily used clusters:\n")
        for _, r in top_clusters.iterrows():
            f.write(f"  Cluster {int(r['cluster_id'])}: {int(r['num_applications'])} applications\n")

    # ---- Bar chart: number of applications per cluster ----
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=cluster_summary.sort_values("num_applications", ascending=False),
                     x="cluster_id", y="num_applications")
    # Add value labels
    for p in ax.patches:
        ax.annotate(format(int(p.get_height()), ","), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 4), textcoords='offset points')
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of applications")
    ax.set_title("Number of applications per cluster")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(bar_png, dpi=150)
    plt.close()

    # ---- Density plot / histogram for largest cluster (by num_apps) ----
    if cluster_summary.shape[0] > 0:
        largest_cluster_id = int(cluster_summary.sort_values("num_applications", ascending=False).iloc[0]["cluster_id"])
        cluster_apps = timeline[timeline["cluster_id"] == largest_cluster_id].copy()
        # compute durations in seconds
        cluster_apps["duration_sec"] = (cluster_apps["end_time"] - cluster_apps["start_time"]).dt.total_seconds().clip(lower=1)
        n = cluster_apps.shape[0]
        if n > 0:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=cluster_apps, x="duration_sec", kde=True)
            plt.xscale("log")
            plt.xlabel("Application duration (seconds, log scale)")
            plt.title(f"Duration distribution for cluster {largest_cluster_id} (n={n})")
            plt.tight_layout()
            plt.savefig(density_png, dpi=150)
            plt.close()
        else:
            # create an empty placeholder if no apps
            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, f"No applications for cluster {largest_cluster_id}", ha="center")
            plt.axis("off")
            plt.savefig(density_png, dpi=150)
            plt.close()
    else:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No clusters found", ha="center")
        plt.axis("off")
        plt.savefig(density_png, dpi=150)
        plt.close()

    print(f"Visuals written: {bar_png}, {density_png}")
    print(f"Stats written: {stats_txt}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-spark", action="store_true", help="Skip Spark processing and use existing CSVs for visuals")
    args = parser.parse_args()

    out_dir = "data/output"
    ensure_output_clean(out_dir)

    timeline_csv = os.path.join(out_dir, "problem2_timeline.csv")
    cluster_summary_csv = os.path.join(out_dir, "problem2_cluster_summary.csv")
    stats_txt = os.path.join(out_dir, "problem2_stats.txt")
    bar_png = os.path.join(out_dir, "problem2_bar_chart.png")
    density_png = os.path.join(out_dir, "problem2_density_plot.png")

    if not args.skip_spark:
        # --------- Use Spark to parse logs and build timeline ----------
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import regexp_extract, col, to_timestamp, min as spark_min, max as spark_max, count

        spark = (SparkSession.builder
                 .appName("Problem2_ClusterUsage")
                 # run locally if not overridden by env/submit command
                 .master(os.environ.get("SPARK_MASTER", "local[*]"))
                 .getOrCreate())

        # path to your logs - adjust if needed
        log_path = "s3a://ahs131-assignment-spark-cluster-logs/data/*/*"
        print(f"Reading logs from: {log_path}")
        logs_df = spark.read.text(log_path).withColumnRenamed("value", "log_entry")

        # extract timestamp string (e.g., "17/03/29 10:04:41") and application id (e.g., application_1485248649253_0001)
        ts_pattern = r"^(\d{2}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2})"
        app_pattern = r"(application_\d+_\d+)"
        ts_col = regexp_extract(col("log_entry"), ts_pattern, 1)
        app_col = regexp_extract(col("log_entry"), app_pattern, 1)

        parsed = logs_df.withColumn("ts_str", ts_col).withColumn("application_id", app_col)
        # keep only rows that have both a timestamp and an application id
        parsed = parsed.filter((col("ts_str") != "") & (col("application_id") != ""))

        # convert to proper timestamp type; logs use "yy/MM/dd HH:mm:ss" (two-digit year)
        parsed = parsed.withColumn("ts", to_timestamp(col("ts_str"), "yy/MM/dd HH:mm:ss"))

        # group to get app start and end
        apps = parsed.groupBy("application_id").agg(
            spark_min("ts").alias("start_time"),
            spark_max("ts").alias("end_time")
        )

        # extract cluster id (digits after application_) and app_number (last group)
        cluster_id_col = regexp_extract(col("application_id"), r"application_(\d+)_\d+", 1)
        app_num_col = regexp_extract(col("application_id"), r"application_\d+_(\d+)", 1)

        apps = apps.withColumn("cluster_id", cluster_id_col).withColumn("app_number", app_num_col)

        # reformat times to strings (ISO) for CSV output
        from pyspark.sql.functions import date_format
        apps_out = apps.select(
            col("cluster_id"),
            col("application_id"),
            col("app_number"),
            date_format(col("start_time"), "yyyy-MM-dd HH:mm:ss").alias("start_time"),
            date_format(col("end_time"),   "yyyy-MM-dd HH:mm:ss").alias("end_time")
        ).orderBy(col("cluster_id"), col("start_time"))

        # write timeline CSV
        write_single_csv_from_spark(apps_out, timeline_csv)
        print(f"Wrote timeline to {timeline_csv}")

        # ---------------- cluster summary ----------------
        cluster_summary_df = apps_out.groupBy("cluster_id").agg(
            count("application_id").alias("num_applications"),
            # for first/last app times we need min/max on start_time/end_time; cast back to timestamp
            spark_min(to_timestamp(col("start_time"), "yyyy-MM-dd HH:mm:ss")).alias("cluster_first_app_ts"),
            spark_max(to_timestamp(col("end_time"), "yyyy-MM-dd HH:mm:ss")).alias("cluster_last_app_ts")
        )

        cluster_summary_out = cluster_summary_df.select(
            col("cluster_id"),
            col("num_applications"),
            date_format(col("cluster_first_app_ts"), "yyyy-MM-dd HH:mm:ss").alias("cluster_first_app"),
            date_format(col("cluster_last_app_ts"), "yyyy-MM-dd HH:mm:ss").alias("cluster_last_app")
        ).orderBy(col("num_applications").desc())

        write_single_csv_from_spark(cluster_summary_out, cluster_summary_csv)
        print(f"Wrote cluster summary to {cluster_summary_csv}")

        # done with Spark
        spark.stop()

    # At this point we should have CSVs; if skip-spark was used, they must already exist.
    # Validate CSVs exist
    if not (os.path.exists(timeline_csv) and os.path.exists(cluster_summary_csv)):
        print("ERROR: required CSV(s) missing. If you used --skip-spark, make sure the timeline and cluster summary CSVs exist in data/output/")
        sys.exit(1)

    # Generate visuals and stats using pandas (fast)
    generate_visuals_and_stats(timeline_csv, cluster_summary_csv, stats_txt, bar_png, density_png)

if __name__ == "__main__":
    main()
