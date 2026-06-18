# Databricks notebook source
# DBTITLE 1,Load environment config
# MAGIC %run ../00-common/01-Env-Config

# COMMAND ----------

# DBTITLE 1,Define table references
gold_schema = "gold"

# Source: gold race_results (already denormalized)
source_table = f"{catalog_name}.{gold_schema}.race_results"
gold_table = f"{catalog_name}.{gold_schema}.constructor_standings"

# Create gold schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{gold_schema}")

# COMMAND ----------

# DBTITLE 1,Import functions
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# DBTITLE 1,Read gold race results
race_results_df = spark.table(source_table)

# COMMAND ----------

# DBTITLE 1,Aggregate constructor stats per season
constructor_standings_df = (
    race_results_df
    .groupBy("season", "constructor_id", "constructor_name", "constructor_nationality")
    .agg(
        F.sum("points").alias("total_points"),
        F.countDistinct("round").alias("races_entered"),
        F.countDistinct("driver_id").alias("num_drivers"),
        F.sum(F.when(F.col("is_winner"), 1).otherwise(0)).alias("wins"),
        F.sum(F.when(F.col("is_podium"), 1).otherwise(0)).alias("podiums"),
        F.sum(F.when(F.col("is_dnf"), 1).otherwise(0)).alias("dnfs"),
        F.min("position").alias("best_finish"),
        F.avg("position").cast("decimal(5,2)").alias("avg_finish_position"),
    )
)

# COMMAND ----------

# DBTITLE 1,Rank constructors per season
season_window = Window.partitionBy("season").orderBy(F.desc("total_points"), F.desc("wins"))

constructor_standings_final_df = (
    constructor_standings_df
    .withColumn("season_rank", F.rank().over(season_window))
    .orderBy("season", "season_rank")
)

# COMMAND ----------

# DBTITLE 1,Write to gold table
(
    constructor_standings_final_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(gold_table)
)

# COMMAND ----------

# DBTITLE 1,Display constructor standings
display(spark.table(gold_table).filter(F.col("season") >= 2020))
