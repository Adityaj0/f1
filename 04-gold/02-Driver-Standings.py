# Databricks notebook source
# DBTITLE 1,Load environment config
# MAGIC %run ../00-common/01-Env-Config

# COMMAND ----------

# DBTITLE 1,Define table references
gold_schema = "gold"

# Source: gold race_results (already denormalized)
source_table = f"{catalog_name}.{gold_schema}.race_results"
gold_table = f"{catalog_name}.{gold_schema}.driver_standings"

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

# DBTITLE 1,Aggregate driver stats per season
driver_standings_df = (
    race_results_df
    .groupBy("season", "driver_id", "driver_name", "driver_nationality", "constructor_name")
    .agg(
        F.sum("points").alias("total_points"),
        F.count("*").alias("races_entered"),
        F.sum(F.when(F.col("is_winner"), 1).otherwise(0)).alias("wins"),
        F.sum(F.when(F.col("is_podium"), 1).otherwise(0)).alias("podiums"),
        F.sum(F.when(F.col("is_dnf"), 1).otherwise(0)).alias("dnfs"),
        F.min("position").alias("best_finish"),
        F.avg("position").cast("decimal(5,2)").alias("avg_finish_position"),
        F.avg("positions_gained").cast("decimal(5,2)").alias("avg_positions_gained"),
    )
)

# COMMAND ----------

# DBTITLE 1,Rank drivers per season
season_window = Window.partitionBy("season").orderBy(F.desc("total_points"), F.desc("wins"))

driver_standings_final_df = (
    driver_standings_df
    .withColumn("season_rank", F.rank().over(season_window))
    .orderBy("season", "season_rank")
)

# COMMAND ----------

# DBTITLE 1,Write to gold table
(
    driver_standings_final_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(gold_table)
)

# COMMAND ----------

# DBTITLE 1,Display driver standings
display(spark.table(gold_table).filter(F.col("season") >= 2020))
