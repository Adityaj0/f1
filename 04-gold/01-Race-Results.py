# Databricks notebook source
# DBTITLE 1,Load environment config
# MAGIC %run ../00-common/01-Env-Config

# COMMAND ----------

# DBTITLE 1,Define table references
gold_schema = "gold"

# Silver source tables
results_table = f"{catalog_name}.{silver_schema}.results"
drivers_table = f"{catalog_name}.{silver_schema}.drivers"
constructors_table = f"{catalog_name}.{silver_schema}.constructors"
races_table = f"{catalog_name}.{silver_schema}.races"
circuits_table = f"{catalog_name}.{silver_schema}.circuits"

# Gold target table
gold_table = f"{catalog_name}.{gold_schema}.race_results"

# Create gold schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_name}.{gold_schema}")

# COMMAND ----------

# DBTITLE 1,Import functions
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# DBTITLE 1,Read silver tables
results_df = spark.table(results_table)
drivers_df = spark.table(drivers_table)
constructors_df = spark.table(constructors_table)
races_df = spark.table(races_table)
circuits_df = spark.table(circuits_table)

# COMMAND ----------

# DBTITLE 1,Join results with all dimension tables
race_results_df = (
    results_df
    .join(
        drivers_df.select("driver_id", "first_name", "last_name", F.col("nationality").alias("driver_nationality")),
        on="driver_id",
        how="left"
    )
    .join(
        constructors_df.select("constructor_id", "constructor_name", F.col("nationality").alias("constructor_nationality")),
        on="constructor_id",
        how="left"
    )
    .join(
        races_df.select("season", "round", "circuit_id", "race_date"),
        on=["season", "round"],
        how="left"
    )
    .join(
        circuits_df.select("circuit_id", "circuit_name", "locality", "country"),
        on="circuit_id",
        how="left"
    )
)

# COMMAND ----------

# DBTITLE 1,Select and compute final columns
race_results_final_df = (
    race_results_df
    .withColumn("driver_name", F.concat_ws(" ", F.col("first_name"), F.col("last_name")))
    .withColumn("positions_gained", F.col("grid") - F.col("position"))
    .withColumn("is_winner", F.when(F.col("position") == 1, True).otherwise(False))
    .withColumn("is_podium", F.when(F.col("position") <= 3, True).otherwise(False))
    .withColumn("is_dnf", F.when(F.col("status") != "Finished", True).otherwise(False))
    .select(
        "season", "round", "race_name", "race_date",
        "circuit_name", "locality", "country",
        "driver_id", "driver_name", "driver_nationality",
        "constructor_id", "constructor_name", "constructor_nationality",
        "grid", "position", "position_text", "points", "laps", "status",
        "positions_gained", "is_winner", "is_podium", "is_dnf"
    )
)

# COMMAND ----------

# DBTITLE 1,Write to gold table
(
    race_results_final_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(gold_table)
)

# COMMAND ----------

# DBTITLE 1,Display gold race results
display(spark.table(gold_table))
