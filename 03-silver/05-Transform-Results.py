# Databricks notebook source
# DBTITLE 1,Load environment config
# MAGIC %run ../00-common/01-Env-Config

# COMMAND ----------

# DBTITLE 1,Define table references
bronze_table = f"{catalog_name}.{bronze_schema}.results"
silver_table = f"{catalog_name}.{silver_schema}.results"

# COMMAND ----------

# DBTITLE 1,Import functions
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Read bronze table
results_df = spark.table(bronze_table)

# COMMAND ----------

# DBTITLE 1,Drop url column
results_dropped_df = results_df.drop("url")

# COMMAND ----------

# DBTITLE 1,Rename columns to snake_case
results_renamed_df = results_dropped_df.withColumnsRenamed({
    "raceName": "race_name",
    "constructorId": "constructor_id",
    "driverId": "driver_id",
    "positionText": "position_text",
})

# COMMAND ----------

# DBTITLE 1,Deduplicate
results_distinct_df = results_renamed_df.dropDuplicates(["season", "round", "driver_id"])

# COMMAND ----------

# DBTITLE 1,Apply transformations
results_final_df = (
    results_distinct_df
    .withColumn("race_name", F.initcap(F.col("race_name")))
    .withColumn("status", F.initcap(F.col("status")))
)

# COMMAND ----------

# DBTITLE 1,Write to silver table
(
    results_final_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(silver_table)
)

# COMMAND ----------

# DBTITLE 1,Display silver table
display(spark.table(silver_table))
