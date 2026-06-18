# Databricks notebook source
# DBTITLE 1,Load environment config
# MAGIC %run ../00-common/01-Env-Config

# COMMAND ----------

# DBTITLE 1,Define table references
bronze_table = f"{catalog_name}.{bronze_schema}.sprints"
silver_table = f"{catalog_name}.{silver_schema}.sprints"

# COMMAND ----------

# DBTITLE 1,Import functions
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Read bronze table
sprints_df = spark.table(bronze_table)

# COMMAND ----------

# DBTITLE 1,Drop url column
sprints_dropped_df = sprints_df.drop("url")

# COMMAND ----------

# DBTITLE 1,Rename columns to snake_case
sprints_renamed_df = sprints_dropped_df.withColumnsRenamed({
    "raceName": "race_name",
    "constructorId": "constructor_id",
    "driverId": "driver_id",
    "positionText": "position_text",
})

# COMMAND ----------

# DBTITLE 1,Deduplicate
sprints_distinct_df = sprints_renamed_df.dropDuplicates(["season", "round", "driver_id"])

# COMMAND ----------

# DBTITLE 1,Apply transformations
sprints_final_df = (
    sprints_distinct_df
    .withColumn("race_name", F.initcap(F.col("race_name")))
    .withColumn("status", F.initcap(F.col("status")))
    .withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
)

# COMMAND ----------

# DBTITLE 1,Write to silver table
(
    sprints_final_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(silver_table)
)

# COMMAND ----------

# DBTITLE 1,Display silver table
display(spark.table(silver_table))
