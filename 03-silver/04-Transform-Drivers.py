# Databricks notebook source
# DBTITLE 1,Load environment config
# MAGIC %run ../00-common/01-Env-Config

# COMMAND ----------

# DBTITLE 1,Define table references
bronze_table = f"{catalog_name}.{bronze_schema}.drivers"
silver_table = f"{catalog_name}.{silver_schema}.drivers"

# COMMAND ----------

# DBTITLE 1,Import functions
from pyspark.sql import functions as F

# COMMAND ----------

# DBTITLE 1,Read bronze table
drivers_df = spark.table(bronze_table)

# COMMAND ----------

# DBTITLE 1,Flatten name struct and drop url
drivers_flattened_df = (
    drivers_df
    .withColumn("first_name", F.col("name.givenName"))
    .withColumn("last_name", F.col("name.familyName"))
    .drop("name", "url")
)

# COMMAND ----------

# DBTITLE 1,Rename columns to snake_case
drivers_renamed_df = drivers_flattened_df.withColumnsRenamed({
    "driverId": "driver_id",
    "dateOfBirth": "date_of_birth",
})

# COMMAND ----------

# DBTITLE 1,Deduplicate on driver_id
drivers_distinct_df = drivers_renamed_df.dropDuplicates(["driver_id"])

# COMMAND ----------

# DBTITLE 1,Apply transformations
drivers_final_df = (
    drivers_distinct_df
    .withColumn("first_name", F.initcap(F.col("first_name")))
    .withColumn("last_name", F.initcap(F.col("last_name")))
    .withColumn("nationality", F.initcap(F.col("nationality")))
)

# COMMAND ----------

# DBTITLE 1,Write to silver table
(
    drivers_final_df
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(silver_table)
)

# COMMAND ----------

# DBTITLE 1,Display silver table
display(spark.table(silver_table))
