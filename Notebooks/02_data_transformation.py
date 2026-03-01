# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 2.1: Create a New Notebook

# COMMAND ----------

# Databricks notebook source
# Housing Price Forecasting Project - Step 2: Data Transformation & Feature Engineering
# This notebook transforms raw data into ML-ready features

print("Starting data transformation...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.2: Load the Raw Data

# COMMAND ----------

# Load the raw housing price data
df = spark.table("main.housing_forecast.housing_price_index_raw")

# Display table schema
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.3: Initial Data Cleaning

# COMMAND ----------

from pyspark.sql.functions import (
    col, to_date, year, month, quarter, 
    lag, avg, stddev, when, lit, round as spark_round
)
from pyspark.sql.window import Window

# Convert REF_DATE to proper date format
df_clean = df.withColumn("date", to_date(col("REF_DATE"), "yyyy-MM"))

# Rename columns for clarity
df_clean = df_clean.withColumnRenamed('New housing price indexes', 'component') \
                    .withColumnRenamed('VALUE','index_value') \
                    .withColumnRenamed('GEO','geography')

# Convert index_value to numeric (in case it's string)
df_clean = df_clean.withColumn('index_value', col('index_value').cast('double'))

# Sort by geography, component, and date
df_clean = df_clean.sort('geography','component','date')

# Display Sample
display(df_clean.select("date", "geography", "component", "index_value").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.4: Filter for Total Housing Index Only

# COMMAND ----------

# Filter for "Total (house and land)" component and exclude \'Canada\' geography
df_filtered = df_clean.filter((col("component") == "Total (house and land)") & (col("geography") != "Canada"))

print(f'Records before filtering: {df_clean.count()}')
print(f'Records after filtering: {df_filtered.count()}')
print(f'Unique Geographies: {df_filtered.select('geography').distinct().count()}')

# Show Sample
display(df_filtered.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.5: Create Temporal Features

# COMMAND ----------

# Extract temporal features
df_features = df_filtered.withColumn('year', year(col('date'))) \
                        .withColumn('month', month(col('date'))) \
                        .withColumn('quarter', quarter(col('date')))

# Create season features
df_features = df_features.withColumn(
    "season",
    when(col("month").isin([12, 1, 2]), "Winter")
    .when(col("month").isin([3, 4, 5]), "Spring")
    .when(col("month").isin([6, 7, 8]), "Summer")
    .otherwise("Fall")
)

# Categorizing Geographical Locations
provinces = ["Ontario", "Quebec", "British Columbia", "Alberta", 
             "Manitoba", "Saskatchewan", "Nova Scotia", "New Brunswick",
             "Prince Edward Island", "Newfoundland and Labrador"]

df_features = df_features.withColumn('geo_type', when(col('geography')=='Canada', 'National') \
    .when(col('geography').isin(provinces), 'Province') \
    .otherwise('Metropolitan'))

# Display Distinct Locations
display(df_features.select('geography','geo_type').distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.5: Create Lagged Features

# COMMAND ----------

# DBTITLE 1,Create Lagged Features
# Window specification for each geography-component combination
# Component is included in case we decide not to filter it the code above (Step 2.4)
window_spec = Window.partitionBy('geography','component').orderBy('date')

# Create lag features (1, 3, 6, 12 months)
df_features = df_features.withColumn('lag_1', lag('index_value',1).over(window_spec)) \
    .withColumn('lag_3', lag('index_value',3).over(window_spec)) \
    .withColumn('lag_6', lag('index_value',6).over(window_spec)) \
    .withColumn('lag_12', lag('index_value',12).over(window_spec))

# Display sample with lags
display(df_features.limit(100))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.7: Create Rolling Statistics

# COMMAND ----------

from pyspark.sql.functions import min as spark_min, max as spark_max

# Define rolling window (12 months)
rolling_window = Window.partitionBy('geography').orderBy('date').rowsBetween(-11,0)

# Calculate rolling statistics
df_features = df_features.withColumn('rolling_mean_12', avg('index_value').over(rolling_window)) \
                            .withColumn('rolling_std_12', stddev('index_value').over(rolling_window)) \
                            .withColumn('rolling_min_12', spark_min('index_value').over(rolling_window)) \
                            .withColumn('rolling_max_12', spark_max('index_value').over(rolling_window))

display(df_features.select('geography','date','rolling_mean_12','rolling_std_12','rolling_min_12','rolling_max_12').limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.8: Create Growth Rate Features

# COMMAND ----------

# Month-over-month change (%)
df_features = df_features.withColumn('mom_change', 
    when(col('lag_1').isNotNull(), 
         ((col('index_value') - col('lag_1')) / col('lag_1')) * 100
    ).otherwise(None)
)

# Year-over-year change (%) - Already correct
df_features = df_features.withColumn('yoy_change', 
    when(col('lag_12').isNotNull(), 
         ((col('index_value') - col('lag_12')) / col('lag_12')) * 100
    ).otherwise(None)
)

# 3-month change (%) - Already correct
df_features = df_features.withColumn('change_3m', 
    when(col('lag_3').isNotNull(), 
         ((col('index_value') - col('lag_3')) / col('lag_3')) * 100
    ).otherwise(None)
)

# 6-month change (%) - Already correct
df_features = df_features.withColumn('change_6m', 
    when(col('lag_6').isNotNull(), 
         ((col('index_value') - col('lag_6')) / col('lag_6')) * 100
    ).otherwise(None)
)

# Display sample
display(df_features.select("date","year","mom_change", "yoy_change", "change_3m", "change_6m").limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.9: Add Economic Indicators

# COMMAND ----------

from pyspark.sql.functions import to_date, col, lit, when, avg
from pyspark.sql.types import DoubleType

# Load filtered interest rate table from Step 1
interest_rate_df = spark.table('main.housing_forecast.interest_rate')

interest_rate_df = interest_rate_df.select(
    col('REF_DATE').alias('date_ir'),
    col("VALUE").alias('interest_rate')
)

interest_rate_df = interest_rate_df.withColumn("date_ir", to_date(col("date_ir"), "yyyy-MM"))
interest_rate_df = interest_rate_df.withColumn("interest_rate", col("interest_rate").cast(DoubleType()))

# Load filtered unemployements rates from Step 1
unemployment_rate_df = spark.table('main.housing_forecast.unemployment_rate')

unemployment_rate_df = unemployment_rate_df.select(
    col("REF_DATE").alias("date_ur"),
    col("VALUE").alias("unemployment_rate")
)

unemployment_rate_df = unemployment_rate_df.withColumn("date_ur", to_date(col("date_ur"), "yyyy-MM"))
unemployment_rate_df = unemployment_rate_df.withColumn("unemployment_rate", col("unemployment_rate").cast(DoubleType()))

# Join interest rate data
df_features = df_features.join(interest_rate_df, 
        df_features.date == interest_rate_df.date_ir, 
        "left_outer")\
    .drop("date_ir")

# Join unemployment rate data
df_features = df_features.join(unemployment_rate_df, 
        df_features.date == unemployment_rate_df.date_ur, 
        "left_outer")\
    .drop("date_ur")

# Handle missing values with mean after join
mean_interest_rate = df_features.agg(avg(col("interest_rate"))).collect()[0][0]
mean_unemployment_rate = df_features.agg(avg(col("unemployment_rate"))).collect()[0][0]

df_features = df_features.na.fill(mean_interest_rate, subset=["interest_rate"])
df_features = df_features.na.fill(mean_unemployment_rate, subset=["unemployment_rate"])

# Display sample
display(df_features.select("date", "year", "interest_rate", "unemployment_rate").limit(5000))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.10: Create Cyclical Features

# COMMAND ----------

import math
from pyspark.sql.functions import sin, cos

# Create sine and cosine features for month cyclicality
# This captures the seasonal pattern in a continuous way, ensuring December transitions smoothly to January
df_features = df_features.withColumn(
    "month_sin",
    spark_round(sin(2 * math.pi * col("month") / 12), 4)
)

df_features = df_features.withColumn(
    "month_cos",
    spark_round(cos(2 * math.pi * col("month") / 12), 4)
)

# Display sample
display(df_features.select("month", "month_sin", "month_cos").limit(13))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.11: Create Trend Feature

# COMMAND ----------

from pyspark.sql.functions import row_number

# Create a trend feature (months since start for each geography)
window_trend = Window.partitionBy('geography').orderBy('date')

df_features = df_features.withColumn('months_since_start', row_number().over(window_trend)-1)

# Display sample
display(df_features.select("date", "geography", "months_since_start").limit(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.12: Create Additional ML Features

# COMMAND ----------

# Volatility (coefficient of variation)
df_features = df_features.withColumn(
    'volatility',
    when(col('rolling_mean_12')!=0,
         (col('rolling_std_12'))/(col('rolling_mean_12'))*100)
    .otherwise(None)
)

# Momentum (deviation from rolling mean) REDACTED DUE TO DATA LEAKAGE
''' df_features = df_features.withColumn(
    'momentum',
    col('index_value') - col('rolling_mean_12')
) '''

# Distance from historical high/low  REDACTED DUE TO DATA LEAKAGE
''' df_features = df_features.withColumn(
    'dist_from_high',
    when(col('rolling_max_12')!=0,
         col('index_value')-col("rolling_max_12"))
    .otherwise(None)
) '''

''' df_features = df_features.withColumn(
    'dist_from_low',
    when(col('rolling_min_12')!=0,
         col('index_value')-col("rolling_min_12"))
    .otherwise(None)
) '''

# Display sample
display(df_features.select("date", "volatility").limit(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.13: Round Numerical Columns

# COMMAND ----------

# Round numerical columns to 2 decimal places
numerical_cols = [
    "index_value", "lag_1", "lag_3", "lag_6", "lag_12",
    "rolling_mean_12", "rolling_std_12", "rolling_min_12", "rolling_max_12",
    "mom_change", "yoy_change", "change_3m", "change_6m",
    "month_sin", "month_cos", "volatility"
]

for col_name in numerical_cols:
    if col_name in df_features.columns:
        df_features = df_features.withColumn(col_name, spark_round(col(col_name), 2))

display(df_features.limit(1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.14: Remove Rows with Missing Values

# COMMAND ----------

# Row Count Before Filtering
print(f'Row count before filtering: {df_features.count()}')

# Remove rows where index_value is null
df_features = df_features.filter(col('index_value').isNotNull())

# Row Count After Filtering
print(f'Row count after filtering: {df_features.count()}')

# Show records with missing values in key features
print("\nRecords with missing lag_12 values:")
print(f"Count: {df_features.filter(col('lag_12').isNull()).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.15: Save the Transformed Data

# COMMAND ----------

# Save as a new table
table_name = "main.housing_forecast.housing_price_index_features"

df_features.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

print(f"Transformed data saved to: {table_name}")
print(f"Total records: {df_features.count()}")
print(f"Total columns: {len(df_features.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2.16: Data Quality Report
# MAGIC Add a final cell to summarize the transformation:

# COMMAND ----------

from pyspark.sql.functions import min as spark_min, max as spark_max, count, when

# Display data quality metrics
print("=== DATA TRANSFORMATION SUMMARY ===")
print(f"\nTotal records: {df_features.count()}")

# Get min and max dates with cleaner names
date_agg = df_features.agg(
    spark_min("date").alias("min_date"),
    spark_max("date").alias("max_date")
).collect()[0]

min_date = date_agg["min_date"]
max_date = date_agg["max_date"]
print(f"Date range: {min_date} to {max_date}")

print(f"Unique geographies: {df_features.select('geography').distinct().count()}")
print(f"Unique components: {df_features.select('component').distinct().count()}")

print("\n=== SAMPLE DATA ===")
display(df_features.select(
    "date", "geography", "index_value",
    "lag_1", "lag_12", "mom_change", "yoy_change",
    "rolling_mean_12", "interest_rate", "volatility"
).limit(20))