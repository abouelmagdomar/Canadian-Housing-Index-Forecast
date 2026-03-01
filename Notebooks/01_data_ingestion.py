# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 1.1: Ingest Housing Data

# COMMAND ----------

df = spark.table("main.housing_forecast.housing_price_index_raw")

# Check the data
print(f"Total records: {df.count()}")
print(f"Columns: {df.columns}")

# Use printSchema instead of display to avoid type inference issues
df.printSchema()

# Show data as a simple table (without display)
df.show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.2: Prepare Interest Rate Table

# COMMAND ----------

# Ingest Interest Rate Data
interest_rates = spark.table('main.housing_forecast.statcan_interest_rate')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.3: Inspect Interest Rate Data

# COMMAND ----------

from pyspark.sql.functions import col

print('='*50)
print('INTEREST RATE DATA INSPECTION')
print('='*50)

# Retreive Date Range
earliest_date = interest_rates.orderBy(col('REF_DATE').asc()).collect()[0][0]
latest_date = interest_rates.orderBy(col('REF_DATE').desc()).collect()[0][0]

print(f'\nEarliest Date: {earliest_date}')
print(f'Latest Date: {latest_date}\n')

# View types of statistics
display(interest_rates.select('Financial market statistics').distinct())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.4: Filter Interest Rate Data

# COMMAND ----------

from pyspark.sql.functions import last_day, lit

# Drop NaN rows
interest_rates = interest_rates.filter(col('VALUE').isNotNull())

# Filter Dates to Match Housing Market Data

earliest_date_h = df.orderBy(col('REF_DATE').asc()).collect()[0][0]
latest_date_h = df.orderBy(col('REF_DATE').desc()).collect()[0][0]
print(f'Earliest Housing Date: {earliest_date_h}')
print(f'Latest Housing Date: {latest_date_h}\n')

interest_rates_filtered = interest_rates.filter(
    (last_day(col('REF_DATE')) >= last_day(lit(earliest_date_h))) &
    (last_day(col('REF_DATE')) <= last_day(lit(latest_date_h)))
)

print(f'ROW COUNT BEFORE DATE FILTER: {interest_rates.count()}')
print(f'ROW COUNT AFTER DATE FILTER: {interest_rates_filtered.count()}')

earliest_date = interest_rates_filtered.orderBy(col('REF_DATE').asc()).collect()[0][0]
latest_date = interest_rates_filtered.orderBy(col('REF_DATE').desc()).collect()[0][0]

print(f'\nEarliest Date: {earliest_date}')
print(f'Latest Date: {latest_date}\n')

# Filter 'Financial market statistics' to 'Target rate'
print(f'ROW COUNT BEFORE STATISTIC FILTER: {interest_rates_filtered.count()}')
interest_rates_filtered = interest_rates_filtered.filter(col('Financial market statistics')=='Target rate').select('REF_DATE','VALUE')
print(f'ROW COUNT AFTER STATISTIC FILTER: {interest_rates_filtered.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.5: Ingest Unemployment Data

# COMMAND ----------

unemployment_rates = spark.table('main.housing_forecast.statcan_unemployment_rate')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.6 Filter Unemployment Data

# COMMAND ----------

# Perform Filtering
unemployment_rates = unemployment_rates.filter(
    (col('GEO') == 'Canada') &
    (col('Labour force characteristics') == 'Unemployment rate') &
    (col('Data type') == 'Seasonally adjusted') &
    (col('Gender') == 'Total - Gender') &
    (col('Age group') == '15 years and over') &
    (col('Statistics') == 'Estimate') &
    (last_day(col('REF_DATE')) >= last_day(lit(earliest_date_h))) &
    (last_day(col('REF_DATE')) <= last_day(lit(latest_date_h)))
).orderBy('REF_DATE').select('REF_DATE','VALUE')

display(unemployment_rates)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.7: Export Filtered Tables

# COMMAND ----------

interest_rates_filtered.write.mode("overwrite").saveAsTable("main.housing_forecast.interest_rate")
unemployment_rates.write.mode("overwrite").saveAsTable("main.housing_forecast.unemployment_rate")