# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 3.1: Creating Notebook

# COMMAND ----------

# Databricks notebook source
# Housing Price Forecasting Project - Step 3: Exploratory Data Analysis
# This notebook analyzes patterns and trends in the housing data

from pyspark.sql.functions import col

print("Starting exploratory data analysis...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.2: Load the Transformed Data

# COMMAND ----------

# Load the transformed features data
df = spark.table("main.housing_forecast.housing_price_index_features")

print(f"Total records: {df.count()}")
print(f"Columns: {df.columns}")
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.3: Convert to Pandas for Analysis

# COMMAND ----------

# Convert to Pandas for easier analysis
# Filter to only recent data to avoid memory issues
df_pandas = df.filter(col("year") >= 1990).toPandas()

print(f"Pandas DataFrame shape: {df_pandas.shape}")
print(f"Columns: {df_pandas.columns.tolist()}")
print(f"Data types:\n{df_pandas.dtypes}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.4: Basic Statistical Summary

# COMMAND ----------

# Summary statistics for key columns
print("=== SUMMARY STATISTICS ===\n")

summary_cols = ["index_value", "lag_1", "lag_12", "mom_change", "yoy_change", 
                "rolling_mean_12", "rolling_std_12", "volatility"]

summary_stats = df_pandas[summary_cols].describe()
print(summary_stats)

# Additional statistics
print("\n=== ADDITIONAL STATISTICS ===")
print(f"Mean index value: {df_pandas['index_value'].mean():.2f}")
print(f"Median index value: {df_pandas['index_value'].median():.2f}")
print(f"Std Dev index value: {df_pandas['index_value'].std():.2f}")
print(f"Min index value: {df_pandas['index_value'].min():.2f}")
print(f"Max index value: {df_pandas['index_value'].max():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.5: Analyze by Geography

# COMMAND ----------

# DBTITLE 1,Analyze by Geography
from pyspark.sql.functions import mean, min, max, stddev, round as spark_round

# Group by geography and calculate statistics
print("=== Statistics by Geography ===\n")

geo_stats = df.groupBy('geography').agg(
    # index_value
    spark_round(min('index_value'), 2).alias('Index Value.Min'),
    spark_round(max('index_value'), 2).alias('Index Value.Max'),
    spark_round(mean('index_value'), 2).alias('Index Value.Mean'),
    spark_round(stddev('index_value'), 2).alias('Index Value.Stddev'),

    # mom_change
    spark_round(mean('mom_change'), 2).alias('MOM Change.Mean'),
    spark_round(stddev('mom_change'), 2).alias('MOM Change.Stddev'),

    # yoy_change
    spark_round(mean('yoy_change'), 2).alias('YOY Change.Mean'),
    spark_round(stddev('yoy_change'), 2).alias('YOY Change.Stddev'),

    # volatility
    spark_round(max('volatility'), 2).alias('Volatility.Max'),
    spark_round(mean('volatility'), 2).alias('Volatility.Mean')
)

display(geo_stats)

highest_price_geo = df_pandas.groupby('geography')['index_value'].mean().idxmax()
lowest_price_geo = df_pandas.groupby('geography')['index_value'].mean().idxmin()

print(f"\nHighest average price: {highest_price_geo}")
print(f"Lowest average price: {lowest_price_geo}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.6: Analyze by Year

# COMMAND ----------

# Group by year and calculate statistics
print("STATISTICS BY YEAR (5-YEAR INTERVALS)\n")

# Create 5-year buckets
df_pandas['year_bucket'] = (df_pandas['year'] // 5 * 5).astype(str) + 's'

year_stats = df_pandas.groupby('year_bucket').agg({
    'index_value': ['mean', 'std', 'min', 'max'],
    'mom_change': 'mean',
    'yoy_change': 'mean',
    'volatility': 'mean'
}).round(2)

print(year_stats)

# Calculate decade-over-decade growth
print("\n=== DECADE-OVER-DECADE GROWTH ===")
for decade in sorted(df_pandas['year_bucket'].unique()):
    decade_data = df_pandas[df_pandas['year_bucket'] == decade]
    avg_price = decade_data['index_value'].mean()
    print(f"{decade}: ${avg_price:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.7: Seasonal Analysis

# COMMAND ----------

# Analyze seasonal patterns
print("=== SEASONAL ANALYSIS ===\n")

seasonal_stats = df_pandas.groupby('season').agg({
    'index_value': ['mean', 'std', 'count'],
    'mom_change': 'mean',
    'yoy_change': 'mean'
}).round(2)

print(seasonal_stats)

# Month-by-month analysis
print("\n=== MONTH-BY-MONTH ANALYSIS ===")
monthly_stats = df_pandas.groupby('month').agg({
    'index_value': ['mean', 'std', 'count'],
    'mom_change': 'mean'
}).round(2)

print(monthly_stats)

# Which months have highest/lowest prices?
highest_month = df_pandas.groupby('month')['index_value'].mean().idxmax()
lowest_month = df_pandas.groupby('month')['index_value'].mean().idxmin()

month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
               7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

print(f"\nHighest average prices: {month_names[highest_month]}")
print(f"Lowest average prices: {month_names[lowest_month]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.8: Correlation Analysis

# COMMAND ----------

import pandas as pd
import numpy as np

# Select numeric columns for correlation
numeric_cols = ['index_value', 'lag_1', 'lag_3', 'lag_6', 'lag_12',
                'rolling_mean_12', 'rolling_std_12', 'mom_change', 'yoy_change',
                'volatility', 'interest_rate', 'unemployment_rate', 'months_since_start']

# Calculate correlation matrix
correlation_matrix = df_pandas[numeric_cols].corr()

print("Correlation with Index Value\n")
correlations_with_target = correlation_matrix['index_value'].sort_values(ascending=False)
print(correlations_with_target)

# Find strongest correlations
print("\nStrongest Correlations (excluding index_value)\n")
strong_corr = correlations_with_target[correlations_with_target.index != 'index_value'].head(10)
for feature, corr in strong_corr.items():
    print(f"{feature}: {corr:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.9: Volatility Analysis
# MAGIC

# COMMAND ----------

# Analyze volatility trends
print("VOLATILITY ANALYSIS\n")

# Volatility by geography
print("Average volatility by geography:")
volatility_by_geo = df_pandas.groupby('geography')['volatility'].agg(['mean', 'std', 'max']).round(2)
print(volatility_by_geo.sort_values('mean', ascending=False))

# Volatility by year
print("\nVOLATILITY BY DECADE")
volatility_by_year = df_pandas.groupby('year_bucket')['volatility'].agg(['mean', 'std', 'max']).round(2)
print(volatility_by_year)

# High volatility periods
print("\nHIGH VOLATILITY PERIODS")
high_vol = df_pandas[df_pandas['volatility'] > df_pandas['volatility'].quantile(0.90)]
print(f"Months with high volatility (top 10%):")
print(high_vol[['date', 'geography', 'volatility', 'index_value']].sort_values('volatility', ascending=False).head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.10: Growth Rate Analysis

# COMMAND ----------

# Analyze growth rates
print("GROWTH RATE ANALYSIS\n")

# Average growth rates by geography
print("Average growth rates by geography:")

growth_by_geo = df.groupBy('geography').agg(
    spark_round(mean(col('mom_change')),2).alias('mom_change (avg)'),
    spark_round(mean(col('change_3m')),2).alias('change_3m (avg)'),
    spark_round(mean(col('change_6m')),2).alias('change_6m (avg)'),
    spark_round(mean(col('yoy_change')),2).alias('yoy_change (avg)')
)

display(growth_by_geo.orderBy(col('yoy_change (avg)').desc()))

# Growth rates by decade
print("\n=== GROWTH RATES BY DECADE ===")
growth_by_decade = df_pandas.groupby('year_bucket').agg({
    'mom_change': 'mean',
    'change_3m': 'mean',
    'change_6m': 'mean',
    'yoy_change': 'mean'
}).round(2)
print(growth_by_decade)

# Identify periods of rapid growth vs. decline
print("\n=== PERIODS OF RAPID GROWTH (Top 10) ===")
rapid_growth = df_pandas.nlargest(10, 'yoy_change')[['date', 'geography', 'yoy_change', 'index_value']]
print(rapid_growth)

print("\n=== PERIODS OF RAPID DECLINE (Bottom 10) ===")
rapid_decline = df_pandas.nsmallest(10, 'yoy_change')[['date', 'geography', 'yoy_change', 'index_value']]
print(rapid_decline)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.11: Feature Importance Preview

# COMMAND ----------

# Which features have the most variation?
print("=== FEATURE VARIATION (Coefficient of Variation) ===\n")

feature_variation = {}
for col in numeric_cols:
    if df_pandas[col].mean() != 0:
        cv = (df_pandas[col].std() / df_pandas[col].mean()) * 100
        feature_variation[col] = cv

# Sort by variation
sorted_variation = sorted(feature_variation.items(), key=lambda x: x[1], reverse=True)
for feature, cv in sorted_variation:
    print(f"{feature}: {cv:.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.12: Missing Data Analysis

# COMMAND ----------

# Check for missing values
print("=== MISSING DATA ANALYSIS ===\n")

missing_counts = df_pandas.isnull().sum()
missing_pct = (missing_counts / len(df_pandas)) * 100

missing_df = pd.DataFrame({
    'Column': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Missing_Percentage': missing_pct.values
})

missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df)
else:
    print("No missing values found!")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.13: Key Insights Summary
# MAGIC

# COMMAND ----------

# Document key insights
print("=== KEY INSIGHTS FROM EDA ===\n")

insights = []

# Insight 1: Overall trend (using available data 1990-2025)
early_period = df_pandas[df_pandas['year'] < 2000]['index_value']
recent_period = df_pandas[df_pandas['year'] >= 2020]['index_value']

if len(early_period) > 0 and len(recent_period) > 0:
    early_mean = early_period.mean()
    recent_mean = recent_period.mean()
    
    if early_mean > 0:
        overall_growth = ((recent_mean - early_mean) / early_mean) * 100
    else:
        overall_growth = 0
else:
    overall_growth = 0

insights.append(f"1. Overall Growth: Housing prices grew {overall_growth:.1f}% from 1990s to 2020s")

# Insight 2: Most volatile geography
most_volatile_geo = df_pandas.groupby('geography')['volatility'].mean().idxmax()
most_volatile_val = df_pandas.groupby('geography')['volatility'].mean().max()
insights.append(f"2. Most Volatile Market: {most_volatile_geo} (avg volatility: {most_volatile_val:.2f}%)")

# Insight 3: Seasonal pattern
seasonal_stats = df_pandas.groupby('season')['index_value'].agg(['mean', 'std', 'min', 'max'])
seasonal_range = seasonal_stats['max'].max() - seasonal_stats['min'].min()
seasonal_avg = df_pandas['index_value'].mean()
seasonal_pct = (seasonal_range / seasonal_avg) * 100
insights.append(f"3. Seasonal Variation: {seasonal_pct:.2f}% variation between seasons")

# Insight 4: Recent trend
recent_growth = df_pandas[df_pandas['year'] >= 2023]['yoy_change'].mean()
insights.append(f"4. Recent Trend: {recent_growth:.2f}% year-over-year growth in 2023-2025")

# Insight 5: Feature correlation
numeric_cols = ['index_value', 'lag_1', 'lag_3', 'lag_6', 'lag_12',
                'rolling_mean_12', 'rolling_std_12', 'mom_change', 'yoy_change',
                'volatility', 'interest_rate', 'unemployment_rate', 'months_since_start']

correlation_matrix = df_pandas[numeric_cols].corr()
correlations_with_target = correlation_matrix['index_value'].sort_values(ascending=False)
top_corr_feature = correlations_with_target[correlations_with_target.index != 'index_value'].index[0]
top_corr_value = correlations_with_target[correlations_with_target.index != 'index_value'].values[0]
insights.append(f"5. Strongest Predictor: {top_corr_feature} (correlation: {top_corr_value:.4f})")

for insight in insights:
    print(insight)

# Additional insight: Show top 5 correlated features
print("\n=== TOP 5 CORRELATED FEATURES ===")
top_5_corr = correlations_with_target[correlations_with_target.index != 'index_value'].head(5)
for feature, corr in top_5_corr.items():
    print(f"{feature}: {corr:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3.14: Save Analysis Results

# COMMAND ----------

analysis_results = {
    'total_records': len(df_pandas),
    'date_range': f"{df_pandas['date'].min()} to {df_pandas['date'].max()}",
    'geographies': df_pandas['geography'].nunique(),
    'avg_price': df_pandas['index_value'].mean(),
    'price_std': df_pandas['index_value'].std(),
    'avg_volatility': df_pandas['volatility'].mean(),
    'avg_yoy_growth': df_pandas['yoy_change'].mean()
}

print("\n=== ANALYSIS SUMMARY ===")
for key, value in analysis_results.items():
    print(f"{key}: {value}")
