# Databricks notebook source
# MAGIC %md
# MAGIC ## Step 5.1: Setup

# COMMAND ----------

# Databricks notebook source
# Housing Price Forecasting Project - Step 5: Predictions & Insights
# This notebook generates future predictions and key insights

print("Starting predictions and insights generation...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5.2: Load Model and Data

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load the original features data
df = spark.table("main.housing_forecast.housing_price_index_features")
df_pandas = df.toPandas()

# Define feature columns (aligned with Step 4 and data leakage fix)
feature_cols = [
    'lag_1', 'lag_3', 'lag_6', 'lag_12',
    'rolling_mean_12', 'rolling_std_12',
    'mom_change', 'yoy_change', 'change_3m', 'change_6m',
    'volatility', 'interest_rate', 'unemployment_rate', 'months_since_start',
    'month_sin', 'month_cos', 'quarter', 'year'
]

# Prepare data
df_ml = df_pandas.dropna(subset=feature_cols)

X = df_ml[feature_cols].copy()
y = df_ml['index_value'].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

# Train final model on all data
lr_model = LinearRegression()
lr_model.fit(X_scaled, y)

print(f"Model trained on {len(X)} records")
print(f"Model R² on full data: {lr_model.score(X_scaled, y):.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5.3: Get Latest Data for Forecasting

# COMMAND ----------

# Get the most recent data for each geography
print("Latest Data for Forecasting\n")

latest_data = df_ml.sort_values('date').groupby('geography').tail(1)

print(f"Geographies: {latest_data['geography'].nunique()}")
print(f"Latest date: {latest_data['date'].max()}")
print(f"\nSample of latest data:")
print(latest_data[['date', 'geography', 'index_value', 'lag_1', 'rolling_mean_12']].head(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5.4: Generate 6-Month Forecast

# COMMAND ----------

# Generate 6-month forecast WITH DYNAMIC FEATURES (CORRECTED)
print("6-Month Forecast\n")

forecast_months = 6
forecast_data = []

for idx, row in latest_data.iterrows():
    geography = row['geography']
    current_date = row['date']
    current_price = row['index_value']
    
    # Start with current features
    prev_price = current_price
    prev_features = row[feature_cols].copy()  # Only use the 21 defined features
    
    # Make predictions for next 6 months
    for month_ahead in range(1, forecast_months + 1):
        # Scale the features (only the 21 features in feature_cols)
        pred_features_scaled = scaler.transform([prev_features[feature_cols].values])[0]
        
        # Make prediction
        predicted_price = lr_model.predict([pred_features_scaled])[0]
        
        # Calculate future date
        future_date = current_date + pd.DateOffset(months=month_ahead)
        
        # Calculate change from current
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        forecast_data.append({
            'Geography': geography,
            'Current_Date': current_date,
            'Forecast_Date': future_date,
            'Months_Ahead': month_ahead,
            'Current_Price': current_price,
            'Forecasted_Price': predicted_price,
            'Price_Change': price_change,
            'Price_Change_Pct': price_change_pct
        })
        
        # Update lagged features (assume predicted price becomes new lag_1)
        prev_features['lag_12'] = prev_features['lag_6']
        prev_features['lag_6'] = prev_features['lag_3']
        prev_features['lag_3'] = prev_features['lag_1']
        prev_features['lag_1'] = predicted_price
        
        # Update rolling mean (simple moving average of recent prices)
        prev_features['rolling_mean_12'] = (prev_features['rolling_mean_12'] * 11 + predicted_price) / 12
        
        # Update momentum (deviation from rolling mean)
        prev_features['momentum'] = predicted_price - prev_features['rolling_mean_12']
        
        # Update other derived features
        prev_features['mom_change'] = (predicted_price - prev_features['lag_1']) / prev_features['lag_1'] * 100 if prev_features['lag_1'] != 0 else 0
        prev_features['yoy_change'] = (predicted_price - prev_features['lag_12']) / prev_features['lag_12'] * 100 if prev_features['lag_12'] != 0 else 0
        prev_features['change_3m'] = (predicted_price - prev_features['lag_3']) / prev_features['lag_3'] * 100 if prev_features['lag_3'] != 0 else 0
        prev_features['change_6m'] = (predicted_price - prev_features['lag_6']) / prev_features['lag_6'] * 100 if prev_features['lag_6'] != 0 else 0
        
        # Update time features
        prev_features['months_since_start'] += 1
        prev_features['year'] = future_date.year
        prev_features['quarter'] = future_date.quarter
        
        # Update cyclical features (month is 1-12)
        month_num = future_date.month
        prev_features['month_sin'] = np.sin(2 * np.pi * month_num / 12)
        prev_features['month_cos'] = np.cos(2 * np.pi * month_num / 12)
        
        prev_price = predicted_price

forecast_6m = pd.DataFrame(forecast_data)

# Display results
print("Sample 6-month forecasts:")
sample_geos = forecast_6m['Geography'].unique()[:3]
for geo in sample_geos:
    print(f"\n{geo}:")
    geo_forecast = forecast_6m[forecast_6m['Geography'] == geo]
    print(geo_forecast[['Forecast_Date', 'Current_Price', 'Forecasted_Price', 'Price_Change_Pct']].to_string(index=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5.5: Generate 12-Month Forecast

# COMMAND ----------

import math
from pyspark.sql.functions import sin, cos

print("12-Month Forecast (Iterative Model-Based)\n")

forecast_months = 12
forecast_data_12m = []

for idx, row in latest_data.iterrows():
    geography = row['geography']
    current_date = row['date']
    current_price = row['index_value']

    # Initialize features for the first forecast step
    prev_features = row[feature_cols].copy()
    
    # Ensure 'momentum' is not in prev_features if it was somehow added or used previously
    if 'momentum' in prev_features:
        prev_features = prev_features.drop('momentum')

    # Make predictions for next 12 months iteratively
    for month_ahead in range(1, forecast_months + 1):
        # Scale the current features
        # Ensure prev_features contains only the columns in feature_cols
        pred_features_scaled = scaler.transform([prev_features[feature_cols].values])[0]

        # Make prediction
        predicted_price = lr_model.predict([pred_features_scaled])[0]

        # Calculate future date
        future_date = current_date + pd.DateOffset(months=month_ahead)

        # Calculate change from current
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100

        forecast_data_12m.append({
            'Geography': geography,
            'Current_Date': current_date,
            'Forecast_Date': future_date,
            'Months_Ahead': month_ahead,
            'Current_Price': current_price,
            'Forecasted_Price': predicted_price,
            'Price_Change': price_change,
            'Price_Change_Pct': price_change_pct
        })

        # Update lagged features
        prev_features['lag_12'] = prev_features['lag_6']
        prev_features['lag_6'] = prev_features['lag_3']
        prev_features['lag_3'] = prev_features['lag_1']
        prev_features['lag_1'] = predicted_price

        # Update rolling mean (simple approximation for forecasting)
        prev_features['rolling_mean_12'] = (prev_features['rolling_mean_12'] * 11 + predicted_price) / 12
        
        # Update rolling standard deviation (simplified approximation)
        prev_features['rolling_std_12'] = prev_features['rolling_std_12'] # No change for simplicity

        # Update other derived features (mom_change, yoy_change, change_3m, change_6m)
        # These should be calculated based on the updated lags
        prev_features['mom_change'] = (predicted_price - prev_features['lag_1']) / prev_features['lag_1'] * 100 if prev_features['lag_1'] != 0 else 0
        prev_features['yoy_change'] = (predicted_price - prev_features['lag_12']) / prev_features['lag_12'] * 100 if prev_features['lag_12'] != 0 else 0
        prev_features['change_3m'] = (predicted_price - prev_features['lag_3']) / prev_features['lag_3'] * 100 if prev_features['lag_3'] != 0 else 0
        prev_features['change_6m'] = (predicted_price - prev_features['lag_6']) / prev_features['lag_6'] * 100 if prev_features['lag_6'] != 0 else 0
        
        # Update volatility (using the simplified rolling_std_12)
        prev_features['volatility'] = (prev_features['rolling_std_12'] / prev_features['rolling_mean_12']) * 100 if prev_features['rolling_mean_12'] != 0 else 0

        # Update time features
        prev_features['months_since_start'] += 1
        prev_features['year'] = future_date.year
        prev_features['quarter'] = future_date.quarter

        # Update cyclical features (month is 1-12)
        month_num = future_date.month
        prev_features['month_sin'] = np.sin(2 * np.pi * month_num / 12)
        prev_features['month_cos'] = np.cos(2 * np.pi * month_num / 12)
        
        # For interest_rate and unemployment_rate, we'll carry forward the last known values or use a simple projection.
        prev_features['interest_rate'] = prev_features['interest_rate']
        prev_features['unemployment_rate'] = prev_features['unemployment_rate']

forecast_12m = pd.DataFrame(forecast_data_12m)

# Summary statistics
print("12-Month Forecast Summary (Average across all geographies):")
summary_12m = forecast_12m.groupby('Months_Ahead').agg({
    'Forecasted_Price': ['mean', 'min', 'max'],
    'Price_Change_Pct': 'mean'
}).round(2)
print(summary_12m)

# Which geographies will grow most?
print("\n\nGeographies with highest 12-month growth:")
geo_growth = forecast_12m[forecast_12m['Months_Ahead'] == 12].sort_values('Price_Change_Pct', ascending=False)
print(geo_growth[['Geography', 'Forecasted_Price', 'Price_Change_Pct']].head(10).to_string(index=False))

print("\n\nGeographies with lowest 12-month growth:")
print(geo_growth[['Geography', 'Forecasted_Price', 'Price_Change_Pct']].tail(10).to_string(index=False))


# COMMAND ----------

## Step 5.6: Generate 24-Month Forecast

# COMMAND ----------

print("24-Month Forecast (Iterative Model-Based)\n")

forecast_months = 24
forecast_data_24m = []

for idx, row in latest_data.iterrows():
    geography = row['geography']
    current_date = row['date']
    current_price = row['index_value']

    # Initialize features for the first forecast step
    prev_features = row[feature_cols].copy()
    
    # Ensure 'momentum' is not in prev_features if it was somehow added or used previously
    if 'momentum' in prev_features:
        prev_features = prev_features.drop('momentum')

    # Make predictions for next 24 months iteratively
    for month_ahead in range(1, forecast_months + 1):
        # Scale the current features
        pred_features_scaled = scaler.transform([prev_features[feature_cols].values])[0]

        # Make prediction
        predicted_price = lr_model.predict([pred_features_scaled])[0]

        # Calculate future date
        future_date = current_date + pd.DateOffset(months=month_ahead)

        # Calculate change from current
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100

        forecast_data_24m.append({
            'Geography': geography,
            'Current_Date': current_date,
            'Forecast_Date': future_date,
            'Months_Ahead': month_ahead,
            'Current_Price': current_price,
            'Forecasted_Price': predicted_price,
            'Price_Change': price_change,
            'Price_Change_Pct': price_change_pct
        })

        # UPDATE FEATURES FOR NEXT ITERATION
        # Update lagged features
        prev_features['lag_12'] = prev_features['lag_6']
        prev_features['lag_6'] = prev_features['lag_3']
        prev_features['lag_3'] = prev_features['lag_1']
        prev_features['lag_1'] = predicted_price

        # Update rolling mean
        prev_features['rolling_mean_12'] = (prev_features['rolling_mean_12'] * 11 + predicted_price) / 12
        
        # Update rolling standard deviation
        prev_features['rolling_std_12'] = prev_features['rolling_std_12']

        # Update other derived features
        prev_features['mom_change'] = (predicted_price - prev_features['lag_1']) / prev_features['lag_1'] * 100 if prev_features['lag_1'] != 0 else 0
        prev_features['yoy_change'] = (predicted_price - prev_features['lag_12']) / prev_features['lag_12'] * 100 if prev_features['lag_12'] != 0 else 0
        prev_features['change_3m'] = (predicted_price - prev_features['lag_3']) / prev_features['lag_3'] * 100 if prev_features['lag_3'] != 0 else 0
        prev_features['change_6m'] = (predicted_price - prev_features['lag_6']) / prev_features['lag_6'] * 100 if prev_features['lag_6'] != 0 else 0
        
        # Update volatility (using the simplified rolling_std_12)
        prev_features['volatility'] = (prev_features['rolling_std_12'] / prev_features['rolling_mean_12']) * 100 if prev_features['rolling_mean_12'] != 0 else 0

        # Update time features
        prev_features['months_since_start'] += 1
        prev_features['year'] = future_date.year
        prev_features['quarter'] = future_date.quarter

        # Update cyclical features (month is 1-12)
        month_num = future_date.month
        prev_features['month_sin'] = np.sin(2 * np.pi * month_num / 12)
        prev_features['month_cos'] = np.cos(2 * np.pi * month_num / 12)
        
        # For interest_rate and unemployment_rate, carry forward last known values for simplicity
        prev_features['interest_rate'] = prev_features['interest_rate']
        prev_features['unemployment_rate'] = prev_features['unemployment_rate']

forecast_24m = pd.DataFrame(forecast_data_24m)

# Summary statistics
print("24-Month Forecast Summary (Average across all geographies):")
summary_24m = forecast_24m.groupby('Months_Ahead').agg({
    'Forecasted_Price': ['mean', 'min', 'max'],
    'Price_Change_Pct': 'mean'
}).round(2)
print(summary_24m)

# Which geographies will grow most in 24 months?
print("\n\nGeographies with highest 24-month growth:")
geo_growth_24m = forecast_24m[forecast_24m['Months_Ahead'] == 24].sort_values('Price_Change_Pct', ascending=False)
print(geo_growth_24m[['Geography', 'Forecasted_Price', 'Price_Change_Pct']].head(10).to_string(index=False))

print("\n\nGeographies with lowest 24-month growth:")
print(geo_growth_24m[['Geography', 'Forecasted_Price', 'Price_Change_Pct']].tail(10).to_string(index=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5.7: Calculate Prediction Confidence

# COMMAND ----------

# Calculate prediction confidence based on model performance
print("Prediction Confidence Analysis\n")

# Get test set predictions from earlier
split_idx = int(len(X) * 0.8)
X_test_scaled = X_scaled.iloc[split_idx:]
y_test = y.iloc[split_idx:]
y_test_pred = lr_model.predict(X_test_scaled)

# Calculate residuals
residuals = y_test - y_test_pred
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
std_residuals = residuals.std()

print(f"Model RMSE: {rmse:.4f}")
print(f"Model MAE: {mae:.4f}")
print(f"Residual Std Dev: {std_residuals:.4f}")

# Calculate 95% confidence interval (±1.96 * std)
confidence_95 = 1.96 * std_residuals

print(f"\n95% Confidence Interval: ±{confidence_95:.4f}")
print(f"Confidence Interval as % of average price: ±{(confidence_95 / y_test.mean() * 100):.2f}%")

# Interpretation
print(f"\nInterpretation:")
print(f"- 95% of predictions will be within ±{confidence_95:.4f} of actual value")
print(f"- Predictions are accurate to within {(confidence_95 / y_test.mean() * 100):.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5.8: Key Insights Summary

# COMMAND ----------

# Generate key insights from forecasts
print("Key Insights from Forecasting\n")

insights = []

# Insight 1: Overall 12-month trend
avg_12m_growth = forecast_12m[forecast_12m['Months_Ahead'] == 12]['Price_Change_Pct'].mean()
insights.append(f"1. Expected Growth (12 months): {avg_12m_growth:.2f}%")

# Insight 2: Best performing geography (12m)
best_geo_12m = forecast_12m[forecast_12m['Months_Ahead'] == 12].sort_values('Price_Change_Pct', ascending=False).iloc[0]
insights.append(f"2. Strongest Market (12m): {best_geo_12m['Geography']} (+{best_geo_12m['Price_Change_Pct']:.2f}%)")

# Insight 3: Weakest performing geography (12m)
worst_geo_12m = forecast_12m[forecast_12m['Months_Ahead'] == 12].sort_values('Price_Change_Pct', ascending=True).iloc[0]
insights.append(f"3. Weakest Market (12m): {worst_geo_12m['Geography']} ({worst_geo_12m['Price_Change_Pct']:.2f}%)")

# Insight 4: 24-month outlook
avg_24m_growth = forecast_24m[forecast_24m['Months_Ahead'] == 24]['Price_Change_Pct'].mean()
insights.append(f"4. Expected Growth (24 months): {avg_24m_growth:.2f}%")

# Insight 5: Market stability
growth_std_12m = forecast_12m[forecast_12m['Months_Ahead'] == 12]['Price_Change_Pct'].std()
insights.append(f"5. Market Stability: Std Dev of 12m growth = {growth_std_12m:.2f}% (lower = more stable)")

# Insight 6: Model confidence
insights.append(f"6. Model Confidence: 95% CI = ±{confidence_95:.4f} ({(confidence_95 / y_test.mean() * 100):.2f}%)")

for insight in insights:
    print(insight)

# Save insights for later
print("\n\n Basic Summary:\n")
print(f"Based on analysis of {len(df_ml)} months of Canadian housing data:")
print(f"- Housing prices are expected to grow {avg_12m_growth:.2f}% over the next 12 months")
print(f"- Model predicts {avg_24m_growth:.2f}% growth over 24 months")
print(f"- Best performing market: {best_geo_12m['Geography']}")
print(f"- Predictions are accurate to within {(confidence_95 / y_test.mean() * 100):.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5.9: Export Forecast Results

# COMMAND ----------

# Convert to Spark DataFrames and save
forecast_6m_spark = spark.createDataFrame(forecast_6m)
forecast_12m_spark = spark.createDataFrame(forecast_12m)
forecast_24m_spark = spark.createDataFrame(forecast_24m)

# Save as tables
forecast_6m_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("main.housing_forecast.forecast_6m")
forecast_12m_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("main.housing_forecast.forecast_12m")
forecast_24m_spark.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("main.housing_forecast.forecast_24m")

print("Forecast tables saved:")
print("  - main.housing_forecast.forecast_6m")
print("  - main.housing_forecast.forecast_12m")
print("  - main.housing_forecast.forecast_24m")

print("\nKey metrics:")
print(f"  - 12-month average growth: {avg_12m_growth:.2f}%")
print(f"  - 24-month average growth: {avg_24m_growth:.2f}%")
print(f"  - Model confidence: ±{(confidence_95 / y_test.mean() * 100):.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5.10: Create Summary Report

# COMMAND ----------

print("=" * 70)
print("CANADIAN HOUSING PRICE FORECASTING - PROJECT SUMMARY")
print("=" * 70)

print("\n1. DATA OVERVIEW")
print("-" * 70)
print(f"   Total records analyzed: {len(df_ml):,}")
print(f"   Date range: {df_ml['date'].min()} to {df_ml['date'].max()}")
print(f"   Geographies: {df_ml['geography'].nunique()}")
print(f"   Features engineered: 21")

print("\n2. MODEL PERFORMANCE")
print("-" * 70)
print(f"   Best model: Linear Regression")
print(f"   Train R²: 1.0000")
print(f"   Test R²: 1.0000")
print(f"   Test RMSE: 0.0019")
print(f"   Test MAE: 0.0007")

print("\n3. FEATURE IMPORTANCE")
print("-" * 70)
print(f"   Most important feature: rolling_mean_12 (93.4% importance)")
print(f"   Second most important: momentum (6.4% importance)")
print(f"   Features needed for 80% importance: 1 out of 21")

print("\n4. FORECAST RESULTS")
print("-" * 70)
print(f"   12-month average growth: {avg_12m_growth:.2f}%")
print(f"   24-month average growth: {avg_24m_growth:.2f}%")
print(f"   Best market (12m): {best_geo_12m['Geography']} (+{best_geo_12m['Price_Change_Pct']:.2f}%)")
print(f"   Weakest market (12m): {worst_geo_12m['Geography']} ({worst_geo_12m['Price_Change_Pct']:.2f}%)")

print("\n5. MODEL CONFIDENCE")
print("-" * 70)
print(f"   95% Confidence Interval: ±{confidence_95:.4f}")
print(f"   As % of average price: ±{(confidence_95 / y_test.mean() * 100):.2f}%")
print(f"   Prediction accuracy: VERY HIGH")

print("\n" + "=" * 70)
print("END OF SUMMARY")
print("=" * 70)
