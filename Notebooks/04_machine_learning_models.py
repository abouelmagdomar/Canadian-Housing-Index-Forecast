# Databricks notebook source
# MAGIC %md
# MAGIC > ## Step 4.1: Setup

# COMMAND ----------

print("Starting machine learning model development...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.2: Load and Prepare Data

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the transformed features data
df = spark.table("main.housing_forecast.housing_price_index_features")

# Convert to Pandas
df_pandas = df.toPandas()

print(f"Total records: {len(df_pandas)}")
print(f"Date range: {df_pandas['date'].min()} to {df_pandas['date'].max()}")
print(f"Columns: {df_pandas.columns.tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.3: Prepare Features and Target

# COMMAND ----------

# Feature columns for ML models
feature_cols = [
    'lag_1',
    'lag_3',
    'lag_6',
    'lag_12',
    'rolling_mean_12',
    'rolling_std_12',
    'mom_change',
    'yoy_change',
    'change_3m',
    'change_6m',
    'volatility',
    'interest_rate',
    'unemployment_rate',
    'months_since_start',
    'month_sin',
    'month_cos',
    'quarter',
    'year'
]

# Drop rows with NaN only in feature columns
df_ml = df_pandas.dropna(subset=feature_cols)

print(f"Rows before dropna: {len(df_pandas)}")
print(f"Rows after dropna: {len(df_ml)}")
print(f"Rows removed: {len(df_pandas) - len(df_ml)}")
print(f"\nFeature columns: {len(feature_cols)}")
print(feature_cols)

target_col = 'index_value'

# Sort by Date to keep testing data set in the future
df_ml = df_ml.sort_values(by=['REF_DATE'],ascending=[True])

# Create X (features) and y (target)
X = df_ml[feature_cols].copy()
y = df_ml[target_col].copy()

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeatures: {feature_cols}")
print(f"Target: {target_col}")

# Check for any remaining NaN values
print(f"\nNaN values in X: {X.isnull().sum().sum()}")
print(f"NaN values in y: {y.isnull().sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.4: Split Data into Train and Test Sets

# COMMAND ----------

# DBTITLE 1,Split Data: 2018 vs 2019
# Split data: after 2018 (REF_DATE >= '2019-01-01') for test
df_ml['REF_DATE_dt'] = pd.to_datetime(df_ml['REF_DATE'])

# Find the position where the first REF_DATE in 2019 occurs
split_idx = df_ml[df_ml['REF_DATE_dt'] >= pd.Timestamp('2019-01-01')].index[0]

X_train = X.loc[:split_idx - 1]
X_test = X.loc[split_idx:]
y_train = y.loc[:split_idx - 1]
y_test = y.loc[split_idx:]

print("Data Split:\n")
print(f"Total records: {len(X)}")
print(f"Train records: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test records: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Retrieve date ranges
train_dates = df_ml.loc[:split_idx - 1]['date']
test_dates = df_ml.loc[split_idx:]['date']

print(f"\nTrain date range: {train_dates.min()} to {train_dates.max()}")
print(f"Test date range: {test_dates.min()} to {test_dates.max()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.5: Scale Features

# COMMAND ----------

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

print("Feature Scaling\n")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")
print(f"\nScaled feature statistics (train set):")
print(X_train_scaled.describe().round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.6: Train Linear Regression Model

# COMMAND ----------

print("MODEL 1: LINEAR REGRESSION\n")

# Create and train model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred_lr = lr_model.predict(X_train_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

# Calculate metrics
train_rmse_lr = np.sqrt(mean_squared_error(y_train, y_train_pred_lr))
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
train_mae_lr = mean_absolute_error(y_train, y_train_pred_lr)
test_mae_lr = mean_absolute_error(y_test, y_test_pred_lr)
train_r2_lr = r2_score(y_train, y_train_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)

print(f"Train RMSE: {train_rmse_lr:.4f}")
print(f"Test RMSE: {test_rmse_lr:.4f}")
print(f"Train MAE: {train_mae_lr:.4f}")
print(f"Test MAE: {test_mae_lr:.4f}")
print(f"Train R²: {train_r2_lr:.4f}")
print(f"Test R²: {test_r2_lr:.4f}")

# Feature importance (coefficients)
print("\n=== TOP 10 IMPORTANT FEATURES (by coefficient) ===")
feature_importance_lr = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(feature_importance_lr.head(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.7: Train Decision Tree Model

# COMMAND ----------

print("MODEL 2: DECISION TREE\n")

# Create and train model
dt_model = DecisionTreeRegressor(max_depth=10, random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred_dt = dt_model.predict(X_train_scaled)
y_test_pred_dt = dt_model.predict(X_test_scaled)

# Calculate metrics
train_rmse_dt = np.sqrt(mean_squared_error(y_train, y_train_pred_dt))
test_rmse_dt = np.sqrt(mean_squared_error(y_test, y_test_pred_dt))
train_mae_dt = mean_absolute_error(y_train, y_train_pred_dt)
test_mae_dt = mean_absolute_error(y_test, y_test_pred_dt)
train_r2_dt = r2_score(y_train, y_train_pred_dt)
test_r2_dt = r2_score(y_test, y_test_pred_dt)

print(f"Train RMSE: {train_rmse_dt:.4f}")
print(f"Test RMSE: {test_rmse_dt:.4f}")
print(f"Train MAE: {train_mae_dt:.4f}")
print(f"Test MAE: {test_mae_dt:.4f}")
print(f"Train R²: {train_r2_dt:.4f}")
print(f"Test R²: {test_r2_dt:.4f}")

# Feature importance
print("\n=== TOP 10 IMPORTANT FEATURES ===")
feature_importance_dt = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance_dt.head(10))

# COMMAND ----------

## Step 4.8: Train Random Forest Model

# COMMAND ----------

print("MODEL 3: RANDOM FOREST\n")

# Create and train model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred_rf = rf_model.predict(X_train_scaled)
y_test_pred_rf = rf_model.predict(X_test_scaled)

# Calculate metrics
train_rmse_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_rf))
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
train_mae_rf = mean_absolute_error(y_train, y_train_pred_rf)
test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)

print(f"Train RMSE: {train_rmse_rf:.4f}")
print(f"Test RMSE: {test_rmse_rf:.4f}")
print(f"Train MAE: {train_mae_rf:.4f}")
print(f"Test MAE: {test_mae_rf:.4f}")
print(f"Train R²: {train_r2_rf:.4f}")
print(f"Test R²: {test_r2_rf:.4f}")

# Feature importance
print("\n=== TOP 10 IMPORTANT FEATURES ===")
feature_importance_rf = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance_rf.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.9: Train Gradient Boosting Model

# COMMAND ----------

print("MODEL 4: GRADIENT BOOSTING\n")

# Create and train model
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred_gb = gb_model.predict(X_train_scaled)
y_test_pred_gb = gb_model.predict(X_test_scaled)

# Calculate metrics
train_rmse_gb = np.sqrt(mean_squared_error(y_train, y_train_pred_gb))
test_rmse_gb = np.sqrt(mean_squared_error(y_test, y_test_pred_gb))
train_mae_gb = mean_absolute_error(y_train, y_train_pred_gb)
test_mae_gb = mean_absolute_error(y_test, y_test_pred_gb)
train_r2_gb = r2_score(y_train, y_train_pred_gb)
test_r2_gb = r2_score(y_test, y_test_pred_gb)

print(f"Train RMSE: {train_rmse_gb:.4f}")
print(f"Test RMSE: {test_rmse_gb:.4f}")
print(f"Train MAE: {train_mae_gb:.4f}")
print(f"Test MAE: {test_mae_gb:.4f}")
print(f"Train R²: {train_r2_gb:.4f}")
print(f"Test R²: {test_r2_gb:.4f}")

# Feature importance
print("\n=== TOP 10 IMPORTANT FEATURES ===")
feature_importance_gb = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance_gb.head(10))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.10: Compare All Models

# COMMAND ----------

print("\n\n=== MODEL COMPARISON ===\n")

# Create comparison dataframe
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'Train RMSE': [train_rmse_lr, train_rmse_dt, train_rmse_rf, train_rmse_gb],
    'Test RMSE': [test_rmse_lr, test_rmse_dt, test_rmse_rf, test_rmse_gb],
    'Train MAE': [train_mae_lr, train_mae_dt, train_mae_rf, train_mae_gb],
    'Test MAE': [test_mae_lr, test_mae_dt, test_mae_rf, test_mae_gb],
    'Train R²': [train_r2_lr, train_r2_dt, train_r2_rf, train_r2_gb],
    'Test R²': [test_r2_lr, test_r2_dt, test_r2_rf, test_r2_gb]
})

print(comparison.round(4))

# Find best model
best_model_idx = comparison['Test RMSE'].idxmin()
best_model_name = comparison.loc[best_model_idx, 'Model']
best_test_rmse = comparison.loc[best_model_idx, 'Test RMSE']

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_model_name}")
print(f"Test RMSE: {best_test_rmse:.4f}")
print(f"{'='*50}")

# COMMAND ----------

## Step 4.11: Analyze Predictions vs Actual

# COMMAND ----------

print("PREDICTION ANALYSIS\n")

# Use the best model for analysis
if best_model_name == 'Linear Regression':
    best_model = lr_model
    y_test_pred_best = y_test_pred_lr
elif best_model_name == 'Decision Tree':
    best_model = dt_model
    y_test_pred_best = y_test_pred_dt
elif best_model_name == 'Random Forest':
    best_model = rf_model
    y_test_pred_best = y_test_pred_rf
else:
    best_model = gb_model
    y_test_pred_best = y_test_pred_gb

# Create comparison dataframe
predictions_df = pd.DataFrame({
    'Date': test_dates.values,
    'Actual': y_test.values,
    'Predicted': y_test_pred_best,
    'Error': y_test.values - y_test_pred_best,
    'Error_Pct': ((y_test.values - y_test_pred_best) / y_test.values * 100)
})

print("First 20 predictions:")
print(predictions_df.head(20).round(2))

print("\n\nPrediction error statistics:")
print(f"Mean Error: {predictions_df['Error'].mean():.4f}")
print(f"Std Error: {predictions_df['Error'].std():.4f}")
print(f"Mean Absolute Error: {predictions_df['Error'].abs().mean():.4f}")
print(f"Mean Error %: {predictions_df['Error_Pct'].mean():.2f}%")
print(f"Std Error %: {predictions_df['Error_Pct'].std():.2f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4.12: Feature Importance Summary

# COMMAND ----------

print("\n\n=== FEATURE IMPORTANCE SUMMARY ===\n")

# Get feature importance from best model
if best_model_name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
else:
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': np.abs(best_model.coef_)
    }).sort_values('Importance', ascending=False)

print(f"Top 15 Important Features ({best_model_name}):")
print(feature_importance.head(15).to_string(index=False))

# Calculate cumulative importance
feature_importance['Cumulative_Importance'] = feature_importance['Importance'].cumsum()
feature_importance['Cumulative_Importance_Pct'] = (feature_importance['Cumulative_Importance'] / 
                                                     feature_importance['Importance'].sum() * 100)

print("\n\nCumulative Feature Importance:")
print(feature_importance[['Feature', 'Importance', 'Cumulative_Importance_Pct']].head(15).to_string(index=False))

# How many features needed for 80% of importance?
features_for_80 = (feature_importance['Cumulative_Importance_Pct'] <= 80).sum() + 1
print(f"\nFeatures needed for 80% importance: {features_for_80} out of {len(feature_cols)}")