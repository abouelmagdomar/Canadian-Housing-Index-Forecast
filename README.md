# Canadian Housing Index Forecast

ML project forecasting the Canadian Housing Market Index using StatCan data. Features robust engineering with real economic indicators (interest rates, unemployment) and cyclical components. Delivers iterative 6, 12, and 24-month regional forecasts while ensuring data integrity and eliminating leakage for reliable market insights.

## Features

-   **Data Ingestion**: Automated loading of historical Canadian housing price index data.
-   **Feature Engineering**: Creation of advanced features including lagged values, rolling statistics, growth rates, cyclical components (month_sin, month_cos), and integration of real economic indicators (interest rates, unemployment).
-   **Data Quality & Remediation**: Implementation of fixes for data leakage and cyclical feature calculation errors.
-   **Geographical Modeling**: Focus on regional trends by excluding aggregated national data.
-   **Machine Learning Model**: Utilizes Linear Regression for forecasting.
-   **Multi-Horizon Forecasting**: Generates 6, 12, and 24-month iterative forecasts.
-   **Insights Generation**: Summarizes key trends, best/worst performing geographies, and model confidence.

## Tech Stack

This project was developed and optimized for the **Databricks platform** and leverages the following technologies:

-   **Apache Spark**: For large-scale data processing and transformation.
-   **Python**: Primary programming language.
-   **Pandas**: For data manipulation within Python.
-   **Scikit-learn**: For machine learning model training (Linear Regression) and preprocessing (StandardScaler).
-   **Statistics Canada (StatCan)**: Authoritative source for economic indicator data.

## Project Structure

The repository is organized as follows:

```
housing_forecast_project/
├── Notebooks/
│   ├── 01_data_ingestion.py
│   ├── 02_data_transformation.py
│   ├── 03_exploratory_data_analysis.py
│   ├── 04_machine_learning_models.py
│   └── 05_predictions_and_insights.py
├── Source Data/
│   ├── 18100205.csv (Raw housing data from StatCan)
│   ├── interest_rate.csv (Historical interest rate data from StatCan)
│   └── unemployment_rate.csv (Historical unemployment rate data from StatCan)
└── Exports/
    ├── forecast_6m.csv
    ├── forecast_12m.csv
    └── forecast_24m.csv
    └── housing_price_index_features.csv
    └── housing_price_index_raw.csv
```

### Important: Modifying Table Destinations

This project uses Spark tables with the format `main.housing_forecast.<table_name>`. **You will need to modify these table destinations** to match your Databricks Unity Catalog or Hive Metastore setup. Specifically:

-   **`01_data_ingestion.py`**: Update the `table_name` variables for `housing_price_index_raw`, `interest_rate`, and `unemployment_rate` to reflect your desired catalog and schema (e.g., `your_catalog.your_schema.housing_price_index_raw`).
-   **`02_data_transformation.py`**: Update the `table_name` for `housing_price_index_features` and the `spark.table()` calls for `interest_rate` and `unemployment_rate` to match your chosen table paths.
-   **`05_predictions_and_insights.py`**: Update the `spark.table()` call for `housing_price_index_features` and the `saveAsTable()` calls for `forecast_6m`, `forecast_12m`, and `forecast_24m` to your desired catalog and schema.
