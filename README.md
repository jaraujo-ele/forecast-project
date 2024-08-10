# 📈 Forecasting Iowa Liquor Sales

![Iowa Liquor Sales](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR4n-jPW98Rw94a1TcaUgawZdTcbv93U2Hgcw&s)


## 📚 Function Definitions:

- **`create_data_table`**: Creates a table with the top-selling items within the specified date range.
- **`create_arima_model`**: Creates an ARIMA_PLUS model.
- **`forecast_model`**: Generates forecasts using the created ARIMA model.
- **`evaluate_model`**: Evaluates the forecast model using MAE, MAPE, MSE, RMSE, and SMAPE.
- **`plot_historical_and_forecast`**: Plots historical and forecasted data.
- **`test_stationarity`**: Tests the stationarity of a time series using the Augmented Dickey-Fuller test.
- **`create_and_forecast_prophet`**: Creates and forecasts using Prophet with hyperparameter tuning.
- **`tune_prophet_model`**: Tunes Prophet hyperparameters and finds the best model.
- **`create_and_train_arima_model`**: Combines the steps to create the training data table and train the ARIMA model.
- **`create_and_insert_forecast_data`**: Inserts the forecast data into a table, evaluates it, and exports it to Google Cloud Storage.

## 🛠️ Workflow Execution:

1. 🗂️ Create the training and test data tables using create_data_table (or within create_and_train_arima_model).
2. 🔍 Create the ARIMA model using create_arima_model (or within create_and_train_arima_model).
3. 📊 Forecast future values using the ARIMA model with forecast_model (or within create_and_insert_forecast_data).
4. 📜 Fetch historical and actual data for comparison.
5. 🧪 Test stationarity for each item's total amount sold using test_stationarity.
6. 📈 Evaluate the ARIMA model using evaluate_model.
7. 🖼️ Plot the historical and forecasted data with plot_historical_and_forecast.
8. 🎯 Tune and forecast using Prophet with the best hyperparameters using create_and_forecast_prophet and tune_prophet_model.
9. 📊 Evaluate the Prophet model using evaluate_model.
10. 📤 Export the forecast data to Google Cloud Storage using export_forecast_data (or within create_and_insert_forecast_data).
