##**Load the required libraries.**
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.float_= np.float64
from google.cloud import bigquery
from google.cloud.bigquery import Client
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error)
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from prophet import Prophet


def create_data_table(client, project_id, dataset_id, table_name, start_date, end_date, limit=3):
    """
    Creates or replaces a data table in BigQuery containing top-selling items within a specified date range.

    Parameters
    ----------
    client : bigquery.Client
        The BigQuery client.
    project_id : str
        The Google Cloud project ID.
    dataset_id : str
        The BigQuery dataset ID.
    table_name : str
        The name of the table to create or replace.
    start_date : str
        The start date for filtering data (format: 'YYYY-MM-DD').
    end_date : str
        The end date for filtering data (format: 'YYYY-MM-DD').
    limit : int, optional
        The number of top-selling items to include (default is 3).
    """
    query = f"""
  CREATE OR REPLACE TABLE `{dataset_id}.{table_name}` AS (
    WITH topsellingitems AS (
        SELECT
            item_description,
            COUNT(item_description) AS cnt_transactions
        FROM
            `bigquery-public-data.iowa_liquor_sales.sales`
        GROUP BY
            item_description
        ORDER BY
            cnt_transactions DESC
        LIMIT {limit}  -- Top N items
    ),
    sales_data AS (
        SELECT
            date,
            item_description AS item_name,
            SUM(bottles_sold) AS total_amount_sold,
            AVG(SUM(bottles_sold)) OVER (PARTITION BY item_description ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_avg_7days,  -- 7-day rolling average
            EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
            CASE
                WHEN EXTRACT(DAYOFWEEK FROM date) IN (1, 7) THEN 1  -- 1=Sunday, 7=Saturday
                ELSE 0
            END AS is_weekend
        FROM
            `bigquery-public-data.iowa_liquor_sales.sales`
        WHERE
            date BETWEEN @START_DATE AND @END_DATE
            AND item_description IN (SELECT item_description FROM topsellingitems)
        GROUP BY
            date, item_name
    )
    SELECT
        date,
        item_name,
        total_amount_sold,
        rolling_avg_7days,
        day_of_week,
        is_weekend
    FROM
        sales_data);
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("START_DATE", "STRING", start_date),
            bigquery.ScalarQueryParameter("END_DATE", "STRING", end_date),
        ]
    )
    job = client.query(query, job_config=job_config)
    print(f"Table {dataset_id}.{table_name} created.")

def create_arima_model(client, dataset_id, model_name):
    """
    Creates or replaces an ARIMA model in BigQuery.

    Parameters
    ----------
    client : bigquery.Client
        The BigQuery client.
    dataset_id : str
        The BigQuery dataset ID.
    model_name : str
        The name of the model to create or replace.
    """
    query = f"""
    CREATE OR REPLACE MODEL {dataset_id}.{model_name}
    OPTIONS(
      MODEL_TYPE='ARIMA_PLUS',
      TIME_SERIES_TIMESTAMP_COL='date',
      TIME_SERIES_DATA_COL='total_amount_sold',
      TIME_SERIES_ID_COL='item_name',
      HOLIDAY_REGION='US',
      AUTO_ARIMA=TRUE,
      DATA_FREQUENCY='DAILY',
      ADJUST_STEP_CHANGES=TRUE
    ) AS
    SELECT
        date,
        item_name,
        total_amount_sold
    FROM
      {dataset_id}.training_data
    """
    job = client.query(query)
    print(f"Model {dataset_id}.{model_name} created.")

def forecast_model(client, dataset_id, model_name, horizon=30, confidence_level=0.9):
    """
    Generates forecasts using an ARIMA model in BigQuery.

    Parameters
    ----------
    client : bigquery.Client
        The BigQuery client.
    dataset_id : str
        The BigQuery dataset ID.
    model_name : str
        The name of the model to use for forecasting.
    horizon : int, optional
        The number of values to forecast (default is 30).
    confidence_level : float, optional
        The confidence level for the forecast (default is 0.9).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the forecasted data.
    """
    query = f'''
        SELECT
          *
        FROM
          ML.FORECAST(MODEL {dataset_id}.{model_name},
                      STRUCT({horizon} AS horizon,
                              {confidence_level} AS confidence_level)
                     )
    '''
    job = client.query(query)
    forecast_df = job.to_dataframe()
    return forecast_df

def evaluate_model(dfactual, dfforecast):
    """
    Evaluates the accuracy of a forecast model.

    Parameters
    ----------
    dfactual : pandas.DataFrame
        The actual data.
    dfforecast : pandas.DataFrame
        The forecasted data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the error metrics for each item.
    """
    dfactual['date'] = pd.to_datetime(dfactual['date']).dt.date
    if 'forecast_timestamp' in dfforecast.columns:
        dfforecast['date'] = pd.to_datetime(dfforecast['forecast_timestamp']).dt.date
    else:
        dfforecast['date'] = pd.to_datetime(dfforecast['date']).dt.date

    item_list = dfactual['item_name'].unique()
    errors = {"item_name": [], "MAE": [], "MAPE": [], "MSE": [], "RMSE": [], "SMAPE": []}

    for i in item_list:
        actual_values = dfactual[dfactual["item_name"] == i].set_index('date')
        forecast_values = dfforecast[dfforecast["item_name"] == i].set_index('date')

        aligned = actual_values.join(forecast_values, how='inner', lsuffix='_actual', rsuffix='_forecast')

        if 'forecast_value' in aligned.columns:
            yhat = 'forecast_value'
            y = 'total_amount_sold'
        else:
            yhat = 'total_amount_sold_forecast'
            y = 'total_amount_sold_actual'

        mae = mean_absolute_error(aligned[y], aligned[yhat])
        mape = (abs((aligned[y] - aligned[yhat]) / aligned[y]).mean()) * 100
        mse = mean_squared_error(aligned[y], aligned[yhat], squared=True)
        rmse = mean_squared_error(aligned[y], aligned[yhat], squared=False)
        smape = (abs(aligned[y] - aligned[yhat]) / (abs(aligned[y]) + abs(aligned[yhat])) * 2).mean() * 100

        errors["item_name"].append(i)
        errors["MAE"].append(mae)
        errors["MAPE"].append(mape)
        errors["MSE"].append(mse)
        errors["RMSE"].append(rmse)
        errors["SMAPE"].append(smape)

    errors_df = pd.DataFrame(errors)
    return errors_df

def plot_historical_and_forecast(input_timeseries, timestamp_col_name, data_col_name, forecast_output=None, actual=None, title=None, plotstartdate=None):
    """
    Plots historical data along with forecasted and actual data.

    Parameters
    ----------
    input_timeseries : pandas.DataFrame
        The input historical data.
    timestamp_col_name : str
        The name of the timestamp column.
    data_col_name : str
        The name of the data column.
    forecast_output : pandas.DataFrame, optional
        The forecasted data (default is None).
    actual : pandas.DataFrame, optional
        The actual data (default is None).
    title : str, optional
        The title of the plot (default is None).
    plotstartdate : str, optional
        The start date for plotting data (default is None).
    """
    if plotstartdate:
        input_timeseries[timestamp_col_name] = pd.to_datetime(input_timeseries[timestamp_col_name])
        input_timeseries = input_timeseries[input_timeseries[timestamp_col_name] >= pd.to_datetime(plotstartdate)]

    input_timeseries = input_timeseries.sort_values(timestamp_col_name)

    plt.figure(figsize=(20, 6))
    plt.plot(input_timeseries[timestamp_col_name], input_timeseries[data_col_name], label='Historical')
    plt.xlabel(timestamp_col_name)
    plt.ylabel(data_col_name)

    if forecast_output is not None:
        forecast_output = forecast_output.sort_values('forecast_timestamp')
        forecast_output['forecast_timestamp'] = pd.to_datetime(forecast_output['forecast_timestamp'])
        x_data = forecast_output['forecast_timestamp']
        y_data = forecast_output['forecast_value']
        confidence_level = forecast_output['confidence_level'].iloc[0] * 100
        low_CI = forecast_output['confidence_interval_lower_bound']
        upper_CI = forecast_output['confidence_interval_upper_bound']
        plt.plot(x_data, y_data, alpha=1, label='Forecast', linestyle='--')
        plt.fill_between(x_data, low_CI, upper_CI, color='#539caf', alpha=0.4, label=f'{confidence_level} confidence interval')

    if actual is not None:
        actual = actual.sort_values(timestamp_col_name)
        plt.plot(actual[timestamp_col_name], actual[data_col_name], label='Actual', linestyle='--')

    plt.title(f'{title}', fontsize=16)
    plt.legend(loc='upper center', prop={'size': 16})

def test_stationarity(timeseries):
    """
    Tests the stationarity of a time series using the Dickey-Fuller test.

    Parameters
    ----------
    timeseries : pandas.Series
        The time series data to test.

    Returns
    -------
    None
    """
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    print(dfoutput)
    if dfoutput['p-value'] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary.")

def plot_historical_and_forecastprophet(input_timeseries, timestamp_col_name, data_col_name, forecast_output=None, actual=None, title=None, plotstartdate=None):
    """
    Plots historical data along with forecasted and actual data.

    Parameters
    ----------
    input_timeseries : pandas.DataFrame
        The input historical data.
    timestamp_col_name : str
        The name of the timestamp column.
    data_col_name : str
        The name of the data column.
    forecast_output : pandas.DataFrame, optional
        The forecasted data (default is None).
    actual : pandas.DataFrame, optional
        The actual data (default is None).
    title : str, optional
        The title of the plot (default is None).
    plotstartdate : str, optional
        The start date for plotting data (default is None).
    """
    if plotstartdate:
        input_timeseries[timestamp_col_name] = pd.to_datetime(input_timeseries[timestamp_col_name])
        input_timeseries = input_timeseries[input_timeseries[timestamp_col_name] >= pd.to_datetime(plotstartdate)]

    input_timeseries = input_timeseries.sort_values(timestamp_col_name)


    plt.figure(figsize=(20, 6))
    plt.plot(input_timeseries[timestamp_col_name], input_timeseries[data_col_name], label='Historical', linewidth=1)
    plt.xlabel(timestamp_col_name)
    plt.ylabel(data_col_name)

    if forecast_output is not None:
        forecast_output = forecast_output.sort_values('ds')
        forecast_output['ds'] = pd.to_datetime(forecast_output['ds'])

        forecast_output = forecast_output[(forecast_output['ds'] >= pd.to_datetime(actual.date.min())) &
                                          (forecast_output['ds'] <= pd.to_datetime(actual.date.max()))]
        x_data = forecast_output['ds']
        y_data = forecast_output['yhat']
        low_CI = forecast_output['yhat_lower']
        upper_CI = forecast_output['yhat_upper']

        plt.plot(x_data, y_data, alpha=1, label='Forecast', linestyle='--', color='orange', linewidth=1)
        plt.fill_between(x_data, low_CI, upper_CI, color='#539caf', alpha=0.4, label='90.0 confidence interval')

    if actual is not None:
        actual = actual.sort_values(timestamp_col_name)
        plt.plot(actual[timestamp_col_name], actual[data_col_name], label='Actual', linestyle='--', color='green', linewidth=1)

    plt.title(f'{title}', fontsize=16)
    plt.legend(loc='upper center', prop={'size': 16})
    plt.show()

def evaluate_profetmodel(dfactual, dfforecast):
    """
    Evaluates the accuracy of a forecast model using actual and forecasted data.

    Parameters
    ----------
    dfactual : pandas.DataFrame
        The actual data.
    dfforecast : pandas.DataFrame
        The forecasted data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the error metrics for each item.
    """
    dfactual['date'] = pd.to_datetime(dfactual['date']).dt.date
    dfforecast['ds'] = pd.to_datetime(dfforecast['ds']).dt.date

    item_list = dfactual['item_name'].unique()
    errors = {"item_name": [], "MAE": [], "MAPE": [], "MSE": [], "RMSE": [], "SMAPE": []}

    for i in item_list:
        actual_values = dfactual[dfactual["item_name"] == i].set_index('date')
        forecast_values = dfforecast.set_index('ds')

        aligned = actual_values.join(forecast_values, how='inner', lsuffix='_actual', rsuffix='_forecast')

        mae = mean_absolute_error(aligned['total_amount_sold'], aligned['yhat'])
        mape = (abs((aligned['total_amount_sold'] - aligned['yhat']) / aligned['total_amount_sold']).mean()) * 100
        mse = mean_squared_error(aligned['total_amount_sold'], aligned['yhat'], squared=True)
        rmse = mean_squared_error(aligned['total_amount_sold'], aligned['yhat'], squared=False)
        smape = (abs(aligned['total_amount_sold'] - aligned['yhat']) / (abs(aligned['total_amount_sold']) + abs(aligned['yhat'])) * 2).mean() * 100

        errors["item_name"].append(i)
        errors["MAE"].append(mae)
        errors["MAPE"].append(mape)
        errors["MSE"].append(mse)
        errors["RMSE"].append(rmse)
        errors["SMAPE"].append(smape)

    errors_df = pd.DataFrame(errors)
    return errors_df

def create_and_forecast_prophet(data, forecast_period=30, changepoint_prior_scale=0.05, seasonality_mode='additive', seasonality_prior_scale=10.0, interval_width=0.90):
    """
    Creates and forecasts using the Prophet model with specified hyperparameters.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data for the Prophet model.
    forecast_period : int, optional
        The number of periods to forecast (default is 30).
    changepoint_prior_scale : float, optional
        The changepoint prior scale for the Prophet model (default is 0.05).
    seasonality_mode : str, optional
        The seasonality mode for the Prophet model (default is 'additive').
    seasonality_prior_scale : float, optional
        The seasonality prior scale for the Prophet model (default is 10.0).
    interval_width : float, optional
        The width of the confidence intervals (default is 0.90).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the forecasted values.
    """
    data = data.rename(columns={'date': 'ds', 'total_amount_sold': 'y'})
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale, seasonality_mode=seasonality_mode, seasonality_prior_scale=seasonality_prior_scale, interval_width=interval_width)
    model.add_country_holidays(country_name='US')
    model.fit(data)
    future = model.make_future_dataframe(periods=forecast_period*2)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def tune_prophet_model(data, param_grid, forecast_period=30):
    """
    Tunes Prophet hyperparameters and finds the best model based on error metrics.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data for the Prophet model.
    param_grid : list of dict
        A list of dictionaries containing hyperparameter combinations to try.
    forecast_period : int, optional
        The number of periods to forecast (default is 30).

    Returns
    -------
    tuple
        A tuple containing the best hyperparameters and the best forecasted values.
    """
    best_mae = float('inf')
    best_params = None
    best_forecast = None

    for params in param_grid:
        forecast = create_and_forecast_prophet(data, forecast_period, **params)
        errors_df = evaluate_profetmodel(data, forecast)
        mae = errors_df['MAE'].mean()

        if mae < best_mae:
            best_mae = mae
            best_params = params
            best_forecast = forecast

    return best_params, best_forecast

def create_and_train_arima_model(project_id, dataset_id, table_name, model_name):
    """
    Creates a training data table and trains an ARIMA model using BigQuery.

    This function performs the following steps:
    1. Creates or replaces a table in the specified dataset that contains
       sales data for the top 3 selling items over the last 3 years.
    2. Creates or replaces an ARIMA model in the specified dataset using
       the training data.

    Args:
        project_id (str): The Google Cloud project ID.
        dataset_id (str): The BigQuery dataset ID where the table and model will be created.
        table_name (str): The name of the table to be created or replaced.
        model_name (str): The name of the ARIMA model to be created or replaced.

    Returns:
        str: A success message indicating that the ARIMA model has been created and trained.
    """
    # Initialize BigQuery client with explicit project ID
    client = bigquery.Client(project=project_id)

    # Step 1: Create the training data table
    create_table_query = f"""
    CREATE OR REPLACE TABLE `{dataset_id}.{table_name}` AS (
        WITH topsellingitems AS (
            SELECT
                item_description,
                COUNT(item_description) AS cnt_transactions
            FROM
                `bigquery-public-data.iowa_liquor_sales.sales`
            GROUP BY
                item_description
            ORDER BY
                cnt_transactions DESC
            LIMIT 3
        )
        SELECT
            date,
            item_description AS item_name,
            SUM(bottles_sold) AS total_amount_sold
        FROM
            `bigquery-public-data.iowa_liquor_sales.sales`
        WHERE
            date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 3 YEAR) AND CURRENT_DATE()
            AND item_description IN (SELECT item_description FROM topsellingitems)
        GROUP BY
            date, item_name
    );
    """

    # Execute the query to create the training data table
    client.query(create_table_query).result()

    # Step 2: Create or replace the ARIMA model
    create_model_query = f"""
    CREATE OR REPLACE MODEL `{dataset_id}.{model_name}`
    OPTIONS(
        MODEL_TYPE='ARIMA_PLUS',
        TIME_SERIES_TIMESTAMP_COL='date',
        TIME_SERIES_DATA_COL='total_amount_sold',
        TIME_SERIES_ID_COL='item_name',
        HOLIDAY_REGION='US',
        AUTO_ARIMA=TRUE,
        DATA_FREQUENCY='DAILY',
        ADJUST_STEP_CHANGES=TRUE
    ) AS
    SELECT
        date,
        item_name,
        total_amount_sold
    FROM
        `{dataset_id}.{table_name}`;
    """

    # Execute the query to create and train the ARIMA model
    client.query(create_model_query).result()

    return "ARIMA model created and trained successfully."


def create_and_insert_forecast_data(project_id, dataset_id, table_name, model_name, gcs_uri):
    """
    Creates a forecast table, inserts forecast data, exports the data to Google Cloud Storage,
    and returns the inserted forecast data as a DataFrame along with a success message.

    This function performs the following steps:
    1. Creates a table in the specified dataset if it doesn't already exist.
       The table schema includes `forecast_timestamp`, `item_name`, `forecast_value`, and `run_date`.
    2. Inserts forecast data into the specified table by running a forecast query on the
       specified ARIMA model. The forecast is generated for a 30-day horizon with a 90% confidence level.
    3. Exports the forecast data from the specified table to a CSV file in a specified
       Google Cloud Storage bucket.
    4. Returns the forecast data as a pandas DataFrame along with a success message.

    Args:
        project_id (str): The Google Cloud project ID.
        dataset_id (str): The BigQuery dataset ID where the table and model are located.
        table_name (str): The name of the table to be created or replaced.
        model_name (str): The name of the ARIMA model to be used for forecasting.
        gcs_uri (str): The URI of the Google Cloud Storage bucket where the forecast data will be exported.

    Returns:
        tuple: A tuple containing:
            - str: A success message indicating that the forecast data table was created, data was inserted,
              and the data was exported successfully.
            - pandas.DataFrame: The forecast data inserted into the table.
    """
    # Initialize BigQuery client with explicit project ID
    client = bigquery.Client(project=project_id)

    # Step 1: Create the forecast table if it doesn't exist
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}.{table_name}` (
        forecast_timestamp TIMESTAMP,
        item_name STRING,
        forecast_value FLOAT64,
        run_date DATE
    );
    """
    client.query(create_table_query).result()

    # Step 2: Insert forecast data into the forecast table
    insert_forecast_query = f"""
    INSERT INTO `{dataset_id}.{table_name}` (forecast_timestamp, item_name, forecast_value, run_date)
    SELECT
        forecast_timestamp,
        item_name,
        forecast_value,
        CURRENT_DATE() AS run_date
    FROM
        ML.FORECAST(MODEL `{dataset_id}.{model_name}`,
                    STRUCT(30 AS horizon,
                           0.90 AS confidence_level))
    ORDER BY
        forecast_timestamp ASC,
        item_name ASC;
    """
    client.query(insert_forecast_query).result()

    # Step 3: Export the forecast data to a Google Cloud Storage bucket
    export_data_query = f"""
    EXPORT DATA OPTIONS(
        uri='{gcs_uri}',
        format='CSV',
        overwrite=true
    ) AS
    SELECT
        *
    FROM
        `{dataset_id}.{table_name}`;
    """
    client.query(export_data_query).result()

    # Step 4: Retrieve the forecast data as a DataFrame
    query_forecast_data = f"""
    SELECT *
    FROM `{dataset_id}.{table_name}`
    ORDER BY forecast_timestamp ASC, item_name ASC;
    """
    query_job = client.query(query_forecast_data)
    df_forecast = query_job.to_dataframe()

    # Return success message and DataFrame
    message = "Forecast data table created, data inserted, and exported successfully."
    return message, df_forecast