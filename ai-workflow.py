import pandas as pd
import os
import json
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo

def get_feature_names(directory):
    all_files = glob(os.path.join(directory, "*.json"))
    first_file = all_files[0]
    
    with open(first_file, 'r') as file:
        data = json.load(file)
        
    feature_names = list(data[0].keys())
    return feature_names

def load_json_files_by_position(directory, feature_names):
    all_files = glob(os.path.join(directory, "*.json"))
    df_list = []
    
    for file in all_files:
        with open(file, 'r') as f:
            data = json.load(f)
            data_by_position = [list(entry.values()) for entry in data]
            df = pd.DataFrame(data_by_position, columns=feature_names)
            df_list.append(df)
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Clean invoice IDs by removing letters
    df['invoice'] = df['invoice'].str.replace(r'\D', '', regex=True)
    
    # Convert relevant columns to numeric types
    df['customer_id'] = pd.to_numeric(df['customer_id'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['times_viewed'] = pd.to_numeric(df['times_viewed'], errors='coerce')
    
    # Create a datetime column
    df['InvoiceDate'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    return df

def create_lag_features(df, lags):
    for lag in lags:
        df[f'revenue_lag_{lag}'] = df['price'].shift(lag)  # Use 'price' instead of 'revenue'
    return df

def retrain_model(country, historical_data):
    # Directory to save models
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create lagged features using the provided function
    lags = list(range(1, 31))  # Creating 30 lags
    historical_data_with_lags = create_lag_features(historical_data, lags)
    
    # Drop rows with NaN values (which are the first 30 rows after creating lags)
    historical_data_with_lags.dropna(inplace=True)

    # Define features (X) and target (y)
    X = historical_data_with_lags.drop('price', axis=1)
    y = historical_data_with_lags['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the Gradient Boosting model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model and historical data together
    last_30_days = historical_data_with_lags[-30:]  # Last 30 days of data
    model_data = {'model': model, 'historical_data': last_30_days}

    # Save the trained model with the country name and last date in the dataset
    last_date = historical_data_with_lags.index[-1].strftime('%Y-%m-%d')
    model_filename = os.path.join(model_dir, f'gradient_boosting_model_{country}_{last_date}.pkl')

    # Remove the old model file if it exists
    old_models = [f for f in os.listdir(model_dir) if f.startswith(f'gradient_boosting_model_{country}_')]
    for old_model in old_models:
        os.remove(os.path.join(model_dir, old_model))

    # Save the new model
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model_data, model_file)

    print(f"Model for {country} retrained and saved as '{model_filename}'")

def load_model_and_last_date(country):
    # Locate the model file based on the country
    model_dir = "saved_models"
    model_file = max([f for f in os.listdir(model_dir) if f.startswith(f"gradient_boosting_model_{country}")])
    model_path = os.path.join(model_dir, model_file)
    
    # Load the model and historical data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    historical_data = model_data['historical_data']
    
    # Extract the last date used in the model
    last_date_str = model_file.split('_')[-1].replace('.pkl', '')
    last_date = pd.to_datetime(last_date_str, format="%Y-%m-%d")
    
    return model, historical_data, last_date

def predict_until_date(country, target_date):
    # Load the model and historical data
    model, historical_data, last_date = load_model_and_last_date(country)
    
    # Check if the target_date is within the historical data
    if target_date <= historical_data.index[-1]:
        # If the target date is within the historical data, return data up to that date with a 'Date' column
        historical_subset = historical_data.loc[:target_date][['price']].reset_index()
        historical_subset.rename(columns={'InvoiceDate': 'Date'}, inplace=True)
        return historical_subset
    
    # If the target date is after the last date in the historical data, start predicting
    current_date = last_date + timedelta(days=1)
    predictions = []
    
    while current_date <= target_date:
        # Use the last 30 days of historical data to make a prediction
        X_new = historical_data.iloc[-1, 1:].values.reshape(1, -1)  # Exclude 'date' and use all lags
        
        # Create a DataFrame with appropriate column names for prediction
        X_new_df = pd.DataFrame(X_new, columns=historical_data.columns[1:])  # Exclude date column
        
        # Predict the next day's price
        predicted_price = model.predict(X_new_df)[0]
        
        # Create a new row for the historical data
        new_row = np.append([predicted_price], X_new[0, :-1])  # Add new prediction, shift the lags
        
        # Ensure the length matches the historical_data columns (including the date column)
        new_series = pd.Series(new_row, index=historical_data.columns[1:], name=current_date)  # Exclude date column
        
        # Insert the new predicted price in the first column (price), and shift the lags accordingly
        historical_data.loc[current_date] = [current_date] + new_row.tolist()
        
        # Record the prediction
        predictions.append((current_date, predicted_price))
        
        # Move to the next day
        current_date += timedelta(days=1)
    
    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['Date', 'Predicted_Price'])
    
    return predictions_df

# def plot_predictions_with_historical(historical_data, predictions):
#     # Combine historical data and predictions for plotting
#     combined_data = pd.concat([historical_data[['price']], predictions.set_index('Date')], axis=0)
    
#     # Plot the historical data and predictions
#     plt.figure(figsize=(14, 7))
#     plt.plot(combined_data.index, combined_data['price'], label='Historical Data')
#     plt.plot(predictions['Date'], predictions['Predicted_Price'], label='Predictions', linestyle='--', color='orange')
#     plt.title('Historical Data and Predictions')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def plot_predictions_with_historical(historical_data, predictions, target_date):
    # Ensure that the dates align by filling in any missing dates
    last_historical_date = historical_data.index[-1]

    if not predictions.empty:
        first_prediction_date = predictions['Date'].iloc[0]
        if first_prediction_date != last_historical_date + pd.Timedelta(days=1):
            date_range = pd.date_range(start=last_historical_date, end=first_prediction_date - pd.Timedelta(days=1))
            filler_data = pd.DataFrame(index=date_range, columns=historical_data.columns)

            # Remove all-NA columns in filler_data to avoid the FutureWarning
            filler_data.dropna(axis=1, how='all', inplace=True)

            historical_data = pd.concat([historical_data, filler_data])

    # Combine historical data and predictions for plotting
    combined_data = historical_data[['price']]
    if not predictions.empty and 'Predicted_Price' in predictions.columns:
        combined_data = pd.concat([combined_data, predictions.set_index('Date')], axis=0)
    
    # Filter the combined data to plot only up to the target date
    combined_data = combined_data.loc[:target_date]

    # Create the plot traces
    trace1 = go.Scatter(x=combined_data.index, y=combined_data['price'], mode='lines', name='Historical Data')

    traces = [trace1]
    if not predictions.empty and 'Predicted_Price' in predictions.columns:
        trace2 = go.Scatter(x=predictions['Date'], y=predictions['Predicted_Price'], mode='lines', name='Predictions', line=dict(dash='dash', color='orange'))
        traces.append(trace2)

    layout = go.Layout(title='Historical Data and Predictions',
                       xaxis={'title': 'Date'},
                       yaxis={'title': 'Price'},
                       hovermode='x')

    fig = go.Figure(data=traces, layout=layout)

    # Display the plot
    pyo.plot(fig, filename='historical_vs_predictions.html')

# Example usage of the module
if __name__ == "__main__":
    directory = 'data'
    feature_names = get_feature_names(directory)
    df = load_json_files_by_position(directory, feature_names)

    country = "United Kingdom"
    model, historical_data, last_date = load_model_and_last_date(country)

    # # You can now use `model`, `historical_data`, and `last_date` for predictions
    # print(f"Model loaded for {country}, with last training date on {last_date}")
    # print("Historical data used for the last 30 days:")
    # print(historical_data)

    # # Example usage
    # country = "Spain"
    target_date = pd.to_datetime("2019-09-28")  # User's desired prediction date

    predictions = predict_until_date(country, target_date)
    # # print(predictions)

    # print(historical_data.tail())
    # print(predictions.head())

    # Plot the historical data and predictions
    plot_predictions_with_historical(historical_data, predictions, target_date)
