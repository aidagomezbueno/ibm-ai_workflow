from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle
import logging
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np

app = Flask(__name__)

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to create lag features
def create_lag_features(df, lags):
    for lag in lags:
        df[f'revenue_lag_{lag}'] = df['price'].shift(lag)
    return df

# Train model endpoint
@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        country = data['country']
        historical_data = pd.DataFrame(data['historical_data'])
        
        # Ensure 'price' column is numeric
        historical_data['price'] = pd.to_numeric(historical_data['price'], errors='coerce')

        # Create lagged features
        lags = list(range(1, 31))
        historical_data_with_lags = create_lag_features(historical_data, lags)
        historical_data_with_lags.dropna(inplace=True)

        # Ensure 'date' column is in datetime format
        historical_data['date'] = pd.to_datetime(historical_data['date'])

        # Set 'date' as index instead of dropping it
        historical_data_with_lags.set_index('date', inplace=True)

        # Define features and target
        X = historical_data_with_lags.drop('price', axis=1)
        y = historical_data_with_lags['price']

        # Check if the dataset is large enough to split
        if len(X) < 5:  
            return jsonify({"error": "Not enough data after creating lag features. Please provide more historical data."}), 400

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model and historical data
        model_dir = "saved_models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        last_date = historical_data_with_lags.index[-1].strftime('%Y-%m-%d')
        model_data = {'model': model, 'historical_data': historical_data_with_lags[-30:]}

        model_filename = os.path.join(model_dir, f'gradient_boosting_model_{country}_{last_date}.pkl')

        old_models = [f for f in os.listdir(model_dir) if f.startswith(f'gradient_boosting_model_{country}_')]
        for old_model in old_models:
            os.remove(os.path.join(model_dir, old_model))

        with open(model_filename, 'wb') as model_file:
            pickle.dump(model_data, model_file)

        logging.info(f"Model for {country} retrained and saved as '{model_filename}'")
        return jsonify({"message": f"Model for {country} retrained and saved as '{model_filename}'"}), 200

    except Exception as e:
        logging.error(f"Error in train_model: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        country = data['country']
        target_date = pd.to_datetime(data['target_date'])

        model, historical_data, last_date = load_model_and_last_date(country)

        if target_date <= historical_data.index[-1]:
            historical_subset = historical_data.loc[:target_date][['price']].reset_index()
            historical_subset.rename(columns={'InvoiceDate': 'Date'}, inplace=True)
            return historical_subset.to_json(orient="records"), 200

        current_date = last_date + timedelta(days=1)
        predictions = []

        while current_date <= target_date:
            X_new = historical_data.iloc[-1, 1:].values.reshape(1, -1)
            X_new_df = pd.DataFrame(X_new, columns=historical_data.columns[1:])
            predicted_price = model.predict(X_new_df)[0]

            new_row = np.append([predicted_price], X_new[0, :-1])
            # new_series = pd.Series(new_row, index=historical_data.columns[1:], name=current_date)
            historical_data.loc[current_date] = [current_date] + new_row.tolist()

            predictions.append((current_date.strftime('%Y-%m-%d'), predicted_price))
            current_date += timedelta(days=1)

        # Extract the last prediction, which corresponds to the target_date
        last_prediction = predictions[-1]
        predicted_date, predicted_value = last_prediction

        # Return the predicted value for the target_date
        return jsonify({"Date": predicted_date, "Predicted_Price": predicted_value}), 200
    
    except Exception as e:
        logging.error(f"Error in predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Log file endpoint
@app.route('/logfile', methods=['GET'])
def get_log_file():
    try:
        with open('app.log', 'r') as file:
            log_data = file.read()
        return log_data, 200
    except Exception as e:
        logging.error(f"Error in get_log_file: {str(e)}")
        return jsonify({"error": str(e)}), 500

def load_model_and_last_date(country):
    model_dir = "saved_models"
    model_file = max([f for f in os.listdir(model_dir) if f.startswith(f"gradient_boosting_model_{country}")])
    model_path = os.path.join(model_dir, model_file)

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    historical_data = model_data['historical_data']

    last_date_str = model_file.split('_')[-1].replace('.pkl', '')
    last_date = pd.to_datetime(last_date_str, format="%Y-%m-%d")

    return model, historical_data, last_date

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
