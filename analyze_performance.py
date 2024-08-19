from flask import jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objs as go
import plotly.offline as pyo
import json
from sklearn.metrics import mean_squared_error


def create_lag_features(df, lags):
    for lag in lags:
        df[f'revenue_lag_{lag}'] = df['price'].shift(lag)
    return df

def train_model(country):
    # try:
    file_path = f'output/{country}.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
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

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error on test set for {country}: {mse}")

    # Create a plot to compare actual vs predicted
    fig = go.Figure()

    # Actual data points in green
    fig.add_trace(go.Scatter(
        x=X_test.index, 
        y=y_test, 
        mode='markers',  
        name='Actual', 
        marker=dict(color='green')  
    ))

    # Predicted data points in blue
    fig.add_trace(go.Scatter(
        x=X_test.index, 
        y=predictions, 
        mode='markers',  
        name='Predicted', 
        marker=dict(color='blue')  
    ))

    fig.update_layout(
        title=f'Actual vs Predicted Prices for {country}',
        xaxis_title='Date',
        yaxis_title='Price'
    )

    # Save the plot as an HTML file
    plot_filename = f'actual_vs_predicted_{country}.html'
    pyo.plot(fig, filename=plot_filename)

if __name__ == "__main__":
    train_model(input('Input the country you wanna analyze (e.g. Spain, Singapore, Portugal, ...): '))