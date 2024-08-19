import pytest
import requests
import json

# URL of the Flask API
BASE_URL = "http://127.0.0.1:5000"

def test_train_model():
    # Load the exact train data from the Spain.json file
    with open('output/Spain.json', 'r') as file:
        train_data = json.load(file)
    
    # Send the POST request to the train endpoint
    response = requests.post(f"{BASE_URL}/train", json=train_data)
    
    # Check that the request was successful
    assert response.status_code == 200
    
    # Check the response message
    response_data = response.json()
    assert "Model for Spain retrained and saved" in response_data["message"]

def test_get_logfile():
    # Send the GET request to retrieve the log file
    response = requests.get(f"{BASE_URL}/logfile")
    
    # Check that the request was successful
    assert response.status_code == 200
    
    # Ensure the response contains log content
    assert "Model for Spain retrained and saved" in response.text

def test_predict():
    # Define the JSON content for prediction
    predict_data = {
        "country": "Spain",
        "target_date": "2019-08-25"
    }
    
    # Send the POST request to the predict endpoint
    response = requests.post(f"{BASE_URL}/predict", json=predict_data)
    
    # Check that the request was successful
    assert response.status_code == 200
    
    # Check the response content
    response_data = response.json()
    assert "Date" in response_data
    assert "Predicted_Price" in response_data