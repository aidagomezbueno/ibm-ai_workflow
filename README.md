# IBM AI Enterprise Workflow

## Overview

This project provides a workflow for training and predicting revenue data using machine learning models. The Flask-based application allows you to either train a new model using historical data, analyze the model’s performance, or use an existing model to predict future revenues for a specific country.

## Prerequisites

- Docker installed on your machine.
- PowerShell (or any terminal that supports command execution).
- Historical data for training a model must contain at least **30 data points** (i.e., 30 rows) for the training to be successful.

## Project Structure

- **data/**: Directory containing raw data (not used directly in this example).
- **output/**: Directory containing JSON and CSV files for various countries, which can be used for training and prediction.
- **saved_models/**: Directory where trained models are saved.
- **app.py**: The main Flask application file.
- **analyze_performance.py**: Script for analyzing the model's performance.
- **environment.yml**: Conda environment file containing dependencies.
- **Dockerfile**: Docker configuration file to set up the environment and run the application.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ibm-ai-workflow.git
cd ibm-ai-workflow
```

### 2. Build and Run the Docker Container

Build the Docker container:

```bash
docker build -t ibm-ai_workflow .
```

Run the Docker container:

```bash
docker run -d -p 5000:5000 --name ibm-ai-container ibm-ai_workflow
```

## Available Endpoints

### 1. **/train**: Train a New Model

The `/train` endpoint allows you to train a new model using historical data in JSON format. The data must include at least 30 rows to successfully train a model.

#### Example of training a model

To train a model, use PowerShell (or another terminal) to send a POST request to the `/train` endpoint:

```powershell
$jsonContent = Get-Content -Path "output\Spain.json" -Raw
Invoke-RestMethod -Uri http://127.0.0.1:5000/train -Method Post -ContentType "application/json" -Body $jsonContent
```

<i>Replace `"output\Spain.json"` with the actual path where the json file is stored.</i>

This will train a model using the data provided in the JSON file and save it under `saved_models/`.

### 2. **/predict**: Predict Revenue

The `/predict` endpoint allows you to predict revenue for a specific date using a pre-trained model. Models for all the proposed countries have already been trained, so you can skip the training step and go directly to this step.

#### Example of predicting revenue

To predict revenue for a specific country and date:

```powershell
$jsonContent = '{
    "country": "Spain",
    "target_date": "2019-08-25"
}'
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType "application/json" -Body $jsonContent
```

<i>Replace `"Spain"` with the country you want to predict revenue for and `"2019-08-25"` with the target date you want to predict.</i>

### 3. **/logfile**: Retrieve the Application Log

The `/logfile` endpoint allows you to retrieve the application logs to review the training or prediction processes.

#### Example of retrieving the log file

To get the log file:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/logfile -UseBasicParsing | Select-Object -ExpandProperty Content
```

This will return the log data from the application, which can be useful for debugging or monitoring the system.

## Available Countries for Prediction

You can choose any of the following countries to make predictions:

- EIRE
- France
- Germany
- Hong_Kong
- Netherlands
- Norway
- Portugal
- Singapore
- Spain
- United_Kingdom

## Notes

- **Pre-Trained Models**: Models have already been trained for all the countries listed above. You can skip the training step and directly proceed to make predictions using the `/predict` endpoint.
- **Training Data**: If you want to retrain a model, ensure that the JSON file you use for training contains at least **30 rows** of historical data.