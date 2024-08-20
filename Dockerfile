# Use an official Python runtime as a parent image
FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libpq-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Create the conda environment using environment.yml
RUN conda env create -f environment.yml && conda clean -afy

# Activate the environment
SHELL ["conda", "run", "-n", "ibm-ai_workflow", "/bin/bash", "-c"]

# Expose the port that the Flask app will run on
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the application when the container launches
CMD ["conda", "run", "--no-capture-output", "-n", "ibm-ai_workflow", "flask", "run", "--host=0.0.0.0"]