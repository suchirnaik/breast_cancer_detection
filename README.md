# breast_cancer_detection
Here’s a shortened version of the README for your project:

Breast Cancer Detection Model

Overview

This project builds a Random Forest model to classify mammogram images as benign or malignant. It uses Azure Machine Learning (Azure ML) to automate data loading, model training, evaluation, and deployment. The entire process is wrapped in an Azure ML pipeline for ease of use and scalability.

Key Features

	•	Data Loading: Loads data from Azure Blob Storage/Datastore.
	•	Model Training: Trains a Random Forest model on the data.
	•	Evaluation: Logs accuracy and classification metrics.
	•	Model Deployment: Option to deploy the trained model to Azure Web Service for real-time predictions.
	•	Manual Trigger: The pipeline can be manually triggered to retrain the model with new data.

Requirements

	1.	Azure Machine Learning Workspace.
	2.	Python 3.7+ and the following libraries:

pip install argparse joblib numpy azureml-core scikit-learn

How to Run

	1.	Upload data (train_data.npy, train_labels.npy, test_data.npy, test_labels.npy) to Azure ML Datastore.
	2.	Create and run the pipeline:

from azureml.core import Workspace, Experiment
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Load workspace
ws = Workspace.from_config()

# Define pipeline steps
train_step = PythonScriptStep(
    name="Train Model",
    script_name="train.py",
    arguments=["--train_data", "train_data.npy", "--train_labels", "train_labels.npy", 
               "--test_data", "test_data.npy", "--test_labels", "test_labels.npy"],
    compute_target="your_compute_target",
    source_directory="./pipeline_scripts"
)

# Create and submit pipeline
pipeline = Pipeline(workspace=ws, steps=[train_step])
pipeline_run = pipeline.submit("breast-cancer-pipeline")
pipeline_run.wait_for_completion(show_output=True)

Model Deployment

After training, the model can be deployed using Azure Container Instances (ACI) for real-time predictions.

Improvements

	•	Set up Azure Event Grid to trigger retraining when new data is uploaded.
	•	Implement hyperparameter tuning with Azure ML Hyperdrive.

