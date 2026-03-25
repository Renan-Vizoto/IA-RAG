import mlflow
from app.infrastructure.configs import settings

def initialize_mlflow():
    mlflow.set_experiment("RAG-Chat-Agent")
    return mlflow

def start_run():
    mlflow.start_run()

def log_metrics(metrics: dict):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)

def end_run():
    mlflow.end_run()

mlflow_client = initialize_mlflow()
