import os
import mlflow
from app.infrastructure.configs import settings


def _configure_env():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY


def initialize_mlflow(experiment: str = "RAG-Chat-Agent"):
    _configure_env()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)
    return mlflow


def initialize_training_mlflow():
    _configure_env()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_TRAINING)
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
