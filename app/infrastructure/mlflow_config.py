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


def log_chat_response(
    response_id: str,
    session_id: str,
    model: str,
    user_message: str,
    answer: str,
    metrics: dict,
):
    with mlflow.start_run(run_name=response_id):
        mlflow.set_tag("session_id", session_id)
        mlflow.set_tag("response_id", response_id)
        mlflow.log_param("model", model)
        mlflow.log_param("user_message", user_message[:500])
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)
        mlflow.log_text(answer, "answer.txt")


mlflow_client = initialize_mlflow()
