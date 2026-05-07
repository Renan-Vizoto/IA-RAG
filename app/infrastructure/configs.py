from pydantic_settings import BaseSettings
from os import getenv

class Settings(BaseSettings):
    MINIO_URL: str = getenv("MINIO_URL", "localhost:9000")
    MILVUS_URL: str = getenv("MILVUS_URL", "http://localhost:19530")
    governance_collection: str = "governance"
    chunk_size: int = getenv("CHUNK_SIZE", 500)
    chunk_overlap: int = getenv("CHUNK_OVERLAP", 20)
    think: bool = getenv("ENABLE_THINK", True)
    OLLAMA_URL: str = getenv("OLLAMA_URL", "http://localhost:11435")

    MLFLOW_TRACKING_URI: str = getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    MLFLOW_EXPERIMENT_TRAINING: str = "dutch-energy-training"
    MLFLOW_S3_ENDPOINT_URL: str = getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    AWS_ACCESS_KEY_ID: str = getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    AWS_SECRET_ACCESS_KEY: str = getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

settings = Settings()
