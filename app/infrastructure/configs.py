from pathlib import Path

from pydantic_settings import BaseSettings
from os import getenv

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EMBEDDING_PATH = (
    _REPO_ROOT / "volumes" / "embeddings" / "paraphrase-multilingual-MiniLM-L12-v2"
)


class Settings(BaseSettings):
    MINIO_URL: str = getenv("MINIO_URL", "localhost:9000")
    MILVUS_URL: str = getenv("MILVUS_URL", "http://localhost:19530")
    governance_collection: str = "governance"
    mlflow_metadata_collection: str = "mlflow_metadata"
    chunk_size: int = getenv("CHUNK_SIZE", 500)
    chunk_overlap: int = getenv("CHUNK_OVERLAP", 20)
    think: bool = getenv("ENABLE_THINK", True)
    OLLAMA_URL: str = getenv("OLLAMA_URL", "http://localhost:11435")
    OLLAMA_MODEL: str = getenv("OLLAMA_MODEL", "gemma4-unsloth")
    OLLAMA_TEMPERATURE: float = float(getenv("OLLAMA_TEMPERATURE", "0.2"))
    OLLAMA_NUM_PREDICT: int = int(getenv("OLLAMA_NUM_PREDICT", "384"))

    RAG_MAX_CONTEXT_CHUNKS: int = int(getenv("RAG_MAX_CONTEXT_CHUNKS", "3"))
    RAG_MAX_CHUNK_CHARS: int = int(getenv("RAG_MAX_CHUNK_CHARS", "400"))

    EMBEDDING_MODEL_PATH: str = getenv(
        "EMBEDDING_MODEL_PATH",
        str(_DEFAULT_EMBEDDING_PATH),
    )
    EMBEDDING_MODEL_HF_ID: str = getenv(
        "EMBEDDING_MODEL_HF_ID",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    EMBEDDING_DIMENSION: int = 384

    MLFLOW_TRACKING_URI: str = getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    MLFLOW_EXPERIMENT_TRAINING: str = "dutch-energy-training"
    MLFLOW_S3_ENDPOINT_URL: str = getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    AWS_ACCESS_KEY_ID: str = getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    AWS_SECRET_ACCESS_KEY: str = getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

    CHAT_DATABASE_URL: str = getenv(
        "CHAT_DATABASE_URL",
        "postgresql://mlflow:mlflow@localhost:5432/rag",
    )

settings = Settings()
