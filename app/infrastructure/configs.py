from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EMBEDDING_PATH = (
    _REPO_ROOT / "volumes" / "embeddings" / "paraphrase-multilingual-MiniLM-L12-v2"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MINIO_URL: str = "localhost:9000"
    MILVUS_URL: str = "http://localhost:19530"
    governance_collection: str = "governance"
    chunk_size: int = 500
    chunk_overlap: int = 20
    think: bool = False
    OLLAMA_URL: str = "http://localhost:11435"
    OLLAMA_MODEL: str = "qwen3.5-2b-unsloth"
    OLLAMA_ALLOWED_MODELS: str = (
        "qwen3.5-2b-unsloth,gemma4-unsloth"
    )
    OLLAMA_TEMPERATURE: float = 0.2
    OLLAMA_NUM_PREDICT: int = 1024
    OLLAMA_KEEP_ALIVE: str = "30m"
    OLLAMA_WARMUP_ENABLED: bool = True

    RAG_MAX_CONTEXT_CHUNKS: int = 3
    RAG_MAX_CHUNK_CHARS: int = 400

    EMBEDDING_MODEL_PATH: str = str(_DEFAULT_EMBEDDING_PATH)
    EMBEDDING_MODEL_HF_ID: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_DIMENSION: int = 384

    MLFLOW_TRACKING_URI: str = "http://localhost:5001"
    MLFLOW_EXPERIMENT_TRAINING: str = "dutch-energy-training"
    MLFLOW_S3_ENDPOINT_URL: str = "http://localhost:9000"
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadmin"

    CHAT_DATABASE_URL: str = "postgresql://mlflow:mlflow@localhost:5434/rag"

    @property
    def ollama_allowed_models(self) -> list[str]:
        return [m.strip() for m in self.OLLAMA_ALLOWED_MODELS.split(",") if m.strip()]


settings = Settings()
