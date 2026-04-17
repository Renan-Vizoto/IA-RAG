from pydantic_settings import BaseSettings
from os import getenv

class Settings(BaseSettings):
    MINIO_URL: str = getenv("MINIO_URL", "localhost:9000")
    MILVUS_URL:str = getenv("MILVUS_URL","http://localhost:19530")
    governance_collection: str = "governance"
    chunk_size:int = getenv("CHUNK_SIZE", 500)
    chunk_overlap:int = getenv("CHUNK_OVERLAP", 20)
    think: bool = getenv("ENABLE_THINK", True)
    OLLAMA_URL: str = getenv("OLLAMA_URL", "http://localhost:11435")
    
settings = Settings()
