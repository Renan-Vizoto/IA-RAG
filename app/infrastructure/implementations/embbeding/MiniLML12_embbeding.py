import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.core.interfaces.embbeding import EmbeddingStrategy
from app.infrastructure.configs import settings


class MiniLML12_Embbeding(EmbeddingStrategy):
    """Loads a multilingual MiniLM embedding model from a local volume path."""

    DIMENSION = settings.EMBEDDING_DIMENSION

    def __init__(self):
        self._model_path = Path(settings.EMBEDDING_MODEL_PATH)
        self._model: SentenceTransformer | None = None

    @property
    def transformer(self) -> SentenceTransformer:
        if self._model is None:
            if not (self._model_path / "config.json").exists():
                raise FileNotFoundError(
                    f"Embedding model not found at {self._model_path}. "
                    "Run ./embedding-setup.sh once to download it into volumes/embeddings."
                )
            self._model = SentenceTransformer(str(self._model_path), device="cpu")
        return self._model

    def embbed_it(self, chunks: list[str]):
        return self.transformer.encode(sentences=chunks)
    
    def embbed_it_for_model(self, chunks: list[str]):
        """
        Convert a list of text strings into embedding vectors for semantic search.
        
        Args:
            chunks: List of text strings to embed
            
        Returns:
            JSON string containing the embedding vectors
        """
        vectors = self.transformer.encode(sentences=chunks)
        return json.dumps(vectors.tolist())
