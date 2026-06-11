from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.core.interfaces.embbeding import EmbeddingStrategy
from app.infrastructure.configs import settings

class SearchService:

    def __init__(self, repo: MilvusRepo, embbeder: EmbeddingStrategy):
        self._repo = repo
        self._embbed = embbeder

    def search(self, text: str) -> str:
        """Busca semântica na collection governance (docs + metadados MLflow)."""
        vector = self._embbed.embbed_it([text])
        top_k = settings.RAG_MAX_CONTEXT_CHUNKS

        hits = self._repo.search(
            settings.governance_collection,
            vector,
            limit=top_k,
            output_fields=["text", "source"],
        )
        return hits
