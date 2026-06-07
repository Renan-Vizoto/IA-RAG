from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.core.interfaces.embbeding import EmbeddingStrategy
from app.infrastructure.configs import settings

class SearchService:

    def __init__(self, repo: MilvusRepo, embbeder: EmbeddingStrategy):
        self._repo = repo
        self._embbed = embbeder

    def search(self, text: str) -> str:
        """Busca semântica na governança do pipeline e nos metadados MLflow indexados."""
        vector = self._embbed.embbed_it([text])
        top_k = settings.RAG_MAX_CONTEXT_CHUNKS

        gov_hits = self._repo.search(
            settings.governance_collection,
            vector,
            limit=top_k,
            output_fields=["text", "source"],
        )
        mlflow_hits = self._repo.search(
            settings.mlflow_metadata_collection,
            vector,
            limit=top_k,
            output_fields=["text", "source", "run_id"],
        )

        merged = []
        for hit_list in gov_hits + mlflow_hits:
            for hit in hit_list:
                merged.append(hit)

        merged.sort(key=lambda h: h.get("distance", 1.0))
        top = merged[:top_k]
        return [top]
