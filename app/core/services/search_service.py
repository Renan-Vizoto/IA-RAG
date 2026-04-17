from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.core.interfaces.embbeding import EmbeddingStrategy
from app.infrastructure.configs import settings

class SearchService:

    def __init__(self, repo: MilvusRepo, embbeder: EmbeddingStrategy):
        self._repo = repo
        self._embbed = embbeder

    def search(self, text: str) -> str:
        """Busca na base de governança do pipeline Dutch Energy."""
        vector = self._embbed.embbed_it([text])
        return self._repo.search(settings.governance_collection, vector)