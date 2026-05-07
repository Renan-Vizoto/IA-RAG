import logging
from app.infrastructure.clients.postgres_client import MLflowSearchClient

logger = logging.getLogger(__name__)


class MLflowSearchService:
    """Recupera dados estruturados do MLflow (PostgreSQL) para uso no RAG."""

    def __init__(self, client: MLflowSearchClient):
        self._client = client

    def search(self, question: str) -> str:
        """
        Retorna texto com metadados de treinamento do MLflow.
        Sempre retorna o resumo do melhor run — o LLM filtra o que é relevante.
        """
        try:
            summary = self._client.get_best_run_summary()
            if not summary or "Nenhum run" in summary:
                return ""
            return f"[Dados estruturados do MLflow/PostgreSQL]\n{summary}"
        except Exception as e:
            logger.warning(f"[MLFLOW_SEARCH] Erro na busca: {e}")
            return ""
