import logging
import time

from pymilvus import MilvusClient

from app.infrastructure.configs import settings

logger = logging.getLogger(__name__)


def _connect_milvus(
    max_retries: int = 30,
    delay_seconds: float = 2.0,
) -> MilvusClient:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return MilvusClient(settings.MILVUS_URL)
        except Exception as exc:
            last_error = exc
            logger.warning(
                "[MILVUS] Conexão falhou (%s/%s): %s",
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                time.sleep(delay_seconds)
    raise last_error  # type: ignore[misc]


milvusClient = _connect_milvus()
