"""
Governance Indexer Worker
Lê os documentos de governança Markdown gerados pelo pipeline Dutch Energy
e os relatórios do MLflow, vetoriza e indexa no Milvus.

Documentos indexados:
  - silver/dutch-energy/governance_silver.md  → processamento e FE do Silver
  - gold/dutch-energy/governance_gold.md      → split, modelo e métricas do Gold
  - gold/dutch-energy/mlflow_report.md        → relatório detalhado do treinamento MLflow

Dispara re-indexação quando os documentos forem atualizados
(detecta mudança via timestamp do governance_gold.md).
"""
import logging
from io import BytesIO
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.core.interfaces.embbeding import EmbeddingStrategy
from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)

_DOCS = [
    ("silver", "dutch-energy/governance_silver.md", "silver_governance"),
    ("gold",   "dutch-energy/governance_gold.md",   "gold_governance"),
    ("gold",   "dutch-energy/mlflow_report.md",     "mlflow_report"),
]

_MAX_CHUNK = 900


class GovernanceIndexer:

    def __init__(
        self,
        storage: StorageBackend,
        repo: MilvusRepo,
        embedder: EmbeddingStrategy,
        schema_builder,
        collection: str,
    ):
        self._storage = storage
        self._repo = repo
        self._embedder = embedder
        self._schema_builder = schema_builder
        self._collection = collection
        self._last_indexed: str | None = None

    async def run(self):
        try:
            current_ts = self._doc_timestamp()
            if current_ts and current_ts == self._last_indexed:
                logger.debug("[GOVERNANCE] Documentos sem mudanças, indexação ignorada.")
                return

            logger.info("[GOVERNANCE] Iniciando indexação dos documentos de governança...")
            chunks, sources = self._collect_chunks()

            if not chunks:
                logger.warning("[GOVERNANCE] Nenhum documento encontrado para indexar.")
                return

            embeddings = self._embedder.embbed_it(chunks)
            data = [
                {"text": text, "text_vector": embeddings[i].tolist(), "source": sources[i]}
                for i, text in enumerate(chunks)
            ]

            self._repo.drop_and_recreate(self._collection, self._schema_builder)
            self._repo.insert(self._collection, data)

            self._last_indexed = current_ts
            logger.info(
                f"[GOVERNANCE] Indexados {len(data)} chunks na collection '{self._collection}'."
            )

        except Exception as e:
            logger.error(f"[GOVERNANCE] Erro na indexação: {e}", exc_info=True)

    def _doc_timestamp(self) -> str | None:
        """Retorna o timestamp de modificação do governance_gold.md como proxy de mudança."""
        try:
            raw = self._storage.get_object("gold", "dutch-energy/governance_gold.md")
            content = raw.read().decode("utf-8")
            # Extrai a data do cabeçalho do doc
            for line in content.splitlines():
                if line.startswith("## Processado em:"):
                    return line.split(":", 1)[1].strip()
            return datetime.now(timezone.utc).isoformat()
        except Exception:
            return None

    def _collect_chunks(self) -> tuple[list[str], list[str]]:
        """Lê os documentos Markdown do MinIO e retorna (chunks, sources)."""
        all_chunks, all_sources = [], []

        for bucket, obj_path, source_label in _DOCS:
            try:
                raw = self._storage.get_object(bucket, obj_path)
                text = raw.read().decode("utf-8")
                for chunk in _split(text, _MAX_CHUNK):
                    all_chunks.append(chunk)
                    all_sources.append(source_label)
                logger.info(f"[GOVERNANCE] Lido: {bucket}/{obj_path}")
            except Exception as e:
                logger.warning(f"[GOVERNANCE] Documento não encontrado ({bucket}/{obj_path}): {e}")

        return all_chunks, all_sources


def _split(text: str, max_len: int) -> list[str]:
    """Divide texto em chunks de no máximo max_len caracteres, quebrando em espaços."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        cut = text.rfind(" ", 0, max_len)
        if cut == -1:
            cut = max_len
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return chunks


def start_worker(
    storage: StorageBackend,
    repo: MilvusRepo,
    embedder: EmbeddingStrategy,
    schema_builder,
    collection: str,
):
    indexer = GovernanceIndexer(storage, repo, embedder, schema_builder, collection)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(indexer.run, "interval", minutes=10)
    scheduler.start()
    logger.info("[GOVERNANCE] Indexador agendado a cada 10 minutos.")
    return indexer
