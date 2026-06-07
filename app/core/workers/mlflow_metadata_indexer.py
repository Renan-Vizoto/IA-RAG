import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.core.interfaces.embbeding import EmbeddingStrategy
from app.infrastructure.clients.postgres_client import MLflowSearchClient
from app.core.services.mlflow_metadata_formatter import (
    content_hash,
    format_run_chunks,
    pick_best_rmse_run_id,
)

logger = logging.getLogger(__name__)

_MAX_RUNS = 20


class MLflowMetadataIndexer:

    def __init__(
        self,
        client: MLflowSearchClient,
        repo: MilvusRepo,
        embedder: EmbeddingStrategy,
        schema_builder,
        collection: str,
    ):
        self._client = client
        self._repo = repo
        self._embedder = embedder
        self._schema_builder = schema_builder
        self._collection = collection

    async def run(self):
        try:
            runs = self._client.search_training_runs(max_results=_MAX_RUNS)
            indexed = self._repo.list_indexed_run_ids(self._collection)
            current_ids: set[str] = set()
            best_run_id = pick_best_rmse_run_id(runs)
            upserted = 0

            for run in runs:
                run_id = run.get("run_id")
                if not run_id:
                    continue
                current_ids.add(run_id)
                h = content_hash(run)
                if indexed.get(run_id) == h:
                    continue

                chunks = format_run_chunks(run, is_best=(run_id == best_run_id))
                texts = [c["text"] for c in chunks]
                embeddings = self._embedder.embbed_it(texts)
                for i, chunk in enumerate(chunks):
                    chunk["text_vector"] = embeddings[i].tolist()
                self._repo.upsert(self._collection, chunks)
                upserted += len(chunks)
                logger.info(f"[MLFLOW_INDEX] Upserted {len(chunks)} chunks for run {run_id}")

            stale = set(indexed.keys()) - current_ids
            if stale:
                self._repo.delete_by_run_ids(self._collection, list(stale))
                logger.info(f"[MLFLOW_INDEX] Removed stale runs: {stale}")

            if upserted:
                logger.info(f"[MLFLOW_INDEX] Sync complete — {upserted} chunks upserted.")
            else:
                logger.debug("[MLFLOW_INDEX] No changes detected.")

        except Exception as e:
            logger.error(f"[MLFLOW_INDEX] Erro na indexação: {e}", exc_info=True)


def start_worker(
    client: MLflowSearchClient,
    repo: MilvusRepo,
    embedder: EmbeddingStrategy,
    schema_builder,
    collection: str,
):
    indexer = MLflowMetadataIndexer(client, repo, embedder, schema_builder, collection)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(indexer.run, "interval", minutes=5)
    scheduler.start()
    logger.info("[MLFLOW_INDEX] Indexador agendado a cada 5 minutos.")
    return indexer
