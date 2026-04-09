import os
import logging
import tempfile

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.services.bucket_service import BucketService
from app.pipeline.storage import MinioStorage
from app.pipeline.silver_extraction import extract_all_tables
from app.pipeline.gold_preprocessing import preprocess_all

logger = logging.getLogger(__name__)

PROCESSED_PREFIX = "pipeline/"


class DataPipelineWorker:

    def __init__(self, bucket: BucketService, storage: MinioStorage):
        self._bucket = bucket
        self._storage = storage
        self._processed_files: set[str] = set()

    async def run(self):
        try:
            files = await self._bucket.list_objects("bronze", None)

            for file_meta in files:
                name = file_meta.object_name
                if not name.lower().endswith(".pdf"):
                    continue

                if name in self._processed_files:
                    continue

                logger.info(f"[DATA PIPELINE] Novo PDF detectado: {name}")
                await self._process_pdf(name)
                self._processed_files.add(name)

        except Exception as e:
            logger.error(f"[DATA PIPELINE] Erro no worker: {e}")

    async def _process_pdf(self, object_name: str):
        # Baixa o PDF do bronze para um arquivo temporario
        file_stream = await self._bucket.get_object(object_name, "bronze")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_stream.read())
            tmp_path = tmp.name

        try:
            # Silver: extrai tabelas
            logger.info(f"[DATA PIPELINE] Extraindo tabelas de {object_name}...")
            silver_tables = extract_all_tables(tmp_path, self._storage)
            for name, df in silver_tables.items():
                logger.info(f"[DATA PIPELINE]   Silver - {name}: {len(df)} rows")

            # Gold: pre-processamento
            logger.info(f"[DATA PIPELINE] Pre-processando dados...")
            gold_tables = preprocess_all(self._storage)
            for name, df in gold_tables.items():
                logger.info(f"[DATA PIPELINE]   Gold - {name}: {len(df)} rows x {len(df.columns)} cols")

            logger.info(f"[DATA PIPELINE] Pipeline concluido para {object_name}: {len(gold_tables)} tabelas")

        finally:
            os.unlink(tmp_path)


def start_worker(bucket_service: BucketService, minio_client):
    storage = MinioStorage(minio_client)
    worker = DataPipelineWorker(bucket_service, storage)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(worker.run, "interval", minutes=5)
    scheduler.start()
    logger.info("[DATA PIPELINE] Worker agendado a cada 5 minutos")
