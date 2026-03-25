from app.core.services.bucket_service import BucketService
from app.core.interfaces.chunking import ChunkingStrategy
from app.core.interfaces.embbeding import EmbeddingStrategy
from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.infrastructure.configs import settings
import numpy as np

from apscheduler.schedulers.asyncio import AsyncIOScheduler

class SilverToGoldWorker:

    def __init__(self, bucket: BucketService, chunk: ChunkingStrategy, embbeding: EmbeddingStrategy, repo: MilvusRepo):
            self._bucketService = bucket
            self._chunkingService = chunk
            self._embbedingService = embbeding
            self._repo = repo

    async def run(self):
        files = await self._bucketService.list_objects("silver", None)

        for file_meta in files:
            print(f"Processing {file_meta.object_name}...")
            file_stream = await self._bucketService.get_object(file_meta.object_name, "silver")

            content = file_stream.read().decode('utf-8')
            
            chunks = self._chunkingService.chunk_it(content)
            embbeds = self._embbedingService.embbed_it(chunks)

            items = self.__to_schema(chunks, embbeds)

            self._repo.insert(settings.collection_name, items)
            print(f"Inserted {len(items)} chunks from {file_meta.object_name} into Milvus.")

    def __to_schema(self, chunks: list[str], embbeds: np.ndarray) -> list[dict]:
            return [dict(text=text, text_vector=embbeds[ind].tolist()) for ind, text in enumerate(chunks)]

def start_worker(bucket: BucketService, chunk: ChunkingStrategy, embbeding: EmbeddingStrategy, repo: MilvusRepo):
    worker = SilverToGoldWorker(bucket, chunk, embbeding, repo)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(worker.run, "interval", minutes=5)
    scheduler.start()