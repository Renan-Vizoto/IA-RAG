from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure.clients.milvus_client import milvusClient
from app.infrastructure.clients.bucket_client import client as minio_client
from app.infrastructure.configs import settings
from app.infrastructure.implementations.schema_builders.milvus_schema import MilvusSchemaBuilder
from app.infrastructure.implementations.embbeding.MiniLML12_embbeding import MiniLML12_Embbeding
from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.pipeline.storage import MinioStorage
from app.core.workers.governance_indexer import start_worker as start_governance_indexer
from app.api.routes import chat


milvus_repo = MilvusRepo(milvusClient)
schema_builder = MilvusSchemaBuilder(milvusClient)
embedder = MiniLML12_Embbeding()
storage = MinioStorage(minio_client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not milvusClient.has_collection(settings.governance_collection):
        schema_builder.build(settings.governance_collection)

    indexer = start_governance_indexer(
        storage=storage,
        repo=milvus_repo,
        embedder=embedder,
        schema_builder=schema_builder,
        collection=settings.governance_collection,
    )

    # Indexação imediata ao subir (não espera o primeiro tick de 10 min)
    await indexer.run()

    yield


app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)


@app.get("/health")
def health():
    return {"ok": "ok"}
