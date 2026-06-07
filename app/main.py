from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure.clients.milvus_client import milvusClient
from app.infrastructure.clients.bucket_client import client as minio_client
from app.infrastructure.clients.postgres_client import MLflowSearchClient
from app.infrastructure.configs import settings
from app.infrastructure.implementations.schema_builders.milvus_schema import MilvusSchemaBuilder
from app.infrastructure.implementations.schema_builders.mlflow_metadata_schema import MLflowMetadataSchemaBuilder
from app.infrastructure.implementations.embbeding.MiniLML12_embbeding import MiniLML12_Embbeding
from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.pipeline.storage import MinioStorage
from app.core.workers.governance_indexer import start_worker as start_governance_indexer
from app.core.workers.mlflow_metadata_indexer import start_worker as start_mlflow_indexer
from app.infrastructure.repositories.chat_session_repo import ChatSessionRepository
from app.api.routes import chat, query
import logging

logger = logging.getLogger(__name__)


milvus_repo = MilvusRepo(milvusClient)
schema_builder = MilvusSchemaBuilder(milvusClient)
mlflow_schema_builder = MLflowMetadataSchemaBuilder(milvusClient)
embedder = MiniLML12_Embbeding()
storage = MinioStorage(minio_client)
mlflow_client = MLflowSearchClient()


chat_session_repo = ChatSessionRepository()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        chat_session_repo.init_schema()
    except Exception as e:
        logger.warning(f"[CHAT_DB] Schema não inicializado: {e}")

    chat.init_chat_dependencies(chat_session_repo, embedder)

    if not milvusClient.has_collection(settings.governance_collection):
        schema_builder.build(settings.governance_collection)

    if not milvusClient.has_collection(settings.mlflow_metadata_collection):
        mlflow_schema_builder.build(settings.mlflow_metadata_collection)

    governance_indexer = start_governance_indexer(
        storage=storage,
        repo=milvus_repo,
        embedder=embedder,
        schema_builder=schema_builder,
        collection=settings.governance_collection,
    )

    mlflow_indexer = start_mlflow_indexer(
        client=mlflow_client,
        repo=milvus_repo,
        embedder=embedder,
        schema_builder=mlflow_schema_builder,
        collection=settings.mlflow_metadata_collection,
    )

    await governance_indexer.run()
    await mlflow_indexer.run()

    try:
        embedder.embbed_it(["warmup"])
        logger.info("[EMBEDDER] Modelo de embedding aquecido.")
    except Exception as e:
        logger.warning(f"[EMBEDDER] Falha no warmup: {e}")

    yield


app = FastAPI(
    title="Dutch Energy RAG API",
    description="API de Retrieval Augmented Generation para análise de consumo de energia elétrica holandesa.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(chat.router)
app.include_router(query.router)


@app.get("/health", tags=["health"], summary="Healthcheck da API")
def health():
    return {"ok": "ok"}
