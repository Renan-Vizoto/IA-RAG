import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.infrastructure.clients.milvus_client import milvusClient
from app.infrastructure.clients.bucket_client import client as minio_client
from app.infrastructure.clients.postgres_client import MLflowSearchClient
from app.infrastructure.configs import settings
from app.infrastructure.implementations.schema_builders.milvus_schema import MilvusSchemaBuilder
from app.infrastructure.implementations.embbeding.MiniLML12_embbeding import MiniLML12_Embbeding
from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.pipeline.storage import MinioStorage
from app.core.workers.governance_indexer import start_worker as start_governance_indexer
from app.infrastructure.repositories.chat_session_repo import ChatSessionRepository
from app.api.docs import (
    OPENAPI_TAGS,
    build_error_responses,
    configure_openapi,
    register_docs_routes,
)
from app.api.routes import chat
from app.api.schemas.health import HealthResponse
import logging

logger = logging.getLogger(__name__)


milvus_repo = MilvusRepo(milvusClient)
schema_builder = MilvusSchemaBuilder(milvusClient)
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

    governance_indexer = start_governance_indexer(
        storage=storage,
        client=mlflow_client,
        repo=milvus_repo,
        embedder=embedder,
        schema_builder=schema_builder,
        collection=settings.governance_collection,
    )

    async def _bootstrap_indexer() -> None:
        try:
            await governance_indexer.run()
        except Exception as e:
            logger.error(f"[GOVERNANCE] Falha na indexação inicial: {e}", exc_info=True)
        try:
            embedder.embbed_it(["warmup"])
            logger.info("[EMBEDDER] Modelo de embedding aquecido.")
        except Exception as e:
            logger.warning(f"[EMBEDDER] Falha no warmup: {e}")

    bootstrap_task = asyncio.create_task(_bootstrap_indexer())

    yield

    bootstrap_task.cancel()
    try:
        await bootstrap_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Dutch Energy RAG API",
    description="API de Retrieval Augmented Generation para análise de consumo de energia elétrica holandesa.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_tags=OPENAPI_TAGS,
)

app.include_router(chat.router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="API healthcheck",
    description=(
        "Returns a minimal liveness payload. This endpoint does not perform a "
        "deep dependency check against Milvus, Ollama, PostgreSQL, MLflow, or MinIO."
    ),
    responses=build_error_responses(500),
)
def health():
    return HealthResponse(ok="ok")


configure_openapi(app)
register_docs_routes(app)
