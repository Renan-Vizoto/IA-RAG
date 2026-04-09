from fastapi import FastAPI
from app.infrastructure.implementations.schema_builders.bucket_schema import BucketSchemaBuilder
from app.infrastructure.implementations.schema_builders.milvus_schema import MilvusSchemaBuilder
from app.api.routes import chat, files
from app.infrastructure.clients.bucket_client import client
from app.infrastructure.clients.milvus_client import milvusClient
from app.infrastructure.configs import settings
from app.core.services.bucket_service import BucketService
from app.core.services.extractor_service import ExtractTextService
from app.core.factories.extractor_factory import ExtractorFactory
from app.infrastructure.implementations.chunking.character_chunking import CharacterChunking
from app.infrastructure.implementations.embbeding.MiniLML12_embbeding import MiniLML12_Embbeding
from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.core.workers.bronze_to_silver import start_worker as start_bronze_to_silver
from app.core.workers.silver_to_gold import start_worker as start_silver_to_gold
from app.core.workers.data_pipeline_worker import start_worker as start_data_pipeline


app = FastAPI()

bucket_service = BucketService(client)
milvus_repo = MilvusRepo(milvusClient)

BucketSchemaBuilder(client).build()

if not milvusClient.has_collection(settings.collection_name):
    MilvusSchemaBuilder(milvusClient).build(settings.collection_name)

start_bronze_to_silver(bucket_service, ExtractTextService(ExtractorFactory()))
start_silver_to_gold(bucket_service, CharacterChunking(), MiniLML12_Embbeding(), milvus_repo)
start_data_pipeline(bucket_service, client)

app.include_router(chat.router)
app.include_router(files.router)

@app.get("/health")
def health():
    return { "ok": "ok" }

