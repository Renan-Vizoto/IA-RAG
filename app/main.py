from fastapi import FastAPI
from app.infrastructure.implementations.schema_builders.bucket_schema import BucketSchemaBuilder
from app.api.routes import chat, files
from app.infrastructure.clients.bucket_client import client
# from app.core.workers.bronze_to_silver import start_worker as start_bronze_to_silver
# from app.core.workers.silver_to_gold import start_worker as start_silver_to_gold


app = FastAPI()
bucketStructure = BucketSchemaBuilder(client)

bucketStructure.build()
# start_bronze_to_silver()
# start_silver_to_gold()

app.include_router(chat.router)
app.include_router(files.router)

@app.get("/health")
def health():
    return { "ok": "ok" }

