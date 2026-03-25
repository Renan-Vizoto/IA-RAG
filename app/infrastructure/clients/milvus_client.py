from pymilvus import MilvusClient
from app.infrastructure.configs import settings

milvusClient = MilvusClient(settings.MILVUS_URL)