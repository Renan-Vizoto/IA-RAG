from minio import Minio
from app.infrastructure.configs import settings

client = Minio(
    settings.MINIO_URL,
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)
