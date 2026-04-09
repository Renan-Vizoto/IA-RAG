import os
import logging
import shutil
from io import BytesIO
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    @abstractmethod
    def ensure_bucket(self, bucket: str): ...

    @abstractmethod
    def put_object(self, bucket: str, name: str, data: BytesIO, content_type: str): ...

    @abstractmethod
    def get_object(self, bucket: str, name: str) -> BytesIO: ...

    @abstractmethod
    def list_objects(self, bucket: str, prefix: str) -> list[str]: ...

    @abstractmethod
    def put_file(self, bucket: str, name: str, file_path: str): ...


class MinioStorage(StorageBackend):
    def __init__(self, client):
        self._client = client

    def ensure_bucket(self, bucket: str):
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)

    def put_object(self, bucket: str, name: str, data: BytesIO, content_type: str):
        size = data.getbuffer().nbytes
        self._client.put_object(bucket, name, data, size, content_type)

    def get_object(self, bucket: str, name: str) -> BytesIO:
        response = self._client.get_object(bucket, name)
        try:
            return BytesIO(response.read())
        finally:
            response.close()
            response.release_conn()

    def list_objects(self, bucket: str, prefix: str) -> list[str]:
        return [
            obj.object_name
            for obj in self._client.list_objects(bucket, prefix=prefix)
            if obj.object_name.endswith(".csv")
        ]

    def put_file(self, bucket: str, name: str, file_path: str):
        with open(file_path, "rb") as f:
            data = BytesIO(f.read())
        self.put_object(bucket, name, data, "application/pdf")


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: str):
        self._base = base_dir

    def _path(self, bucket: str, name: str = "") -> str:
        return os.path.join(self._base, bucket, name)

    def ensure_bucket(self, bucket: str):
        os.makedirs(self._path(bucket), exist_ok=True)

    def put_object(self, bucket: str, name: str, data: BytesIO, content_type: str):
        path = self._path(bucket, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data.getvalue())

    def get_object(self, bucket: str, name: str) -> BytesIO:
        path = self._path(bucket, name)
        with open(path, "rb") as f:
            return BytesIO(f.read())

    def list_objects(self, bucket: str, prefix: str) -> list[str]:
        base = self._path(bucket)
        prefix_dir = os.path.join(base, prefix) if prefix else base
        if not os.path.exists(prefix_dir):
            return []
        result = []
        for root, _, files in os.walk(prefix_dir):
            for file in files:
                if file.endswith(".csv"):
                    full = os.path.join(root, file)
                    rel = os.path.relpath(full, base).replace("\\", "/")
                    result.append(rel)
        return result

    def put_file(self, bucket: str, name: str, file_path: str):
        dest = self._path(bucket, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(file_path, dest)


def create_storage(use_minio: bool = True, base_dir: str = "data") -> StorageBackend:
    if use_minio:
        try:
            from minio import Minio
            from app.infrastructure.configs import settings
            client = Minio(
                settings.MINIO_URL,
                access_key="minioadmin",
                secret_key="minioadmin",
                secure=False,
            )
            client.list_buckets()
            logger.info("Usando MinIO como storage backend")
            return MinioStorage(client)
        except Exception as e:
            logger.warning(f"MinIO indisponivel ({e}), usando storage local")

    logger.info(f"Usando storage local: {os.path.abspath(base_dir)}")
    return LocalStorage(base_dir)
