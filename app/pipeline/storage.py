import os
import json
import logging
import shutil
from io import BytesIO
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    @abstractmethod
    def ensure_bucket(self, bucket: str, versioning: bool = False): ...

    @abstractmethod
    def put_object(self, bucket: str, name: str, data: BytesIO, content_type: str, metadata: dict = None): ...

    @abstractmethod
    def get_object(self, bucket: str, name: str) -> BytesIO: ...

    @abstractmethod
    def list_objects(self, bucket: str, prefix: str) -> list[str]: ...

    @abstractmethod
    def put_file(self, bucket: str, name: str, file_path: str, metadata: dict = None): ...

    @abstractmethod
    def stat_object(self, bucket: str, name: str) -> dict: ...


class MinioStorage(StorageBackend):
    def __init__(self, client):
        self._client = client

    def ensure_bucket(self, bucket: str, versioning: bool = False):
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
        
        if versioning:
            from minio.versioningconfig import VersioningConfig, ENABLED
            self._client.set_bucket_versioning(bucket, VersioningConfig(ENABLED))

    def put_object(self, bucket: str, name: str, data: BytesIO, content_type: str, metadata: dict = None):
        size = data.getbuffer().nbytes
        self._client.put_object(bucket, name, data, size, content_type, metadata=metadata)

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

    def put_file(self, bucket: str, name: str, file_path: str, metadata: dict = None):
        with open(file_path, "rb") as f:
            data = BytesIO(f.read())
        # Tenta detectar content-type ou assume octet-stream
        self.put_object(bucket, name, data, "application/octet-stream", metadata=metadata)

    def stat_object(self, bucket: str, name: str) -> dict:
        try:
            res = self._client.stat_object(bucket, name)
            # Minio retorna metadados com prefixo x-amz-meta-
            return {k.replace("x-amz-meta-", ""): v for k, v in res.metadata.items()}
        except Exception:
            return {}


class LocalStorage(StorageBackend):
    def __init__(self, base_dir: str):
        self._base = base_dir

    def _path(self, bucket: str, name: str = "") -> str:
        return os.path.join(self._base, bucket, name)

    def ensure_bucket(self, bucket: str, versioning: bool = False):
        os.makedirs(self._path(bucket), exist_ok=True)
        if versioning:
            logger.info(f"Local storage nao suporta versionamento nativo (bucket: {bucket})")

    def put_object(self, bucket: str, name: str, data: BytesIO, content_type: str, metadata: dict = None):
        path = self._path(bucket, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data.getvalue())
        if metadata:
            meta_path = path + ".meta"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

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

    def put_file(self, bucket: str, name: str, file_path: str, metadata: dict = None):
        dest = self._path(bucket, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(file_path, dest)
        if metadata:
            meta_path = dest + ".meta"
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

    def stat_object(self, bucket: str, name: str) -> dict:
        meta_path = self._path(bucket, name) + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return {}


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
