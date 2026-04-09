from minio import Minio
from app.core.interfaces.schema_builder import SchemaBuilder


class BucketSchemaBuilder(SchemaBuilder):        

    def __init__(self, client: Minio):
        self._client = client

    @property
    def client(self):
        return self._client
    
    def build(self):
        needed_buckets = ("bronze", "silver", "gold")
        for b in needed_buckets:
            if not self.client.bucket_exists(b):
                self.client.make_bucket(b)
            self.client.enable_object_versioning(b)
