from pymilvus import MilvusClient
from app.core.entities.documents import MilvusSchema
from app.core.interfaces.vector_repository import VectorRepository, SearchItem

class MilvusRepo(VectorRepository):
    
    def __init__(self, client: MilvusClient):
        self._client = client

    @property
    def client(self):
        return self._client

    def insert(self, collection: str, data: list[MilvusSchema]):
        self.client.insert(
            collection_name=collection,
            data=data
        )

    def search(self, collection: str, vector: list[float]) -> list[list[SearchItem]]:
        return self.client.search(
            collection_name=collection,
            anns_field="text_vector",
            data=vector,
            limit=5,
            search_params={"metric_type": "COSINE"},
            output_fields=["text", "source"]
        )

    def drop_and_recreate(self, collection: str, schema_builder):
        """Remove todos os dados da collection recriando-a do zero."""
        if self.client.has_collection(collection):
            self.client.drop_collection(collection)
        schema_builder.build(collection)