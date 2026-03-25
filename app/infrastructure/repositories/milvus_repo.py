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
        """Perform a vector search against a Milvus collection.

        Args:
            collection: Name of the Milvus collection to query.
            vector: A single embedding (list of floats) or a batch of embeddings.
                    The method expects the vector(s) you obtained from the
                    embedding service. Milvus will compare this against the
                    stored "text_vector" field.

        Returns:
            A nested list of :class:`SearchItem` instances. Each sub-list
            corresponds to one query vector (useful when passing multiple
            vectors at once) and contains the top-k results, ordered by
            increasing distance (smaller is more similar).
        """
        return self.client.search(
            collection_name=collection,
            anns_field="text_vector",
            data=vector,
            limit=3,
            search_params={"metric_type": "COSINE"},
            output_fields=["text"]
        )