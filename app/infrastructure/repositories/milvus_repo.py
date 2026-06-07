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

    def upsert(self, collection: str, data: list[dict]):
        self.client.upsert(
            collection_name=collection,
            data=data,
        )

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        output_fields: list[str] | None = None,
    ) -> list[list[SearchItem]]:
        if not self.client.has_collection(collection):
            return [[]]
        fields = output_fields or ["text", "source"]
        return self.client.search(
            collection_name=collection,
            anns_field="text_vector",
            data=vector,
            limit=limit,
            search_params={"metric_type": "COSINE"},
            output_fields=fields,
        )

    def list_indexed_run_ids(self, collection: str) -> dict[str, str]:
        if not self.client.has_collection(collection):
            return {}
        results = self.client.query(
            collection_name=collection,
            filter='section == "overview"',
            output_fields=["run_id", "content_hash"],
        )
        return {r["run_id"]: r["content_hash"] for r in results}

    def delete_by_run_ids(self, collection: str, run_ids: list[str]):
        if not run_ids or not self.client.has_collection(collection):
            return
        ids_expr = ", ".join(f'"{rid}"' for rid in run_ids)
        self.client.delete(
            collection_name=collection,
            filter=f"run_id in [{ids_expr}]",
        )

    def drop_and_recreate(self, collection: str, schema_builder):
        """Remove todos os dados da collection recriando-a do zero."""
        if self.client.has_collection(collection):
            self.client.drop_collection(collection)
        schema_builder.build(collection)
