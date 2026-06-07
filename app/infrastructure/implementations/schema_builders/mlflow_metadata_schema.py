from pymilvus import DataType
from app.infrastructure.clients.milvus_client import MilvusClient
from app.core.interfaces.schema_builder import SchemaBuilder


class MLflowMetadataSchemaBuilder(SchemaBuilder):

    def __init__(self, client: MilvusClient):
        self._client = client

    @property
    def client(self):
        return self._client

    def build(self, name: str):
        schema = self._client.create_schema()

        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=128,
        )
        schema.add_field(
            field_name="run_id",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="section",
            datatype=DataType.VARCHAR,
            max_length=32,
        )
        schema.add_field(
            field_name="text_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=384,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=2048,
        )
        schema.add_field(
            field_name="source",
            datatype=DataType.VARCHAR,
            max_length=64,
        )
        schema.add_field(
            field_name="content_hash",
            datatype=DataType.VARCHAR,
            max_length=64,
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="text_vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )
        self._client.load_collection(collection_name=name)
