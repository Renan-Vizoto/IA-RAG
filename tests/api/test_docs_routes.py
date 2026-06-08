import sys
from unittest.mock import MagicMock

sys.modules.setdefault(
    "app.infrastructure.clients.milvus_client",
    MagicMock(milvusClient=MagicMock()),
)
sys.modules.setdefault("psycopg2", MagicMock())
sys.modules.setdefault("psycopg2.extras", MagicMock())

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestDocsRoutes:

    def test_scalar_docs_route_is_registered(self):
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Scalar" in response.text or "api-reference" in response.text

    def test_swagger_remains_available_on_secondary_route(self):
        response = client.get("/swagger")

        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_openapi_contains_enriched_contract(self):
        schema = client.get("/openapi.json").json()

        assert "/chat/message" in schema["paths"]
        assert "/chat/trace/{response_id}" in schema["paths"]
        assert "/query" in schema["paths"]
        assert "/metadata" in schema["paths"]
        assert schema["paths"]["/chat/message"]["post"]["description"]
        assert "examples" in schema["components"]["schemas"]["ChatResponse"]
