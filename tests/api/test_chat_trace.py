import sys
from unittest.mock import MagicMock

# Evita conexão Milvus ao importar rotas de chat
sys.modules.setdefault(
    "app.infrastructure.clients.milvus_client",
    MagicMock(milvusClient=MagicMock()),
)

from unittest.mock import patch
import pytest
from fastapi import HTTPException

from app.api.routes.chat import get_trace


class TestChatTraceEndpoint:

    def test_trace_retorna_404_quando_nao_encontrado(self):
        import app.api.routes.chat as chat_routes
        chat_routes.session_repo = MagicMock()
        chat_routes.session_repo.get_response.return_value = None

        with pytest.raises(HTTPException) as exc:
            get_trace("unknown-id")
        assert exc.value.status_code == 404

    def test_trace_retorna_dados_com_milvus_hits(self):
        import app.api.routes.chat as chat_routes
        chat_routes.session_repo = MagicMock()
        chat_routes.session_repo.get_response.return_value = {
            "response_id": "resp-1",
            "session_id": "sess-1",
            "chat_id": "chat-1",
            "model": "gemma4-unsloth",
            "user_message": "Qual o RMSE?",
            "answer": "O RMSE foi 89.1",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "response_time_seconds": 1.2,
            "confidence_score": 0.8,
            "created_at": "2025-01-01T00:00:00+00:00",
            "milvus_hits": [
                {
                    "milvus_id": "mlflow:abc:metrics",
                    "collection": "mlflow_metadata",
                    "source": "mlflow_metadata",
                    "distance": 0.15,
                    "text_preview": "RMSE na validação: 89.1",
                }
            ],
        }

        result = get_trace("resp-1")
        assert result.response_id == "resp-1"
        assert result.chat_id == "chat-1"
        assert result.tokens.input_tokens == 10
        assert len(result.milvus_hits) == 1
        assert result.milvus_hits[0].collection == "mlflow_metadata"

    def test_trace_retorna_503_sem_repo(self):
        import app.api.routes.chat as chat_routes
        chat_routes.session_repo = None

        with pytest.raises(HTTPException) as exc:
            get_trace("resp-1")
        assert exc.value.status_code == 503
