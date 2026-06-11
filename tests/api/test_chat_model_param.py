import sys
from unittest.mock import MagicMock

sys.modules.setdefault(
    "app.infrastructure.clients.milvus_client",
    MagicMock(milvusClient=MagicMock()),
)

import pytest
from fastapi import HTTPException

from app.api.routes.chat import ChatRequest, send_message
from app.api.schemas.chat_response import ChatResponse, TokenUsage


class TestChatModelParam:

    def test_modelo_invalido_retorna_400(self):
        import app.api.routes.chat as chat_routes

        chat_routes.chatService = MagicMock()
        chat_routes.chatService.send_message.side_effect = ValueError(
            "Modelo 'foo' não permitido."
        )

        with pytest.raises(HTTPException) as exc:
            send_message(
                ChatRequest(message="oi", model="foo"),
                response=MagicMock(),
                session_id="sess-1",
            )

        assert exc.value.status_code == 400

    def test_modelo_valido_repassa_para_service(self):
        import app.api.routes.chat as chat_routes

        chat_routes.chatService = MagicMock()
        chat_routes.chatService.send_message.return_value = ChatResponse(
            search_results=[],
            agent_thoughts="",
            answer="ok",
            response_time_seconds=1.0,
            confidence_score=0.0,
            session_id="sess-1",
            chat_id="chat-1",
            title="New chat",
            message_count=1,
            response_id="r1",
            model="gemma4-unsloth",
            tokens=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            chat_tokens=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        )

        response = send_message(
            ChatRequest(message="oi", model="gemma4-unsloth", chat_id="chat-1"),
            response=MagicMock(),
            session_id="sess-1",
        )

        chat_routes.chatService.send_message.assert_called_once_with(
            "oi", "sess-1", model="gemma4-unsloth", chat_id="chat-1"
        )
        assert response.model == "gemma4-unsloth"
