import sys
from unittest.mock import MagicMock

sys.modules.setdefault(
    "app.infrastructure.clients.milvus_client",
    MagicMock(milvusClient=MagicMock()),
)

import pytest
from fastapi import HTTPException

from app.api.routes.chat import (
    create_chat,
    list_chats,
    get_chat_messages,
    CreateChatRequest,
)


class TestChatChatsEndpoints:

    def test_create_chat_retorna_chat_id(self):
        import app.api.routes.chat as chat_routes

        chat_routes.session_repo = MagicMock()
        chat_routes.session_repo.create_chat.return_value = {
            "chat_id": "chat-1",
            "title": "New chat",
            "created_at": "2025-01-01T00:00:00+00:00",
        }

        result = create_chat(
            CreateChatRequest(),
            response=MagicMock(),
            session_id="sess-1",
        )

        assert result.chat_id == "chat-1"
        chat_routes.session_repo.create_chat.assert_called_once()

    def test_list_chats_retorna_resumo(self):
        import app.api.routes.chat as chat_routes

        chat_routes.session_repo = MagicMock()
        chat_routes.session_repo.list_chats.return_value = [
            {
                "chat_id": "chat-1",
                "title": "RMSE question",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "message_count": 2,
                "created_at": "2025-01-01T00:00:00+00:00",
                "updated_at": "2025-01-02T00:00:00+00:00",
            }
        ]

        result = list_chats(response=MagicMock(), session_id="sess-1")

        assert len(result.chats) == 1
        assert result.chats[0].chat_id == "chat-1"
        assert result.chats[0].total_tokens == 15

    def test_get_chat_messages_retorna_historico(self):
        import app.api.routes.chat as chat_routes

        chat_routes.session_repo = MagicMock()
        chat_routes.session_repo.get_chat.return_value = {
            "chat_id": "chat-1",
            "session_id": "sess-1",
        }
        chat_routes.session_repo.get_messages.return_value = [
            {
                "message_id": "msg-1",
                "role": "user",
                "content": "olá",
                "created_at": "2025-01-01T00:00:00+00:00",
            }
        ]

        result = get_chat_messages("chat-1", session_id="sess-1")

        assert result.chat_id == "chat-1"
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"

    def test_get_chat_messages_404_sem_cookie(self):
        with pytest.raises(HTTPException) as exc:
            get_chat_messages("chat-1", session_id=None)
        assert exc.value.status_code == 404

    def test_get_chat_messages_404_chat_de_outra_sessao(self):
        import app.api.routes.chat as chat_routes

        chat_routes.session_repo = MagicMock()
        chat_routes.session_repo.get_chat.return_value = {
            "chat_id": "chat-1",
            "session_id": "outra-sessao",
        }

        with pytest.raises(HTTPException) as exc:
            get_chat_messages("chat-1", session_id="sess-1")
        assert exc.value.status_code == 404
