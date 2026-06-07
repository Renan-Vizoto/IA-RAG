"""
Testes do ChatService.
Verifica rastreio de resposta, tokens por resposta/sessão e persistência.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.core.services.chat_service import ChatService
from app.api.schemas.chat_response import ChatResponse


@pytest.fixture
def mock_agent_result():
    from langchain_core.messages import AIMessage, HumanMessage

    ai_msg = AIMessage(
        content="O RMSE do modelo foi 89.1.",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    return {
        "messages": [
            HumanMessage(content="Qual o RMSE?"),
            ai_msg,
        ]
    }


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.model = "gemma4-unsloth"
    return model


@pytest.fixture
def mock_search_tool():
    from langchain_core.tools import tool

    @tool
    def search(query: str) -> str:
        """Busca semântica no Milvus."""
        return "[]"

    return search


@pytest.fixture
def mock_session_repo():
    repo = MagicMock()
    repo.get_session_tokens.side_effect = [
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
    ]
    return repo


class TestChatService:

    @patch("app.core.services.chat_service.log_chat_response")
    @patch("app.core.services.chat_service.create_agent")
    def test_retorna_chat_response_com_tracking(
        self, mock_create_agent, mock_log, mock_model, mock_search_tool,
        mock_agent_result, mock_session_repo
    ):
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model, tools=[mock_search_tool], session_repo=mock_session_repo
        )
        response = service.send_message("Qual o modelo?", session_id="sess-1")

        assert isinstance(response, ChatResponse)
        assert response.session_id == "sess-1"
        assert response.model == "gemma4-unsloth"
        assert response.response_id
        assert response.tokens.input_tokens == 10
        assert response.tokens.output_tokens == 5
        assert response.tokens.total_tokens == 15
        assert response.session_tokens.total_tokens == 15
        mock_session_repo.ensure_session.assert_called_once_with("sess-1")
        mock_session_repo.save_response.assert_called_once()
        mock_log.assert_called_once()

    @patch("app.core.services.chat_service.log_chat_response")
    @patch("app.core.services.chat_service.create_agent")
    def test_session_tokens_acumulam(
        self, mock_create_agent, mock_log, mock_model, mock_search_tool,
        mock_agent_result, mock_session_repo
    ):
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model, tools=[mock_search_tool], session_repo=mock_session_repo
        )
        r1 = service.send_message("Pergunta 1", session_id="sess-abc")
        r2 = service.send_message("Pergunta 2", session_id="sess-abc")

        assert r1.session_tokens.total_tokens == 15
        assert r2.session_tokens.total_tokens == 42
        assert mock_session_repo.add_session_tokens.call_count == 2

    @patch("app.core.services.chat_service.log_chat_response")
    @patch("app.core.services.chat_service.create_agent")
    def test_mensagem_original_enviada_sem_prefixo_mlflow(
        self, mock_create_agent, mock_log, mock_model, mock_search_tool, mock_agent_result
    ):
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(model=mock_model, tools=[mock_search_tool])
        service.send_message("Qual o RMSE?", session_id="test-session")

        invoked_messages = agent.invoke.call_args[0][0]["messages"]
        last_human = invoked_messages[-1]
        assert last_human.content == "Qual o RMSE?"

    @patch("app.core.services.chat_service.log_chat_response")
    @patch("app.core.services.chat_service.create_agent")
    def test_historico_de_sessao_mantido(
        self, mock_create_agent, mock_log, mock_model, mock_search_tool, mock_agent_result
    ):
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(model=mock_model, tools=[mock_search_tool])
        service.send_message("Pergunta 1", session_id="sess-abc")
        service.send_message("Pergunta 2", session_id="sess-abc")

        assert "sess-abc" in service.chats


class TestExtractTokenUsage:

    def test_extrai_tokens_de_ai_message(self):
        from langchain_core.messages import AIMessage

        messages = [
            AIMessage(
                content="resposta",
                usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
            )
        ]
        inp, out, total = ChatService._extract_token_usage(messages)
        assert inp == 20
        assert out == 8
        assert total == 28

    def test_sem_usage_retorna_none(self):
        from langchain_core.messages import AIMessage

        messages = [AIMessage(content="resposta")]
        assert ChatService._extract_token_usage(messages) == (None, None, None)
