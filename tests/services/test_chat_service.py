"""
Testes do ChatService.
Verifica que o serviço busca no PostgreSQL (MLflow) antes do Milvus
e combina os contextos corretamente.
"""
from unittest.mock import MagicMock, patch, call

import pytest

from app.core.services.chat_service import ChatService
from app.core.services.mlflow_search_service import MLflowSearchService
from app.api.schemas.chat_response import ChatResponse


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def mock_agent_result():
    """Resultado simulado do agente LangChain."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    ai_msg = AIMessage(content="O RMSE do modelo foi 89.1.")
    return {
        "messages": [
            HumanMessage(content="Qual o RMSE?"),
            ai_msg,
        ]
    }


@pytest.fixture
def mock_model(mock_agent_result):
    """Mock do ChatOllama."""
    return MagicMock()


@pytest.fixture
def mock_search_tool():
    from langchain_core.tools import tool

    @tool
    def search(query: str) -> str:
        """Busca semântica no Milvus."""
        return "[]"

    return search


@pytest.fixture
def mock_mlflow_service():
    service = MagicMock(spec=MLflowSearchService)
    service.search.return_value = "[Dados estruturados do MLflow/PostgreSQL]\nMelhor run: run-id-0\n  rmse: 89.1"
    return service


@pytest.fixture
def mock_mlflow_service_empty():
    service = MagicMock(spec=MLflowSearchService)
    service.search.return_value = ""
    return service


# ──────────────────────────────────────────────
# Testes
# ──────────────────────────────────────────────

class TestChatService:

    @patch("app.core.services.chat_service.create_agent")
    def test_busca_postgres_antes_do_milvus(
        self, mock_create_agent, mock_model, mock_search_tool,
        mock_mlflow_service, mock_agent_result
    ):
        """ChatService deve chamar MLflowSearchService antes de enviar ao agente."""
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model,
            tools=[mock_search_tool],
            mlflow_search_service=mock_mlflow_service,
        )
        service.send_message("Qual o RMSE?", session_id="test-session")

        mock_mlflow_service.search.assert_called_once_with("Qual o RMSE?")

    @patch("app.core.services.chat_service.create_agent")
    def test_contexto_postgresql_incluido_na_mensagem(
        self, mock_create_agent, mock_model, mock_search_tool,
        mock_mlflow_service, mock_agent_result
    ):
        """Contexto do PostgreSQL deve ser prefixado na mensagem enviada ao agente."""
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model,
            tools=[mock_search_tool],
            mlflow_search_service=mock_mlflow_service,
        )
        service.send_message("Qual o RMSE?", session_id="test-session")

        invoked_messages = agent.invoke.call_args[0][0]["messages"]
        last_human = invoked_messages[-1]
        assert "MLflow" in last_human.content or "PostgreSQL" in last_human.content
        assert "Qual o RMSE?" in last_human.content

    @patch("app.core.services.chat_service.create_agent")
    def test_sem_contexto_postgresql_mensagem_normal(
        self, mock_create_agent, mock_model, mock_search_tool,
        mock_mlflow_service_empty, mock_agent_result
    ):
        """Quando MLflow não retorna contexto, mensagem original vai ao agente sem prefixo."""
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model,
            tools=[mock_search_tool],
            mlflow_search_service=mock_mlflow_service_empty,
        )
        service.send_message("Qual o RMSE?", session_id="test-session")

        invoked_messages = agent.invoke.call_args[0][0]["messages"]
        last_human = invoked_messages[-1]
        assert last_human.content == "Qual o RMSE?"

    @patch("app.core.services.chat_service.create_agent")
    def test_funciona_sem_mlflow_service(
        self, mock_create_agent, mock_model, mock_search_tool, mock_agent_result
    ):
        """ChatService deve funcionar normalmente quando mlflow_search_service=None."""
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model,
            tools=[mock_search_tool],
            mlflow_search_service=None,
        )
        # Não deve lançar exceção
        response = service.send_message("Pergunta simples", session_id="test-session")
        assert response is not None

    @patch("app.core.services.chat_service.create_agent")
    def test_retorna_chat_response(
        self, mock_create_agent, mock_model, mock_search_tool,
        mock_mlflow_service, mock_agent_result
    ):
        """send_message deve retornar um ChatResponse válido."""
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model,
            tools=[mock_search_tool],
            mlflow_search_service=mock_mlflow_service,
        )
        response = service.send_message("Qual o modelo?", session_id="sess-1")

        assert isinstance(response, ChatResponse)
        assert response.session_id == "sess-1"
        assert response.response_time_seconds >= 0

    @patch("app.core.services.chat_service.create_agent")
    def test_historico_de_sessao_mantido(
        self, mock_create_agent, mock_model, mock_search_tool,
        mock_mlflow_service, mock_agent_result
    ):
        """Histórico de mensagens deve crescer a cada interação da sessão."""
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model,
            tools=[mock_search_tool],
            mlflow_search_service=mock_mlflow_service,
        )
        service.send_message("Pergunta 1", session_id="sess-abc")
        service.send_message("Pergunta 2", session_id="sess-abc")

        # Após 2 mensagens, histórico da sessão existe
        assert "sess-abc" in service.chats

    @patch("app.core.services.chat_service.create_agent")
    def test_sessoes_independentes(
        self, mock_create_agent, mock_model, mock_search_tool,
        mock_mlflow_service, mock_agent_result
    ):
        """Sessões diferentes devem ter históricos isolados."""
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model,
            tools=[mock_search_tool],
            mlflow_search_service=mock_mlflow_service,
        )
        service.send_message("Pergunta A", session_id="sessao-1")
        service.send_message("Pergunta B", session_id="sessao-2")

        assert "sessao-1" in service.chats
        assert "sessao-2" in service.chats

    @patch("app.core.services.chat_service.create_agent")
    def test_mlflow_search_chamado_com_mensagem_original(
        self, mock_create_agent, mock_model, mock_search_tool,
        mock_mlflow_service, mock_agent_result
    ):
        """MLflowSearchService.search deve receber a mensagem original do usuário."""
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = ChatService(
            model=mock_model,
            tools=[mock_search_tool],
            mlflow_search_service=mock_mlflow_service,
        )
        service.send_message("Quais features foram usadas?", session_id="s1")

        mock_mlflow_service.search.assert_called_once_with("Quais features foram usadas?")
