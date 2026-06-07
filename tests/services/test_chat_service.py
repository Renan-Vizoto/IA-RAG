"""
Testes do ChatService.
Verifica rastreio de resposta, tokens por resposta/sessão e persistência.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.core.services.chat_service import ChatService
from app.api.schemas.chat_response import ChatResponse
from app.infrastructure.configs import settings


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
def mock_search_tool():
    from langchain_core.tools import tool

    @tool
    def search(query: str) -> str:
        """Busca semântica no Milvus."""
        return "[]"

    return search


@pytest.fixture
def mock_search_service():
    service = MagicMock()
    service.search.return_value = [[
        {
            "id": "hit-1",
            "distance": 0.12,
            "entity": {"text": "Modelo XGBoost treinado no gold.", "source": "gold_governance"},
        }
    ]]
    return service


@pytest.fixture
def mock_session_repo():
    repo = MagicMock()
    repo.get_session_tokens.side_effect = [
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        {"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
    ]
    return repo


@pytest.fixture
def mock_model_manager():
    return MagicMock()


@pytest.fixture
def service_factory(mock_search_tool, mock_model_manager):
    def _build(**kwargs):
        return ChatService(
            tools=[mock_search_tool],
            ollama_model_manager=mock_model_manager,
            **kwargs,
        )
    return _build


class TestChatService:

    @patch("app.core.services.chat_service.log_chat_response")
    @patch("app.core.services.chat_service.create_agent")
    @patch("app.core.services.chat_service.create_chat_model")
    def test_retorna_chat_response_com_tracking(
        self, mock_create_chat_model, mock_create_agent, mock_log,
        mock_agent_result, mock_session_repo, mock_search_service,
        service_factory,
    ):
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = service_factory(
            search_service=mock_search_service,
            session_repo=mock_session_repo,
        )
        response = service.send_message("Qual o modelo?", session_id="sess-1")

        assert isinstance(response, ChatResponse)
        assert response.session_id == "sess-1"
        assert response.model == settings.OLLAMA_MODEL
        assert response.response_id
        assert response.tokens.input_tokens == 10
        assert len(response.search_results) == 1
        mock_search_service.search.assert_called_once_with("Qual o modelo?")

    @patch("app.core.services.chat_service.log_chat_response")
    @patch("app.core.services.chat_service.create_agent")
    @patch("app.core.services.chat_service.create_chat_model")
    def test_modelo_explicito_na_requisicao(
        self, mock_create_chat_model, mock_create_agent, mock_log,
        mock_agent_result, service_factory, mock_model_manager,
    ):
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = service_factory()
        response = service.send_message(
            "Pergunta",
            session_id="sess-1",
            model="gemma4-unsloth",
        )

        assert response.model == "gemma4-unsloth"
        mock_model_manager.ensure_loaded.assert_called_once_with("gemma4-unsloth")

    @patch("app.core.services.chat_service.log_chat_response")
    @patch("app.core.services.chat_service.create_agent")
    @patch("app.core.services.chat_service.create_chat_model")
    def test_mesmo_modelo_nao_forca_reload_duplo(
        self, mock_create_chat_model, mock_create_agent, mock_log,
        mock_agent_result, service_factory, mock_model_manager,
    ):
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = service_factory()
        service.send_message("Pergunta 1", session_id="sess-1", model="gemma4-unsloth")
        service.send_message("Pergunta 2", session_id="sess-1", model="gemma4-unsloth")

        assert mock_model_manager.ensure_loaded.call_count == 2
        mock_model_manager.ensure_loaded.assert_called_with("gemma4-unsloth")

    def test_modelo_fora_da_allowlist_levanta_erro(self, service_factory):
        service = service_factory()

        with pytest.raises(ValueError, match="não permitido"):
            service.send_message("oi", session_id="sess-1", model="modelo-inexistente")

    @patch("app.core.services.chat_service.log_chat_response")
    @patch("app.core.services.chat_service.create_agent")
    @patch("app.core.services.chat_service.create_chat_model")
    def test_mensagem_enviada_com_contexto_da_busca_obrigatoria(
        self, mock_create_chat_model, mock_create_agent, mock_log,
        mock_agent_result, mock_search_service, service_factory,
    ):
        agent = MagicMock()
        agent.invoke.return_value = mock_agent_result
        mock_create_agent.return_value = agent

        service = service_factory(search_service=mock_search_service)
        service.send_message("Qual o RMSE?", session_id="test-session")

        invoked_messages = agent.invoke.call_args[0][0]["messages"]
        last_human = invoked_messages[-1]
        assert "Qual o RMSE?" in last_human.content
        assert "--- CONTEXTO ---" in last_human.content


class TestMandatorySearch:

    def test_fallback_hits_quando_agente_nao_chama_ferramenta(self, mock_search_tool, service_factory):
        from langchain_core.messages import AIMessage, HumanMessage

        service = service_factory()
        fallback_hits = [
            service._dict_to_milvus_hit({
                "id": "1",
                "distance": 0.2,
                "entity": {"text": "trecho", "source": "mlflow_metadata"},
            })
        ]
        parsed = service._parse_agent_output(
            {
                "messages": [
                    HumanMessage(content="pergunta"),
                    AIMessage(content="resposta final"),
                ]
            },
            fallback_hits=fallback_hits,
        )
        assert len(parsed.result) == 1
        assert parsed.result[0].source == "mlflow_metadata"


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

    def test_extrai_tokens_apenas_do_turno_atual(self):
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="pergunta antiga"),
            AIMessage(
                content="resposta antiga",
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            ),
            HumanMessage(content="pergunta nova"),
            AIMessage(
                content="resposta nova",
                usage_metadata={"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
            ),
        ]
        service = ChatService(tools=[])
        current_turn = service._get_current_turn_messages(messages)
        inp, out, total = ChatService._extract_token_usage(current_turn)
        assert inp == 30
        assert out == 10
        assert total == 40


class TestAnswerSanitization:

    def test_remove_tags_de_fonte_e_run_id(self):
        raw = (
            "O modelo foi XGBoost [mlflow_metadata]. "
            "Run ID: a1888f014bb04b61ba4c245bb58552c8 [mlflow_metadata]."
        )
        cleaned = ChatService._sanitize_answer(raw)
        assert "[mlflow_metadata]" not in cleaned
        assert "a1888f014bb04b61ba4c245bb58552c8" not in cleaned
        assert "XGBoost" in cleaned

    def test_redact_sensitive_text_no_contexto(self):
        text = "Run ID MLflow: a1888f014bb04b61ba4c245bb58552c8. RMSE: 89.1"
        redacted = ChatService._redact_sensitive_text(text)
        assert "a1888f014bb04b61ba4c245bb58552c8" not in redacted
        assert "RMSE: 89.1" in redacted
