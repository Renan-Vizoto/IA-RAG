from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.repositories.chat_session_repo import ChatSessionRepository


@pytest.fixture
def repo():
    return ChatSessionRepository(database_url="postgresql://test:test@localhost/rag")


class TestChatSessionRepository:

    @patch("app.infrastructure.repositories.chat_session_repo.psycopg2.connect")
    def test_init_schema_cria_database_e_tabelas(self, mock_connect, repo):
        admin_conn = MagicMock()
        app_conn = MagicMock()
        admin_cur = MagicMock()
        app_cur = MagicMock()
        mock_connect.side_effect = [admin_conn, app_conn]
        admin_conn.cursor.return_value.__enter__ = lambda s: admin_cur
        admin_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        app_conn.cursor.return_value.__enter__ = lambda s: app_cur
        app_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        admin_cur.fetchone.return_value = None

        repo.init_schema()

        assert mock_connect.call_count == 2
        admin_cur.execute.assert_any_call(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            ("rag",),
        )
        app_cur.execute.assert_called_once()
        assert "CREATE TABLE IF NOT EXISTS chat_sessions" in app_cur.execute.call_args[0][0]

    @patch("app.infrastructure.repositories.chat_session_repo.psycopg2.connect")
    def test_ensure_session_insere_se_nao_existir(self, mock_connect, repo):
        conn = MagicMock()
        cur = MagicMock()
        mock_connect.return_value = conn
        conn.cursor.return_value.__enter__ = lambda s: cur
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        repo.ensure_session("sess-123")

        cur.execute.assert_called_once()
        sql = cur.execute.call_args[0][0]
        assert "INSERT INTO chat_sessions" in sql
        assert cur.execute.call_args[0][1] == ("sess-123",)

    @patch("app.infrastructure.repositories.chat_session_repo.psycopg2.connect")
    def test_add_session_tokens_atualiza_acumulado(self, mock_connect, repo):
        conn = MagicMock()
        cur = MagicMock()
        mock_connect.return_value = conn
        conn.cursor.return_value.__enter__ = lambda s: cur
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        repo.add_session_tokens("sess-1", 10, 5, 15)

        sql = cur.execute.call_args[0][0]
        assert "input_tokens = input_tokens + %s" in sql
        assert cur.execute.call_args[0][1] == (10, 5, 15, "sess-1")

    @patch("app.infrastructure.repositories.chat_session_repo.psycopg2.connect")
    def test_get_session_tokens_retorna_totais(self, mock_connect, repo):
        conn = MagicMock()
        cur = MagicMock()
        mock_connect.return_value = conn
        conn.cursor.return_value.__enter__ = lambda s: cur
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        cur.fetchone.return_value = {
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
        }

        totals = repo.get_session_tokens("sess-1")

        assert totals == {
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
        }

    @patch("app.infrastructure.repositories.chat_session_repo.psycopg2.connect")
    def test_save_milvus_hits_insere_por_hit(self, mock_connect, repo):
        conn = MagicMock()
        cur = MagicMock()
        mock_connect.return_value = conn
        conn.cursor.return_value.__enter__ = lambda s: cur
        conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        repo.save_milvus_hits("resp-1", [
            {
                "id": "hit-1",
                "collection": "mlflow_metadata",
                "source": "mlflow_metadata",
                "distance": 0.2,
                "text": "RMSE do modelo",
            }
        ])

        assert cur.execute.call_count == 1
        sql = cur.execute.call_args[0][0]
        assert "chat_response_milvus_hits" in sql
