"""
Testes do MLflowSearchClient e MLflowSearchService.
Verifica busca no PostgreSQL via MLflow API (mockado).
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.infrastructure.clients.postgres_client import MLflowSearchClient
from app.core.services.mlflow_search_service import MLflowSearchService


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

def _make_runs_df(n: int = 2) -> pd.DataFrame:
    """Cria DataFrame simulando retorno do mlflow.search_runs."""
    rows = []
    for i in range(n):
        rows.append({
            "run_id": f"run-id-{i}",
            "status": "FINISHED",
            "start_time": pd.Timestamp("2025-01-15"),
            "metrics.rmse": 89.1 - i * 5,
            "metrics.mae": 39.5 - i * 2,
            "metrics.r2": 0.743 + i * 0.01,
            "metrics.mape": 17.6,
            "params.n_estimators": "500",
            "params.learning_rate": "0.05",
            "params.algorithm": "XGBoost",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Testes do MLflowSearchClient
# ──────────────────────────────────────────────

class TestMLflowSearchClient:

    @patch("app.infrastructure.clients.postgres_client.mlflow")
    def test_search_runs_retorna_lista_de_dicts(self, mock_mlflow):
        """search_training_runs deve retornar lista de dicts com run_id e métricas."""
        mock_mlflow.search_runs.return_value = _make_runs_df(2)

        client = MLflowSearchClient()
        runs = client.search_training_runs()

        assert isinstance(runs, list)
        assert len(runs) == 2
        assert "run_id" in runs[0]
        assert "metric_rmse" in runs[0]
        assert "param_n_estimators" in runs[0]

    @patch("app.infrastructure.clients.postgres_client.mlflow")
    def test_search_runs_dataframe_vazio_retorna_lista_vazia(self, mock_mlflow):
        """search_training_runs deve retornar [] quando não há runs."""
        mock_mlflow.search_runs.return_value = pd.DataFrame()

        client = MLflowSearchClient()
        runs = client.search_training_runs()

        assert runs == []

    @patch("app.infrastructure.clients.postgres_client.mlflow")
    def test_search_runs_captura_excecao(self, mock_mlflow):
        """search_training_runs deve retornar [] em caso de erro (MLflow indisponível)."""
        mock_mlflow.search_runs.side_effect = Exception("Connection refused")

        client = MLflowSearchClient()
        runs = client.search_training_runs()

        assert runs == []

    @patch("app.infrastructure.clients.postgres_client.mlflow")
    def test_get_best_run_summary_seleciona_menor_rmse(self, mock_mlflow):
        """get_best_run_summary deve selecionar o run com menor RMSE."""
        mock_mlflow.search_runs.return_value = _make_runs_df(3)

        client = MLflowSearchClient()
        summary = client.get_best_run_summary()

        # O run com menor RMSE é o run-id-2 (89.1 - 2*5 = 79.1)
        assert "run-id-2" in summary

    @patch("app.infrastructure.clients.postgres_client.mlflow")
    def test_get_best_run_summary_sem_runs(self, mock_mlflow):
        """get_best_run_summary deve retornar mensagem clara quando não há runs."""
        mock_mlflow.search_runs.return_value = pd.DataFrame()

        client = MLflowSearchClient()
        summary = client.get_best_run_summary()

        assert "Nenhum run" in summary

    @patch("app.infrastructure.clients.postgres_client.mlflow")
    def test_get_best_run_summary_contem_metricas(self, mock_mlflow):
        """Summary deve conter métricas do melhor run."""
        mock_mlflow.search_runs.return_value = _make_runs_df(1)

        client = MLflowSearchClient()
        summary = client.get_best_run_summary()

        assert "rmse" in summary.lower()
        assert "run-id-0" in summary

    @patch("app.infrastructure.clients.postgres_client.mlflow")
    def test_configura_tracking_uri_no_init(self, mock_mlflow):
        """MLflowSearchClient deve configurar tracking URI ao instanciar."""
        client = MLflowSearchClient()
        mock_mlflow.set_tracking_uri.assert_called_once()

    @patch("app.infrastructure.clients.postgres_client.mlflow")
    def test_search_runs_usa_experimento_de_treinamento(self, mock_mlflow):
        """search_training_runs deve buscar no experimento correto."""
        mock_mlflow.search_runs.return_value = _make_runs_df(1)

        client = MLflowSearchClient()
        client.search_training_runs()

        call_kwargs = mock_mlflow.search_runs.call_args
        experiment_names = call_kwargs[1].get("experiment_names") or call_kwargs[0][0]
        assert "dutch-energy-training" in str(experiment_names)


# ──────────────────────────────────────────────
# Testes do MLflowSearchService
# ──────────────────────────────────────────────

class TestMLflowSearchService:

    def test_search_retorna_contexto_quando_ha_runs(self):
        """search() deve retornar texto com contexto do PostgreSQL."""
        client = MagicMock()
        client.get_best_run_summary.return_value = "Melhor run: run-id-0\n  RMSE: 89.1"

        service = MLflowSearchService(client)
        result = service.search("Qual foi o RMSE do modelo?")

        assert result != ""
        assert "PostgreSQL" in result or "MLflow" in result
        assert "RMSE" in result

    def test_search_retorna_vazio_quando_sem_runs(self):
        """search() deve retornar string vazia quando não há runs."""
        client = MagicMock()
        client.get_best_run_summary.return_value = "Nenhum run de treinamento encontrado no MLflow."

        service = MLflowSearchService(client)
        result = service.search("Métricas do modelo?")

        assert result == ""

    def test_search_retorna_vazio_em_caso_de_erro(self):
        """search() deve retornar string vazia em caso de exceção."""
        client = MagicMock()
        client.get_best_run_summary.side_effect = Exception("Erro de conexão")

        service = MLflowSearchService(client)
        result = service.search("Qualquer pergunta")

        assert result == ""

    def test_search_sempre_chama_client(self):
        """search() deve sempre consultar o client independente da pergunta."""
        client = MagicMock()
        client.get_best_run_summary.return_value = "Resultado"

        service = MLflowSearchService(client)
        service.search("pergunta 1")
        service.search("pergunta 2")

        assert client.get_best_run_summary.call_count == 2

    def test_search_prefixo_identifica_fonte_postgresql(self):
        """O contexto retornado deve identificar claramente a fonte (PostgreSQL/MLflow)."""
        client = MagicMock()
        client.get_best_run_summary.return_value = "Melhor run: run-id-0"

        service = MLflowSearchService(client)
        result = service.search("algo")

        assert "MLflow" in result or "PostgreSQL" in result
