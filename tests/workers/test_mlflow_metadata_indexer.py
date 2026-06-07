import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.workers.mlflow_metadata_indexer import MLflowMetadataIndexer


RUN_A = {
    "run_id": "run-a",
    "status": "FINISHED",
    "start_time": "2025-01-01",
    "metric_rmse": 90.0,
    "param_algorithm": "XGBoost",
}

RUN_B = {
    "run_id": "run-b",
    "status": "FINISHED",
    "start_time": "2025-01-02",
    "metric_rmse": 80.0,
    "param_algorithm": "XGBoost",
}


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.search_training_runs.return_value = [RUN_A, RUN_B]
    return client


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.list_indexed_run_ids.return_value = {}
    return repo


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embbed_it.side_effect = lambda texts: np.random.rand(len(texts), 384)
    return embedder


@pytest.fixture
def indexer(mock_client, mock_repo, mock_embedder):
    return MLflowMetadataIndexer(
        client=mock_client,
        repo=mock_repo,
        embedder=mock_embedder,
        schema_builder=MagicMock(),
        collection="mlflow_metadata",
    )


class TestMLflowMetadataIndexer:

    def test_novo_run_faz_upsert(self, indexer, mock_repo):
        asyncio.run(indexer.run())
        assert mock_repo.upsert.called
        mock_repo.drop_and_recreate.assert_not_called()

    def test_run_inalterado_nao_faz_upsert(self, indexer, mock_repo, mock_client):
        from app.core.services.mlflow_metadata_formatter import content_hash

        mock_repo.list_indexed_run_ids.return_value = {
            "run-a": content_hash(RUN_A),
            "run-b": content_hash(RUN_B),
        }
        asyncio.run(indexer.run())
        mock_repo.upsert.assert_not_called()

    def test_run_removido_e_deletado(self, indexer, mock_repo, mock_client):
        mock_client.search_training_runs.return_value = [RUN_B]
        mock_repo.list_indexed_run_ids.return_value = {
            "run-a": "oldhash",
            "run-b": "hashb",
        }
        asyncio.run(indexer.run())
        mock_repo.delete_by_run_ids.assert_called_once()
        deleted_ids = mock_repo.delete_by_run_ids.call_args[0][1]
        assert "run-a" in deleted_ids

    def test_metrica_alterada_faz_upsert(self, indexer, mock_repo, mock_client):
        from app.core.services.mlflow_metadata_formatter import content_hash

        mock_repo.list_indexed_run_ids.return_value = {"run-a": "stale-hash"}
        mock_client.search_training_runs.return_value = [RUN_A]
        asyncio.run(indexer.run())
        mock_repo.upsert.assert_called_once()
