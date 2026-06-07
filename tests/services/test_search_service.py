from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.services.search_service import SearchService


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.search.side_effect = [
        [[{"id": "1", "distance": 0.3, "entity": {"text": "gov doc", "source": "gold_governance"}}]],
        [[{"id": "2", "distance": 0.1, "entity": {"text": "mlflow run", "source": "mlflow_metadata", "run_id": "r1"}}]],
    ]
    return repo


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embbed_it.return_value = np.random.rand(1, 384)
    return embedder


class TestSearchService:

    def test_merge_ordena_por_distancia(self, mock_repo, mock_embedder):
        service = SearchService(mock_repo, mock_embedder)
        result = service.search("qual o rmse?")

        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0]["distance"] == 0.1
        assert result[0][1]["distance"] == 0.3

    def test_busca_em_duas_collections(self, mock_repo, mock_embedder):
        service = SearchService(mock_repo, mock_embedder)
        service.search("treinamento")

        assert mock_repo.search.call_count == 2
