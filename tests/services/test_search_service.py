from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.services.search_service import SearchService


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    repo.search.return_value = [
        [
            {"id": "1", "distance": 0.1, "entity": {"text": "mlflow run", "source": "mlflow_metadata"}},
            {"id": "2", "distance": 0.3, "entity": {"text": "gov doc", "source": "gold_governance"}},
        ]
    ]
    return repo


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embbed_it.return_value = np.random.rand(1, 384)
    return embedder


class TestSearchService:

    def test_retorna_hits_da_collection_governance(self, mock_repo, mock_embedder):
        service = SearchService(mock_repo, mock_embedder)
        result = service.search("qual o rmse?")

        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0]["distance"] == 0.1

    def test_busca_apenas_governance(self, mock_repo, mock_embedder):
        service = SearchService(mock_repo, mock_embedder)
        service.search("treinamento")

        mock_repo.search.assert_called_once()
