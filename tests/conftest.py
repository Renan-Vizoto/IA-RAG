"""
conftest.py raiz: mock de módulos pesados (sentence_transformers, minio, etc.)
para que os testes possam importar o código sem precisar dos serviços instalados.
"""
import sys
from unittest.mock import MagicMock

# Mocka módulos pesados/externos antes que qualquer módulo os importe
_HEAVY_MOCKS = [
    "sentence_transformers",
    "minio",
    "minio.versioningconfig",
    "pdfplumber",
    "pypdf",
    # mlflow é mockado globalmente para evitar conexão ao servidor na importação.
    # Os testes que precisam de comportamento específico usam @patch localmente.
    "mlflow",
    "mlflow.xgboost",
]

for mod in _HEAVY_MOCKS:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# SentenceTransformer precisa de um mock específico para encode() retornar numpy
import numpy as np
_st_mock = MagicMock()
_st_mock.SentenceTransformer.return_value.encode.return_value = np.zeros((1, 384))
sys.modules["sentence_transformers"] = _st_mock
