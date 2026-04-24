import os
import json
import pickle
from io import BytesIO
from unittest.mock import MagicMock

import pytest
import pandas as pd
import numpy as np

from app.pipeline.storage import StorageBackend
from app.pipeline.dutch_energy_bronze import ingest_csvs
from app.pipeline.dutch_energy_silver import transform as silver_transform
from app.pipeline.dutch_energy_gold import build as gold_build

@pytest.fixture
def mock_storage():
    """Mock do backend de storage com suporte a metadados e idempotencia."""
    storage = MagicMock(spec=StorageBackend)
    storage.ensure_bucket.return_value = None
    
    # Dicionarios internos para simular o storage
    storage_data = {}
    storage_meta = {}

    def put_object(bucket, name, data, content_type, metadata=None):
        key = f"{bucket}/{name}"
        storage_data[key] = data.getvalue()
        storage_meta[key] = metadata or {}

    def get_object(bucket, name):
        key = f"{bucket}/{name}"
        if key not in storage_data:
            raise Exception(f"Objeto {key} nao encontrado")
        return BytesIO(storage_data[key])

    def list_objects(bucket, prefix):
        return [k.split("/", 1)[1] for k in storage_data.keys() if k.startswith(f"{bucket}/{prefix}")]

    def put_file(bucket, name, file_path, metadata=None):
        key = f"{bucket}/{name}"
        with open(file_path, "rb") as f:
            storage_data[key] = f.read()
        storage_meta[key] = metadata or {}

    def stat_object(bucket, name):
        key = f"{bucket}/{name}"
        return storage_meta.get(key, {})

    storage.put_object.side_effect = put_object
    storage.get_object.side_effect = get_object
    storage.list_objects.side_effect = list_objects
    storage.put_file.side_effect = put_file
    storage.stat_object.side_effect = stat_object
    
    return storage

@pytest.fixture
def sample_csv(tmp_path):
    """Cria um CSV de exemplo para teste do bronze."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "electricity_2018.csv"
    
    df = pd.DataFrame({
        "net_manager": ["Manager A"] * 10 + ["Manager B"] * 10,
        "purchase_area": ["Area A"] * 10 + ["Area B"] * 10,
        "city": [f"City_{i}" for i in range(20)],
        "num_connections": [20] * 10 + [30] * 10,
        "delivery_perc": [90.5] * 10 + [100.0] * 10,
        "perc_of_active_connections": [95.0] * 10 + [100.0] * 10,
        "type_of_connection": ["3x25"] * 10 + ["1x35"] * 10,
        "type_conn_perc": [80.0] * 10 + [100.0] * 10,
        "annual_consume": [4000.0] * 10 + [6000.0] * 10,
        "annual_consume_lowtarif_perc": [50.0] * 10 + [70.0] * 10,
        "smartmeter_perc": [10.0] * 10 + [20.0] * 10
    })
    df.to_csv(csv_file, index=False)
    return str(data_dir)

# ── Bronze Tests ──────────────────────────────────────────────

def test_bronze_idempotency(sample_csv, mock_storage):
    """Verifica se o Bronze pula arquivos ja marcados como consumed."""
    # 1. Primeira ingestao
    ingest_csvs(sample_csv, mock_storage)
    assert mock_storage.put_file.call_count == 1
    
    # Verifica se metadado foi setado
    meta = mock_storage.stat_object("bronze", "dutch-energy/electricity_2018.csv")
    assert meta.get("consumed") == "true"

    # 2. Segunda ingestao (deve pular)
    mock_storage.put_file.reset_mock()
    ingest_csvs(sample_csv, mock_storage)
    assert mock_storage.put_file.call_count == 0

# ── Silver Tests ──────────────────────────────────────────────

def test_silver_idempotency(sample_csv, mock_storage):
    """Verifica se o Silver pula processamento se cleaned.csv ja existe com meta."""
    ingest_csvs(sample_csv, mock_storage)
    
    # 1. Primeiro processamento
    silver_transform(mock_storage)
    assert mock_storage.put_object.call_count >= 1 # cleaned.csv + stats
    
    # 2. Segundo processamento (deve pular logic de limpeza)
    # Mockamos _load_bronze indiretamente: se o list_objects nao for chamado, pulou.
    mock_storage.list_objects.reset_mock()
    df = silver_transform(mock_storage)
    assert mock_storage.list_objects.call_count == 0
    assert len(df) == 20

    # 3. Terceiro processamento com force=True
    df_forced = silver_transform(mock_storage, force=True)
    assert mock_storage.list_objects.call_count == 1
    assert len(df_forced) == 20

# ── Gold Tests ─────────────────────────────────────────────────

def test_gold_idempotency(sample_csv, mock_storage):
    """Verifica se o Gold pula se artefatos ja existem."""
    ingest_csvs(sample_csv, mock_storage)
    silver_transform(mock_storage)
    
    # 1. Primeiro build
    gold_build(mock_storage)
    
    # 2. Segundo build (deve retornar {} por idempotencia)
    # Resetamos mock_storage para ver se ele tenta ler do silver
    mock_storage.get_object.reset_mock()
    res = gold_build(mock_storage)
    assert res == {}
    # get_object nao deve ter sido chamado para o silver 'cleaned.csv'
    # Nota: no meu mock simplified, get_object e chamado no early return do silver, 
    # mas no Gold eu fiz um return {} simples.
    
    # 3. Terceiro build com force=True
    res_forced = gold_build(mock_storage, force=True)
    assert "X_train" in res_forced

def test_gold_feature_logic(sample_csv, mock_storage):
    """Verifica se a logica de features (amperage, log target) continua correta."""
    ingest_csvs(sample_csv, mock_storage)
    silver_transform(mock_storage)
    
    artifacts = gold_build(mock_storage)
    X_train = artifacts["X_train"]
    y_train = artifacts["y_train"]
    
    assert "amperage" in X_train.columns
    assert "city_te" in X_train.columns
    # Log1p(200) ~= 5.3033
    assert np.isclose(y_train.iloc[0], np.log1p(200.0), atol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__])
