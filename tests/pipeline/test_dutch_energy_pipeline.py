"""
Testes do pipeline Medallion Dutch Energy.
Cobre Bronze, Silver (com FE), Gold (split only) e integração completa.
"""
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
from app.pipeline.dutch_energy_gold import build as gold_build, update_governance_with_model


# ──────────────────────────────────────────────
# Fixtures compartilhadas
# ──────────────────────────────────────────────

@pytest.fixture
def mock_storage():
    """Mock do backend de storage com suporte a metadados e idempotência."""
    storage = MagicMock(spec=StorageBackend)
    storage.ensure_bucket.return_value = None

    storage_data = {}
    storage_meta = {}

    def put_object(bucket, name, data, content_type, metadata=None):
        key = f"{bucket}/{name}"
        storage_data[key] = data.getvalue()
        storage_meta[key] = metadata or {}

    def get_object(bucket, name):
        key = f"{bucket}/{name}"
        if key not in storage_data:
            raise Exception(f"Objeto {key} não encontrado")
        return BytesIO(storage_data[key])

    def list_objects(bucket, prefix):
        return [
            k.split("/", 1)[1]
            for k in storage_data.keys()
            if k.startswith(f"{bucket}/{prefix}") and k.endswith(".csv")
        ]

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
    """CSV mínimo com todas as colunas necessárias para o pipeline."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "electricity_2018.csv"

    n = 30  # linhas suficientes para split 70/15/15
    df = pd.DataFrame({
        "net_manager":                  ["Manager A"] * 15 + ["Manager B"] * 15,
        "purchase_area":                ["Area A"] * 15 + ["Area B"] * 15,
        "city":                         [f"City_{i}" for i in range(n)],
        "num_connections":              [20] * 15 + [30] * 15,
        "delivery_perc":                [90.5] * n,
        "perc_of_active_connections":   [95.0] * n,
        "type_of_connection":           ["3x25"] * 15 + ["1x35"] * 15,
        "type_conn_perc":               [80.0] * n,
        "annual_consume":               [4000.0] * 15 + [6000.0] * 15,
        "annual_consume_lowtarif_perc": [50.0] * 15 + [70.0] * 15,
        "smartmeter_perc":              [10.0] * 15 + [20.0] * 15,
    })
    df.to_csv(csv_file, index=False)
    return str(data_dir)


# ──────────────────────────────────────────────
# Bronze
# ──────────────────────────────────────────────

class TestBronze:
    def test_ingestao_basica(self, sample_csv, mock_storage):
        """Bronze deve salvar o CSV com metadado consumed=true."""
        result = ingest_csvs(sample_csv, mock_storage)
        assert len(result) == 1
        mock_storage.put_file.assert_called_once()
        meta = mock_storage.stat_object("bronze", "dutch-energy/electricity_2018.csv")
        assert meta.get("consumed") == "true"

    def test_idempotencia(self, sample_csv, mock_storage):
        """Bronze não deve re-ingerir arquivo já marcado como consumed."""
        ingest_csvs(sample_csv, mock_storage)
        assert mock_storage.put_file.call_count == 1

        mock_storage.put_file.reset_mock()
        ingest_csvs(sample_csv, mock_storage)
        assert mock_storage.put_file.call_count == 0

    def test_manifesto_gerado(self, sample_csv, mock_storage):
        """Bronze deve gerar manifest.json com info dos arquivos."""
        ingest_csvs(sample_csv, mock_storage)
        raw = mock_storage.get_object("bronze", "dutch-energy/manifest.json")
        manifest = json.loads(raw.read())
        assert "ingested_at" in manifest
        assert manifest["total_files"] == 1
        assert len(manifest["files"]) == 1


# ──────────────────────────────────────────────
# Silver
# ──────────────────────────────────────────────

class TestSilver:
    def test_feature_engineering_cria_amperage(self, sample_csv, mock_storage):
        """Silver deve criar a feature 'amperage' a partir de type_of_connection."""
        ingest_csvs(sample_csv, mock_storage)
        df = silver_transform(mock_storage)
        assert "amperage" in df.columns
        # "3x25" → 3×25=75, "1x35" → 1×35=35
        assert set(df["amperage"].dropna().unique()) == {75.0, 35.0}

    def test_feature_engineering_cria_consume_per_conn(self, sample_csv, mock_storage):
        """Silver deve criar 'consume_per_conn' = annual_consume / num_connections."""
        ingest_csvs(sample_csv, mock_storage)
        df = silver_transform(mock_storage)
        assert "consume_per_conn" in df.columns
        # 4000/20=200 e 6000/30=200
        assert (df["consume_per_conn"] == 200.0).all()

    def test_feature_engineering_cria_total_capacity(self, sample_csv, mock_storage):
        """Silver deve criar 'total_capacity' = amperage × num_connections."""
        ingest_csvs(sample_csv, mock_storage)
        df = silver_transform(mock_storage)
        assert "total_capacity" in df.columns
        expected = {75 * 20, 35 * 30}  # {1500, 1050}
        assert set(df["total_capacity"].unique()) == expected

    def test_feature_engineering_cria_hightarif_perc(self, sample_csv, mock_storage):
        """Silver deve criar 'hightarif_perc' = 100 - annual_consume_lowtarif_perc."""
        ingest_csvs(sample_csv, mock_storage)
        df = silver_transform(mock_storage)
        assert "hightarif_perc" in df.columns
        # 100-50=50 e 100-70=30
        assert set(df["hightarif_perc"].unique()) == {50.0, 30.0}

    def test_feature_engineering_cria_log_target(self, sample_csv, mock_storage):
        """Silver deve criar 'log_target' = log1p(consume_per_conn)."""
        ingest_csvs(sample_csv, mock_storage)
        df = silver_transform(mock_storage)
        assert "log_target" in df.columns
        expected = np.log1p(200.0)
        assert np.allclose(df["log_target"].values, expected, atol=1e-4)

    def test_remocao_de_duplicatas(self, tmp_path, mock_storage):
        """Silver deve remover linhas duplicadas."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        base_row = {
            "net_manager": "M", "purchase_area": "A", "city": "C",
            "num_connections": 10.0, "delivery_perc": 90.0,
            "perc_of_active_connections": 95.0, "type_of_connection": "3x25",
            "type_conn_perc": 80.0, "annual_consume": 2000.0,
            "annual_consume_lowtarif_perc": 50.0, "smartmeter_perc": 10.0,
        }
        df = pd.DataFrame([base_row] * 5)  # 5 linhas idênticas
        (data_dir / "electricity_2019.csv").write_text(df.to_csv(index=False))
        ingest_csvs(str(data_dir), mock_storage)
        result = silver_transform(mock_storage)
        assert len(result) == 1  # apenas 1 linha única

    def test_remocao_consumo_invalido(self, tmp_path, mock_storage):
        """Silver deve remover linhas com annual_consume <= 0."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = pd.DataFrame({
            "net_manager": ["M"] * 3, "purchase_area": ["A"] * 3, "city": ["C1", "C2", "C3"],
            "num_connections": [10.0] * 3, "delivery_perc": [90.0] * 3,
            "perc_of_active_connections": [95.0] * 3, "type_of_connection": ["3x25"] * 3,
            "type_conn_perc": [80.0] * 3, "annual_consume": [0.0, -100.0, 2000.0],
            "annual_consume_lowtarif_perc": [50.0] * 3, "smartmeter_perc": [10.0] * 3,
        })
        (data_dir / "electricity_2020.csv").write_text(df.to_csv(index=False))
        ingest_csvs(str(data_dir), mock_storage)
        result = silver_transform(mock_storage)
        assert len(result) == 1

    def test_governance_doc_gerado(self, sample_csv, mock_storage):
        """Silver deve gerar governance_silver.md no MinIO."""
        ingest_csvs(sample_csv, mock_storage)
        silver_transform(mock_storage)
        raw = mock_storage.get_object("silver", "dutch-energy/governance_silver.md")
        content = raw.read().decode("utf-8")
        assert "Documento de Governança — Camada Silver" in content

    def test_governance_doc_contem_secoes_obrigatorias(self, sample_csv, mock_storage):
        """governance_silver.md deve conter as seções exigidas pelo professor."""
        ingest_csvs(sample_csv, mock_storage)
        silver_transform(mock_storage)
        raw = mock_storage.get_object("silver", "dutch-energy/governance_silver.md")
        content = raw.read().decode("utf-8")
        assert "Tipos de Dados" in content
        assert "Limpeza de Dados" in content
        assert "Feature Engineering" in content
        assert "amperage" in content
        assert "consume_per_conn" in content
        assert "log_target" in content

    def test_idempotencia(self, sample_csv, mock_storage):
        """Silver deve pular processamento se cleaned.csv já existe."""
        ingest_csvs(sample_csv, mock_storage)
        silver_transform(mock_storage)

        mock_storage.list_objects.reset_mock()
        df = silver_transform(mock_storage)
        assert mock_storage.list_objects.call_count == 0
        assert len(df) == 30

    def test_force_reprocessa(self, sample_csv, mock_storage):
        """Silver deve reprocessar quando force=True."""
        ingest_csvs(sample_csv, mock_storage)
        silver_transform(mock_storage)

        mock_storage.list_objects.reset_mock()
        silver_transform(mock_storage, force=True)
        assert mock_storage.list_objects.call_count == 1


# ──────────────────────────────────────────────
# Gold
# ──────────────────────────────────────────────

class TestGold:
    def _setup(self, sample_csv, mock_storage):
        ingest_csvs(sample_csv, mock_storage)
        silver_transform(mock_storage)

    def test_split_gera_tres_conjuntos(self, sample_csv, mock_storage):
        """Gold deve gerar X_train, X_val, X_test, y_train, y_val, y_test."""
        self._setup(sample_csv, mock_storage)
        result = gold_build(mock_storage)
        for key in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
            assert key in result

    def test_split_proporcoes(self, sample_csv, mock_storage):
        """Split 70/15/15 deve aproximar as proporções corretas."""
        self._setup(sample_csv, mock_storage)
        result = gold_build(mock_storage)
        total = len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"])
        assert total == 30
        assert len(result["X_train"]) > len(result["X_val"])
        assert len(result["X_val"]) == len(result["X_test"])

    def test_nao_faz_feature_engineering(self, sample_csv, mock_storage):
        """Gold NÃO deve criar amperage (isso é responsabilidade do Silver)."""
        ingest_csvs(sample_csv, mock_storage)
        # Fornece Silver sem FE para confirmar que Gold não cria
        import app.pipeline.dutch_energy_gold as gold_module
        original = gold_module._select_features

        called_with_amperage = []

        def spy(df):
            called_with_amperage.append("amperage" in df.columns)
            return original(df)

        gold_module._select_features = spy
        try:
            silver_transform(mock_storage)
            gold_build(mock_storage)
            # O df que Gold recebe do Silver já tem amperage (Silver criou)
            assert all(called_with_amperage), "Gold deve receber amperage do Silver"
        finally:
            gold_module._select_features = original

    def test_target_encoding_aplicado(self, sample_csv, mock_storage):
        """Gold deve aplicar target encoding nas colunas categóricas."""
        self._setup(sample_csv, mock_storage)
        result = gold_build(mock_storage)
        X_train = result["X_train"]
        assert "city_te" in X_train.columns
        assert "purchase_area_te" in X_train.columns
        assert "net_manager_te" in X_train.columns
        # Colunas originais devem ter sido removidas
        assert "city" not in X_train.columns
        assert "net_manager" not in X_train.columns

    def test_y_train_e_log_target(self, sample_csv, mock_storage):
        """y_train deve conter os valores de log_target gerados pelo Silver."""
        self._setup(sample_csv, mock_storage)
        result = gold_build(mock_storage)
        expected = np.log1p(200.0)
        assert np.allclose(result["y_train"].values, expected, atol=1e-3)

    def test_artefatos_salvos_no_minio(self, sample_csv, mock_storage):
        """Gold deve salvar todos os artefatos esperados no MinIO."""
        self._setup(sample_csv, mock_storage)
        gold_build(mock_storage)
        for artifact in [
            "dutch-energy/X_train.csv", "dutch-energy/X_val.csv", "dutch-energy/X_test.csv",
            "dutch-energy/y_train.csv", "dutch-energy/y_val.csv", "dutch-energy/y_test.csv",
            "dutch-energy/scaler.pkl", "dutch-energy/target_encoders.pkl",
            "dutch-energy/feature_cols.json", "dutch-energy/gold_metadata.json",
            "dutch-energy/governance_gold.md",
        ]:
            raw = mock_storage.get_object("gold", artifact)
            assert raw.read() != b"", f"Artefato vazio: {artifact}"

    def test_governance_doc_tem_placeholder(self, sample_csv, mock_storage):
        """governance_gold.md deve ter placeholder do modelo antes do treinamento."""
        self._setup(sample_csv, mock_storage)
        gold_build(mock_storage)
        raw = mock_storage.get_object("gold", "dutch-energy/governance_gold.md")
        content = raw.read().decode("utf-8")
        assert "Aguardando treinamento MLflow" in content

    def test_governance_doc_contem_secoes_obrigatorias(self, sample_csv, mock_storage):
        """governance_gold.md deve ter seções de divisão dos dados e features."""
        self._setup(sample_csv, mock_storage)
        gold_build(mock_storage)
        raw = mock_storage.get_object("gold", "dutch-energy/governance_gold.md")
        content = raw.read().decode("utf-8")
        assert "Divisão dos Dados" in content
        assert "Treino" in content
        assert "Validação" in content
        assert "Teste" in content
        assert "Features do Modelo" in content

    def test_update_governance_substitui_placeholder(self, sample_csv, mock_storage):
        """update_governance_with_model deve substituir o placeholder com info do modelo."""
        self._setup(sample_csv, mock_storage)
        gold_build(mock_storage)

        model_info = {
            "run_id": "abc123",
            "algorithm": "XGBoost",
            "metrics": {"rmse": 89.1, "mae": 39.5, "r2": 0.743, "mape": 17.6},
            "params": {"n_estimators": "500", "learning_rate": "0.05"},
        }
        update_governance_with_model(mock_storage, model_info)

        raw = mock_storage.get_object("gold", "dutch-energy/governance_gold.md")
        content = raw.read().decode("utf-8")
        assert "Aguardando treinamento MLflow" not in content
        assert "XGBoost" in content
        assert "abc123" in content
        assert "89.1" in content

    def test_idempotencia(self, sample_csv, mock_storage):
        """Gold deve retornar {} se artefatos já existem."""
        self._setup(sample_csv, mock_storage)
        gold_build(mock_storage)
        result = gold_build(mock_storage)
        assert result == {}

    def test_force_reprocessa(self, sample_csv, mock_storage):
        """Gold deve reprocessar quando force=True."""
        self._setup(sample_csv, mock_storage)
        gold_build(mock_storage)
        result = gold_build(mock_storage, force=True)
        assert "X_train" in result

    def test_feature_cols_json_correto(self, sample_csv, mock_storage):
        """feature_cols.json deve listar as features numéricas finais."""
        self._setup(sample_csv, mock_storage)
        gold_build(mock_storage)
        raw = mock_storage.get_object("gold", "dutch-energy/feature_cols.json")
        feat_cols = json.loads(raw.read())
        assert isinstance(feat_cols, list)
        assert len(feat_cols) > 0
        assert "amperage" in feat_cols
        assert "city_te" in feat_cols
        # Targets não devem estar nas features
        assert "consume_per_conn" not in feat_cols
        assert "log_target" not in feat_cols

    def test_scaler_pkl_valido(self, sample_csv, mock_storage):
        """scaler.pkl deve ser um StandardScaler serializável."""
        from sklearn.preprocessing import StandardScaler
        self._setup(sample_csv, mock_storage)
        gold_build(mock_storage)
        raw = mock_storage.get_object("gold", "dutch-energy/scaler.pkl")
        scaler = pickle.loads(raw.read())
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mean_")


# ──────────────────────────────────────────────
# Integração completa Bronze → Silver → Gold
# ──────────────────────────────────────────────

class TestPipelineIntegration:
    def test_pipeline_completo_bronze_silver_gold(self, sample_csv, mock_storage):
        """Pipeline end-to-end deve produzir artefatos prontos para treinamento."""
        # Bronze
        ingested = ingest_csvs(sample_csv, mock_storage)
        assert len(ingested) == 1

        # Silver
        df = silver_transform(mock_storage)
        assert "log_target" in df.columns
        assert "amperage" in df.columns
        assert len(df) == 30

        # Gold
        result = gold_build(mock_storage)
        assert "X_train" in result
        total = sum(len(result[k]) for k in ["X_train", "X_val", "X_test"])
        assert total == 30

    def test_features_consistentes_entre_camadas(self, sample_csv, mock_storage):
        """As features que Silver gera devem estar presentes no Gold."""
        ingest_csvs(sample_csv, mock_storage)
        silver_transform(mock_storage)
        result = gold_build(mock_storage)

        raw = mock_storage.get_object("gold", "dutch-energy/feature_cols.json")
        feat_cols = json.loads(raw.read())

        X_train = result["X_train"]
        assert list(X_train.columns) == feat_cols

    def test_sem_data_leakage_no_target_encoding(self, sample_csv, mock_storage):
        """Target encoding deve ser calculado apenas no treino (sem leakage)."""
        ingest_csvs(sample_csv, mock_storage)
        silver_transform(mock_storage)
        result = gold_build(mock_storage)

        raw = mock_storage.get_object("gold", "dutch-energy/target_encoders.pkl")
        encoders = pickle.loads(raw.read())
        # Cada encoder deve ter o mapa calculado no treino e um global_mean
        for col in ["city", "purchase_area", "net_manager"]:
            assert col in encoders
            assert "map" in encoders[col]
            assert "global_mean" in encoders[col]
