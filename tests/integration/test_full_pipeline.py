"""
Testes de integração do pipeline Dutch Energy.

Usam implementações REAIS:
  - LocalStorage: escrita/leitura em filesystem real (sem MinIO)
  - Pandas / NumPy / scikit-learn: processamento real dos dados
  - XGBoost: treinamento real com dados sintéticos
  - MLflow: mockado apenas para a camada de tracking server
             (XGBoost treina de verdade; só evitamos conexão HTTP ao servidor)

O que diferencia dos testes unitários:
  - Nenhum mock de lógica de dados
  - As features são realmente calculadas
  - Os arquivos são escritos e lidos de/para o disco
  - Os documentos de governança têm conteúdo real e verificável
  - O modelo XGBoost é treinado de verdade e faz predições reais
  - A cadeia Bronze → Silver → Gold → Training é executada de ponta a ponta
"""
import json
import os
import pickle
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from app.pipeline.dutch_energy_bronze import ingest_csvs
from app.pipeline.dutch_energy_gold import build as gold_build, update_governance_with_model
from app.pipeline.dutch_energy_silver import transform as silver_transform
from app.pipeline.dutch_energy_train import train


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _read_file(local_storage, bucket: str, path: str) -> str:
    raw = local_storage.get_object(bucket, path)
    return raw.read().decode("utf-8")


def _read_json(local_storage, bucket: str, path: str) -> dict | list:
    raw = local_storage.get_object(bucket, path)
    return json.loads(raw.read())


def _read_pkl(local_storage, bucket: str, path: str):
    raw = local_storage.get_object(bucket, path)
    return pickle.loads(raw.read())


def _read_csv_from_storage(local_storage, bucket: str, path: str) -> pd.DataFrame:
    raw = local_storage.get_object(bucket, path)
    return pd.read_csv(raw)


# ──────────────────────────────────────────────
# Testes de integração: Bronze
# ──────────────────────────────────────────────

class TestBronzeIntegration:

    def test_csv_salvo_em_disco(self, data_dir, local_storage, tmp_path):
        """Bronze deve escrever o CSV no filesystem (LocalStorage)."""
        ingest_csvs(data_dir, local_storage)
        csv_path = Path(tmp_path) / "bronze" / "dutch-energy" / "electricity_2018.csv"
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

    def test_manifest_tem_conteudo_real(self, data_dir, local_storage):
        """manifest.json deve listar os arquivos realmente ingeridos."""
        ingest_csvs(data_dir, local_storage)
        manifest = _read_json(local_storage, "bronze", "dutch-energy/manifest.json")
        assert manifest["total_files"] == 1
        # files é lista de strings (caminhos object_name)
        assert any("electricity_2018.csv" in f for f in manifest["files"])

    def test_metadado_consumed_persistido(self, data_dir, local_storage, tmp_path):
        """Metadado consumed=true deve ser persistido em arquivo .meta."""
        ingest_csvs(data_dir, local_storage)
        meta_path = (
            Path(tmp_path) / "bronze" / "dutch-energy" / "electricity_2018.csv.meta"
        )
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta.get("consumed") == "true"


# ──────────────────────────────────────────────
# Testes de integração: Silver
# ──────────────────────────────────────────────

class TestSilverIntegration:

    def test_feature_engineering_com_dados_reais(self, data_dir, local_storage):
        """Silver deve aplicar FE real: amperage varia por tipo de conexão."""
        ingest_csvs(data_dir, local_storage)
        df = silver_transform(local_storage)

        # "3x25" → 75,  "1x35" → 35
        assert set(df["amperage"].dropna().unique()) == {75.0, 35.0}

    def test_consume_per_conn_calculado_corretamente(self, data_dir, local_storage):
        """consume_per_conn = annual_consume / num_connections deve ser positivo."""
        ingest_csvs(data_dir, local_storage)
        df = silver_transform(local_storage)
        assert (df["consume_per_conn"] > 0).all()
        # Verifica fórmula para uma amostra
        sample = df.iloc[0]
        expected = sample["annual_consume"] / sample["num_connections"]
        assert abs(sample["consume_per_conn"] - expected) < 1e-3

    def test_log_target_e_log1p_de_consume_per_conn(self, data_dir, local_storage):
        """log_target deve ser exatamente log1p(consume_per_conn)."""
        ingest_csvs(data_dir, local_storage)
        df = silver_transform(local_storage)
        expected = np.log1p(df["consume_per_conn"])
        assert np.allclose(df["log_target"].values, expected.values, atol=1e-5)

    def test_outliers_removidos_do_dataset(self, data_dir, local_storage):
        """Silver deve remover outliers de consume_per_conn (P99.5 calculado antes da remoção)."""
        ingest_csvs(data_dir, local_storage)
        # Calcula o threshold original a partir do CSV bruto do Bronze
        raw = local_storage.get_object("bronze", "dutch-energy/electricity_2018.csv")
        df_raw = pd.read_csv(raw)
        df_raw = df_raw[df_raw["annual_consume"].notna() & (df_raw["annual_consume"] > 0)]
        df_raw = df_raw[df_raw["num_connections"].notna() & (df_raw["num_connections"] > 0)]
        cpc_raw = df_raw["annual_consume"] / df_raw["num_connections"]
        threshold = cpc_raw.quantile(0.995)

        df = silver_transform(local_storage)
        # Após remoção, consume_per_conn deve estar abaixo do threshold original
        assert (df["consume_per_conn"] <= threshold + 1e-6).all()

    def test_cleaned_csv_salvo_em_disco(self, data_dir, local_storage, tmp_path):
        """cleaned.csv deve ser escrito no filesystem."""
        ingest_csvs(data_dir, local_storage)
        silver_transform(local_storage)
        csv_path = Path(tmp_path) / "silver" / "dutch-energy" / "cleaned.csv"
        assert csv_path.exists()
        df_disk = pd.read_csv(csv_path)
        assert len(df_disk) > 0
        assert "log_target" in df_disk.columns

    def test_governance_silver_escrito_em_disco(self, data_dir, local_storage, tmp_path):
        """governance_silver.md deve ser escrito em disco com conteúdo real."""
        ingest_csvs(data_dir, local_storage)
        silver_transform(local_storage)
        md_path = Path(tmp_path) / "silver" / "dutch-energy" / "governance_silver.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        # Verifica conteúdo dinâmico real (estatísticas calculadas)
        assert re.search(r"Registros iniciais.*\d+", content)
        assert re.search(r"Registros finais.*\d+", content)
        assert "amperage" in content
        assert "consume_per_conn" in content

    def test_governance_silver_tipos_de_dados_corretos(self, data_dir, local_storage, tmp_path):
        """governance_silver.md deve listar os tipos reais das colunas."""
        ingest_csvs(data_dir, local_storage)
        silver_transform(local_storage)
        md_path = Path(tmp_path) / "silver" / "dutch-energy" / "governance_silver.md"
        content = md_path.read_text(encoding="utf-8")
        assert "float32" in content  # tipos reais do processamento

    def test_cleaning_stats_json_salvo(self, data_dir, local_storage):
        """cleaning_stats.json deve ser gerado com estatísticas reais."""
        ingest_csvs(data_dir, local_storage)
        silver_transform(local_storage)
        stats = _read_json(local_storage, "silver", "dutch-energy/cleaning_stats.json")
        assert stats["initial_rows"] == 200
        assert stats["final_rows"] <= 200
        assert stats["final_rows"] > 0

    def test_silver_idempotencia_le_do_disco(self, data_dir, local_storage):
        """Segunda chamada deve ler cleaned.csv do disco (não reprocessar)."""
        ingest_csvs(data_dir, local_storage)
        df1 = silver_transform(local_storage)
        df2 = silver_transform(local_storage)
        # Conteúdo idêntico
        assert len(df1) == len(df2)
        assert list(df1.columns) == list(df2.columns)


# ──────────────────────────────────────────────
# Testes de integração: Gold
# ──────────────────────────────────────────────

class TestGoldIntegration:

    def _setup(self, data_dir, local_storage):
        ingest_csvs(data_dir, local_storage)
        silver_transform(local_storage)

    def test_artefatos_escritos_em_disco(self, data_dir, local_storage, tmp_path):
        """Gold deve escrever todos os artefatos no filesystem."""
        self._setup(data_dir, local_storage)
        gold_build(local_storage)

        base = Path(tmp_path) / "gold" / "dutch-energy"
        for fname in [
            "X_train.csv", "X_val.csv", "X_test.csv",
            "y_train.csv", "y_val.csv", "y_test.csv",
            "scaler.pkl", "target_encoders.pkl",
            "feature_cols.json", "gold_metadata.json",
            "governance_gold.md",
        ]:
            p = base / fname
            assert p.exists(), f"Artefato ausente: {fname}"
            assert p.stat().st_size > 0

    def test_split_com_dados_reais(self, data_dir, local_storage):
        """Split deve respeitar proporção 70/15/15 com dados reais."""
        self._setup(data_dir, local_storage)
        result = gold_build(local_storage)
        n_train = len(result["X_train"])
        n_val   = len(result["X_val"])
        n_test  = len(result["X_test"])
        total   = n_train + n_val + n_test
        assert total > 0
        assert abs(n_train / total - 0.70) < 0.05
        assert abs(n_val   / total - 0.15) < 0.05
        assert abs(n_test  / total - 0.15) < 0.05

    def test_scaler_fitted_apenas_no_treino(self, data_dir, local_storage):
        """StandardScaler deve ter mean_ e scale_ ajustados no treino."""
        self._setup(data_dir, local_storage)
        gold_build(local_storage)
        scaler = _read_pkl(local_storage, "gold", "dutch-energy/scaler.pkl")
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mean_")
        assert len(scaler.mean_) > 0

    def test_target_encoders_tem_mapa_real(self, data_dir, local_storage):
        """target_encoders.pkl deve conter mapas calculados do treino real."""
        self._setup(data_dir, local_storage)
        gold_build(local_storage)
        encoders = _read_pkl(local_storage, "gold", "dutch-energy/target_encoders.pkl")
        for col in ["city", "purchase_area", "net_manager"]:
            assert col in encoders
            assert len(encoders[col]["map"]) > 0  # mapa não vazio

    def test_x_train_sem_nans(self, data_dir, local_storage):
        """X_train não deve ter valores NaN."""
        self._setup(data_dir, local_storage)
        result = gold_build(local_storage)
        assert not result["X_train"].isnull().any().any()

    def test_feature_cols_sao_as_reais(self, data_dir, local_storage):
        """feature_cols.json deve listar as colunas reais do X_train."""
        self._setup(data_dir, local_storage)
        result = gold_build(local_storage)
        feat_cols = _read_json(local_storage, "gold", "dutch-energy/feature_cols.json")
        assert list(result["X_train"].columns) == feat_cols

    def test_governance_gold_tem_placeholder(self, data_dir, local_storage, tmp_path):
        """governance_gold.md deve ter placeholder antes do treinamento."""
        self._setup(data_dir, local_storage)
        gold_build(local_storage)
        md = (Path(tmp_path) / "gold" / "dutch-energy" / "governance_gold.md").read_text()
        assert "Aguardando treinamento MLflow" in md
        # Deve ter os números reais de linhas
        assert re.search(r"Treino.*\d+", md)

    def test_gold_metadata_json_contem_info_real(self, data_dir, local_storage):
        """gold_metadata.json deve refletir o split real feito."""
        self._setup(data_dir, local_storage)
        result = gold_build(local_storage)
        meta = _read_json(local_storage, "gold", "dutch-energy/gold_metadata.json")
        assert meta["split"]["train"] == len(result["X_train"])
        assert meta["split"]["val"]   == len(result["X_val"])
        assert meta["split"]["test"]  == len(result["X_test"])
        assert meta["seed"] == 42


# ──────────────────────────────────────────────
# Testes de integração: Training (XGBoost real)
# ──────────────────────────────────────────────

class TestTrainingIntegration:

    def _setup(self, data_dir, local_storage):
        ingest_csvs(data_dir, local_storage)
        silver_transform(local_storage)
        gold_build(local_storage)

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_xgboost_treina_de_verdade(self, mock_mlflow, mock_pickle, data_dir, local_storage):
        """XGBoost deve ser instanciado e treinado com os dados reais do Gold."""
        trained_models = []

        def capture_pickle(obj):
            trained_models.append(obj)
            return b"fake"

        mock_pickle.side_effect = capture_pickle
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="r1"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        self._setup(data_dir, local_storage)
        train(local_storage)

        # pickle.dumps patches the shared module, so gold's scaler/encoders
        # may also be captured; filter to find the XGBoost model specifically
        xgb_models = [m for m in trained_models if isinstance(m, xgb.XGBRegressor)]
        assert len(xgb_models) == 1

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_modelo_faz_predicoes_validas(self, mock_mlflow, mock_pickle, data_dir, local_storage):
        """Modelo treinado deve fazer predições numéricas válidas no conjunto de validação."""
        trained_models = []

        def capture_pickle(obj):
            trained_models.append(obj)
            return b"fake"

        mock_pickle.side_effect = capture_pickle
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="r1"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        self._setup(data_dir, local_storage)
        result = train(local_storage)

        # Filter to the XGBoost model (pickle patches shared module; gold also serializes scaler/encoders)
        xgb_models = [m for m in trained_models if isinstance(m, xgb.XGBRegressor)]
        assert len(xgb_models) == 1, f"Expected 1 XGBRegressor, got {len(xgb_models)}"
        model = xgb_models[0]
        # Carrega X_val para fazer predição extra
        X_val = _read_csv_from_storage(local_storage, "gold", "dutch-energy/X_val.csv").values
        preds = model.predict(X_val)

        assert len(preds) == len(X_val)
        assert np.isfinite(preds).all()
        assert preds.min() > 0  # log_target > 0

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_metricas_reais_sao_razoaveis(self, mock_mlflow, mock_pickle, data_dir, local_storage):
        """Métricas reais do XGBoost treinado devem ser numericamente razoáveis."""
        mock_pickle.return_value = b"fake"
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="r1"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        self._setup(data_dir, local_storage)
        result = train(local_storage)

        metrics = result["metrics"]
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert -1.0 <= metrics["r2"] <= 1.0  # R² pode ser negativo com dados ruins
        assert metrics["mape"] >= 0

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_relatorio_tem_metricas_reais(self, mock_mlflow, mock_pickle, data_dir, local_storage, tmp_path):
        """mlflow_report.md deve conter os valores reais de RMSE/MAE/R²."""
        mock_pickle.return_value = b"fake"
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="run-real-123"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        self._setup(data_dir, local_storage)
        result = train(local_storage)

        md_path = Path(tmp_path) / "gold" / "dutch-energy" / "mlflow_report.md"
        content = md_path.read_text(encoding="utf-8")

        # Métricas reais devem aparecer no relatório
        for metric_name in ["RMSE", "MAE", "R2"]:
            assert metric_name in content
        # Run ID capturado deve aparecer
        assert "run-real-123" in content

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_governance_gold_atualizado_com_xgboost(self, mock_mlflow, mock_pickle, data_dir, local_storage, tmp_path):
        """governance_gold.md deve ser atualizado com nome real do algoritmo."""
        mock_pickle.return_value = b"fake"
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="r42"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        self._setup(data_dir, local_storage)
        train(local_storage)

        md_path = Path(tmp_path) / "gold" / "dutch-energy" / "governance_gold.md"
        content = md_path.read_text(encoding="utf-8")
        assert "Aguardando treinamento MLflow" not in content
        assert "XGBoost" in content
        assert "r42" in content

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_mlflow_recebeu_params_e_metricas_reais(self, mock_mlflow, mock_pickle, data_dir, local_storage):
        """MLflow deve ter recebido os hiperparâmetros e métricas calculadas."""
        mock_pickle.return_value = b"fake"
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="r1"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        self._setup(data_dir, local_storage)
        train(local_storage)

        # Parâmetros
        params_logged = mock_mlflow.log_params.call_args[0][0]
        assert params_logged["n_estimators"] == 500

        # Métricas — valores reais (não mockados)
        metrics_logged = mock_mlflow.log_metrics.call_args[0][0]
        assert isinstance(metrics_logged["rmse"], float)
        assert isinstance(metrics_logged["r2"], float)
        assert np.isfinite(metrics_logged["rmse"])


# ──────────────────────────────────────────────
# Teste de integração: Pipeline completo E2E
# ──────────────────────────────────────────────

class TestPipelineE2E:

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_bronze_silver_gold_train_e2e(
        self, mock_mlflow, mock_pickle, data_dir, local_storage, tmp_path
    ):
        """
        Pipeline completo Bronze → Silver → Gold → Training.
        Verifica que cada camada produz artefatos consumíveis pela próxima.
        """
        mock_pickle.return_value = b"model_bytes"
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="e2e-run"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Bronze
        ingested = ingest_csvs(data_dir, local_storage)
        assert len(ingested) == 1

        # Silver — lê do Bronze, escreve features reais
        df_silver = silver_transform(local_storage)
        assert "log_target" in df_silver.columns
        assert "amperage" in df_silver.columns
        assert len(df_silver) > 0

        # Gold — lê do Silver, faz split real
        result = gold_build(local_storage)
        assert "X_train" in result
        n_train = len(result["X_train"])

        # Training — lê do Gold, treina XGBoost real
        model_info = train(local_storage)
        assert model_info["algorithm"] == "XGBoost"
        assert "rmse" in model_info["metrics"]

        # Verifica que todos os 3 documentos de governança existem em disco
        base = Path(tmp_path)
        assert (base / "silver" / "dutch-energy" / "governance_silver.md").exists()
        assert (base / "gold"   / "dutch-energy" / "governance_gold.md").exists()
        assert (base / "gold"   / "dutch-energy" / "mlflow_report.md").exists()

        # governance_gold.md atualizado (placeholder substituído)
        gov_gold = (base / "gold" / "dutch-energy" / "governance_gold.md").read_text()
        assert "Aguardando treinamento MLflow" not in gov_gold
        assert "XGBoost" in gov_gold

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_dados_consistentes_entre_camadas(
        self, mock_mlflow, mock_pickle, data_dir, local_storage
    ):
        """Features que Silver cria devem estar presentes no X_train do Gold."""
        mock_pickle.return_value = b"fake"
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="r1"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ingest_csvs(data_dir, local_storage)
        silver_transform(local_storage)
        result = gold_build(local_storage)

        X_train = result["X_train"]
        # Features derivadas pelo Silver devem chegar ao Gold
        assert "amperage" in X_train.columns
        assert "total_capacity" in X_train.columns
        assert "hightarif_perc" in X_train.columns
        # Target encoding deve ter gerado as colunas _te
        assert "city_te" in X_train.columns

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_todos_os_docs_de_governanca_tem_conteudo_real(
        self, mock_mlflow, mock_pickle, data_dir, local_storage, tmp_path
    ):
        """Cada documento de governança deve ter dados reais (números, timestamps)."""
        mock_pickle.return_value = b"fake"
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="check-run"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ingest_csvs(data_dir, local_storage)
        silver_transform(local_storage)
        gold_build(local_storage)
        train(local_storage)

        base = Path(tmp_path)

        # governance_silver.md — deve ter número real de registros
        silver_md = (base / "silver" / "dutch-energy" / "governance_silver.md").read_text()
        assert re.search(r"\d{2,}", silver_md), "Deve ter números reais"
        assert "Processado em: 20" in silver_md  # timestamp real

        # governance_gold.md — deve ter tamanhos reais do split
        gold_md = (base / "gold" / "dutch-energy" / "governance_gold.md").read_text()
        assert re.search(r"Treino.*\d{2,}", gold_md)
        assert "XGBoost" in gold_md

        # mlflow_report.md — deve ter métricas reais
        mlflow_md = (base / "gold" / "dutch-energy" / "mlflow_report.md").read_text()
        assert re.search(r"RMSE.*\d+\.\d+", mlflow_md)
        assert "check-run" in mlflow_md

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_idempotencia_de_ponta_a_ponta(
        self, mock_mlflow, mock_pickle, data_dir, local_storage
    ):
        """Rodar o pipeline duas vezes não deve dobrar os dados ou corromper artefatos."""
        mock_pickle.return_value = b"fake"
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="r1"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ingest_csvs(data_dir, local_storage)
        df1 = silver_transform(local_storage)
        result1 = gold_build(local_storage)

        # Segunda rodada
        ingest_csvs(data_dir, local_storage)
        df2 = silver_transform(local_storage)
        result2 = gold_build(local_storage)

        assert len(df1) == len(df2)
        assert result2 == {}  # Gold pulado por idempotência

    @patch("app.pipeline.dutch_energy_train.pickle.dumps")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_governance_indexer_le_docs_reais(
        self, mock_mlflow, mock_pickle, data_dir, local_storage, tmp_path
    ):
        """GovernanceIndexer deve conseguir ler os docs reais gerados pelo pipeline."""
        from app.core.workers.governance_indexer import GovernanceIndexer

        mock_pickle.return_value = b"fake"
        mock_mlflow.start_run.return_value.__enter__ = lambda s: MagicMock(info=MagicMock(run_id="idx-run"))
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ingest_csvs(data_dir, local_storage)
        silver_transform(local_storage)
        gold_build(local_storage)
        train(local_storage)

        # Usa um indexer com repo e embedder mockados mas storage REAL
        mock_repo = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embbed_it.side_effect = lambda chunks: np.random.rand(len(chunks), 384)

        indexer = GovernanceIndexer(
            storage=local_storage,
            repo=mock_repo,
            embedder=mock_embedder,
            schema_builder=MagicMock(),
            collection="governance",
        )
        chunks, sources = indexer._collect_chunks()

        # Todos os 3 documentos devem ser encontrados e chunkeados
        assert "silver_governance" in sources
        assert "gold_governance" in sources
        assert "mlflow_report" in sources
        assert len(chunks) >= 3  # pelo menos 1 chunk por doc

        # O conteúdo dos chunks deve vir dos docs reais
        all_text = " ".join(chunks)
        assert "XGBoost" in all_text          # do mlflow_report.md
        assert "amperage" in all_text          # do governance_silver.md
        assert "Divisão dos Dados" in all_text # do governance_gold.md
