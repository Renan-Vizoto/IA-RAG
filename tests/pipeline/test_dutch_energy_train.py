"""
Testes do training step (dutch_energy_train.py).
Usa mocks para MLflow, XGBoost e pickle — sem necessidade de serviços externos.
"""
import json
import pickle
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import pandas as pd

from app.pipeline.storage import StorageBackend


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

N_TRAIN, N_VAL, N_FEATURES = 50, 10, 8
FEAT_COLS = [f"feat_{i}" for i in range(N_FEATURES)]
FAKE_MODEL_BYTES = b"fake-serialized-model"


@pytest.fixture
def mock_storage_with_gold():
    """Storage com artefatos Gold pré-carregados."""
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

    def stat_object(bucket, name):
        return storage_meta.get(f"{bucket}/{name}", {})

    storage.put_object.side_effect = put_object
    storage.get_object.side_effect = get_object
    storage.stat_object.side_effect = stat_object

    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.random((N_TRAIN, N_FEATURES)), columns=FEAT_COLS)
    y_train = pd.Series(rng.random(N_TRAIN) * 2 + 4, name="log_target")
    X_val   = pd.DataFrame(rng.random((N_VAL, N_FEATURES)), columns=FEAT_COLS)
    y_val   = pd.Series(rng.random(N_VAL) * 2 + 4, name="log_target")

    def to_buf(df):
        buf = BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return buf

    storage_data["gold/dutch-energy/X_train.csv"]      = to_buf(X_train).getvalue()
    storage_data["gold/dutch-energy/y_train.csv"]      = to_buf(y_train.to_frame()).getvalue()
    storage_data["gold/dutch-energy/X_val.csv"]        = to_buf(X_val).getvalue()
    storage_data["gold/dutch-energy/y_val.csv"]        = to_buf(y_val.to_frame()).getvalue()
    storage_data["gold/dutch-energy/feature_cols.json"] = json.dumps(FEAT_COLS).encode()

    gov_doc = "# Doc Gold\n## 5. Modelo (preenchido após treinamento)\n[Aguardando treinamento MLflow]"
    storage_data["gold/dutch-energy/governance_gold.md"] = gov_doc.encode()

    return storage


@pytest.fixture
def mock_run():
    run = MagicMock()
    run.info.run_id = "test-run-id-123"
    return run


def _patches(mock_run):
    """Decoradores comuns para todos os testes: mlflow + XGBoost + pickle."""
    return [
        patch("app.pipeline.dutch_energy_train.mlflow"),
        patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor"),
        patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES),
    ]


def _setup_mocks(mock_pickle_dumps, MockXGB, mock_mlflow, mock_run):
    """Configura comportamento padrão dos mocks."""
    mock_mlflow.start_run.return_value.__enter__ = lambda s: mock_run
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    model_instance = MagicMock()
    model_instance.predict.return_value = np.ones(N_VAL) * 5.0
    MockXGB.return_value = model_instance
    return model_instance


# ──────────────────────────────────────────────
# Testes
# ──────────────────────────────────────────────

class TestDutchEnergyTrain:

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_carrega_dados_do_gold(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """train() deve carregar X_train, y_train, X_val, y_val do MinIO."""
        model_instance = _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        train(mock_storage_with_gold)

        model_instance.fit.assert_called_once()
        X_fit = model_instance.fit.call_args[0][0]
        assert X_fit.shape == (N_TRAIN, N_FEATURES)

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_loga_params_no_mlflow(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """train() deve logar parâmetros do XGBoost no MLflow."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        train(mock_storage_with_gold)

        mock_mlflow.log_params.assert_called()
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert "n_estimators" in logged_params
        assert "learning_rate" in logged_params
        assert "max_depth" in logged_params

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_loga_metricas_no_mlflow(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """train() deve logar RMSE, MAE, R², MAPE no MLflow."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        train(mock_storage_with_gold)

        mock_mlflow.log_metrics.assert_called()
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        for metric in ["rmse", "mae", "r2", "mape"]:
            assert metric in logged_metrics

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_salva_modelo_no_minio(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """train() deve salvar model.pkl no MinIO com conteúdo não-vazio."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        train(mock_storage_with_gold)

        raw = mock_storage_with_gold.get_object("gold", "dutch-energy/model.pkl")
        model_bytes = raw.read()
        assert model_bytes == FAKE_MODEL_BYTES

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_salva_relatorio_no_minio(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """train() deve salvar mlflow_report.md no MinIO."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        train(mock_storage_with_gold)

        raw = mock_storage_with_gold.get_object("gold", "dutch-energy/mlflow_report.md")
        content = raw.read().decode("utf-8")
        assert "Relatório MLflow" in content
        assert "dutch-energy-training" in content

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_relatorio_contem_secoes_obrigatorias(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """mlflow_report.md deve conter seções de algoritmo, hiperparâmetros e métricas."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        train(mock_storage_with_gold)

        raw = mock_storage_with_gold.get_object("gold", "dutch-energy/mlflow_report.md")
        content = raw.read().decode("utf-8")
        assert "Algoritmo Utilizado" in content
        assert "Hiperparâmetros" in content
        assert "Métricas de Avaliação" in content
        assert "Dados de Treinamento" in content
        assert "XGBoost" in content

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_atualiza_governance_gold(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """train() deve atualizar governance_gold.md removendo o placeholder."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        train(mock_storage_with_gold)

        raw = mock_storage_with_gold.get_object("gold", "dutch-energy/governance_gold.md")
        content = raw.read().decode("utf-8")
        assert "Aguardando treinamento MLflow" not in content
        assert "XGBoost" in content

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_retorna_dict_com_run_id_e_metricas(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """train() deve retornar dict com run_id, algorithm e metrics."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        result = train(mock_storage_with_gold)

        assert "run_id" in result
        assert "algorithm" in result
        assert "metrics" in result
        assert "params" in result
        assert result["algorithm"] == "XGBoost"
        for metric in ["rmse", "mae", "r2", "mape"]:
            assert metric in result["metrics"]

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_configura_mlflow_tracking_uri(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """train() deve configurar tracking URI e experiment antes do run."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        train(mock_storage_with_gold)

        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_metricas_sao_numericas(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """Todas as métricas retornadas devem ser float arredondados."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        result = train(mock_storage_with_gold)

        for k, v in result["metrics"].items():
            assert isinstance(v, float), f"Métrica {k} deve ser float, got {type(v)}"

    @patch("app.pipeline.dutch_energy_train.pickle.dumps", return_value=FAKE_MODEL_BYTES)
    @patch("app.pipeline.dutch_energy_train.xgb.XGBRegressor")
    @patch("app.pipeline.dutch_energy_train.mlflow")
    def test_mlflow_log_artifact_chamado(self, mock_mlflow, MockXGB, mock_pickle, mock_storage_with_gold, mock_run):
        """train() deve logar o relatório markdown como artefato no MLflow."""
        _setup_mocks(mock_pickle, MockXGB, mock_mlflow, mock_run)

        from app.pipeline.dutch_energy_train import train
        train(mock_storage_with_gold)

        mock_mlflow.log_artifact.assert_called_once()
