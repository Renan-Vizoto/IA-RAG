"""
Training Step - Dutch Energy Dataset
Carrega os artefatos do Gold, treina o modelo XGBoost com MLflow tracking
e salva relatórios no MinIO.

Fluxo:
  1. Carrega X_train, y_train, X_val, y_val do MinIO (gold)
  2. Treina XGBoost Regressor
  3. Loga params e métricas no MLflow tracking server
  4. Salva modelo (.pkl) no MinIO gold
  5. Gera mlflow_report.md e salva no MinIO gold
  6. Atualiza governance_gold.md com informações do modelo
"""
import json
import pickle
import logging
import tempfile
import os
from io import BytesIO
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.xgboost
import xgboost as xgb

from app.pipeline.storage import StorageBackend
from app.infrastructure.configs import settings
from app.pipeline.dutch_energy_gold import update_governance_with_model

logger = logging.getLogger(__name__)

GOLD_BUCKET = "gold"
GOLD_PREFIX = "dutch-energy/"

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 42,
    "n_jobs": -1,
}


def train(storage: StorageBackend) -> dict:
    """
    Treina XGBoost com MLflow tracking e salva artefatos no MinIO.
    Retorna dicionário com run_id e métricas.
    """
    _configure_mlflow()

    X_train, y_train, X_val, y_val, feat_cols = _load_gold(storage)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"[TRAIN] MLflow run iniciado: {run_id}")

        mlflow.log_params(XGBOOST_PARAMS)
        mlflow.log_param("algorithm", "XGBoost")
        mlflow.log_param("n_features", len(feat_cols))
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))

        model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        metrics = _evaluate(model, X_val, y_val)
        mlflow.log_metrics(metrics)
        logger.info(f"[TRAIN] Métricas validação: {metrics}")

        mlflow.xgboost.log_model(model, "xgboost_model")

        # Relatório markdown
        report = _build_report(run_id, metrics, len(X_train), len(X_val), feat_cols)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(report)
            report_path = f.name

        mlflow.log_artifact(report_path, artifact_path="reports")
        os.unlink(report_path)

    # Salva no MinIO
    _save_model(model, storage)
    _save_report(report, storage)

    model_info = {
        "run_id": run_id,
        "algorithm": "XGBoost",
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "params": {k: str(v) for k, v in XGBOOST_PARAMS.items()},
    }

    # Atualiza governance_gold.md
    update_governance_with_model(storage, model_info)

    logger.info(f"[TRAIN] Treinamento concluído. Run ID: {run_id}")
    return model_info


# ──────────────────────────────────────────────
# Internos
# ──────────────────────────────────────────────

def _configure_mlflow():
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL
    os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_TRAINING)


def _load_gold(storage: StorageBackend):
    def read_csv(name: str) -> pd.DataFrame:
        raw = storage.get_object(GOLD_BUCKET, f"{GOLD_PREFIX}{name}.csv")
        return pd.read_csv(raw)

    X_train = read_csv("X_train").values
    y_train = read_csv("y_train").values.ravel()
    X_val   = read_csv("X_val").values
    y_val   = read_csv("y_val").values.ravel()

    feat_raw = storage.get_object(GOLD_BUCKET, f"{GOLD_PREFIX}feature_cols.json")
    feat_cols: list[str] = json.loads(feat_raw.read().decode("utf-8"))

    logger.info(
        f"[TRAIN] Dados carregados: train={X_train.shape}, val={X_val.shape}, "
        f"features={len(feat_cols)}"
    )
    return X_train, y_train, X_val, y_val, feat_cols


def _evaluate(model, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    preds = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
    mae  = float(mean_absolute_error(y_val, preds))
    r2   = float(r2_score(y_val, preds))
    # MAPE em espaço original (desfaz log1p)
    y_orig    = np.expm1(y_val)
    p_orig    = np.expm1(np.clip(preds, -10, 20))
    mask      = y_orig > 0
    mape      = float(np.mean(np.abs((y_orig[mask] - p_orig[mask]) / y_orig[mask])) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def _build_report(run_id: str, metrics: dict, n_train: int, n_val: int, feat_cols: list[str]) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    param_rows = "\n".join(f"| {k} | {v} |" for k, v in XGBOOST_PARAMS.items())
    metric_rows = "\n".join(f"| {k.upper()} | {v:.4f} |" for k, v in metrics.items())

    return f"""# Relatório MLflow — Treinamento Dutch Energy
## Experimento: dutch-energy-training
## Run ID: {run_id}
## Data: {ts}

---

## 1. Algoritmo Utilizado

- **Modelo**: XGBoost Regressor
- **Objetivo**: minimizar RMSE em espaço log (target = log1p(consume_per_conn))
- **Selecionado com base em**: melhor R² e RMSE nos notebooks de análise exploratória

---

## 2. Hiperparâmetros

| Parâmetro | Valor |
|-----------|-------|
{param_rows}

---

## 3. Dados de Treinamento

| Conjunto | Registros |
|----------|-----------|
| Treino   | {n_train:,} |
| Validação| {n_val:,} |
| Features | {len(feat_cols)} |

---

## 4. Métricas de Avaliação (conjunto de validação)

| Métrica | Valor |
|---------|-------|
{metric_rows}

> Métricas calculadas no espaço original (desfazendo log1p) para MAPE;
> RMSE e MAE calculados no espaço log.

---

## 5. Features Utilizadas

{chr(10).join(f'- {f}' for f in feat_cols)}
"""


def _save_model(model, storage: StorageBackend):
    storage.ensure_bucket(GOLD_BUCKET)
    model_bytes = BytesIO(pickle.dumps(model))
    storage.put_object(
        GOLD_BUCKET, f"{GOLD_PREFIX}model.pkl", model_bytes, "application/octet-stream"
    )
    logger.info(f"[TRAIN] Modelo salvo no MinIO: {GOLD_PREFIX}model.pkl")


def _save_report(report: str, storage: StorageBackend):
    report_buf = BytesIO(report.encode("utf-8"))
    storage.put_object(
        GOLD_BUCKET, f"{GOLD_PREFIX}mlflow_report.md", report_buf, "text/markdown"
    )
    logger.info(f"[TRAIN] Relatório salvo no MinIO: {GOLD_PREFIX}mlflow_report.md")
