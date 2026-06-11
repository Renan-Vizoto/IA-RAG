import hashlib
import json
from app.infrastructure.configs import settings

_HEADER = (
    "Metadados de treinamento MLflow — Pipeline Dutch Energy — consumo elétrico holandês."
)

_METRIC_LABELS = {
    "val_rmse": "RMSE (erro quadrático médio) na validação",
    "val_mae": "MAE (erro absoluto médio) na validação",
    "val_r2": "R² (coeficiente de determinação) na validação",
    "val_mape": "MAPE (erro percentual absoluto médio) na validação",
    "rmse": "RMSE (erro quadrático médio) na validação",
    "mae": "MAE (erro absoluto médio) na validação",
    "r2": "R² (coeficiente de determinação) na validação",
    "mape": "MAPE (erro percentual absoluto médio) na validação",
    "test_rmse": "RMSE (erro quadrático médio) no teste",
    "test_mae": "MAE (erro absoluto médio) no teste",
    "test_r2": "R² (coeficiente de determinação) no teste",
    "test_mape": "MAPE no conjunto de teste",
    "train_duration_sec": "Duração do treinamento em segundos",
}

_METRIC_ORDER = (
    "val_rmse", "val_mae", "val_r2", "val_mape",
    "rmse", "mae", "r2", "mape",
    "test_rmse", "test_mae", "test_r2", "test_mape",
    "train_duration_sec",
)


def content_hash(run: dict) -> str:
    payload = {k: v for k, v in sorted(run.items()) if k != "start_time"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _rmse_value(run: dict) -> float | None:
    for key in ("metric_val_rmse", "metric_test_rmse", "metric_rmse"):
        if key in run:
            return run[key]
    return None


def pick_best_rmse_run_id(runs: list[dict]) -> str | None:
    runs_with_rmse = [r for r in runs if _rmse_value(r) is not None]
    if not runs_with_rmse:
        return None
    best = min(runs_with_rmse, key=lambda r: _rmse_value(r))
    return best.get("run_id")


def _format_metric_chunk(
    run_id: str,
    algorithm: str,
    metric_key: str,
    value,
    is_best: bool,
) -> str:
    label = _METRIC_LABELS.get(metric_key, metric_key)
    best_note = "Melhor run de treinamento. " if is_best else ""
    return (
        f"{_HEADER}\n"
        f"{best_note}"
        f"Desempenho do modelo treinado — Run ID: {run_id}.\n"
        f"Modelo: {algorithm}. Métrica de avaliação do treinamento.\n"
        f"{label}: {value}.\n"
        f"Qual foi o desempenho do modelo? Métrica {metric_key}, {label}."
    )


def format_run_chunks(run: dict, is_best: bool = False) -> list[dict]:
    run_id = run.get("run_id", "unknown")
    h = content_hash(run)
    metrics = {k.replace("metric_", ""): v for k, v in run.items() if k.startswith("metric_")}
    params = {k.replace("param_", ""): v for k, v in run.items() if k.startswith("param_")}
    algorithm = params.get("algorithm", "XGBoost")

    best_prefix = "Melhor run de treinamento (menor RMSE). " if is_best else ""

    overview = (
        f"{_HEADER}\n"
        f"{best_prefix}"
        f"Este é um run de treinamento do modelo de machine learning no experimento "
        f"'{settings.MLFLOW_EXPERIMENT_TRAINING}'.\n"
        f"Run ID MLflow: {run_id}. Status: {run.get('status', '?')}. "
        f"Data de início: {run.get('start_time', '?')}.\n"
        f"Algoritmo utilizado: {algorithm}. "
        f"Treinamento do modelo, experimento MLflow, modelo treinado, run de machine learning."
    )

    param_lines = [f"- {k}: {v}" for k, v in sorted(params.items())]
    params_text = "\n".join(param_lines) if param_lines else "- Parâmetros não disponíveis."
    n_features = params.get("n_features")
    features_note = (
        f" O modelo utilizou {n_features} features (variáveis de entrada)."
        if n_features else ""
    )

    params_chunk = (
        f"{_HEADER}\n"
        f"Configuração e hiperparâmetros do treinamento — Run ID: {run_id}.\n"
        f"O modelo {algorithm} foi configurado com os seguintes parâmetros de treinamento"
        f"{features_note}:\n"
        f"Hiperparâmetros, configuração do modelo, parâmetros de treinamento:\n"
        f"{params_text}"
    )

    sections: list[tuple[str, str]] = [("overview", overview)]

    ordered_keys = [k for k in _METRIC_ORDER if k in metrics]
    ordered_keys.extend(sorted(k for k in metrics if k not in _METRIC_ORDER))

    for metric_key in ordered_keys:
        section = f"metric_{metric_key}"
        text = _format_metric_chunk(run_id, algorithm, metric_key, metrics[metric_key], is_best)
        sections.append((section, text))

    sections.append(("params", params_chunk))

    return [
        {
            "id": f"mlflow:{run_id}:{section}",
            "run_id": run_id,
            "section": section,
            "text": text,
            "source": "mlflow_metadata",
            "content_hash": h,
        }
        for section, text in sections
    ]
