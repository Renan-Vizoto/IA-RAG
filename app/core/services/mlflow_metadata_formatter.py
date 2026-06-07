import hashlib
import json
from app.infrastructure.configs import settings

_HEADER = (
    "Metadados de treinamento MLflow — Pipeline Dutch Energy — consumo elétrico holandês."
)


def content_hash(run: dict) -> str:
    payload = {k: v for k, v in sorted(run.items()) if k != "start_time"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def pick_best_rmse_run_id(runs: list[dict]) -> str | None:
    runs_with_rmse = [r for r in runs if "metric_rmse" in r]
    if not runs_with_rmse:
        return None
    best = min(runs_with_rmse, key=lambda r: r["metric_rmse"])
    return best.get("run_id")


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

    metric_lines = []
    val_metrics = {k: v for k, v in metrics.items() if not k.startswith("test_") and k != "train_duration_sec"}
    test_metrics = {k: v for k, v in metrics.items() if k.startswith("test_")}

    labels = {
        "rmse": "RMSE (erro quadrático médio) na validação",
        "mae": "MAE (erro absoluto médio) na validação",
        "r2": "R² (coeficiente de determinação) na validação",
        "mape": "MAPE (erro percentual absoluto médio) na validação",
        "test_rmse": "RMSE (erro quadrático médio) no teste",
        "test_mae": "MAE (erro absoluto médio) no teste",
        "test_r2": "R² (coeficiente de determinação) no teste",
        "test_mape": "MAPE no conjunto de teste",
    }

    for key, val in val_metrics.items():
        label = labels.get(key, key)
        metric_lines.append(f"- {label}: {val}")

    for key, val in test_metrics.items():
        label = labels.get(key, key)
        metric_lines.append(f"- {label}: {val}")

    metrics_text = "\n".join(metric_lines) if metric_lines else "- Métricas não disponíveis."

    metrics_chunk = (
        f"{_HEADER}\n"
        f"Desempenho do modelo treinado — Run ID: {run_id}.\n"
        f"Qual foi o desempenho do modelo? Quão preciso foi o treinamento?\n"
        f"Métricas de avaliação do modelo:\n{metrics_text}"
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

    sections = [
        ("overview", overview),
        ("metrics", metrics_chunk),
        ("params", params_chunk),
    ]

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
