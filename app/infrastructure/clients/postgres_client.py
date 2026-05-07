import logging
import mlflow
import pandas as pd
from app.infrastructure.configs import settings

logger = logging.getLogger(__name__)


class MLflowSearchClient:
    """Busca runs de treinamento no MLflow tracking server (backend PostgreSQL)."""

    def __init__(self):
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    def search_training_runs(self, max_results: int = 5) -> list[dict]:
        """Retorna os runs mais recentes do experimento de treinamento."""
        try:
            df: pd.DataFrame = mlflow.search_runs(
                experiment_names=[settings.MLFLOW_EXPERIMENT_TRAINING],
                order_by=["start_time DESC"],
                max_results=max_results,
            )
            if df.empty:
                return []

            runs = []
            for _, row in df.iterrows():
                run = {
                    "run_id": row.get("run_id", ""),
                    "status": row.get("status", ""),
                    "start_time": str(row.get("start_time", "")),
                }
                # Métricas
                for col in df.columns:
                    if col.startswith("metrics."):
                        key = col.replace("metrics.", "")
                        val = row.get(col)
                        if pd.notna(val):
                            run[f"metric_{key}"] = round(float(val), 4)
                # Parâmetros
                for col in df.columns:
                    if col.startswith("params."):
                        key = col.replace("params.", "")
                        val = row.get(col)
                        if pd.notna(val):
                            run[f"param_{key}"] = str(val)
                runs.append(run)
            return runs
        except Exception as e:
            logger.warning(f"[MLFLOW_CLIENT] Erro ao buscar runs: {e}")
            return []

    def get_best_run_summary(self) -> str:
        """Retorna texto descritivo do melhor run (menor RMSE) para uso no RAG."""
        runs = self.search_training_runs(max_results=10)
        if not runs:
            return "Nenhum run de treinamento encontrado no MLflow."

        # Ordena pelo menor RMSE
        runs_with_rmse = [r for r in runs if "metric_rmse" in r]
        if runs_with_rmse:
            best = min(runs_with_rmse, key=lambda r: r["metric_rmse"])
        else:
            best = runs[0]

        lines = [
            f"Melhor run de treinamento (ID: {best.get('run_id', '?')}):",
            f"  Status: {best.get('status', '?')}",
            f"  Data: {best.get('start_time', '?')}",
        ]
        metrics = {k: v for k, v in best.items() if k.startswith("metric_")}
        if metrics:
            lines.append("  Métricas:")
            for k, v in metrics.items():
                lines.append(f"    {k.replace('metric_', '')}: {v}")
        params = {k: v for k, v in best.items() if k.startswith("param_")}
        if params:
            lines.append("  Parâmetros:")
            for k, v in params.items():
                lines.append(f"    {k.replace('param_', '')}: {v}")

        return "\n".join(lines)
