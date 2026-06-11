from app.core.services.mlflow_metadata_formatter import (
    content_hash,
    format_run_chunks,
    pick_best_rmse_run_id,
)


SAMPLE_RUN = {
    "run_id": "abc123",
    "status": "FINISHED",
    "start_time": "2025-01-15",
    "metric_rmse": 89.1,
    "metric_mae": 39.5,
    "metric_r2": 0.74,
    "metric_test_rmse": 91.0,
    "param_algorithm": "XGBoost",
    "param_n_estimators": "100",
    "param_max_depth": "6",
    "param_n_features": "12",
}


class TestMLflowMetadataFormatter:

    def test_cada_metrica_gera_chunk_proprio(self):
        chunks = format_run_chunks(SAMPLE_RUN, is_best=True)
        sections = {c["section"] for c in chunks}
        assert "overview" in sections
        assert "params" in sections
        assert "metrics_summary" in sections
        assert "metric_rmse" in sections
        assert "metric_mae" in sections
        assert "metric_r2" in sections
        assert "metric_test_rmse" in sections
        assert "metrics" not in sections
        assert len(chunks) == 7

    def test_ids_deterministicos(self):
        chunks = format_run_chunks(SAMPLE_RUN)
        ids = [c["id"] for c in chunks]
        assert ids == [
            "mlflow:abc123:overview",
            "mlflow:abc123:metrics_summary",
            "mlflow:abc123:metric_rmse",
            "mlflow:abc123:metric_mae",
            "mlflow:abc123:metric_r2",
            "mlflow:abc123:metric_test_rmse",
            "mlflow:abc123:params",
        ]

    def test_metrics_summary_agrega_todas_metricas(self):
        summary = next(
            c for c in format_run_chunks(SAMPLE_RUN, is_best=True)
            if c["section"] == "metrics_summary"
        )
        assert "Modelos treinados e métricas" in summary["text"]
        assert "89.1" in summary["text"]
        assert "39.5" in summary["text"]
        assert "0.74" in summary["text"]

    def test_overview_marca_melhor_run(self):
        overview = format_run_chunks(SAMPLE_RUN, is_best=True)[0]["text"]
        assert "Melhor run de treinamento (menor RMSE)" in overview
        assert "treinamento do modelo" in overview.lower() or "machine learning" in overview.lower()

    def test_chunk_rmse_tem_semantica(self):
        rmse = next(c for c in format_run_chunks(SAMPLE_RUN) if c["section"] == "metric_rmse")
        assert "erro quadrático médio" in rmse["text"]
        assert "89.1" in rmse["text"]
        assert "desempenho do modelo" in rmse["text"].lower()

    def test_params_tem_hiperparametros(self):
        params = next(c for c in format_run_chunks(SAMPLE_RUN) if c["section"] == "params")
        assert "hiperparâmetros" in params["text"].lower()
        assert "n_estimators" in params["text"]

    def test_content_hash_muda_com_metrica(self):
        h1 = content_hash(SAMPLE_RUN)
        changed = {**SAMPLE_RUN, "metric_rmse": 50.0}
        h2 = content_hash(changed)
        assert h1 != h2

    def test_pick_best_rmse(self):
        runs = [
            {"run_id": "r1", "metric_val_rmse": 100},
            {"run_id": "r2", "metric_val_rmse": 80},
        ]
        assert pick_best_rmse_run_id(runs) == "r2"

    def test_val_rmse_chunk(self):
        run = {
            "run_id": "abc123",
            "status": "FINISHED",
            "start_time": "2025-01-15",
            "metric_val_rmse": 0.3432,
            "metric_val_mae": 0.2568,
            "param_algorithm": "XGBoost",
        }
        rmse = next(c for c in format_run_chunks(run) if c["section"] == "metric_val_rmse")
        assert "0.3432" in rmse["text"]
        assert "erro quadrático médio" in rmse["text"]
