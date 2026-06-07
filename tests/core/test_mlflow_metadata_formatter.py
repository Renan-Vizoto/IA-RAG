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

    def test_chunks_tem_tres_secoes(self):
        chunks = format_run_chunks(SAMPLE_RUN, is_best=True)
        assert len(chunks) == 3
        sections = {c["section"] for c in chunks}
        assert sections == {"overview", "metrics", "params"}

    def test_ids_deterministicos(self):
        chunks = format_run_chunks(SAMPLE_RUN)
        ids = [c["id"] for c in chunks]
        assert ids == [
            "mlflow:abc123:overview",
            "mlflow:abc123:metrics",
            "mlflow:abc123:params",
        ]

    def test_overview_marca_melhor_run(self):
        overview = format_run_chunks(SAMPLE_RUN, is_best=True)[0]["text"]
        assert "Melhor run de treinamento (menor RMSE)" in overview
        assert "treinamento do modelo" in overview.lower() or "machine learning" in overview.lower()

    def test_metrics_tem_semantica_rmse(self):
        metrics = next(c for c in format_run_chunks(SAMPLE_RUN) if c["section"] == "metrics")
        assert "erro quadrático médio" in metrics["text"]
        assert "desempenho do modelo" in metrics["text"].lower()

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
            {"run_id": "r1", "metric_rmse": 100},
            {"run_id": "r2", "metric_rmse": 80},
        ]
        assert pick_best_rmse_run_id(runs) == "r2"
