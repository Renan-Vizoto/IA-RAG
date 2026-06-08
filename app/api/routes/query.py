from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import List

from app.infrastructure.clients.postgres_client import MLflowSearchClient
from app.api.routes import chat as chat_routes
from app.api.docs import build_error_responses

router = APIRouter(
    prefix="",
    tags=["query", "metadata"],
    responses={404: {"description": "Not found"}},
)

_mlflow_client = MLflowSearchClient()


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Qual foi o RMSE do modelo treinado?",
                }
            ]
        }
    )

    query: str = Field(
        description="Natural language text to embed and search in Milvus.",
        examples=["Qual foi o RMSE do modelo treinado?"],
    )

class QueryResultItem(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "mlflow:run:metrics",
                    "distance": 0.12,
                    "text": "RMSE de validacao: 89.1. MAE: 41.2.",
                    "source": "mlflow_metadata",
                }
            ]
        }
    )

    id: str | None = Field(default=None, description="Milvus hit identifier.")
    distance: float | None = Field(
        default=None,
        description="Cosine distance returned by Milvus. Lower values are more similar.",
    )
    text: str = Field(description="Retrieved text snippet.")
    source: str | None = Field(default=None, description="Indexed source label.")

class QueryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "Qual foi o RMSE do modelo treinado?",
                    "results": [
                        {
                            "id": "mlflow:run:metrics",
                            "distance": 0.12,
                            "text": "RMSE de validacao: 89.1. MAE: 41.2.",
                            "source": "mlflow_metadata",
                        }
                    ],
                    "total": 1,
                }
            ]
        }
    )

    query: str = Field(description="Echo of the submitted query.")
    results: List[QueryResultItem] = Field(
        description="Flattened Milvus hits sorted by ascending distance."
    )
    total: int = Field(description="Number of returned items.")

class RunInfo(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "run_id": "a1888f014bb04b61ba4c245bb58552c8",
                    "status": "FINISHED",
                    "start_time": "2026-06-08 12:34:56+00:00",
                    "metrics": {"rmse": 89.1, "mae": 41.2, "r2": 0.87},
                    "params": {"model": "xgboost", "max_depth": "6"},
                }
            ]
        }
    )

    run_id: str = Field(description="MLflow run ID.")
    status: str = Field(description="MLflow run status.")
    start_time: str = Field(description="Run start time converted to string.")
    metrics: dict = Field(
        description="Metric names to numeric values, for example rmse, mae, and r2."
    )
    params: dict = Field(description="MLflow parameter names to string values.")

class MetadataResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "experiment": "dutch-energy-training",
                    "total_runs": 1,
                    "best_run": {
                        "run_id": "a1888f014bb04b61ba4c245bb58552c8",
                        "status": "FINISHED",
                        "start_time": "2026-06-08 12:34:56+00:00",
                        "metrics": {"rmse": 89.1, "mae": 41.2, "r2": 0.87},
                        "params": {"model": "xgboost", "max_depth": "6"},
                    },
                    "all_runs": [
                        {
                            "run_id": "a1888f014bb04b61ba4c245bb58552c8",
                            "status": "FINISHED",
                            "start_time": "2026-06-08 12:34:56+00:00",
                            "metrics": {"rmse": 89.1, "mae": 41.2, "r2": 0.87},
                            "params": {"model": "xgboost", "max_depth": "6"},
                        }
                    ],
                }
            ]
        }
    )

    experiment: str = Field(description="Configured MLflow training experiment name.")
    total_runs: int = Field(description="Number of runs returned by MLflow.")
    best_run: RunInfo | None = Field(
        description="Best run by lowest rmse, fallback newest run, or null."
    )
    all_runs: List[RunInfo] = Field(description="Runs returned by MLflow.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic vector search",
    description=(
        "Embeds the submitted query, searches the governance and MLflow metadata "
        "Milvus collections, merges the hits, sorts by ascending cosine distance, "
        "and returns up to RAG_MAX_CONTEXT_CHUNKS results."
    ),
    responses=build_error_responses(422, 500, 503),
)
def query(request: QueryRequest):
    """
    Realiza busca semântica na base vetorial (Milvus) usando os documentos
    de governança indexados. Retorna os trechos mais relevantes para a consulta.
    """
    if chat_routes.searchService is None:
        raise HTTPException(status_code=503, detail="SearchService não inicializado")
    raw = chat_routes.searchService.search(request.query)

    items: List[QueryResultItem] = []
    for hit_list in raw:
        for hit in hit_list:
            items.append(QueryResultItem(
                id=str(hit.get("id", "")),
                distance=hit.get("distance"),
                text=hit.get("entity", {}).get("text", hit.get("text", "")),
                source=hit.get("entity", {}).get("source", hit.get("source", "")),
            ))

    return QueryResponse(
        query=request.query,
        results=items,
        total=len(items),
    )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Training model metadata",
    description=(
        "Returns MLflow training runs from the configured experiment. Metrics and "
        "params are normalized by removing the MLflow `metrics.` and `params.` "
        "prefixes. `best_run` is selected by the lowest `rmse` metric when present."
    ),
    responses=build_error_responses(500),
)
def metadata():
    """
    Retorna metadados dos experimentos de treinamento registrados no MLflow:
    parâmetros, métricas (RMSE, MAE, R², MAPE) e informações dos runs.
    """
    runs = _mlflow_client.search_training_runs(max_results=20)

    run_infos: List[RunInfo] = []
    for r in runs:
        metrics = {k.replace("metric_", ""): v for k, v in r.items() if k.startswith("metric_")}
        params = {k.replace("param_", ""): v for k, v in r.items() if k.startswith("param_")}
        run_infos.append(RunInfo(
            run_id=r.get("run_id", ""),
            status=r.get("status", ""),
            start_time=r.get("start_time", ""),
            metrics=metrics,
            params=params,
        ))

    best: RunInfo | None = None
    runs_with_rmse = [ri for ri in run_infos if "rmse" in ri.metrics]
    if runs_with_rmse:
        best = min(runs_with_rmse, key=lambda ri: ri.metrics["rmse"])
    elif run_infos:
        best = run_infos[0]

    from app.infrastructure.configs import settings
    return MetadataResponse(
        experiment=settings.MLFLOW_EXPERIMENT_TRAINING,
        total_runs=len(run_infos),
        best_run=best,
        all_runs=run_infos,
    )
