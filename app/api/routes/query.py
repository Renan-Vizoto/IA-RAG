from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from app.infrastructure.clients.postgres_client import MLflowSearchClient
from app.api.routes import chat as chat_routes

router = APIRouter(
    prefix="",
    tags=["query", "metadata"],
    responses={404: {"description": "Not found"}},
)

_mlflow_client = MLflowSearchClient()


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QueryResultItem(BaseModel):
    id: str | None = None
    distance: float | None = None
    text: str
    source: str | None = None

class QueryResponse(BaseModel):
    query: str
    results: List[QueryResultItem]
    total: int

class RunInfo(BaseModel):
    run_id: str
    status: str
    start_time: str
    metrics: dict
    params: dict

class MetadataResponse(BaseModel):
    experiment: str
    total_runs: int
    best_run: RunInfo | None
    all_runs: List[RunInfo]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="Busca semântica vetorial")
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


@router.get("/metadata", response_model=MetadataResponse, summary="Metadados do modelo treinado")
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
