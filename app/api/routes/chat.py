from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated
import uuid

from app.core.services.chat_service import ChatService
from app.core.services.search_service import SearchService
from app.core.interfaces.embbeding import EmbeddingStrategy
from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.infrastructure.repositories.chat_session_repo import ChatSessionRepository
from app.infrastructure.clients import milvus_client
from langchain_core.tools import tool
from app.api.schemas.chat_response import ChatResponse, ChatTraceResponse, TokenUsage, MilvusHitTrace
from app.api.docs import build_error_responses

class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "Qual modelo foi treinado e qual foi o RMSE?",
                    "model": "gemma4-unsloth",
                },
                {
                    "message": "Explique o pipeline bronze, silver e gold.",
                    "model": None,
                },
            ]
        }
    )

    message: str = Field(
        description="User message sent to the LangChain RAG agent.",
        examples=["Qual modelo foi treinado e qual foi o RMSE?"],
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional Ollama model override. Must be listed in "
            "OLLAMA_ALLOWED_MODELS; when omitted, OLLAMA_MODEL is used."
        ),
        examples=["gemma4-unsloth"],
    )

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

repo = MilvusRepo(milvus_client.milvusClient)
searchService: SearchService | None = None
search_tool = None

session_repo: ChatSessionRepository | None = None
chatService: ChatService | None = None


def _build_search_tool(service: SearchService):
    @tool
    def search(query: str) -> list:
        """Busca semântica em governança do pipeline e metadados MLflow.

        Chame SEMPRE antes de responder perguntas sobre modelo treinado,
        métricas (RMSE, MAE, R²), hiperparâmetros ou etapas do pipeline.
        """
        raw = service.search(query)
        return ChatService.redact_raw_search_hits(raw)

    return search


def init_chat_dependencies(
    chat_session_repo: ChatSessionRepository,
    embedder: EmbeddingStrategy,
):
    global session_repo, chatService, searchService, search_tool
    session_repo = chat_session_repo
    searchService = SearchService(repo, embedder)
    search_tool = _build_search_tool(searchService)
    chatService = ChatService(
        tools=[search_tool],
        session_repo=session_repo,
    )


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a RAG chat message",
    description=(
        "Sends a user message to the LangChain agent. The agent may call the "
        "internal semantic search tool before generating the final answer with "
        "Ollama. If no `session_id` cookie is provided, the API creates a UUID "
        "session and returns it as an HttpOnly cookie with a seven-day lifetime."
    ),
    responses=build_error_responses(400, 422, 500, 503),
)
def send_message(
    request: ChatRequest,
    response: Response,
    session_id: Annotated[str | None, Cookie()] = None
):
    if chatService is None:
        raise HTTPException(status_code=503, detail="Chat não inicializado")

    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 7
        )
    try:
        return chatService.send_message(
            request.message,
            session_id,
            model=request.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/trace/{response_id}",
    response_model=ChatTraceResponse,
    summary="Get chat response trace",
    description=(
        "Returns persisted observability data for a previous chat response, "
        "including the original user message, sanitized answer, token usage, "
        "response latency, confidence score, and retrieved Milvus hit previews."
    ),
    responses=build_error_responses(404, 500, 503),
)
def get_trace(response_id: str):
    if not session_repo:
        raise HTTPException(status_code=503, detail="Repositório de sessão não inicializado")

    data = session_repo.get_response(response_id)
    if not data:
        raise HTTPException(status_code=404, detail="Resposta não encontrada")

    return ChatTraceResponse(
        response_id=data["response_id"],
        session_id=data["session_id"],
        model=data["model"],
        user_message=data["user_message"],
        answer=data["answer"],
        tokens=TokenUsage(
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            total_tokens=data.get("total_tokens"),
        ),
        response_time_seconds=data.get("response_time_seconds"),
        confidence_score=data.get("confidence_score"),
        milvus_hits=[
            MilvusHitTrace(
                milvus_id=h.get("milvus_id"),
                collection=h.get("collection", "governance"),
                source=h.get("source"),
                distance=h.get("distance"),
                text_preview=h.get("text_preview"),
            )
            for h in data.get("milvus_hits", [])
        ],
        created_at=str(data.get("created_at", "")),
    )
