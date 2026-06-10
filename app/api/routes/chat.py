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
from app.api.schemas.chat import (
    ChatSummary,
    ChatMessage,
    CreateChatRequest,
    CreateChatResponse,
    ChatListResponse,
    ChatMessagesResponse,
)
from app.api.docs import build_error_responses

class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "Qual modelo foi treinado e qual foi o RMSE?",
                    "model": "gemma4-unsloth",
                    "chat_id": None,
                },
                {
                    "message": "Explique o pipeline bronze, silver e gold.",
                    "model": None,
                    "chat_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
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
    chat_id: str | None = Field(
        default=None,
        description=(
            "Target chat thread. When omitted, the active chat_id cookie is used; "
            "if neither is present, a new chat is created and returned."
        ),
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


_COOKIE_MAX_AGE = 60 * 60 * 24 * 7


def _ensure_session_id(
    response: Response,
    session_id: str | None,
) -> str:
    if session_id:
        return session_id
    session_id = str(uuid.uuid4())
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=_COOKIE_MAX_AGE,
    )
    return session_id


def _resolve_request_chat_id(
    body_chat_id: str | None,
    cookie_chat_id: str | None,
) -> str | None:
    return body_chat_id or cookie_chat_id


def _set_chat_cookie(response: Response, chat_id: str) -> None:
    response.set_cookie(
        key="chat_id",
        value=chat_id,
        httponly=True,
        samesite="lax",
        max_age=_COOKIE_MAX_AGE,
    )


def _row_to_chat_summary(row: dict) -> ChatSummary:
    return ChatSummary(
        chat_id=row["chat_id"],
        title=row["title"],
        input_tokens=int(row.get("input_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        total_tokens=int(row.get("total_tokens") or 0),
        message_count=int(row.get("message_count") or 0),
        created_at=str(row.get("created_at", "")),
        updated_at=str(row.get("updated_at", "")),
    )


def _row_to_chat_message(row: dict) -> ChatMessage:
    return ChatMessage(
        message_id=row["message_id"],
        role=row["role"],
        content=row["content"],
        created_at=str(row.get("created_at", "")),
    )


def _get_chat_for_session(chat_id: str, session_id: str) -> dict:
    if not session_repo:
        raise HTTPException(status_code=503, detail="Repositório de sessão não inicializado")
    chat = session_repo.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat não encontrado")
    if chat["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Chat não encontrado")
    return chat


@router.post(
    "/chats",
    response_model=CreateChatResponse,
    summary="Create a new chat",
    description="Creates an empty chat thread for the current anonymous session.",
    responses=build_error_responses(500, 503),
)
def create_chat(
    request: CreateChatRequest,
    response: Response,
    session_id: Annotated[str | None, Cookie()] = None,
):
    if not session_repo:
        raise HTTPException(status_code=503, detail="Repositório de sessão não inicializado")

    session_id = _ensure_session_id(response, session_id)
    chat_id = str(uuid.uuid4())
    title = request.title or "New chat"
    row = session_repo.create_chat(session_id, chat_id, title=title)
    return CreateChatResponse(
        chat_id=row["chat_id"],
        title=row["title"],
        created_at=str(row.get("created_at", "")),
    )


@router.get(
    "/chats",
    response_model=ChatListResponse,
    summary="List chats for session",
    description="Returns all chat threads for the current anonymous session.",
    responses=build_error_responses(500, 503),
)
def list_chats(
    response: Response,
    session_id: Annotated[str | None, Cookie()] = None,
):
    if not session_repo:
        raise HTTPException(status_code=503, detail="Repositório de sessão não inicializado")

    session_id = _ensure_session_id(response, session_id)
    rows = session_repo.list_chats(session_id)
    return ChatListResponse(chats=[_row_to_chat_summary(row) for row in rows])


@router.get(
    "/chats/{chat_id}/messages",
    response_model=ChatMessagesResponse,
    summary="Get chat message history",
    description="Returns persisted user/assistant messages for a chat owned by the session.",
    responses=build_error_responses(404, 500, 503),
)
def get_chat_messages(
    chat_id: str,
    session_id: Annotated[str | None, Cookie()] = None,
):
    if not session_id:
        raise HTTPException(status_code=404, detail="Chat não encontrado")

    _get_chat_for_session(chat_id, session_id)
    rows = session_repo.get_messages(chat_id)
    return ChatMessagesResponse(
        chat_id=chat_id,
        messages=[_row_to_chat_message(row) for row in rows],
    )


@router.post(
    "/message",
    response_model=ChatResponse,
    summary="Send a RAG chat message",
    description=(
        "Sends a user message to the LangChain agent. The agent may call the "
        "internal semantic search tool before generating the final answer with "
        "Ollama. If no `session_id` cookie is provided, the API creates a UUID "
        "session and returns it as an HttpOnly cookie with a seven-day lifetime. "
        "When `chat_id` is omitted, the `chat_id` cookie is used. If neither is "
        "present, a new chat is created. The resolved `chat_id` is returned in "
        "the response body and as an HttpOnly cookie."
    ),
    responses=build_error_responses(400, 422, 500, 503),
)
def send_message(
    request: ChatRequest,
    response: Response,
    session_id: Annotated[str | None, Cookie()] = None,
    chat_id: Annotated[str | None, Cookie()] = None,
):
    if chatService is None:
        raise HTTPException(status_code=503, detail="Chat não inicializado")

    session_id = _ensure_session_id(response, session_id)
    resolved_chat_id = _resolve_request_chat_id(request.chat_id, chat_id)
    try:
        result = chatService.send_message(
            request.message,
            session_id,
            model=request.model,
            chat_id=resolved_chat_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _set_chat_cookie(response, result.chat_id)
    return result


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
        chat_id=data.get("chat_id"),
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
