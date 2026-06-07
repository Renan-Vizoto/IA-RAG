from fastapi import APIRouter, Cookie, HTTPException, Response
from pydantic import BaseModel
from typing import Annotated
import uuid

from app.core.services.chat_service import ChatService
from app.core.services.search_service import SearchService
from app.infrastructure.implementations.embbeding.MiniLML12_embbeding import MiniLML12_Embbeding
from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.infrastructure.repositories.chat_session_repo import ChatSessionRepository
from app.infrastructure.clients import ollama, milvus_client
from langchain_core.tools import tool
from app.api.schemas.chat_response import ChatResponse, ChatTraceResponse, TokenUsage, MilvusHitTrace

class ChatRequest(BaseModel):
    message: str

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

embbeder = MiniLML12_Embbeding()
repo = MilvusRepo(milvus_client.milvusClient)
searchService = SearchService(repo, embbeder)
search_tool = tool(searchService.search)

session_repo: ChatSessionRepository | None = None
chatService: ChatService | None = None


def init_chat_dependencies(chat_session_repo: ChatSessionRepository):
    global session_repo, chatService
    session_repo = chat_session_repo
    chatService = ChatService(
        ollama.client,
        [search_tool],
        search_service=searchService,
        session_repo=session_repo,
    )


if chatService is None:
    chatService = ChatService(
        ollama.client,
        [search_tool],
        search_service=searchService,
    )


@router.post("/message", response_model=ChatResponse)
def send_message(
    request: ChatRequest,
    response: Response,
    session_id: Annotated[str | None, Cookie()] = None
):
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24 * 7
        )
    return chatService.send_message(request.message, session_id)


@router.get("/trace/{response_id}", response_model=ChatTraceResponse, summary="Rastreio da resposta")
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
