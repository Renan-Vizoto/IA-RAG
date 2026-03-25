from fastapi import APIRouter, Cookie, Response
from pydantic import BaseModel
from typing import Annotated
import uuid

from app.core.services.chat_service import ChatService
from app.core.services.search_service import SearchService
from app.infrastructure.implementations.embbeding.MiniLML12_embbeding import MiniLML12_Embbeding
from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.infrastructure.clients import ollama, milvus_client
from langchain_core.tools import tool
from app.api.schemas.chat_response import ChatResponse

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

chatService = ChatService(ollama.client, [search_tool], search_service=searchService)

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