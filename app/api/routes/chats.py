from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, AIMessage

from app.infrastructure.database import get_db
from app.core.entities.chat_models import Chat, Message
from app.api.schemas.chat_schemas import (
    ChatOut, ChatWithMessagesOut, CreateChatRequest, UpdateChatRequest,
    SendMessageRequest
)
from app.api.schemas.chat_response import ChatResponse
from app.core.services.chat_service import ChatService
from app.core.services.search_service import SearchService
from app.infrastructure.implementations.embbeding.MiniLML12_embbeding import MiniLML12_Embbeding
from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.infrastructure.clients import ollama, milvus_client
from langchain_core.tools import tool

router = APIRouter(
    prefix="/chats",
    tags=["chats"],
    responses={404: {"description": "Not found"}},
)

embbeder = MiniLML12_Embbeding()
repo = MilvusRepo(milvus_client.milvusClient)
searchService = SearchService(repo, embbeder)
search_tool = tool(searchService.search)
chatService = ChatService(ollama.client, [search_tool], search_service=searchService)


def _to_langchain_messages(db_messages: list[Message]) -> list:
    result = []
    for msg in db_messages:
        if msg.role == "user":
            result.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            result.append(AIMessage(content=msg.content))
    return result


@router.post("", response_model=ChatOut, status_code=201)
def create_chat(request: CreateChatRequest, db: Session = Depends(get_db)):
    chat = Chat(title=request.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


@router.get("", response_model=list[ChatOut])
def list_chats(db: Session = Depends(get_db)):
    return db.query(Chat).order_by(Chat.updated_at.desc()).all()


@router.get("/{chat_id}", response_model=ChatWithMessagesOut)
def get_chat(chat_id: str, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.put("/{chat_id}", response_model=ChatOut)
def update_chat(chat_id: str, request: UpdateChatRequest, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat.title = request.title
    chat.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(chat)
    return chat


@router.delete("/{chat_id}", status_code=204)
def delete_chat(chat_id: str, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    db.delete(chat)
    db.commit()


@router.post("/{chat_id}/messages", response_model=ChatResponse)
def send_message(chat_id: str, request: SendMessageRequest, db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Load history and convert to LangChain messages
    langchain_history = _to_langchain_messages(chat.messages)

    # Call the agent
    response = chatService.send_message(request.message, chat_id, langchain_history)

    # Persist user message and assistant answer
    db.add(Message(chat_id=chat_id, role="user", content=request.message))
    db.add(Message(chat_id=chat_id, role="assistant", content=response.answer))

    # Bump chat updated_at
    chat.updated_at = datetime.now(timezone.utc)
    db.commit()

    return response
