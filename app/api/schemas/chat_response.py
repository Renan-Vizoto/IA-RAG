from pydantic import BaseModel
from typing import List

class SearchResult(BaseModel):
    id: str | None = None
    distance: float | None = None
    text: str
    source: str | None = None
    collection: str | None = None

class TokenUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

class ChatResponse(BaseModel):
    search_results: List[SearchResult]
    agent_thoughts: str
    answer: str
    response_time_seconds: float
    confidence_score: float
    session_id: str
    message_count: int
    response_id: str
    model: str
    tokens: TokenUsage
    session_tokens: TokenUsage

class MilvusHitTrace(BaseModel):
    milvus_id: str | None = None
    collection: str
    source: str | None = None
    distance: float | None = None
    text_preview: str | None = None

class ChatTraceResponse(BaseModel):
    response_id: str
    session_id: str
    model: str
    user_message: str
    answer: str
    tokens: TokenUsage
    response_time_seconds: float | None = None
    confidence_score: float | None = None
    milvus_hits: List[MilvusHitTrace]
    created_at: str
