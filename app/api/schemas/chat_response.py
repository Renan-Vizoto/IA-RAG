from pydantic import BaseModel
from typing import List

class SearchResult(BaseModel):
    id: str | None = None
    distance: float | None = None
    text: str

class ChatResponse(BaseModel):
    search_results: List[SearchResult]
    agent_thoughts: str
    answer: str
    response_time_seconds: float
    confidence_score: float
    session_id: str
    message_count: int
