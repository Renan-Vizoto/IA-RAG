from pydantic import BaseModel
from pydantic import ConfigDict, Field
from typing import List

class SearchResult(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "hit-1",
                    "distance": 0.12,
                    "text": "Modelo XGBoost treinado na camada gold. RMSE: 89.1.",
                    "source": "mlflow_metadata",
                    "collection": "governance",
                }
            ]
        }
    )

    id: str | None = Field(default=None, description="Milvus hit identifier.")
    distance: float | None = Field(
        default=None,
        description="Cosine distance returned by Milvus. Lower values are more similar.",
    )
    text: str = Field(
        description="Retrieved text snippet after sensitive MLflow IDs are redacted."
    )
    source: str | None = Field(
        default=None,
        description="Indexed source label, for example gold_governance or mlflow_report.",
    )
    collection: str | None = Field(
        default=None,
        description="Milvus collection inferred from the source label.",
    )

class TokenUsage(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "input_tokens": 120,
                    "output_tokens": 48,
                    "total_tokens": 168,
                }
            ]
        }
    )

    input_tokens: int | None = Field(
        default=None,
        description="Prompt/input tokens reported for the LLM call.",
    )
    output_tokens: int | None = Field(
        default=None,
        description="Completion/output tokens reported for the LLM call.",
    )
    total_tokens: int | None = Field(
        default=None,
        description="Total tokens. Computed from input plus output when needed.",
    )

class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "search_results": [
                        {
                            "id": "hit-1",
                            "distance": 0.12,
                            "text": "Modelo XGBoost treinado na camada gold. RMSE: 89.1.",
                            "source": "mlflow_metadata",
                            "collection": "governance",
                        }
                    ],
                    "agent_thoughts": "",
                    "answer": "O modelo treinado foi XGBoost e o RMSE foi 89.1.",
                    "response_time_seconds": 1.25,
                    "confidence_score": 0.88,
                    "session_id": "8b2d3ff5-2f4a-4a1e-b23b-4d61b67a61bb",
                    "chat_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "title": "Qual modelo foi treinado?",
                    "message_count": 4,
                    "response_id": "64682936-5f57-4772-baf8-dd6f546a4c98",
                    "model": "gemma4-unsloth",
                    "tokens": {
                        "input_tokens": 120,
                        "output_tokens": 48,
                        "total_tokens": 168,
                    },
                    "chat_tokens": {
                        "input_tokens": 120,
                        "output_tokens": 48,
                        "total_tokens": 168,
                    },
                }
            ]
        }
    )

    search_results: List[SearchResult] = Field(
        description="Retrieval evidence used in the current agent turn."
    )
    agent_thoughts: str = Field(
        description="Extracted model reasoning when present. Intended for diagnostics."
    )
    answer: str = Field(description="Final sanitized answer returned to the client.")
    response_time_seconds: float = Field(
        description="Wall-clock time spent processing the message."
    )
    confidence_score: float = Field(
        description="Score derived from retrieved vector distances, clamped from 0 to 1."
    )
    session_id: str = Field(description="Chat session ID from the HttpOnly cookie.")
    chat_id: str = Field(description="Active chat thread ID for this conversation.")
    title: str = Field(
        description="Current chat thread title after persistence and auto-title rules."
    )
    message_count: int = Field(
        description="Number of user messages in this chat thread."
    )
    response_id: str = Field(
        description="Unique response ID for this assistant turn."
    )
    model: str = Field(description="Resolved Ollama model used for generation.")
    tokens: TokenUsage = Field(description="Token usage for the current turn.")
    chat_tokens: TokenUsage = Field(
        description="Persisted cumulative token usage for this chat."
    )
