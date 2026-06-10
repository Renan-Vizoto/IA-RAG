from pydantic import BaseModel, ConfigDict, Field


class ChatSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chat_id: str
    title: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    message_count: int = 0
    created_at: str
    updated_at: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: str
    role: str
    content: str
    created_at: str


class CreateChatRequest(BaseModel):
    title: str | None = Field(
        default=None,
        description="Optional chat title. Defaults to 'New chat'.",
    )


class CreateChatResponse(BaseModel):
    chat_id: str
    title: str
    created_at: str


class ChatListResponse(BaseModel):
    chats: list[ChatSummary]


class ChatMessagesResponse(BaseModel):
    chat_id: str
    messages: list[ChatMessage]
