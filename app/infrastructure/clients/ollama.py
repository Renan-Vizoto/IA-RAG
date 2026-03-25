from langchain_ollama import ChatOllama
from app.infrastructure.configs import settings

client = ChatOllama(
    model="qwen3.5-unsloth",
    base_url=settings.OLLAMA_URL,
    think=settings.think,
)
