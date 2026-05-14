from langchain_ollama import ChatOllama
from app.infrastructure.configs import settings

client = ChatOllama(
    model="gemma4-unsloth",
    base_url=settings.OLLAMA_URL,
)
