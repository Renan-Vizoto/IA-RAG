from langchain_ollama import ChatOllama
from app.infrastructure.configs import settings

client = ChatOllama(
    model=settings.OLLAMA_MODEL,
    base_url=settings.OLLAMA_URL,
)
