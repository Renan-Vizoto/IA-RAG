from langchain_ollama import ChatOllama

from app.infrastructure.configs import settings


def create_chat_model(model_name: str) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url=settings.OLLAMA_URL,
        temperature=settings.OLLAMA_TEMPERATURE,
        num_predict=settings.OLLAMA_NUM_PREDICT,
        keep_alive=settings.OLLAMA_KEEP_ALIVE,
        reasoning=settings.think,
    )


client = create_chat_model(settings.OLLAMA_MODEL)
