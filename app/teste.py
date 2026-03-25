from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent

from app.infrastructure.clients.milvus_client import milvusClient
from app.infrastructure.repositories.milvus_repo import BaseRepo
from app.infrastructure.implementations.embbeding.MiniLML12_embbeding import EmbbedingService
from app.infrastructure.configs import settings

# --- Dependencies ---

repo = BaseRepo(milvusClient)
service = EmbbedingService()

# --- Tools ---

@tool
def search(text: str) -> str:
    """Search the document database for the given text query."""
    vector = service.embbed_it([text])
    return str(repo.search(settings.collection_name, vector))

# --- LLM ---

llm = ChatOllama(
    model="qwen3.5-unsloth",
    base_url="http://localhost:11435",
    think=False,
)

# --- Agent ---

agent = create_agent(llm, tools=[search])

# --- Run ---

if __name__ == "__main__":
    messages = [
        SystemMessage("You are a library assistant. Always search before answering."),
        HumanMessage("What's written in the paragraph where it says: to produce fully autonomous agents"),
    ]

    result = agent.invoke({"messages": messages})

    for msg in result["messages"]:
        role = msg.__class__.__name__
        content = msg.content or getattr(msg, "tool_calls", "")
        print(f"[{role}]: {content}\n")

    print("=== Final Answer ===")
    print(result["messages"][-1].content)