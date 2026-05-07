from typing import Dict, List, Callable, Any, Union, Optional
import time
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from app.infrastructure.mlflow_config import start_run, log_metrics, log_params, end_run
from app.api.schemas.chat_response import ChatResponse
from pydantic import BaseModel

Message = Union[HumanMessage, SystemMessage]

class MilvusHit(BaseModel):
    id: str
    distance: float
    text: str
class ToolStep(BaseModel):
    tool: str
    input: str | dict
    result: list[MilvusHit] | str

class ParsedResponse(BaseModel):
    thinking: str
    answer: str
    result: Any

class ChatService:
    chats: Dict[str, List[Message]] = {}

    def __init__(self, model: ChatOllama, tools: List[Callable[..., Any]], search_service=None, mlflow_search_service=None):
        self._agent_executor = create_agent(
            model=model,
            tools=tools,
            system_prompt=(
                "Você é um assistente de governança do pipeline de machine learning Dutch Energy. "
                "Sempre use a ferramenta de busca antes de responder. "
                "Você tem acesso a dois tipos de fonte de dados: "
                "(1) dados estruturados do PostgreSQL/MLflow com metadados e métricas de treinamento, "
                "e (2) documentos de governança semânticos no banco vetorial (Milvus). "
                "Responda em português, com base nos dados recuperados sobre ingestão (bronze), "
                "processamento (silver), split de dados (gold) e treinamento do modelo. "
                "Se não encontrar informação relevante, diga claramente que não há dados disponíveis."
            )
        )
        self._search_service = search_service
        self._mlflow_search_service = mlflow_search_service

    def _get_or_create_chat(self, session_id: str) -> List[Message]:
        chat = self.chats.get(session_id)
        if chat is None:
            chat = []
            self.chats[session_id] = chat
        return chat

    def send_message(self, message: str, session_id: str) -> ChatResponse:
        start_time = time.time()

        # 1º: busca estruturada no PostgreSQL via MLflow
        postgres_context = ""
        if self._mlflow_search_service:
            postgres_context = self._mlflow_search_service.search(message)

        chat_history = self._get_or_create_chat(session_id)

        # Inclui contexto do PostgreSQL na mensagem enviada ao agente
        user_content = message
        if postgres_context:
            user_content = f"{postgres_context}\n\n---\nPergunta do usuário: {message}"

        messages = chat_history + [HumanMessage(content=user_content)]

        result = self._agent_executor.invoke({
            "messages": messages
        })

        print(result)

        parsed_response = self._parse_agent_output(result)

        self.chats[session_id] = result.get("messages", messages)

        response_time = time.time() - start_time
        confidence_score = 0.0
        if hasattr(parsed_response, 'result') and parsed_response.result:
            avg_distance = sum([r.distance or 0 for r in parsed_response.result]) / len(parsed_response.result)
            confidence_score = max(0.0, min(1.0, 1.0 - avg_distance))

        start_run()
        log_params({
            "model_name": "qwen3.5-unsloth",
            "session_id": session_id,
            "message_count": len(self.chats[session_id])
        })
        log_metrics({
            "response_time_seconds": response_time,
            "confidence_score": confidence_score,
            "search_result_count": len(parsed_response.result)
        })
        end_run()

        return ChatResponse(
            search_results=[r.model_dump() for r in parsed_response.result] if parsed_response.result else [],
            agent_thoughts=parsed_response.thinking,
            answer=parsed_response.answer,
            response_time_seconds=response_time,
            confidence_score=confidence_score,
            session_id=session_id,
            message_count=len(self.chats[session_id])
        )
    
    def _get_current_turn_messages(self, messages: List[Any]) -> List[Any]:
        last_human_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                last_human_idx = i
        return messages[last_human_idx:] if last_human_idx >= 0 else messages

    def _extract_thinking(self, messages: List[Any]) -> str:
        think_blocks = []
        for msg in messages:
            if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
                content = msg.content if isinstance(msg.content, str) else ""
                think_matches = re.findall(r"<think>(.*?)</think>", content, re.DOTALL)
                for match in think_matches:
                    if match.strip():
                        think_blocks.append(match.strip())
        return "\n".join(think_blocks) if think_blocks else ""

    def _flatten_hits(self, items: list) -> list:
        flat = []
        for item in items:
            if isinstance(item, list):
                flat.extend(self._flatten_hits(item))
            else:
                flat.append(item)
        return flat

    def _parse_tool_content(self, content_str: str) -> list:
        if not content_str:
            return []
        
        import json
        try:
            return json.loads(content_str)
        except json.JSONDecodeError:
            import ast
            try:
                return ast.literal_eval(content_str)
            except Exception:
                return []

    def _extract_search_results(self, messages: List[Any]) -> List[MilvusHit]:
        search_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
                try:
                    res = self._parse_tool_content(getattr(msg, "content", ""))
                    if isinstance(res, list):
                        flat_res = self._flatten_hits(res)
                        for hit in flat_res:
                            if isinstance(hit, dict):
                                text = hit.get("text", "")
                                if "entity" in hit and isinstance(hit["entity"], dict):
                                    text = text or hit["entity"].get("text", "")
                                    
                                search_results.append(MilvusHit(
                                    id=str(hit.get("id", "")),
                                    distance=float(hit.get("distance", 0.0)),
                                    text=str(text)
                                ))
                except Exception as e:
                    print(f"Error parsing tool message: {e}")
        return search_results

    def _parse_agent_output(self, result: dict) -> ParsedResponse:
        messages = result.get("messages", [])
        current_turn_msgs = self._get_current_turn_messages(messages)
        
        final_ai_message = messages[-1] if messages else AIMessage(content="")
        raw_output = final_ai_message.content if hasattr(final_ai_message, "content") and isinstance(final_ai_message.content, str) else ""
        
        thinking = self._extract_thinking(current_turn_msgs)
        search_results = self._extract_search_results(current_turn_msgs)
        
        answer = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        
        return ParsedResponse(
            thinking=thinking,
            answer=answer,
            result=search_results
        )



