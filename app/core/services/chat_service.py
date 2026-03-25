from typing import Dict, List, Callable, Any, Union
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
    def __init__(self, model: ChatOllama, tools: List[Callable[..., Any]], search_service=None):
        self._agent_executor = create_agent(
            model=model,
            tools=tools,
            system_prompt="You are a library assistant. Always search before answering."
        )
        self._search_service = search_service

    def send_message(self, message: str, chat_id: str, chat_history: List[Message] | None = None) -> ChatResponse:
        start_time = time.time()

        history = chat_history or []
        messages = history + [HumanMessage(content=message)]

        result = self._agent_executor.invoke({
            "messages": messages
        })

        print(result)

        parsed_response = self._parse_agent_output(result)

        all_messages = result.get("messages", messages)
        response_time = time.time() - start_time
        confidence_score = 0.0
        if hasattr(parsed_response, 'result') and parsed_response.result:
            avg_distance = sum([r.distance or 0 for r in parsed_response.result]) / len(parsed_response.result)
            confidence_score = max(0.0, min(1.0, 1.0 - avg_distance))

        start_run()
        log_params({
            "model_name": "qwen3.5-unsloth",
            "chat_id": chat_id,
            "message_count": len(all_messages)
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
            session_id=chat_id,
            message_count=len(all_messages)
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



