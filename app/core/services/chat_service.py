from typing import Dict, List, Callable, Any, Union, Optional
import logging
import time
import re
import uuid
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from app.infrastructure.mlflow_config import log_chat_response
from app.infrastructure.configs import settings
from app.api.schemas.chat_response import ChatResponse, TokenUsage
from app.infrastructure.repositories.chat_session_repo import ChatSessionRepository
from pydantic import BaseModel

logger = logging.getLogger(__name__)

Message = Union[HumanMessage, SystemMessage]

class MilvusHit(BaseModel):
    id: str
    distance: float
    text: str
    source: str | None = None
    collection: str | None = None

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

    def __init__(
        self,
        model: ChatOllama,
        tools: List[Callable[..., Any]],
        search_service=None,
        session_repo: Optional[ChatSessionRepository] = None,
    ):
        self._model_name = getattr(model, "model", None) or settings.OLLAMA_MODEL
        self._agent_executor = create_agent(
            model=model,
            tools=tools,
            system_prompt=(
                "Você é um assistente de governança do pipeline de machine learning Dutch Energy. "
                "Sempre use a ferramenta de busca antes de responder. "
                "A busca vetorial retorna documentos de governança e metadados de treinamento MLflow. "
                "Fontes disponíveis: silver_governance e gold_governance (pipeline bronze/silver/gold), "
                "mlflow_report (relatório markdown do treinamento) e mlflow_metadata "
                "(métricas, hiperparâmetros e run IDs dos experimentos no PostgreSQL). "
                "Responda em português, com base nos dados recuperados sobre ingestão (bronze), "
                "processamento (silver), split de dados (gold) e treinamento do modelo. "
                "Se não encontrar informação relevante, diga claramente que não há dados disponíveis."
            )
        )
        self._search_service = search_service
        self._session_repo = session_repo

    def _get_or_create_chat(self, session_id: str) -> List[Message]:
        chat = self.chats.get(session_id)
        if chat is None:
            chat = []
            self.chats[session_id] = chat
        return chat

    def _persist_session_data(
        self,
        response_id: str,
        session_id: str,
        message: str,
        parsed_response: ParsedResponse,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        response_time: float,
        confidence_score: float,
    ) -> TokenUsage:
        default_session = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        if not self._session_repo:
            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        try:
            self._session_repo.ensure_session(session_id)
            self._session_repo.save_response(
                response_id=response_id,
                session_id=session_id,
                model=self._model_name,
                user_message=message,
                answer=parsed_response.answer,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                response_time_seconds=response_time,
                confidence_score=confidence_score,
            )
            hits = [
                h.model_dump() for h in (parsed_response.result or [])
            ]
            self._session_repo.save_milvus_hits(response_id, hits)
            self._session_repo.add_session_tokens(
                session_id,
                input_tokens or 0,
                output_tokens or 0,
                total_tokens or 0,
            )
            totals = self._session_repo.get_session_tokens(session_id)
            return TokenUsage(**totals)
        except Exception as e:
            logger.warning(f"[CHAT_DB] Falha ao persistir sessão: {e}")
            return default_session

    def send_message(self, message: str, session_id: str) -> ChatResponse:
        response_id = str(uuid.uuid4())
        start_time = time.time()

        chat_history = self._get_or_create_chat(session_id)
        messages = chat_history + [HumanMessage(content=message)]

        result = self._agent_executor.invoke({
            "messages": messages
        })

        parsed_response = self._parse_agent_output(result)

        self.chats[session_id] = result.get("messages", messages)

        response_time = time.time() - start_time
        confidence_score = 0.0
        if hasattr(parsed_response, 'result') and parsed_response.result:
            avg_distance = sum([r.distance or 0 for r in parsed_response.result]) / len(parsed_response.result)
            confidence_score = max(0.0, min(1.0, 1.0 - avg_distance))

        input_tokens, output_tokens, total_tokens = self._extract_token_usage(
            result.get("messages", [])
        )

        response_tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        session_tokens = self._persist_session_data(
            response_id=response_id,
            session_id=session_id,
            message=message,
            parsed_response=parsed_response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            response_time=response_time,
            confidence_score=confidence_score,
        )

        log_chat_response(
            response_id=response_id,
            session_id=session_id,
            model=self._model_name,
            user_message=message,
            answer=parsed_response.answer,
            metrics={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "session_input_tokens": session_tokens.input_tokens,
                "session_output_tokens": session_tokens.output_tokens,
                "session_total_tokens": session_tokens.total_tokens,
                "response_time_seconds": response_time,
                "confidence_score": confidence_score,
                "search_result_count": len(parsed_response.result) if parsed_response.result else 0,
                "message_count": len(self.chats[session_id]),
            },
        )

        return ChatResponse(
            search_results=[r.model_dump() for r in parsed_response.result] if parsed_response.result else [],
            agent_thoughts=parsed_response.thinking,
            answer=parsed_response.answer,
            response_time_seconds=response_time,
            confidence_score=confidence_score,
            session_id=session_id,
            message_count=len(self.chats[session_id]),
            response_id=response_id,
            model=self._model_name,
            tokens=response_tokens,
            session_tokens=session_tokens,
        )

    @staticmethod
    def _extract_token_usage(messages: List[Any]) -> tuple[int | None, int | None, int | None]:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        found = False

        for msg in messages:
            if not (isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai"):
                continue
            usage = getattr(msg, "usage_metadata", None) or {}
            if not usage:
                continue
            found = True
            input_tokens += usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            output_tokens += usage.get("output_tokens") or usage.get("completion_tokens") or 0
            total_tokens += usage.get("total_tokens") or 0

        if not found:
            return None, None, None
        if total_tokens == 0 and (input_tokens or output_tokens):
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens

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
                gemma_matches = re.findall(r"<\|channel>thought\n(.*?)(?:\s*$|\s*<\|)", content, re.DOTALL)
                for match in gemma_matches:
                    if match.strip():
                        think_blocks.append(match.strip())
                qwen_tag = "redacted_thinking"
                qwen_matches = re.findall(
                    rf"<{qwen_tag}>(.*?)</{qwen_tag}>", content, re.DOTALL
                )
                for match in qwen_matches:
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

    @staticmethod
    def _infer_collection(source: str | None) -> str:
        if source == "mlflow_metadata":
            return settings.mlflow_metadata_collection
        return settings.governance_collection

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
                                entity = hit.get("entity", {}) if isinstance(hit.get("entity"), dict) else {}
                                text = hit.get("text", "") or entity.get("text", "")
                                source = entity.get("source") or hit.get("source")
                                collection = self._infer_collection(source)

                                search_results.append(MilvusHit(
                                    id=str(hit.get("id", "")),
                                    distance=float(hit.get("distance", 0.0)),
                                    text=str(text),
                                    source=source,
                                    collection=collection,
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

        answer = re.sub(r"<\|channel>thought\n.*?(?=\S|\Z)", "", raw_output, flags=re.DOTALL)
        qwen_tag = "redacted_thinking"
        answer = re.sub(rf"<{qwen_tag}>.*?</{qwen_tag}>", "", answer, flags=re.DOTALL).strip()

        return ParsedResponse(
            thinking=thinking,
            answer=answer,
            result=search_results
        )
