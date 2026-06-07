from typing import Dict, List, Callable, Any, Union, Optional
import logging
import time
import re
import uuid
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from app.infrastructure.mlflow_config import log_chat_response
from app.infrastructure.configs import settings
from app.core.prompt_loader import load_prompt
from app.infrastructure.clients.ollama import create_chat_model
from app.infrastructure.clients.ollama_model_manager import OllamaModelManager, model_manager
from app.api.schemas.chat_response import ChatResponse, TokenUsage
from app.infrastructure.repositories.chat_session_repo import ChatSessionRepository
from pydantic import BaseModel

logger = logging.getLogger(__name__)

Message = Union[HumanMessage, SystemMessage]

_RUN_ID_LINE_PATTERN = re.compile(
    r"(?i)run\s*id(?:\s*mlflow)?\s*:\s*[a-f0-9]{32}\.?\s*",
)
_HEX_ID_PATTERN = re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE)
_SOURCE_TAG_PATTERN = re.compile(
    r"\s*\[(?:silver_governance|gold_governance|mlflow_report|mlflow_metadata)\]",
    re.IGNORECASE,
)

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
        tools: List[Callable[..., Any]],
        search_service=None,
        session_repo: Optional[ChatSessionRepository] = None,
        ollama_model_manager: OllamaModelManager | None = None,
    ):
        self._default_model = settings.OLLAMA_MODEL
        self._allowed_models = set(settings.ollama_allowed_models)
        self._agents: dict[str, Any] = {}
        self._tools = tools
        self._system_prompt = load_prompt("rag_system")
        self._search_service = search_service
        self._session_repo = session_repo
        self._model_manager = ollama_model_manager or model_manager

    def _resolve_model(self, model: str | None) -> str:
        resolved = model or self._default_model
        if resolved not in self._allowed_models:
            allowed = ", ".join(sorted(self._allowed_models))
            raise ValueError(
                f"Modelo '{resolved}' não permitido. Modelos disponíveis: {allowed}"
            )
        return resolved

    def _get_agent(self, model_name: str):
        if model_name not in self._agents:
            llm = create_chat_model(model_name)
            self._agents[model_name] = create_agent(
                model=llm,
                tools=self._tools,
                system_prompt=self._system_prompt,
            )
        return self._agents[model_name]

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
        model_name: str,
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
                model=model_name,
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

    def send_message(
        self,
        message: str,
        session_id: str,
        model: str | None = None,
    ) -> ChatResponse:
        response_id = str(uuid.uuid4())
        start_time = time.time()
        resolved_model = self._resolve_model(model)

        self._model_manager.ensure_loaded(resolved_model)
        agent = self._get_agent(resolved_model)

        chat_history = self._get_or_create_chat(session_id)

        presearch_hits: list[MilvusHit] = []
        if self._search_service:
            try:
                raw_hits = self._search_service.search(message)
                presearch_hits = self._hits_from_raw_search(raw_hits)
            except Exception as e:
                logger.warning(f"[RAG] Falha na busca vetorial obrigatória: {e}")

        user_content = self._build_user_message(message, presearch_hits)
        messages = chat_history + [HumanMessage(content=user_content)]

        result = agent.invoke({
            "messages": messages
        })

        parsed_response = self._parse_agent_output(result, fallback_hits=presearch_hits)

        self.chats[session_id] = result.get("messages", messages)

        response_time = time.time() - start_time
        confidence_score = 0.0
        if hasattr(parsed_response, 'result') and parsed_response.result:
            avg_distance = sum([r.distance or 0 for r in parsed_response.result]) / len(parsed_response.result)
            confidence_score = max(0.0, min(1.0, 1.0 - avg_distance))

        current_turn_messages = self._get_current_turn_messages(
            result.get("messages", [])
        )
        input_tokens, output_tokens, total_tokens = self._extract_token_usage(
            current_turn_messages
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
            model_name=resolved_model,
        )

        log_chat_response(
            response_id=response_id,
            session_id=session_id,
            model=resolved_model,
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
            model=resolved_model,
            tokens=response_tokens,
            session_tokens=session_tokens,
        )

    @staticmethod
    def _redact_sensitive_text(text: str) -> str:
        text = _RUN_ID_LINE_PATTERN.sub("", text)
        text = _HEX_ID_PATTERN.sub("", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    @staticmethod
    def _sanitize_answer(answer: str) -> str:
        answer = _SOURCE_TAG_PATTERN.sub("", answer)
        answer = _RUN_ID_LINE_PATTERN.sub("", answer)
        answer = _HEX_ID_PATTERN.sub("", answer)
        answer = re.sub(r"\s{2,}", " ", answer)
        answer = re.sub(r"\s+([.,;])", r"\1", answer)
        return answer.strip()

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

    def _dict_to_milvus_hit(self, hit: dict) -> MilvusHit | None:
        if not isinstance(hit, dict):
            return None
        entity = hit.get("entity", {}) if isinstance(hit.get("entity"), dict) else {}
        text = hit.get("text", "") or entity.get("text", "")
        if not text:
            return None
        source = entity.get("source") or hit.get("source")
        return MilvusHit(
            id=str(hit.get("id", "")),
            distance=float(hit.get("distance", 0.0)),
            text=str(text),
            source=source,
            collection=self._infer_collection(source),
        )

    def _hits_from_raw_search(self, raw: Any) -> list[MilvusHit]:
        hits: list[MilvusHit] = []
        for item in self._flatten_hits(raw if isinstance(raw, list) else [raw]):
            parsed = self._dict_to_milvus_hit(item)
            if parsed:
                hits.append(parsed)
        return hits

    @staticmethod
    def _truncate_chunk_text(text: str, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _format_hits_for_prompt(self, hits: list[MilvusHit]) -> str:
        if not hits:
            return "(vazio)"

        limited = hits[: settings.RAG_MAX_CONTEXT_CHUNKS]
        lines = []
        for index, hit in enumerate(limited, start=1):
            source = hit.source or "desconhecida"
            text = self._truncate_chunk_text(hit.text, settings.RAG_MAX_CHUNK_CHARS)
            text = self._redact_sensitive_text(text)
            lines.append(f"[{index}]\n{text}")
        return "\n\n".join(lines)

    def _build_user_message(self, message: str, hits: list[MilvusHit]) -> str:
        context = self._format_hits_for_prompt(hits)
        return (
            "--- CONTEXTO ---\n"
            f"{context}\n"
            "--- FIM ---\n\n"
            f"PERGUNTA: {message.strip()}\n\n"
            "RESPOSTA (tom cordial, máx. 5 frases, sem ids técnicos nem tags de fonte):"
        )

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
                        for hit in self._hits_from_raw_search(res):
                            search_results.append(hit)
                except Exception as e:
                    print(f"Error parsing tool message: {e}")
        return search_results

    def _parse_agent_output(
        self,
        result: dict,
        fallback_hits: list[MilvusHit] | None = None,
    ) -> ParsedResponse:
        messages = result.get("messages", [])
        current_turn_msgs = self._get_current_turn_messages(messages)

        final_ai_message = messages[-1] if messages else AIMessage(content="")
        raw_output = final_ai_message.content if hasattr(final_ai_message, "content") and isinstance(final_ai_message.content, str) else ""

        thinking = self._extract_thinking(current_turn_msgs)
        search_results = self._extract_search_results(current_turn_msgs)
        if not search_results and fallback_hits:
            search_results = fallback_hits

        answer = re.sub(r"<\|channel>thought\n.*?(?=\S|\Z)", "", raw_output, flags=re.DOTALL)
        qwen_tag = "redacted_thinking"
        answer = re.sub(rf"<{qwen_tag}>.*?</{qwen_tag}>", "", answer, flags=re.DOTALL).strip()
        answer = self._sanitize_answer(answer)

        return ParsedResponse(
            thinking=thinking,
            answer=answer,
            result=search_results
        )
