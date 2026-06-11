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
_PIPELINE_SEARCH_PATTERN = re.compile(
    r"modelo|métric|metric|rmse|mae|r²|r2|mape|trein|pipeline|xgboost|feature|hiper|"
    r"gold|silver|bronze|mlflow|consume|dados|limpos|limpeza|dataset|"
    r"pré-?process|preprocess|transform|encoding|normaliz",
    re.IGNORECASE,
)
_IDENTITY_PATTERN = re.compile(
    r"^(?:olá|oi|bom dia|boa tarde|boa noite[!.]?\s*)?"
    r"(?:quem é você|quem é vc|o que você é|o que é você|o que você faz|"
    r"qual é sua função|qual sua função|apresente-se|se apresente|quem sou eu)",
    re.IGNORECASE,
)
_METRICS_QUESTION_PATTERN = re.compile(
    r"métric|metric|desempenho|avaliação|rmse|mae|mape|r²|r2",
    re.IGNORECASE,
)
_PREPROCESSING_QUESTION_PATTERN = re.compile(
    r"pré-?process|preprocess|limpeza|limpos|transform|encoding|normaliz|prepar",
    re.IGNORECASE,
)
_IDENTITY_ANSWER = (
    "Olá! Sou o WattTrack, assistente de governança do pipeline Dutch Energy. "
    "Posso explicar as etapas bronze, silver e gold, a limpeza dos dados, "
    "o pré-processamento e o treinamento do modelo."
)
_EMPTY_SEARCH_PHRASE = "Não encontrei essa informação nos dados disponíveis"
_SEARCH_POOL_SIZE = 12
_MAX_RETURNED_HITS = 3
_SEARCH_RETRY_NUDGE_PREFIX = "INSTRUÇÃO INTERNA: chame a ferramenta search"
_FORCED_SEARCH_PREFIX = "Contexto da busca semântica (use somente isto):"
_EMPTY_SEARCH_ANSWER = (
    "Não encontrei essa informação nos dados disponíveis. "
    "Posso ajudar com outra pergunta sobre o pipeline?"
)
_BOGUS_TOOL_ANSWERS = frozenset({"search"})
_QWEN_THINKING_TAGS = ("think", "redacted_thinking")
_GEMMA_THOUGHT_PATTERN = re.compile(
    r"<\|channel>thought\n.*?(?:\n\n|\Z)",
    re.DOTALL,
)
_QWEN_TOOL_CALL_PATTERN = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_GEMMA_TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call>.*?<tool_call\|>",
    re.DOTALL,
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

    def __init__(
        self,
        tools: List[Callable[..., Any]],
        session_repo: Optional[ChatSessionRepository] = None,
        ollama_model_manager: OllamaModelManager | None = None,
        search_fn: Callable[[str], Any] | None = None,
    ):
        self._chat_histories: Dict[str, List[Message]] = {}
        self._default_model = settings.OLLAMA_MODEL
        self._allowed_models = set(settings.ollama_allowed_models)
        self._agents: dict[str, Any] = {}
        self._tools = tools
        self._search_fn = search_fn
        self._system_prompt = load_prompt("rag_system")
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

    def _resolve_chat_id(self, session_id: str, chat_id: str | None) -> str:
        if not self._session_repo:
            return chat_id or str(uuid.uuid4())
        return self._session_repo.ensure_chat(session_id, chat_id)

    def _load_chat_history(self, chat_id: str) -> List[Message]:
        cached = self._chat_histories.get(chat_id)
        if cached is not None:
            return cached

        history: List[Message] = []
        if self._session_repo:
            try:
                for row in self._session_repo.get_messages(chat_id):
                    if row["role"] == "user":
                        history.append(HumanMessage(content=row["content"]))
                    elif row["role"] == "assistant":
                        history.append(AIMessage(content=row["content"]))
            except Exception as e:
                logger.warning(f"[CHAT_DB] Falha ao carregar histórico: {e}")

        self._chat_histories[chat_id] = history
        return history

    _DEFAULT_CHAT_TITLES = frozenset({"new chat", "novo chat"})

    def _maybe_auto_title(self, chat_id: str, message: str) -> None:
        if not self._session_repo:
            return
        try:
            chat = self._session_repo.get_chat(chat_id)
            if not chat:
                return
            current = (chat.get("title") or "").strip()
            if current.lower() not in self._DEFAULT_CHAT_TITLES:
                return
            title = message.strip()[:50] or current
            self._session_repo.update_chat_title(chat_id, title)
        except Exception as e:
            logger.warning(f"[CHAT_DB] Falha ao definir título do chat: {e}")

    def _persist_chat_data(
        self,
        response_id: str,
        session_id: str,
        chat_id: str,
        message: str,
        parsed_response: ParsedResponse,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        response_time: float,
        confidence_score: float,
        model_name: str,
    ) -> TokenUsage:
        default_chat = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        if not self._session_repo:
            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        try:
            self._session_repo.ensure_session(session_id)
            self._session_repo.save_message(
                str(uuid.uuid4()), chat_id, "user", message
            )
            self._session_repo.save_message(
                str(uuid.uuid4()), chat_id, "assistant", parsed_response.answer
            )
            self._maybe_auto_title(chat_id, message)
            self._session_repo.touch_chat(chat_id)

            self._session_repo.save_response(
                response_id=response_id,
                session_id=session_id,
                chat_id=chat_id,
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
            self._session_repo.add_chat_tokens(
                chat_id,
                input_tokens or 0,
                output_tokens or 0,
                total_tokens or 0,
            )
            totals = self._session_repo.get_chat_tokens(chat_id)
            return TokenUsage(**totals)
        except Exception as e:
            logger.warning(f"[CHAT_DB] Falha ao persistir chat: {e}")
            return default_chat

    def send_message(
        self,
        message: str,
        session_id: str,
        model: str | None = None,
        chat_id: str | None = None,
    ) -> ChatResponse:
        response_id = str(uuid.uuid4())
        start_time = time.time()
        resolved_model = self._resolve_model(model)

        resolved_chat_id = self._resolve_chat_id(session_id, chat_id)

        chat_history = self._load_chat_history(resolved_chat_id)

        if self._is_identity_question(message):
            parsed_response = ParsedResponse(
                thinking="",
                answer=_IDENTITY_ANSWER,
                result=[],
            )
            result_messages = chat_history + [
                HumanMessage(content=message),
                AIMessage(content=_IDENTITY_ANSWER),
            ]
            used_forced_search = False
        elif self._is_structured_pipeline_question(message):
            topic_hits = self._topic_search(message)
            answer = (
                self._build_structured_answer(message, topic_hits)
                if topic_hits
                else _EMPTY_SEARCH_ANSWER
            )
            parsed_response = ParsedResponse(
                thinking="",
                answer=answer,
                result=topic_hits[:_MAX_RETURNED_HITS],
            )
            result_messages = chat_history + [
                HumanMessage(content=message),
                AIMessage(content=answer),
            ]
            used_forced_search = False
        else:
            self._model_manager.ensure_loaded(resolved_model)
            agent = self._get_agent(resolved_model)
            messages = chat_history + [HumanMessage(content=message)]

            result = agent.invoke({"messages": messages})
            result_messages = result.get("messages", messages)
            parsed_response = self._parse_agent_output({"messages": result_messages})
            used_forced_search = False

            if self._requires_search(message) and not parsed_response.result:
                forced_hits = self._force_search(message)
                if forced_hits:
                    logger.info("[CHAT] Busca automática — modelo não chamou search.")
                    used_forced_search = True
                    context = self._format_hits_for_context(forced_hits)
                    follow_up = messages + [
                        HumanMessage(
                            content=(
                                f"{_FORCED_SEARCH_PREFIX}\n{context}\n\n"
                                f"Responda em português, no máximo 3 frases curtas: {message}"
                            )
                        )
                    ]
                    result = agent.invoke({"messages": follow_up})
                    result_messages = result.get("messages", follow_up)
                    parsed_response = self._parse_agent_output({"messages": result_messages})
                    parsed_response = parsed_response.model_copy(update={"result": forced_hits})
                elif self._should_retry_search(message, result_messages, parsed_response):
                    logger.info("[CHAT] Modelo não chamou search; tentando novamente com instrução explícita.")
                    nudge = HumanMessage(
                        content=(
                            f"{_SEARCH_RETRY_NUDGE_PREFIX} com query relacionada a: {message}"
                        )
                    )
                    result = agent.invoke({"messages": messages + [nudge]})
                    result_messages = self._strip_search_retry_nudge(
                        result.get("messages", messages + [nudge])
                    )
                    parsed_response = self._parse_agent_output({"messages": result_messages})

            if self._is_bogus_answer(parsed_response.answer):
                if parsed_response.result:
                    parsed_response = parsed_response.model_copy(
                        update={"answer": self._fallback_answer_from_hits(
                            parsed_response.result, message
                        )}
                    )
                else:
                    parsed_response = parsed_response.model_copy(
                        update={"answer": _EMPTY_SEARCH_ANSWER}
                    )

            parsed_response = parsed_response.model_copy(
                update={
                    "answer": self._refine_answer_for_question(
                        message,
                        parsed_response.answer,
                        parsed_response.result or [],
                    )
                }
            )

        if used_forced_search:
            updated_history = chat_history + [
                HumanMessage(content=message),
                AIMessage(content=parsed_response.answer),
            ]
        else:
            updated_history = self._compact_chat_history(result_messages)
        self._chat_histories[resolved_chat_id] = updated_history
        user_message_count = self._count_user_messages(updated_history)

        response_time = time.time() - start_time
        confidence_score = 0.0
        if hasattr(parsed_response, 'result') and parsed_response.result:
            avg_distance = sum([r.distance or 0 for r in parsed_response.result]) / len(parsed_response.result)
            confidence_score = max(0.0, min(1.0, 1.0 - avg_distance))

        current_turn_messages = self._get_current_turn_messages(result_messages)
        input_tokens, output_tokens, total_tokens = self._extract_token_usage(
            current_turn_messages
        )

        response_tokens = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        chat_tokens = self._persist_chat_data(
            response_id=response_id,
            session_id=session_id,
            chat_id=resolved_chat_id,
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
            chat_id=resolved_chat_id,
            model=resolved_model,
            user_message=message,
            answer=parsed_response.answer,
            metrics={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "chat_input_tokens": chat_tokens.input_tokens,
                "chat_output_tokens": chat_tokens.output_tokens,
                "chat_total_tokens": chat_tokens.total_tokens,
                "response_time_seconds": response_time,
                "confidence_score": confidence_score,
                "search_result_count": len(parsed_response.result) if parsed_response.result else 0,
                "message_count": user_message_count,
            },
        )

        chat_row = (
            self._session_repo.get_chat(resolved_chat_id)
            if self._session_repo
            else None
        )
        chat_title = (chat_row or {}).get("title") or "New chat"

        return ChatResponse(
            search_results=[r.model_dump() for r in parsed_response.result] if parsed_response.result else [],
            agent_thoughts=parsed_response.thinking,
            answer=parsed_response.answer,
            response_time_seconds=response_time,
            confidence_score=confidence_score,
            session_id=session_id,
            chat_id=resolved_chat_id,
            title=chat_title,
            message_count=user_message_count,
            response_id=response_id,
            model=resolved_model,
            tokens=response_tokens,
            chat_tokens=chat_tokens,
        )

    @staticmethod
    def _redact_sensitive_text(text: str) -> str:
        text = _RUN_ID_LINE_PATTERN.sub("", text)
        text = _HEX_ID_PATTERN.sub("", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    @classmethod
    def _sanitize_answer(cls, answer: str) -> str:
        answer = cls._strip_model_artifacts(answer)
        answer = _SOURCE_TAG_PATTERN.sub("", answer)
        answer = _RUN_ID_LINE_PATTERN.sub("", answer)
        answer = _HEX_ID_PATTERN.sub("", answer)
        answer = re.sub(r"\s{2,}", " ", answer)
        answer = re.sub(r"\s+([.,;])", r"\1", answer)
        return answer.strip()

    def _search_pool(self, query: str) -> list[MilvusHit]:
        if not self._search_fn:
            return []
        try:
            raw = self._search_fn(query, limit=_SEARCH_POOL_SIZE)
        except TypeError:
            raw = self._search_fn(query)
        redacted = self.redact_raw_search_hits(raw)
        return self._hits_from_raw_search(redacted)

    def _force_search(self, query: str) -> list[MilvusHit]:
        return self._search_pool(query)

    @staticmethod
    def _is_structured_pipeline_question(message: str) -> bool:
        return (
            ChatService._is_metrics_question(message)
            or ChatService._is_preprocessing_question(message)
        )

    @staticmethod
    def _search_queries_for_message(message: str) -> list[str]:
        if ChatService._is_metrics_question(message):
            return [
                "modelos treinados métricas RMSE MAE R² MAPE validação teste XGBoost",
                "resumo métricas de avaliação do treinamento MLflow",
                message,
            ]
        if ChatService._is_preprocessing_question(message):
            return [
                "pré-processamento Target Encoding StandardScaler normalização gold",
                "limpeza de dados silver duplicatas outliers consumo inválido",
                message,
            ]
        return [message]

    def _topic_search(self, message: str) -> list[MilvusHit]:
        merged: dict[str, MilvusHit] = {}
        for query in self._search_queries_for_message(message):
            for hit in self._search_pool(query):
                merged[hit.id] = hit
        ranked = self._rerank_hits(message, list(merged.values()))
        return self._dedupe_hits(ranked)[:_MAX_RETURNED_HITS]

    @staticmethod
    def _dedupe_hits(hits: list[MilvusHit]) -> list[MilvusHit]:
        seen: set[str] = set()
        unique: list[MilvusHit] = []
        for hit in hits:
            signature = re.sub(r"\s+", " ", hit.text[:160].lower())
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(hit)
        return unique

    @classmethod
    def _is_low_value_hit(cls, hit: MilvusHit, message: str) -> bool:
        text = hit.text.strip().lower()
        if re.search(r"processado em:\s*\d{4}", text):
            if "limpeza" not in text and "target encoding" not in text:
                return True
        if cls._is_metrics_question(message):
            if "train_duration" in text or "duração do treinamento em segundos" in text:
                return True
            if "hiperparâmetro" in text and "métricas de avaliação" not in text:
                return True
        if cls._is_preprocessing_question(message):
            if "variável alvo" in text and "target encoding" not in text:
                return True
            if hit.source == "mlflow_metadata" and not any(
                token in text for token in ("target encoding", "standardscaler", "limpeza", "pré-processamento")
            ):
                return True
        return False

    def _rerank_hits(self, message: str, hits: list[MilvusHit]) -> list[MilvusHit]:
        def score(hit: MilvusHit) -> float:
            value = 1.0 - float(hit.distance or 0)
            text = hit.text.lower()
            if self._is_metrics_question(message):
                if "modelos treinados e métricas" in text:
                    value += 4.0
                if "métricas de avaliação do treinamento" in text and "- " in hit.text:
                    value += 3.0
                for token in ("rmse", "mae", "r²", "r2", "mape"):
                    if token in text:
                        value += 0.5
                if "train_duration" in text or "duração do treinamento" in text:
                    value -= 4.0
                if "hiperparâmetro" in text or "learning_rate" in text:
                    value -= 2.0
            if self._is_preprocessing_question(message):
                if "pré-processamento pós-split" in text or "pré-processamento na camada gold" in text:
                    value += 3.0
                if "target encoding" in text:
                    value += 2.0
                if "limpeza de dados" in text:
                    value += 2.5
                if "standardscaler" in text or "normalização" in text:
                    value += 1.5
                if "processado em:" in text and "target encoding" not in text:
                    value -= 4.0
                if "variável alvo" in text:
                    value -= 2.5
            return value

        filtered = [hit for hit in hits if not self._is_low_value_hit(hit, message)]
        return sorted(filtered, key=score, reverse=True)

    def _build_structured_answer(self, message: str, hits: list[MilvusHit]) -> str:
        if self._is_metrics_question(message):
            return self._build_metrics_answer(hits)
        if self._is_preprocessing_question(message):
            return self._build_preprocessing_answer(hits)
        return self._fallback_answer_from_hits(hits, message)

    @classmethod
    def _detect_model_name(cls, hits: list[MilvusHit]) -> str:
        for hit in hits:
            if "xgboost" in hit.text.lower():
                return "XGBoost"
        return "modelo"

    @classmethod
    def _build_metrics_answer(cls, hits: list[MilvusHit]) -> str:
        model = cls._detect_model_name(hits)
        for hit in hits:
            if "métricas de avaliação do treinamento" not in hit.text.lower():
                continue
            metrics: list[str] = []
            for line in hit.text.splitlines():
                stripped = line.strip()
                if not stripped.startswith("- "):
                    continue
                entry = stripped[2:].strip().rstrip(".")
                lower = entry.lower()
                if "duração" in lower or "train_duration" in lower:
                    continue
                metrics.append(entry)
            if metrics:
                return f"O modelo treinado foi {model}. Métricas: {'; '.join(metrics[:6])}."

        metrics = cls._extract_metrics_from_hits(hits)
        metrics = [
            metric for metric in metrics
            if "duração" not in metric.lower() and "train_duration" not in metric.lower()
        ]
        if metrics:
            return f"O modelo treinado foi {model}. Métricas: {'; '.join(metrics[:6])}."
        return _EMPTY_SEARCH_ANSWER

    @classmethod
    def _build_preprocessing_answer(cls, hits: list[MilvusHit]) -> str:
        sentences: list[str] = []

        for hit in hits:
            lower = hit.text.lower()
            if "limpeza de dados" not in lower and "limpeza de dados na camada silver" not in lower:
                continue
            cleaning = cls._summarize_cleaning(hit.text)
            if cleaning:
                sentences.append(f"Na Silver, {cleaning}")
                break

        gold_bits: list[str] = []
        for hit in hits:
            lower = hit.text.lower()
            if "target encoding" in lower and "target encoding" not in " ".join(gold_bits).lower():
                gold_bits.append(
                    "Target Encoding em city, purchase_area e net_manager "
                    "(médias calculadas apenas no treino)"
                )
            if ("standardscaler" in lower or "normalização" in lower) and "standardscaler" not in " ".join(gold_bits).lower():
                gold_bits.append(
                    "normalização com StandardScaler (fit no treino, transform em todos os conjuntos)"
                )
        if gold_bits:
            sentences.append(f"No Gold, {', e '.join(gold_bits)}.")

        if sentences:
            return ". ".join(sentence.rstrip(".") for sentence in sentences) + "."

        snippets = cls._extract_preprocessing_from_hits(hits)
        if snippets:
            return ". ".join(snippets[:4])[:400]
        return _EMPTY_SEARCH_ANSWER

    @staticmethod
    def _summarize_cleaning(text: str) -> str:
        rules: list[str] = []
        for line in text.splitlines():
            stripped = line.strip().lstrip("|").strip()
            if not stripped or stripped.startswith("---") or stripped.startswith("Operação"):
                continue
            lower = stripped.lower()
            if "duplicat" in lower:
                rules.append("duplicatas exatas")
            elif "consumo anual inválido" in lower or "consumo anual invalido" in lower:
                rules.append("consumo anual inválido (nulo ou ≤ 0)")
            elif "conex" in lower and "inválid" in lower:
                rules.append("conexões inválidas")
            elif "outlier" in lower:
                rules.append("outliers de consumo anual acima do percentil 99,5")
        if rules:
            return "a limpeza removeu " + ", ".join(dict.fromkeys(rules))
        if "limpeza" in text.lower():
            return "foram aplicadas regras de limpeza de duplicatas, consumo inválido e outliers"
        return ""

    @staticmethod
    def _format_hits_for_context(hits: list[MilvusHit]) -> str:
        return "\n\n---\n\n".join(hit.text for hit in hits if hit.text)

    @staticmethod
    def _is_identity_question(message: str) -> bool:
        return bool(_IDENTITY_PATTERN.search(message.strip()))

    @staticmethod
    def _is_metrics_question(message: str) -> bool:
        return bool(_METRICS_QUESTION_PATTERN.search(message))

    @staticmethod
    def _is_preprocessing_question(message: str) -> bool:
        return bool(_PREPROCESSING_QUESTION_PATTERN.search(message))

    @classmethod
    def _extract_metrics_from_hits(cls, hits: list[MilvusHit]) -> list[str]:
        metrics: list[str] = []
        seen: set[str] = set()
        metric_line = re.compile(
            r"^[-•]?\s*((?:RMSE|MAE|R²|MAPE|R2)[^:]{0,80}):\s*([\d.]+)",
            re.IGNORECASE | re.MULTILINE,
        )
        inline_metric = re.compile(
            r"((?:RMSE|MAE|R²|MAPE|R2)[^:\n]{0,80}):\s*([\d.]+)",
            re.IGNORECASE,
        )
        for hit in hits:
            for match in metric_line.finditer(hit.text):
                label = re.sub(r"\s+", " ", match.group(1).strip())
                value = match.group(2).strip()
                entry = f"{label}: {value}"
                key = entry.lower()
                if "duração" in key or "train_duration" in key:
                    continue
                if key not in seen:
                    seen.add(key)
                    metrics.append(entry)
            for match in inline_metric.finditer(hit.text):
                label = re.sub(r"\s+", " ", match.group(1).strip())
                value = match.group(2).strip()
                entry = f"{label}: {value}"
                key = entry.lower()
                if "duração" in key or "train_duration" in key:
                    continue
                if key not in seen:
                    seen.add(key)
                    metrics.append(entry)
        return metrics

    @classmethod
    def _extract_preprocessing_from_hits(cls, hits: list[MilvusHit]) -> list[str]:
        snippets: list[str] = []
        keywords = (
            "limpeza", "target encoding", "standardscaler", "normalização",
            "pré-processamento", "duplicat", "outlier", "removid",
        )
        for hit in hits:
            lower = hit.text.lower()
            if not any(token in lower for token in keywords):
                continue
            for line in hit.text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("|---"):
                    continue
                line_lower = stripped.lower()
                if any(token in line_lower for token in keywords):
                    if stripped not in snippets:
                        snippets.append(stripped.lstrip("-• ").strip())
        return snippets[:6]

    @classmethod
    def _fallback_answer_from_hits(cls, hits: list[MilvusHit], message: str = "") -> str:
        if cls._is_metrics_question(message):
            metrics = cls._extract_metrics_from_hits(hits)
            if metrics:
                model = "XGBoost"
                for hit in hits:
                    if "xgboost" in hit.text.lower():
                        model = "XGBoost"
                        break
                return (
                    f"O modelo treinado foi {model}. "
                    f"Métricas: {'; '.join(metrics[:4])}."
                )

        if cls._is_preprocessing_question(message):
            snippets = cls._extract_preprocessing_from_hits(hits)
            if snippets:
                return " ".join(snippets[:4])[:400]

        for hit in hits:
            text = hit.text.lower()
            if "rmse" in text or "erro quadrático" in text:
                for line in hit.text.splitlines():
                    if "rmse" in line.lower() or "erro quadrático" in line.lower():
                        value = line.split(":", 1)[-1].strip().rstrip(".")
                        if value:
                            return f"O modelo treinado é XGBoost e o RMSE na validação foi {value}."
        for hit in hits:
            lower = hit.text.lower()
            if any(
                token in lower
                for token in ("limpeza", "limpos", "removido", "registros finais", "após limpeza")
            ):
                lines = [
                    line.strip()
                    for line in hit.text.splitlines()
                    if line.strip() and not line.strip().startswith("|---")
                ]
                if lines:
                    return " ".join(lines[:5])[:400]
        snippet = hits[0].text.splitlines()[-1] if hits else ""
        return snippet[:280] if snippet else _EMPTY_SEARCH_ANSWER

    @classmethod
    def _refine_answer_for_question(
        cls,
        message: str,
        answer: str,
        hits: list[MilvusHit],
    ) -> str:
        if cls._is_identity_question(message):
            return _IDENTITY_ANSWER

        if _EMPTY_SEARCH_PHRASE in answer:
            substantive = answer.split(_EMPTY_SEARCH_PHRASE)[0].strip().rstrip(".")
            if substantive and len(substantive) > 40:
                answer = substantive + "."

        if cls._is_preprocessing_question(message):
            if cls._mentions_model_metrics(answer) and hits:
                fallback = cls._fallback_answer_from_hits(hits, message)
                if fallback != _EMPTY_SEARCH_ANSWER:
                    return cls._sanitize_answer(fallback)
            answer = re.sub(
                r"\s*(?:O modelo|O desempenho|Atingiu|Utilizou|foi treinado).*",
                "",
                answer,
                flags=re.IGNORECASE,
            ).strip()

        if cls._is_metrics_question(message) and hits:
            structured = cls._build_metrics_answer(hits)
            if structured != _EMPTY_SEARCH_ANSWER:
                return structured

        if cls._is_preprocessing_question(message) and hits:
            structured = cls._build_preprocessing_answer(hits)
            if structured != _EMPTY_SEARCH_ANSWER:
                return structured

        return cls._sanitize_answer(answer)

    @staticmethod
    def _mentions_model_metrics(answer: str) -> bool:
        lower = answer.lower()
        return any(token in lower for token in ("rmse", "mae", "mape", "r²", "r2", "xgboost"))

    @classmethod
    def _is_bogus_answer(cls, answer: str) -> bool:
        cleaned = cls._strip_model_artifacts(answer).strip()
        if not cleaned:
            return True
        if cleaned.lower() in _BOGUS_TOOL_ANSWERS:
            return True
        lower = cleaned.lower()
        if any(f"<{tag}>" in lower for tag in _QWEN_THINKING_TAGS):
            return True
        if "chame a ferramenta" in lower and "search" in lower:
            return True
        if "minha primeira ação" in lower or "query será" in lower:
            return True
        return False

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

    @staticmethod
    def _is_internal_human_message(content: str) -> bool:
        return (
            content.startswith(_SEARCH_RETRY_NUDGE_PREFIX)
            or content.startswith(_FORCED_SEARCH_PREFIX)
        )

    def _get_current_turn_messages(self, messages: List[Any]) -> List[Any]:
        last_human_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                if isinstance(msg, HumanMessage) and self._is_internal_human_message(
                    str(msg.content)
                ):
                    continue
                last_human_idx = i
        if last_human_idx < 0:
            return messages
        return self._strip_search_retry_nudge(messages[last_human_idx:])

    @staticmethod
    def _requires_search(message: str) -> bool:
        return bool(_PIPELINE_SEARCH_PATTERN.search(message))

    def _turn_used_search(self, messages: List[Any]) -> bool:
        for msg in self._get_current_turn_messages(messages):
            if isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
                return True
            if isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
                if getattr(msg, "tool_calls", None):
                    return True
        return False

    def _should_retry_search(
        self,
        message: str,
        messages: List[Any],
        parsed: ParsedResponse,
    ) -> bool:
        if not self._requires_search(message):
            return False
        if self._turn_used_search(messages) and parsed.result:
            return False
        if self._turn_used_search(messages) and not parsed.result:
            return True
        answer = parsed.answer.strip().lower()
        if answer in _BOGUS_TOOL_ANSWERS:
            return True
        return not self._turn_used_search(messages)

    @classmethod
    def _strip_search_retry_nudge(cls, messages: List[Any]) -> List[Any]:
        return [
            msg
            for msg in messages
            if not (
                isinstance(msg, HumanMessage)
                and cls._is_internal_human_message(str(msg.content))
            )
        ]

    @classmethod
    def _compact_chat_history(cls, messages: List[Any]) -> List[Message]:
        """Mantém só diálogo user/assistant; descarta tool calls e chunks RAG antigos."""
        compact: List[Message] = []
        for msg in messages:
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human":
                if isinstance(msg, HumanMessage) and cls._is_internal_human_message(
                    str(msg.content)
                ):
                    continue
                compact.append(HumanMessage(content=str(msg.content)))
            elif isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai":
                tool_calls = getattr(msg, "tool_calls", None) or []
                content = cls._sanitize_answer(
                    cls._strip_model_artifacts(cls._message_text_content(msg))
                )
                if tool_calls and not content:
                    continue
                if content:
                    compact.append(AIMessage(content=content))
        return compact

    @staticmethod
    def _count_user_messages(messages: List[Any]) -> int:
        return sum(
            1
            for msg in messages
            if isinstance(msg, HumanMessage) or getattr(msg, "type", "") == "human"
        )

    @staticmethod
    def _message_text_content(msg: Any) -> str:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts)
        return str(content) if content else ""

    @classmethod
    def _strip_model_artifacts(cls, text: str) -> str:
        text = _GEMMA_THOUGHT_PATTERN.sub("", text)
        for tag in _QWEN_THINKING_TAGS:
            text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)
            text = re.sub(rf"<{tag}>.*", "", text, flags=re.DOTALL)
        text = _QWEN_TOOL_CALL_PATTERN.sub("", text)
        text = _GEMMA_TOOL_CALL_PATTERN.sub("", text)
        return text.strip()

    @classmethod
    def _extract_inline_thinking(cls, content: str) -> list[str]:
        blocks: list[str] = []
        for match in re.findall(
            r"<\|channel>thought\n(.*?)(?:\n\n|\Z)",
            content,
            re.DOTALL,
        ):
            if match.strip():
                blocks.append(match.strip())
        for tag in _QWEN_THINKING_TAGS:
            for match in re.findall(
                rf"<{tag}>(.*?)</{tag}>",
                content,
                re.DOTALL,
            ):
                if match.strip():
                    blocks.append(match.strip())
        return blocks

    def _extract_thinking(self, messages: List[Any]) -> str:
        think_blocks: list[str] = []
        for msg in messages:
            if not (isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai"):
                continue
            reasoning = (getattr(msg, "additional_kwargs", None) or {}).get(
                "reasoning_content"
            )
            if isinstance(reasoning, str) and reasoning.strip():
                think_blocks.append(reasoning.strip())
            think_blocks.extend(
                self._extract_inline_thinking(self._message_text_content(msg))
            )
        return "\n".join(think_blocks) if think_blocks else ""

    def _get_final_answer_text(self, messages: List[Any]) -> str:
        answer = ""
        for msg in messages:
            if not (isinstance(msg, AIMessage) or getattr(msg, "type", "") == "ai"):
                continue
            content = self._strip_model_artifacts(self._message_text_content(msg))
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls and not content:
                continue
            if content:
                answer = content
        return answer

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
            text=self._redact_sensitive_text(str(text)),
            source=source,
            collection=self._infer_collection(source),
        )

    @classmethod
    def redact_raw_search_hits(cls, raw: Any) -> Any:
        """Remove IDs sensíveis dos hits antes de expor ao agente."""
        if isinstance(raw, list):
            return [cls.redact_raw_search_hits(item) for item in raw]
        if isinstance(raw, dict):
            redacted = dict(raw)
            if "text" in redacted and isinstance(redacted["text"], str):
                redacted["text"] = cls._redact_sensitive_text(redacted["text"])
            entity = redacted.get("entity")
            if isinstance(entity, dict) and isinstance(entity.get("text"), str):
                redacted["entity"] = {
                    **entity,
                    "text": cls._redact_sensitive_text(entity["text"]),
                }
            return redacted
        return raw

    def _hits_from_raw_search(self, raw: Any) -> list[MilvusHit]:
        return self.hits_from_search(raw)

    @classmethod
    def hits_from_search(cls, raw: Any) -> list[MilvusHit]:
        hits: list[MilvusHit] = []
        service = cls(tools=[])
        for item in service._flatten_hits(raw if isinstance(raw, list) else [raw]):
            parsed = service._dict_to_milvus_hit(item)
            if parsed:
                hits.append(parsed)
        return hits

    @staticmethod
    def _parse_tool_content(content: Any) -> list:
        if not content:
            return []
        if isinstance(content, list):
            return content
        if isinstance(content, dict):
            return [content]
        if not isinstance(content, str):
            content = str(content)

        import json
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            import ast
            try:
                parsed = ast.literal_eval(content)
            except Exception:
                logger.warning("[CHAT] Falha ao interpretar conteúdo da tool: %s", content[:200])
                return []
        return parsed if isinstance(parsed, list) else [parsed]

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
                    logger.warning("Error parsing tool message: %s", e)
        return search_results

    def _parse_agent_output(self, result: dict) -> ParsedResponse:
        messages = result.get("messages", [])
        current_turn_msgs = self._get_current_turn_messages(messages)

        thinking = self._extract_thinking(current_turn_msgs)
        search_results = self._extract_search_results(current_turn_msgs)
        answer = self._sanitize_answer(self._get_final_answer_text(current_turn_msgs))

        return ParsedResponse(
            thinking=thinking,
            answer=answer,
            result=search_results
        )
