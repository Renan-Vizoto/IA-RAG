"""
Governance Indexer Worker
Lê os documentos de governança Markdown gerados pelo pipeline Dutch Energy,
os relatórios do MLflow e os metadados de runs de treinamento, vetoriza e
indexa tudo na collection governance do Milvus.

Documentos indexados:
  - silver/dutch-energy/governance_silver.md  → processamento e FE do Silver
  - gold/dutch-energy/governance_gold.md      → split, modelo e métricas do Gold
  - gold/dutch-energy/mlflow_report.md        → relatório detalhado do treinamento MLflow
  - runs MLflow (PostgreSQL)                  → overview, métricas e hiperparâmetros

Dispara re-indexação quando os documentos ou os runs MLflow mudarem.
"""
import hashlib
import json
import logging
import re
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.core.interfaces.embbeding import EmbeddingStrategy
from app.infrastructure.clients.postgres_client import MLflowSearchClient
from app.core.services.mlflow_metadata_formatter import (
    format_run_chunks,
    pick_best_rmse_run_id,
)
from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)

_DOCS = [
    ("silver", "dutch-energy/governance_silver.md", "silver_governance"),
    ("gold",   "dutch-energy/governance_gold.md",   "gold_governance"),
    ("gold",   "dutch-energy/mlflow_report.md",     "mlflow_report"),
]

_MAX_CHUNK = 750
_MAX_TEXT_LEN = 1020  # limite do campo varchar no Milvus
_MAX_MLFLOW_RUNS = 20
_MIN_CHUNK_LEN = 80

_SECTION_KEYWORDS = {
    "limpeza de dados": "Limpeza de dados na camada Silver: duplicatas, consumo inválido, outliers.",
    "pré-processamento": "Pré-processamento na camada Gold: Target Encoding e StandardScaler.",
    "feature engineering": "Feature engineering na camada Silver.",
    "divisão dos dados": "Divisão treino, validação e teste na camada Gold.",
}


class GovernanceIndexer:

    def __init__(
        self,
        storage: StorageBackend,
        client: MLflowSearchClient,
        repo: MilvusRepo,
        embedder: EmbeddingStrategy,
        schema_builder,
        collection: str,
    ):
        self._storage = storage
        self._client = client
        self._repo = repo
        self._embedder = embedder
        self._schema_builder = schema_builder
        self._collection = collection
        self._last_indexed: str | None = None

    async def run(self):
        try:
            fingerprint = self._index_fingerprint()
            if fingerprint and fingerprint == self._last_indexed:
                logger.debug("[GOVERNANCE] Documentos e runs MLflow sem mudanças, indexação ignorada.")
                return

            logger.info("[GOVERNANCE] Iniciando indexação dos documentos de governança e MLflow...")
            chunks, sources = self._collect_chunks()

            if not chunks:
                logger.warning("[GOVERNANCE] Nenhum documento encontrado para indexar.")
                return

            embeddings = self._embedder.embbed_it(chunks)
            data = [
                {
                    "text": _cap_text(text),
                    "text_vector": embeddings[i].tolist(),
                    "source": sources[i],
                }
                for i, text in enumerate(chunks)
            ]

            self._repo.replace_all(self._collection, self._schema_builder, data)

            self._last_indexed = fingerprint
            logger.info(
                f"[GOVERNANCE] Indexados {len(data)} chunks na collection '{self._collection}'."
            )

        except Exception as e:
            logger.error(f"[GOVERNANCE] Erro na indexação: {e}", exc_info=True)

    def _index_fingerprint(self) -> str | None:
        doc_ts = self._doc_timestamp()
        runs = self._client.search_training_runs(max_results=_MAX_MLFLOW_RUNS)
        runs_fp = _runs_fingerprint(runs)
        if doc_ts:
            return f"{doc_ts}|{runs_fp}"
        return runs_fp if runs else None

    def _doc_timestamp(self) -> str | None:
        """Retorna o timestamp de modificação do governance_gold.md como proxy de mudança."""
        try:
            raw = self._storage.get_object("gold", "dutch-energy/governance_gold.md")
            content = raw.read().decode("utf-8")
            for line in content.splitlines():
                if line.startswith("## Processado em:"):
                    return line.split(":", 1)[1].strip()
            return datetime.now(timezone.utc).isoformat()
        except Exception:
            return None

    def _collect_chunks(self) -> tuple[list[str], list[str]]:
        """Lê documentos Markdown e runs MLflow, retornando (chunks, sources)."""
        all_chunks, all_sources = [], []

        for bucket, obj_path, source_label in _DOCS:
            try:
                raw = self._storage.get_object(bucket, obj_path)
                text = raw.read().decode("utf-8")
                for chunk in _split_markdown(text, _MAX_CHUNK, source_label):
                    all_chunks.append(chunk)
                    all_sources.append(source_label)
                logger.info(f"[GOVERNANCE] Lido: {bucket}/{obj_path}")
            except Exception as e:
                logger.warning(f"[GOVERNANCE] Documento não encontrado ({bucket}/{obj_path}): {e}")

        mlflow_chunks, mlflow_sources = self._collect_mlflow_chunks()
        all_chunks.extend(mlflow_chunks)
        all_sources.extend(mlflow_sources)

        return all_chunks, all_sources

    def _collect_mlflow_chunks(self) -> tuple[list[str], list[str]]:
        runs = self._client.search_training_runs(max_results=_MAX_MLFLOW_RUNS)
        if not runs:
            return [], []

        best_run_id = pick_best_rmse_run_id(runs)
        chunks, sources = [], []
        for run in runs:
            for chunk in format_run_chunks(run, is_best=(run.get("run_id") == best_run_id)):
                chunks.append(chunk["text"])
                sources.append(chunk["source"])
        logger.info(f"[GOVERNANCE] Indexando {len(runs)} runs MLflow ({len(chunks)} chunks).")
        return chunks, sources


def _runs_fingerprint(runs: list[dict]) -> str:
    payload = [{k: v for k, v in sorted(run.items())} for run in runs]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _is_low_value_chunk(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < _MIN_CHUNK_LEN:
        return True
    if "[Aguardando treinamento" in stripped:
        return True
    if re.fullmatch(r"#+ [^\n]+", stripped):
        return True
    if re.search(r"processado em:\s*\d{4}", stripped, re.IGNORECASE):
        body = re.sub(r".*processado em:[^\n-]*", "", stripped, flags=re.IGNORECASE).strip(" -")
        if len(body) < 50:
            return True
    return False


def _document_title(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _section_title(section: str) -> str:
    for line in section.splitlines():
        if re.match(r"#{2,3}\s", line):
            return re.sub(r"^#+\s*", "", line).strip()
    return ""


def _cap_text(text: str, max_len: int = _MAX_TEXT_LEN) -> str:
    stripped = text.strip()
    if len(stripped) <= max_len:
        return stripped
    cut = stripped.rfind(" ", 0, max_len)
    if cut == -1:
        cut = max_len
    return stripped[:cut].strip()


def _section_keywords(section_title: str) -> str | None:
    lower = section_title.lower()
    for key, keywords in _SECTION_KEYWORDS.items():
        if key in lower:
            return keywords
    return None


def _enrich_chunk(text: str, source_label: str, doc_title: str, section_title: str) -> str:
    parts: list[str] = []
    section_kw = _section_keywords(section_title)
    if section_kw:
        parts.append(section_kw)
    elif source_label == "mlflow_metadata":
        parts.append("Modelos treinados e métricas MLflow do pipeline Dutch Energy.")
    if section_title:
        parts.append(section_title + ".")
    parts.append(text.strip())
    return _cap_text("\n".join(parts))


def _split_markdown(text: str, max_len: int, source_label: str = "") -> list[str]:
    """Divide markdown por seções (##) e depois por tamanho, preservando contexto semântico."""
    doc_title = _document_title(text)
    sections: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"#{2,3}\s", line) and current:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    chunks: list[str] = []
    for section in sections:
        section_title = _section_title(section)
        for piece in _split(section, max_len):
            enriched = _enrich_chunk(piece, source_label, doc_title, section_title)
            if not _is_low_value_chunk(enriched):
                chunks.append(enriched)
    return chunks


def _split(text: str, max_len: int) -> list[str]:
    """Divide texto em chunks de no máximo max_len caracteres, quebrando em espaços."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        cut = text.rfind(" ", 0, max_len)
        if cut == -1:
            cut = max_len
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return chunks


def start_worker(
    storage: StorageBackend,
    client: MLflowSearchClient,
    repo: MilvusRepo,
    embedder: EmbeddingStrategy,
    schema_builder,
    collection: str,
):
    indexer = GovernanceIndexer(storage, client, repo, embedder, schema_builder, collection)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(indexer.run, "interval", minutes=10)
    scheduler.start()
    logger.info("[GOVERNANCE] Indexador agendado a cada 10 minutos.")
    return indexer
