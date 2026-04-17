"""
Governance Indexer Worker
Lê os artefatos JSON gerados pelo pipeline Dutch Energy (bronze/silver/gold),
converte para texto legível, embeda e indexa no Milvus.

Artefatos indexados:
  - bronze/dutch-energy/manifest.json        → ingestão (quais arquivos, quando)
  - silver/dutch-energy/cleaning_stats.json  → limpeza (duplicatas, outliers, etc.)
  - gold/dutch-energy/gold_metadata.json     → features, split, scaler, encoders
  - gold/dutch-energy/feature_cols.json      → lista exata de features do modelo

Dispara a re-indexação quando o artefato gold mais recente for mais novo
que o ultimo timestamp indexado (evita re-indexar sem mudanças).
"""
import json
import logging
from io import BytesIO
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.infrastructure.repositories.milvus_repo import MilvusRepo
from app.core.interfaces.embbeding import EmbeddingStrategy
from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)

# Localização dos artefatos no MinIO
_ARTIFACTS = [
    ("bronze", "dutch-energy/manifest.json",       "bronze_manifest"),
    ("silver", "dutch-energy/cleaning_stats.json", "silver_cleaning"),
    ("gold",   "dutch-energy/gold_metadata.json",  "gold_metadata"),
    ("gold",   "dutch-energy/feature_cols.json",   "gold_features"),
]

_MAX_CHUNK = 900   # chars (Milvus max_length=1024, margem de segurança)


class GovernanceIndexer:

    def __init__(
        self,
        storage: StorageBackend,
        repo: MilvusRepo,
        embedder: EmbeddingStrategy,
        schema_builder,
        collection: str,
    ):
        self._storage = storage
        self._repo = repo
        self._embedder = embedder
        self._schema_builder = schema_builder
        self._collection = collection
        self._last_indexed: str | None = None   # ISO timestamp do ultimo gold_metadata

    async def run(self):
        try:
            current_ts = self._artifact_timestamp()
            if current_ts and current_ts == self._last_indexed:
                logger.debug("[GOVERNANCE] Artefatos sem mudancas, indexacao ignorada.")
                return

            logger.info("[GOVERNANCE] Iniciando indexacao dos artefatos de governanca...")
            chunks, sources = self._collect_chunks()

            if not chunks:
                logger.warning("[GOVERNANCE] Nenhum artefato encontrado para indexar.")
                return

            embeddings = self._embedder.embbed_it(chunks)
            data = [
                {"text": text, "text_vector": embeddings[i].tolist(), "source": sources[i]}
                for i, text in enumerate(chunks)
            ]

            self._repo.drop_and_recreate(self._collection, self._schema_builder)
            self._repo.insert(self._collection, data)

            self._last_indexed = current_ts
            logger.info(
                f"[GOVERNANCE] Indexados {len(data)} chunks na collection '{self._collection}'."
            )

        except Exception as e:
            logger.error(f"[GOVERNANCE] Erro na indexacao: {e}", exc_info=True)

    # ──────────────────────────────────────────────
    # Internos
    # ──────────────────────────────────────────────

    def _artifact_timestamp(self) -> str | None:
        """Retorna o 'built_at' do gold_metadata (proxy de quando o pipeline rodou)."""
        try:
            raw = self._storage.get_object("gold", "dutch-energy/gold_metadata.json")
            meta = json.loads(raw.read().decode("utf-8"))
            return meta.get("built_at")
        except Exception:
            return None

    def _collect_chunks(self) -> tuple[list[str], list[str]]:
        """Lê todos os artefatos e retorna (chunks_texto, sources)."""
        all_chunks, all_sources = [], []

        for bucket, obj_path, source_label in _ARTIFACTS:
            try:
                raw = self._storage.get_object(bucket, obj_path)
                payload = json.loads(raw.read().decode("utf-8"))
                text = _artifact_to_text(source_label, payload)
                for chunk in _split(text, _MAX_CHUNK):
                    all_chunks.append(chunk)
                    all_sources.append(source_label)
                logger.info(f"[GOVERNANCE] Lido: {bucket}/{obj_path}")
            except Exception as e:
                logger.warning(f"[GOVERNANCE] Artefato nao encontrado ({bucket}/{obj_path}): {e}")

        return all_chunks, all_sources


# ──────────────────────────────────────────────
# Conversão JSON → texto legível
# ──────────────────────────────────────────────

def _artifact_to_text(source: str, payload: dict) -> str:
    if source == "bronze_manifest":
        return _manifest_to_text(payload)
    if source == "silver_cleaning":
        return _cleaning_to_text(payload)
    if source == "gold_metadata":
        return _gold_meta_to_text(payload)
    if source == "gold_features":
        # payload é uma lista de strings
        if isinstance(payload, list):
            return f"Features do modelo Dutch Energy ({len(payload)} no total): {', '.join(payload)}."
        return str(payload)
    return json.dumps(payload, ensure_ascii=False)


def _manifest_to_text(m: dict) -> str:
    ts = m.get("ingested_at", "desconhecido")
    total = m.get("total_files", 0)
    files = m.get("files", [])
    names = [f.get("object_name", "?").split("/")[-1] for f in files]
    sizes = [f"{f.get('size_bytes', 0):,} bytes" for f in files]
    detail = "; ".join(f"{n} ({s})" for n, s in zip(names, sizes))
    return (
        f"Ingestao Bronze do dataset Dutch Energy realizada em {ts}. "
        f"{total} arquivo(s) ingerido(s): {detail}."
    )


def _cleaning_to_text(s: dict) -> str:
    initial  = s.get("initial_rows", 0)
    dups     = s.get("removed_duplicates", 0)
    invalid  = s.get("removed_invalid_consume", 0)
    inv_conn = s.get("removed_invalid_connections", 0)
    outliers = s.get("removed_outliers", 0)
    thresh   = s.get("outlier_threshold_kwh", 0)
    final    = s.get("final_rows", 0)
    ts       = s.get("saved_at", "desconhecido")
    removed  = initial - final
    return (
        f"Limpeza Silver do dataset Dutch Energy concluida em {ts}. "
        f"Registros iniciais: {initial:,}. "
        f"Duplicatas removidas: {dups:,}. "
        f"Consumo invalido (nulo ou negativo) removido: {invalid:,}. "
        f"Conexoes invalidas removidas: {inv_conn:,}. "
        f"Outliers removidos: {outliers:,} (threshold: {thresh:,.2f} kWh, percentil 99.5). "
        f"Registros finais apos limpeza: {final:,} "
        f"(total removido: {removed:,})."
    )


def _gold_meta_to_text(m: dict) -> str:
    ts       = m.get("built_at", "desconhecido")
    target   = m.get("target", "?")
    n_feat   = m.get("n_features", 0)
    feats    = ", ".join(m.get("feature_cols", []))
    split    = m.get("split", {})
    tr, val, te = split.get("train", 0), split.get("val", 0), split.get("test", 0)
    ratios   = split.get("ratios", {})
    scaler   = m.get("scaler", "?")
    seed     = m.get("seed", "?")
    cats     = ", ".join(m.get("cat_cols_encoded", []))
    return (
        f"Pipeline Gold do dataset Dutch Energy construido em {ts}. "
        f"Variavel alvo: {target}. "
        f"{n_feat} features selecionadas: {feats}. "
        f"Split (seed={seed}): treino={tr:,} ({ratios.get('train',0)*100:.0f}%), "
        f"validacao={val:,} ({ratios.get('val',0)*100:.0f}%), "
        f"teste={te:,} ({ratios.get('test',0)*100:.0f}%). "
        f"Normalizacao: {scaler} (ajustado apenas no treino). "
        f"Colunas categoricas com label encoding: {cats}."
    )


def _split(text: str, max_len: int) -> list[str]:
    """Divide texto em chunks de no maximo max_len caracteres, quebrando em espacos."""
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
    repo: MilvusRepo,
    embedder: EmbeddingStrategy,
    schema_builder,
    collection: str,
):
    indexer = GovernanceIndexer(storage, repo, embedder, schema_builder, collection)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(indexer.run, "interval", minutes=10)
    scheduler.start()
    logger.info("[GOVERNANCE] Indexador agendado a cada 10 minutos.")
    return indexer
