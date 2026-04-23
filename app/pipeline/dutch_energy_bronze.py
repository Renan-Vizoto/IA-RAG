"""
Bronze Layer - Dutch Energy Dataset
Recebe os CSVs brutos do Kaggle e os armazena sem nenhuma transformacao.
Aplica idempotencia usando metadados do MinIO (consumed=true) e versionamento.
"""
import os
import json
import logging
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path

from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)

BRONZE_BUCKET = "bronze"
BRONZE_PREFIX = "dutch-energy/"


def ingest_csvs(data_dir: str, storage: StorageBackend) -> list[str]:
    """
    Varre data_dir em busca de CSVs de eletricidade e os envia raw para o bronze.
    Retorna lista dos object names salvos/já existentes.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {data_path.resolve()}")

    elec_files = sorted(set(data_path.glob("*electr*")))

    if not elec_files:
        raise FileNotFoundError(
            f"Nenhum CSV de eletricidade encontrado em {data_path.resolve()}."
        )

    # Garante bucket com versionamento habilitado
    storage.ensure_bucket(BRONZE_BUCKET, versioning=True)

    ingested = []
    skipped = 0

    for fp in elec_files:
        object_name = f"{BRONZE_PREFIX}{fp.name}"

        # 1. Verifica se ja foi consumido via metadados
        meta = storage.stat_object(BRONZE_BUCKET, object_name)
        if meta.get("consumed") == "true":
            logger.info(f"[BRONZE] Pulo (ja consumido): {fp.name}")
            skipped += 1
            ingested.append(object_name)
            continue

        # 2. Upload com metadado consumed=true
        storage.put_file(
            BRONZE_BUCKET, 
            object_name, 
            str(fp), 
            metadata={"consumed": "true", "ingested_at": datetime.now(timezone.utc).isoformat()}
        )

        size = fp.stat().st_size
        logger.info(f"[BRONZE] Ingerido: {fp.name} -> {object_name} ({size:,} bytes)")
        ingested.append(object_name)

    # Salva manifesto com metadados da ingestao
    manifest = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(data_path.resolve()),
        "files": ingested,
        "total_files": len(ingested),
        "new_files": len(ingested) - skipped,
        "skipped_files": skipped
    }
    manifest_bytes = BytesIO(json.dumps(manifest, indent=2).encode("utf-8"))
    storage.put_object(BRONZE_BUCKET, f"{BRONZE_PREFIX}manifest.json", manifest_bytes, "application/json")

    logger.info(f"[BRONZE] Concluido. Novos: {manifest['new_files']}, Skipped: {skipped}")

    return ingested


