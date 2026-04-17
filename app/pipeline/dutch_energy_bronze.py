"""
Bronze Layer - Dutch Energy Dataset
Recebe os CSVs brutos do Kaggle e os armazena sem nenhuma transformacao.
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
    Retorna lista dos object names salvos.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Diretorio nao encontrado: {data_path.resolve()}")

    # *electr* cobre tanto "electric" quanto "electricity" em qualquer nome de arquivo
    elec_files = sorted(set(data_path.glob("*electr*")))

    if not elec_files:
        raise FileNotFoundError(
            f"Nenhum CSV de eletricidade encontrado em {data_path.resolve()}. "
            "Baixe o dataset do Kaggle (dutch-energy) e extraia os CSVs aqui."
        )

    storage.ensure_bucket(BRONZE_BUCKET)

    ingested = []
    for fp in elec_files:
        object_name = f"{BRONZE_PREFIX}{fp.name}"
        storage.put_file(BRONZE_BUCKET, object_name, str(fp))
        size = fp.stat().st_size
        logger.info(f"[BRONZE] {fp.name} -> {object_name} ({size:,} bytes)")
        ingested.append(object_name)

    # Salva manifesto com metadados da ingestao
    manifest = {
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(data_path.resolve()),
        "files": [
            {
                "object_name": obj,
                "source_file": str(elec_files[i]),
                "size_bytes": elec_files[i].stat().st_size,
            }
            for i, obj in enumerate(ingested)
        ],
        "total_files": len(ingested),
    }
    manifest_bytes = BytesIO(json.dumps(manifest, indent=2).encode("utf-8"))
    storage.put_object(BRONZE_BUCKET, f"{BRONZE_PREFIX}manifest.json", manifest_bytes, "application/json")
    logger.info(f"[BRONZE] Manifesto salvo. Total: {len(ingested)} arquivo(s).")

    return ingested
