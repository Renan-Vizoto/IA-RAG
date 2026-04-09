import os
import logging

from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)


def ingest_pdf(pdf_path: str, storage: StorageBackend) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    storage.ensure_bucket("bronze")

    filename = os.path.basename(pdf_path)
    storage.put_file("bronze", filename, pdf_path)

    size = os.path.getsize(pdf_path)
    logger.info(f"[BRONZE] Uploaded '{filename}' ({size:,} bytes)")
    return filename
