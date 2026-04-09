import sys
import logging

from app.pipeline.storage import create_storage
from app.pipeline.bronze_ingestion import ingest_pdf
from app.pipeline.silver_extraction import extract_all_tables
from app.pipeline.gold_preprocessing import preprocess_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(pdf_path: str):
    storage = create_storage(use_minio=True)

    # ── Bronze ──
    logger.info("=" * 50)
    logger.info("BRONZE - Ingestao do PDF bruto")
    logger.info("=" * 50)
    object_name = ingest_pdf(pdf_path, storage)
    logger.info(f"PDF armazenado como: {object_name}")

    # ── Silver ──
    logger.info("=" * 50)
    logger.info("SILVER - Extracao de tabelas")
    logger.info("=" * 50)
    silver_tables = extract_all_tables(pdf_path, storage)
    for name, df in silver_tables.items():
        logger.info(f"  {name}: {len(df)} rows x {len(df.columns)} cols")

    # ── Gold ──
    logger.info("=" * 50)
    logger.info("GOLD - Pre-processamento e limpeza")
    logger.info("=" * 50)
    gold_tables = preprocess_all(storage)
    for name, df in gold_tables.items():
        logger.info(f"  {name}: {len(df)} rows x {len(df.columns)} cols")
        logger.info(f"    Colunas: {list(df.columns)}")
        logger.info(f"    Tipos: {dict(df.dtypes)}")

    logger.info("=" * 50)
    logger.info("Pipeline concluido com sucesso!")
    logger.info(f"Tabelas processadas: {len(gold_tables)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m app.pipeline.run_pipeline <caminho_do_pdf>")
        sys.exit(1)

    run(sys.argv[1])
