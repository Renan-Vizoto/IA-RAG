"""
Orquestrador do pipeline Medaliao - Dutch Energy Dataset

Uso:
    python -m app.pipeline.run_dutch_energy_pipeline --data-dir notebooks/data

Layers:
    Bronze   -> armazena CSVs brutos sem transformacao
    Silver   -> limpeza + feature engineering + gera governance_silver.md
    Gold     -> split 70/15/15 + encoding + scaling + gera governance_gold.md
    Training -> XGBoost com MLflow + salva mlflow_report.md + atualiza governance_gold.md
"""
import sys
import argparse
import logging

from app.pipeline.storage import create_storage
from app.pipeline.dutch_energy_bronze import ingest_csvs
from app.pipeline.dutch_energy_silver import transform
from app.pipeline.dutch_energy_gold import build
from app.pipeline.dutch_energy_train import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 55


def run(data_dir: str, use_minio: bool = True, force: bool = False):
    storage = create_storage(use_minio=use_minio, base_dir="data")

    # ── Bronze ──────────────────────────────────────────────
    logger.info(SEPARATOR)
    logger.info("BRONZE - Ingestao dos CSVs brutos")
    logger.info(SEPARATOR)
    ingested = ingest_csvs(data_dir, storage)
    logger.info(f"Arquivos ingeridos/verificados: {len(ingested)}")

    # ── Silver ──────────────────────────────────────────────
    logger.info(SEPARATOR)
    logger.info("SILVER - Limpeza e validacao")
    logger.info(SEPARATOR)
    silver_df = transform(storage, force=force)
    logger.info(f"Dataset em memoria: {len(silver_df):,} registros")

    # ── Gold ─────────────────────────────────────────────────
    logger.info(SEPARATOR)
    logger.info("GOLD - Split 70/15/15 + encoding + normalizacao")
    logger.info(SEPARATOR)
    gold = build(storage, force=force)
    if gold:
        logger.info("Artefatos prontos para treinamento:")
        for name, arr in gold.items():
            shape = arr.shape if hasattr(arr, "shape") else (len(arr),)
            logger.info(f"  {name}: {shape}")
    else:
        logger.info("Gold Layer pulada (ja processada).")

    # ── Training ─────────────────────────────────────────────
    logger.info(SEPARATOR)
    logger.info("TRAINING - XGBoost + MLflow")
    logger.info(SEPARATOR)
    model_info = train(storage)
    logger.info(f"Modelo treinado. Run ID: {model_info.get('run_id')}")
    logger.info(f"Métricas: {model_info.get('metrics')}")

    logger.info(SEPARATOR)
    logger.info("Pipeline Dutch Energy concluido com sucesso!")
    logger.info(SEPARATOR)

    return gold


def _parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Medaliao - Dutch Energy")
    parser.add_argument(
        "--data-dir",
        default="notebooks/data",
        help="Diretorio com os CSVs de eletricidade do Kaggle (default: notebooks/data)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Usa storage local ao inves do MinIO",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força re-processamento das camadas Silver e Gold",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(data_dir=args.data_dir, use_minio=not args.local, force=args.force)
