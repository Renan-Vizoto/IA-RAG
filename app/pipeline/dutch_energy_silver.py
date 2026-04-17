"""
Silver Layer - Dutch Energy Dataset
Le os CSVs brutos do bronze, consolida, limpa e valida os dados.

Transformacoes aplicadas:
  1. Leitura seletiva de colunas (USE_COLS) com dtype otimizado (float32, category)
  2. Extracao do ano a partir do nome do arquivo (por arquivo, durante a leitura)
  3. Type casting de colunas numericas (pd.to_numeric + float32)
  4. Remocao de registros com annual_consume nulo ou <= 0
  5. Remocao de linhas completamente duplicadas
  6. Remocao de outliers extremos (acima do percentil 99.5 de annual_consume)
  7. Validacoes adicionais (num_connections > 0, percentuais em [0,100])
  8. Reset de index e salvamento no silver
"""
import re
import gc
import json
import logging
from io import BytesIO
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)

BRONZE_BUCKET = "bronze"
BRONZE_PREFIX = "dutch-energy/"
SILVER_BUCKET = "silver"
SILVER_PREFIX = "dutch-energy/"

TARGET = "annual_consume"
OUTLIER_QUANTILE = 0.995

# Apenas as colunas que o modelo realmente usa (espelha USE_COLS do notebook)
USE_COLS = [
    "net_manager", "purchase_area", "city", "num_connections",
    "delivery_perc", "perc_of_active_connections", "type_of_connection",
    "type_conn_perc", "annual_consume", "annual_consume_lowtarif_perc",
    "smartmeter_perc",
]

NUM_COLS = [
    "num_connections", "delivery_perc", "perc_of_active_connections",
    "type_conn_perc", "annual_consume", "annual_consume_lowtarif_perc",
    "smartmeter_perc",
]

CAT_COLS = ["net_manager", "purchase_area", "city", "type_of_connection"]

PCT_COLS = [c for c in NUM_COLS if "perc" in c]


def transform(storage: StorageBackend) -> pd.DataFrame:
    """
    Executa o pipeline Silver completo: le do bronze, limpa e salva no silver.
    Retorna o DataFrame limpo.
    """
    raw_df = _load_bronze(storage)
    df, stats = _clean(raw_df)
    del raw_df; gc.collect()
    _save_silver(df, stats, storage)
    return df


# ──────────────────────────────────────────────
# Internos
# ──────────────────────────────────────────────

def _load_bronze(storage: StorageBackend) -> pd.DataFrame:
    """
    Le todos os CSVs de eletricidade do bronze.
    - Carrega apenas USE_COLS (colunas que o modelo usa)
    - Converte numericas para float32 e categoricas para category
    - Extrai o ano do nome do arquivo por arquivo (antes do concat)
    """
    csv_files = storage.list_objects(BRONZE_BUCKET, BRONZE_PREFIX)
    csv_files = [f for f in csv_files if f.endswith(".csv")]

    if not csv_files:
        raise RuntimeError(
            "Nenhum CSV encontrado no bronze. Execute primeiro a camada Bronze."
        )

    frames = []
    for obj_name in sorted(csv_files):
        raw = storage.get_object(BRONZE_BUCKET, obj_name)

        # Le o header para saber quais USE_COLS existem neste arquivo
        header = pd.read_csv(raw, nrows=0).columns.tolist()
        raw.seek(0)
        cols = [c for c in USE_COLS if c in header]

        # Le tudo como string para evitar erros de cast silenciosos
        chunk = pd.read_csv(raw, usecols=cols, dtype=str, low_memory=False)

        # Cast numericos -> float32
        for col in NUM_COLS:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")

        # Cast categoricos -> category (economiza memoria)
        for col in CAT_COLS:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("category")

        # Extrai o ano do nome do arquivo (ex: electricity_2018.csv -> 2018)
        filename = obj_name.split("/")[-1]
        match = re.search(r"(20\d{2})", filename)
        if match:
            chunk["year"] = np.int16(int(match.group(1)))

        n = len(chunk)
        logger.info(f"[SILVER] {filename}: {n:,} registros")
        frames.append(chunk)

    combined = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()

    mem_mb = combined.memory_usage(deep=True).sum() / 1e6
    logger.info(f"[SILVER] Total concatenado: {len(combined):,} registros | {mem_mb:.1f} MB")
    return combined


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica todas as regras de limpeza. Retorna (df_limpo, stats)."""
    stats = {"initial_rows": len(df)}

    # 1. Remove duplicatas exatas
    before = len(df)
    df = df.drop_duplicates().copy()
    stats["removed_duplicates"] = before - len(df)
    logger.info(f"[SILVER] Duplicatas removidas: {stats['removed_duplicates']:,}")

    # 2. Remove registros sem consumo ou consumo invalido
    before = len(df)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)].copy()
    stats["removed_invalid_consume"] = before - len(df)
    logger.info(f"[SILVER] Removidos sem consumo valido: {stats['removed_invalid_consume']:,}")

    # 3. Remove registros com num_connections invalido
    if "num_connections" in df.columns:
        before = len(df)
        df = df[df["num_connections"].notna() & (df["num_connections"] > 0)].copy()
        stats["removed_invalid_connections"] = before - len(df)
        logger.info(f"[SILVER] Removidos com num_connections invalido: {stats['removed_invalid_connections']:,}")

    # 4. Remove percentuais fora do range [0, 100]
    for col in PCT_COLS:
        if col in df.columns:
            before = len(df)
            df = df[df[col].isna() | df[col].between(0, 100)].copy()
            removed = before - len(df)
            if removed:
                logger.info(f"[SILVER] Removidos com {col} fora de [0,100]: {removed:,}")

    # 5. Remove outliers extremos de annual_consume (> p99.5)
    q_high = float(df[TARGET].quantile(OUTLIER_QUANTILE))
    before = len(df)
    df = df[df[TARGET] <= q_high].copy()
    stats["removed_outliers"] = before - len(df)
    stats["outlier_threshold_kwh"] = round(q_high, 2)
    logger.info(
        f"[SILVER] Outliers removidos (>{q_high:.0f} kWh, p{OUTLIER_QUANTILE*100:.1f}): "
        f"{stats['removed_outliers']:,}"
    )

    df = df.reset_index(drop=True)
    stats["final_rows"] = len(df)
    stats["final_cols"] = len(df.columns)

    logger.info(
        f"[SILVER] Dataset limpo: {stats['final_rows']:,} registros | "
        f"removidos no total: {stats['initial_rows'] - stats['final_rows']:,}"
    )
    return df, stats


def _save_silver(df: pd.DataFrame, stats: dict, storage: StorageBackend):
    storage.ensure_bucket(SILVER_BUCKET)

    # Dataset limpo (categoricas como string para CSV)
    df_out = df.copy()
    for col in CAT_COLS:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str)

    csv_buf = BytesIO()
    df_out.to_csv(csv_buf, index=False, encoding="utf-8")
    csv_buf.seek(0)
    storage.put_object(SILVER_BUCKET, f"{SILVER_PREFIX}cleaned.csv", csv_buf, "text/csv")
    logger.info(f"[SILVER] Salvo: {SILVER_PREFIX}cleaned.csv ({len(df):,} rows)")

    # Estatisticas de limpeza
    stats["saved_at"] = datetime.now(timezone.utc).isoformat()
    stats_buf = BytesIO(json.dumps(stats, indent=2).encode("utf-8"))
    storage.put_object(SILVER_BUCKET, f"{SILVER_PREFIX}cleaning_stats.json", stats_buf, "application/json")
    logger.info(f"[SILVER] Salvo: {SILVER_PREFIX}cleaning_stats.json")
