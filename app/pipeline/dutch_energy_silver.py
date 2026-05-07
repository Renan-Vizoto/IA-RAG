"""
Silver Layer - Dutch Energy Dataset
Le os CSVs brutos do bronze, consolida, limpa, aplica feature engineering
e gera o documento de governança da camada.

Transformacoes aplicadas:
  1. Leitura seletiva de colunas (USE_COLS) com dtype otimizado (float32, category)
  2. Extracao do ano a partir do nome do arquivo
  3. Type casting de colunas numericas (pd.to_numeric + float32)
  4. Remocao de registros com annual_consume nulo ou <= 0
  5. Remocao de linhas completamente duplicadas
  6. Remocao de outliers extremos (acima do percentil 99.5 de annual_consume)
  7. Validacoes adicionais (num_connections > 0, percentuais em [0,100])
  8. Feature Engineering:
     - amperage: extraido de type_of_connection (ex: "3x25" -> 75)
     - consume_per_conn: annual_consume / num_connections (target)
     - total_capacity: amperage * num_connections
     - hightarif_perc: 100 - annual_consume_lowtarif_perc
     - log_target: log1p(consume_per_conn)
  9. Geracao de governance_silver.md salvo no MinIO
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

TARGET_RAW = "annual_consume"
TARGET = "consume_per_conn"
OUTLIER_QUANTILE = 0.995

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

_COL_DESCRIPTIONS = {
    "net_manager": "Gestora de rede elétrica",
    "purchase_area": "Área de compra de energia",
    "city": "Cidade",
    "num_connections": "Número de conexões",
    "delivery_perc": "% de entrega de energia",
    "perc_of_active_connections": "% de conexões ativas",
    "type_of_connection": "Tipo de conexão (ex: 3x25)",
    "type_conn_perc": "% do tipo de conexão",
    "annual_consume": "Consumo anual total (kWh)",
    "annual_consume_lowtarif_perc": "% consumo em tarifa baixa",
    "smartmeter_perc": "% de medidores inteligentes",
    "amperage": "Amperagem total extraída de type_of_connection",
    "consume_per_conn": "Consumo por conexão — variável alvo (kWh/conn)",
    "total_capacity": "Capacidade total (amperage × num_connections)",
    "hightarif_perc": "% consumo em tarifa alta (100 - lowtarif_perc)",
    "log_target": "Transformação log1p de consume_per_conn",
    "year": "Ano extraído do nome do arquivo CSV",
}


def transform(storage: StorageBackend, force: bool = False) -> pd.DataFrame:
    """
    Executa o pipeline Silver completo: le do bronze, limpa, aplica FE e salva.
    Retorna o DataFrame processado.
    Aplica idempotencia (pula se ja processado e nao forcado).
    """
    obj_name = f"{SILVER_PREFIX}cleaned.csv"

    if not force:
        meta = storage.stat_object(SILVER_BUCKET, obj_name)
        if meta.get("silver_completed") == "true":
            logger.info("[SILVER] Pulo: Dataset ja processado.")
            raw = storage.get_object(SILVER_BUCKET, obj_name)
            return pd.read_csv(raw, low_memory=False)

    raw_df = _load_bronze(storage)
    df, stats = _clean(raw_df)
    del raw_df; gc.collect()
    df, fe_stats = _feature_engineering(df)
    _save_silver(df, stats, fe_stats, storage)
    return df


# ──────────────────────────────────────────────
# Internos
# ──────────────────────────────────────────────

def _load_bronze(storage: StorageBackend) -> pd.DataFrame:
    csv_files = storage.list_objects(BRONZE_BUCKET, BRONZE_PREFIX)
    csv_files = [f for f in csv_files if f.endswith(".csv")]

    if not csv_files:
        raise RuntimeError(
            "Nenhum CSV encontrado no bronze. Execute primeiro a camada Bronze."
        )

    frames = []
    for obj_name in sorted(csv_files):
        raw = storage.get_object(BRONZE_BUCKET, obj_name)

        header = pd.read_csv(raw, nrows=0).columns.tolist()
        raw.seek(0)
        cols = [c for c in USE_COLS if c in header]

        chunk = pd.read_csv(raw, usecols=cols, dtype=str, low_memory=False)

        for col in NUM_COLS:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")

        for col in CAT_COLS:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("category")

        filename = obj_name.split("/")[-1]
        match = re.search(r"(20\d{2})", filename)
        if match:
            chunk["year"] = np.int16(int(match.group(1)))

        logger.info(f"[SILVER] {filename}: {len(chunk):,} registros")
        frames.append(chunk)

    combined = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()

    mem_mb = combined.memory_usage(deep=True).sum() / 1e6
    logger.info(f"[SILVER] Total concatenado: {len(combined):,} registros | {mem_mb:.1f} MB")
    return combined


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica regras de limpeza. Retorna (df_limpo, stats)."""
    stats = {"initial_rows": len(df)}

    before = len(df)
    df = df.drop_duplicates().copy()
    stats["removed_duplicates"] = before - len(df)

    before = len(df)
    df = df[df[TARGET_RAW].notna() & (df[TARGET_RAW] > 0)].copy()
    stats["removed_invalid_consume"] = before - len(df)

    if "num_connections" in df.columns:
        before = len(df)
        df = df[df["num_connections"].notna() & (df["num_connections"] > 0)].copy()
        stats["removed_invalid_connections"] = before - len(df)

    for col in PCT_COLS:
        if col in df.columns:
            before = len(df)
            df = df[df[col].isna() | df[col].between(0, 100)].copy()
            removed = before - len(df)
            if removed:
                logger.info(f"[SILVER] Removidos com {col} fora de [0,100]: {removed:,}")

    q_high = float(df[TARGET_RAW].quantile(OUTLIER_QUANTILE))
    before = len(df)
    df = df[df[TARGET_RAW] <= q_high].copy()
    stats["removed_outliers_annual_consume"] = before - len(df)
    stats["outlier_threshold_annual_consume_kwh"] = round(q_high, 2)

    df = df.reset_index(drop=True)
    stats["rows_after_cleaning"] = len(df)
    logger.info(
        f"[SILVER] Apos limpeza: {stats['rows_after_cleaning']:,} registros "
        f"(removidos: {stats['initial_rows'] - stats['rows_after_cleaning']:,})"
    )
    return df, stats


def _parse_amperage(conn_str) -> float:
    """Extrai amperagem total de strings como '3x25' -> 75, '1x35' -> 35."""
    if pd.isna(conn_str) or conn_str == "nan":
        return np.nan
    m = re.match(r"(\d+)x(\d+)", str(conn_str).strip())
    if m:
        return int(m.group(1)) * int(m.group(2))
    return np.nan


def _feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica feature engineering. Retorna (df_enriquecido, stats)."""
    df = df.copy()
    fe_stats = {}

    # Amperagem
    if "type_of_connection" in df.columns:
        df["amperage"] = df["type_of_connection"].apply(_parse_amperage).astype("float32")
        fe_stats["amperage_null_count"] = int(df["amperage"].isna().sum())
        logger.info(f"[SILVER] 'amperage' criada ({fe_stats['amperage_null_count']:,} nulos)")

    # Target: consume_per_conn
    df[TARGET] = (df[TARGET_RAW] / df["num_connections"].replace(0, np.nan)).astype("float32")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df = df[df[TARGET] > 0].copy()

    # Outliers do target
    q_cpc = df[TARGET].quantile(OUTLIER_QUANTILE)
    before = len(df)
    df = df[df[TARGET] <= q_cpc].copy()
    fe_stats["removed_outliers_consume_per_conn"] = before - len(df)
    fe_stats["outlier_threshold_consume_per_conn"] = round(float(q_cpc), 2)
    logger.info(
        f"[SILVER] Outliers {TARGET} removidos (>{q_cpc:.1f}): "
        f"{fe_stats['removed_outliers_consume_per_conn']:,}"
    )

    # Capacidade total
    if "amperage" in df.columns:
        df["total_capacity"] = (df["amperage"] * df["num_connections"]).astype("float32")

    # % tarifa alta
    if "annual_consume_lowtarif_perc" in df.columns:
        df["hightarif_perc"] = (100.0 - df["annual_consume_lowtarif_perc"]).astype("float32")

    # Log do target
    df["log_target"] = np.log1p(df[TARGET]).astype("float32")

    fe_stats["final_rows"] = len(df)
    fe_stats["final_cols"] = len(df.columns)
    fe_stats["features_created"] = ["amperage", "consume_per_conn", "total_capacity", "hightarif_perc", "log_target"]

    logger.info(f"[SILVER] Feature engineering concluido: {fe_stats['final_rows']:,} registros finais")
    return df, fe_stats


def _build_governance_doc(stats: dict, fe_stats: dict, df: pd.DataFrame) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    initial = stats.get("initial_rows", 0)
    dups = stats.get("removed_duplicates", 0)
    invalid_consume = stats.get("removed_invalid_consume", 0)
    invalid_conn = stats.get("removed_invalid_connections", 0)
    outliers_annual = stats.get("removed_outliers_annual_consume", 0)
    threshold_annual = stats.get("outlier_threshold_annual_consume_kwh", 0)
    rows_clean = stats.get("rows_after_cleaning", 0)
    outliers_cpc = fe_stats.get("removed_outliers_consume_per_conn", 0)
    threshold_cpc = fe_stats.get("outlier_threshold_consume_per_conn", 0)
    final_rows = fe_stats.get("final_rows", 0)
    total_removed = initial - final_rows

    # Tipos de dados por coluna
    type_rows = []
    for col in df.columns:
        desc = _COL_DESCRIPTIONS.get(col, "")
        type_rows.append(f"| {col} | {str(df[col].dtype)} | {desc} |")
    types_table = "\n".join(type_rows)

    doc = f"""# Documento de Governança — Camada Silver
## Dataset: Dutch Energy Electricity Consumption
## Processado em: {ts}

---

## 1. Tipos de Dados

| Coluna | Tipo Final | Descrição |
|--------|-----------|-----------|
{types_table}

---

## 2. Limpeza de Dados

| Operação | Registros Removidos |
|----------|-------------------|
| Registros iniciais | {initial:,} |
| Duplicatas exatas removidas | {dups:,} |
| Consumo anual inválido (nulo ou ≤ 0) | {invalid_consume:,} |
| Conexões inválidas (nulo ou ≤ 0) | {invalid_conn:,} |
| Outliers annual_consume (> P99.5 = {threshold_annual:,.2f} kWh) | {outliers_annual:,} |
| **Registros após limpeza** | **{rows_clean:,}** |

---

## 3. Feature Engineering

| Feature Criada | Origem | Transformação Aplicada |
|---------------|--------|----------------------|
| amperage | type_of_connection | Regex `(\\d+)x(\\d+)` → fase × amperagem |
| consume_per_conn | annual_consume, num_connections | annual_consume / num_connections |
| total_capacity | amperage, num_connections | amperage × num_connections |
| hightarif_perc | annual_consume_lowtarif_perc | 100 − lowtarif_perc |
| log_target | consume_per_conn | log1p(consume_per_conn) |

### Outliers do Target (consume_per_conn)
- Threshold P99.5: **{threshold_cpc:,.2f} kWh/conexão**
- Registros removidos: **{outliers_cpc:,}**

---

## 4. Resumo Final

| Métrica | Valor |
|---------|-------|
| Registros iniciais | {initial:,} |
| Total removido | {total_removed:,} |
| **Registros finais** | **{final_rows:,}** |
| Colunas finais | {fe_stats.get('final_cols', 0)} |

---

## 5. Colunas Descartadas para o Modelo

- `type_of_connection`: substituída pela feature numérica `amperage`
- `annual_consume`: substituída pelo target `consume_per_conn` e `log_target`
"""
    return doc


def _save_silver(df: pd.DataFrame, stats: dict, fe_stats: dict, storage: StorageBackend):
    storage.ensure_bucket(SILVER_BUCKET)
    ts = datetime.now(timezone.utc).isoformat()

    # Dataset processado (categoricas como string para CSV)
    df_out = df.copy()
    for col in CAT_COLS:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str)

    csv_buf = BytesIO()
    df_out.to_csv(csv_buf, index=False, encoding="utf-8")
    csv_buf.seek(0)

    storage.put_object(
        SILVER_BUCKET,
        f"{SILVER_PREFIX}cleaned.csv",
        csv_buf,
        "text/csv",
        metadata={"silver_completed": "true", "transformed_at": ts},
    )
    logger.info(f"[SILVER] Salvo: {SILVER_PREFIX}cleaned.csv ({len(df):,} rows)")

    # Estatísticas (mantido para compatibilidade)
    combined_stats = {**stats, **fe_stats, "saved_at": ts}
    stats_buf = BytesIO(json.dumps(combined_stats, indent=2).encode("utf-8"))
    storage.put_object(
        SILVER_BUCKET, f"{SILVER_PREFIX}cleaning_stats.json", stats_buf, "application/json"
    )

    # Documento de governança
    gov_doc = _build_governance_doc(stats, fe_stats, df_out)
    gov_buf = BytesIO(gov_doc.encode("utf-8"))
    storage.put_object(
        SILVER_BUCKET, f"{SILVER_PREFIX}governance_silver.md", gov_buf, "text/markdown"
    )
    logger.info(f"[SILVER] Salvo: {SILVER_PREFIX}governance_silver.md")
