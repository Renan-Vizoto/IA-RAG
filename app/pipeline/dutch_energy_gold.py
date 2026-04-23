"""
Gold Layer - Dutch Energy Dataset
Le o dataset limpo do silver e produz os artefatos prontos para treinamento:

  - Feature engineering (amperage, total_capacity, hightarif_perc, consume_per_conn)
  - Target Transformation (log1p de consume_per_conn)
  - Target Encoding para colunas categoricas de alta cardinalidade
  - Split 70 / 15 / 15  (train / val / test)
  - Normalizacao com StandardScaler (fit apenas no train)
  - Salva no gold:
      X_train.csv, X_val.csv, X_test.csv
      y_train.csv, y_val.csv, y_test.csv
      scaler.pkl, target_encoders.pkl, feature_cols.json, gold_metadata.json
"""
import re
import json
import pickle
import logging
from io import BytesIO
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)

SILVER_BUCKET = "silver"
SILVER_PREFIX = "dutch-energy/"
GOLD_BUCKET = "gold"
GOLD_PREFIX = "dutch-energy/"

# Novo target de modelagem conforme notebook
TARGET = "consume_per_conn"
ANNUAL_CONSUME_COL = "annual_consume"
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Colunas que receberao Target Encoding (alta cardinalidade)
CAT_COLS_TE = ["city", "purchase_area", "net_manager"]

# Colunas a excluir do modelo (strings brutas e identificadores)
EXCLUDE_FROM_FEATURES = [
    TARGET,
    ANNUAL_CONSUME_COL,
    "city",
    "net_manager",
    "purchase_area",
    "type_of_connection",
    "log_target",
]


def build(storage: StorageBackend, force: bool = False) -> dict[str, pd.DataFrame]:
    """
    Executa o pipeline Gold completo.
    Retorna dict com chaves: X_train, X_val, X_test, y_train, y_val, y_test.
    Aplica idempotencia (pula se ja processado e nao forçado).
    """
    meta_obj = f"{GOLD_PREFIX}gold_metadata.json"
    if not force:
        meta = storage.stat_object(GOLD_BUCKET, meta_obj)
        if meta.get("gold_completed") == "true":
            logger.info("[GOLD] Pulo: Artefatos ja gerados.")
            return {} 

    df = _load_silver(storage)
    
    # 1. Feature Engineering Inicial
    df = _feature_engineering(df)
    
    # 2. Selecao de Features Numericas
    feat_cols = _select_initial_features(df)
    
    # 3. Split
    splits = _split(df, feat_cols)
    
    # 4. Target Encoding (pos-split para evitar leakage)
    splits, target_encoders = _apply_target_encoding(splits)
    
    # 5. Atualiza feat_cols para incluir as novas colunas encoded
    final_feat_cols = [c for c in splits["X_train"].columns if c not in EXCLUDE_FROM_FEATURES]
    
    # 6. Scaling
    X_train_s, X_val_s, X_test_s, scaler = _scale(splits, final_feat_cols)
    
    # 7. Salvamento
    _save_gold(splits, X_train_s, X_val_s, X_test_s, scaler, target_encoders, final_feat_cols, storage)

    return {
        "X_train": pd.DataFrame(X_train_s, columns=final_feat_cols),
        "X_val":   pd.DataFrame(X_val_s,   columns=final_feat_cols),
        "X_test":  pd.DataFrame(X_test_s,  columns=final_feat_cols),
        "y_train": splits["y_train"],
        "y_val":   splits["y_val"],
        "y_test":  splits["y_test"],
    }


# ──────────────────────────────────────────────
# Internos
# ──────────────────────────────────────────────

def _load_silver(storage: StorageBackend) -> pd.DataFrame:
    obj_name = f"{SILVER_PREFIX}cleaned.csv"
    try:
        raw = storage.get_object(SILVER_BUCKET, obj_name)
        df = pd.read_csv(raw, low_memory=False)
        logger.info(f"[GOLD] Lido do silver: {len(df):,} registros, {df.shape[1]} colunas")
        return df
    except Exception as e:
        raise RuntimeError(
            f"Nao foi possivel ler {obj_name} do silver. "
            "Execute primeiro a camada Silver."
        ) from e


def _parse_amperage(conn_str):
    """Extrai amperagem total de strings como '3x25' -> 75, '1x35' -> 35."""
    if pd.isna(conn_str) or conn_str == "nan":
        return np.nan
    m = re.match(r"(\d+)x(\d+)", str(conn_str).strip())
    if m:
        return int(m.group(1)) * int(m.group(2))
    return np.nan


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as transformacoes de feature engineering do notebook."""
    df = df.copy()

    # Amperagem a partir do type_of_connection
    if "type_of_connection" in df.columns:
        df["amperage"] = df["type_of_connection"].apply(_parse_amperage).astype("float32")
        logger.info("[GOLD] Feature 'amperage' criada")

    # Consumo por conexao (TARGET)
    if "num_connections" in df.columns and TARGET not in df.columns:
        df[TARGET] = (df[ANNUAL_CONSUME_COL] / df["num_connections"].replace(0, np.nan)).astype("float32")
    
    # Remove registros onde o target e invalido
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df = df[df[TARGET] > 0].copy()

    # Outliers de consume_per_conn (p99.5)
    q_cpc = df[TARGET].quantile(0.995)
    before = len(df)
    df = df[df[TARGET] <= q_cpc].copy()
    logger.info(f"[GOLD] Outliers {TARGET} removidos (>{q_cpc:.1f}): {before - len(df):,}")

    # Interacao: amperagem * num_connections (capacidade total)
    if "amperage" in df.columns and "num_connections" in df.columns:
        df["total_capacity"] = (df["amperage"] * df["num_connections"]).astype("float32")

    # Proporcao de tarifa alta
    if "annual_consume_lowtarif_perc" in df.columns:
        df["hightarif_perc"] = (100.0 - df["annual_consume_lowtarif_perc"]).astype("float32")

    # Log transformation do target (estabiliza variancia)
    df["log_target"] = np.log1p(df[TARGET]).astype("float32")
    logger.info("[GOLD] Target transformado com log1p")

    return df


def _select_initial_features(df: pd.DataFrame) -> list[str]:
    """Seleciona colunas numericas iniciais."""
    numeric_types = [np.float64, np.int64, np.float32, np.int32, np.int16, float, int]
    feat_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
        and df[c].dtype in numeric_types
    ]
    # Remove constantes
    feat_cols = [c for c in feat_cols if df[c].nunique() > 1]
    return feat_cols


def _split(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Split 70/15/15."""
    # Colunas necessarias para o target encoding posterior
    cols_to_keep = feat_cols + CAT_COLS_TE + ["log_target"]
    model_df = df[cols_to_keep].dropna(subset=feat_cols + ["log_target"]).copy()
    
    X = model_df.drop(columns=["log_target"])
    y = model_df["log_target"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=SEED
    )
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_frac, random_state=SEED
    )

    logger.info(
        f"[GOLD] Split -> Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def _apply_target_encoding(splits: dict) -> tuple[dict, dict]:
    """Aplica target encoding nas colunas categoricas de alta cardinalidade."""
    X_train = splits["X_train"].copy()
    X_val = splits["X_val"].copy()
    X_test = splits["X_test"].copy()
    y_train = splits["y_train"]

    global_mean = y_train.mean()
    target_encoders = {}

    for col in CAT_COLS_TE:
        if col in X_train.columns:
            # Calcula a media do log_target por categoria apenas no TREINO
            te_map = pd.Series(y_train.values, index=X_train[col].values)
            te_map = te_map.groupby(level=0).mean()
            
            enc_col = f"{col}_te"
            X_train[enc_col] = X_train[col].map(te_map).fillna(global_mean).astype("float32")
            X_val[enc_col] = X_val[col].map(te_map).fillna(global_mean).astype("float32")
            X_test[enc_col] = X_test[col].map(te_map).fillna(global_mean).astype("float32")
            
            target_encoders[col] = {
                "map": te_map.to_dict(),
                "global_mean": global_mean
            }
            logger.info(f"[GOLD] Target encoding aplicado: {col} -> {enc_col}")

    # Remove colunas categoricas brutas
    X_train = X_train.drop(columns=CAT_COLS_TE)
    X_val = X_val.drop(columns=CAT_COLS_TE)
    X_test = X_test.drop(columns=CAT_COLS_TE)

    splits["X_train"] = X_train
    splits["X_val"] = X_val
    splits["X_test"] = X_test

    return splits, target_encoders


def _scale(splits: dict, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Ajusta o StandardScaler apenas no train."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(splits["X_train"][feat_cols])
    X_val_s   = scaler.transform(splits["X_val"][feat_cols])
    X_test_s  = scaler.transform(splits["X_test"][feat_cols])
    return X_train_s, X_val_s, X_test_s, scaler


def _df_to_csv_bytes(df) -> BytesIO:
    buf = BytesIO()
    if isinstance(df, pd.Series):
        df.to_csv(buf, index=False, header=True, encoding="utf-8")
    elif isinstance(df, np.ndarray):
        pd.DataFrame(df).to_csv(buf, index=False, encoding="utf-8")
    else:
        df.to_csv(buf, index=False, encoding="utf-8")
    buf.seek(0)
    return buf


def _save_gold(
    splits: dict,
    X_train_s: np.ndarray,
    X_val_s: np.ndarray,
    X_test_s: np.ndarray,
    scaler: StandardScaler,
    target_encoders: dict,
    feat_cols: list[str],
    storage: StorageBackend,
):
    storage.ensure_bucket(GOLD_BUCKET)

    # X e y salvos normalmente
    for name, arr in [("X_train", X_train_s), ("X_val", X_val_s), ("X_test", X_test_s)]:
        df_out = pd.DataFrame(arr, columns=feat_cols)
        storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}{name}.csv", _df_to_csv_bytes(df_out), "text/csv")

    for name in ["y_train", "y_val", "y_test"]:
        storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}{name}.csv", _df_to_csv_bytes(splits[name]), "text/csv")

    # Scaler
    storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}scaler.pkl", BytesIO(pickle.dumps(scaler)), "application/octet-stream")

    # Target Encoders
    storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}target_encoders.pkl", BytesIO(pickle.dumps(target_encoders)), "application/octet-stream")

    # Feature names
    storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}feature_cols.json", BytesIO(json.dumps(feat_cols).encode("utf-8")), "application/json")

    # Metadados
    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "target_transform": "log1p",
        "feature_cols": feat_cols,
        "split": {"train": len(splits["X_train"]), "val": len(splits["X_val"]), "test": len(splits["X_test"])},
        "seed": SEED,
        "te_cols": list(target_encoders.keys()),
    }
    storage.put_object(
        GOLD_BUCKET, 
        f"{GOLD_PREFIX}gold_metadata.json", 
        BytesIO(json.dumps(metadata, indent=2).encode("utf-8")), 
        "application/json",
        metadata={"gold_completed": "true", "built_at": metadata["built_at"]}
    )
    logger.info("[GOLD] Todos os artefatos salvos com sucesso")
