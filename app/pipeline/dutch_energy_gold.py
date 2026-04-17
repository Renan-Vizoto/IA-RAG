"""
Gold Layer - Dutch Energy Dataset
Le o dataset limpo do silver e produz os artefatos prontos para treinamento:

  - Feature engineering (consume_per_connection, zipcode_range, label encoding)
  - Split 70 / 15 / 15  (train / val / test)
  - Normalizacao com StandardScaler (fit apenas no train)
  - Salva no gold:
      X_train.csv, X_val.csv, X_test.csv
      y_train.csv, y_val.csv, y_test.csv
      scaler.pkl  (StandardScaler serializado)
      encoders.pkl (LabelEncoders serializados)
      feature_cols.json
      gold_metadata.json
"""
import json
import pickle
import logging
from io import BytesIO
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)

SILVER_BUCKET = "silver"
SILVER_PREFIX = "dutch-energy/"
GOLD_BUCKET = "gold"
GOLD_PREFIX = "dutch-energy/"

TARGET = "annual_consume"
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Colunas categoricas que receberao label encoding
CAT_COLS = ["net_manager", "purchase_area", "city", "type_of_connection"]

# Colunas que nao entram no modelo (identifiers e raw categoricas)
# Nota: zipcode_from/to e street nao sao mais carregadas desde o silver
EXCLUDE_FROM_FEATURES = [
    TARGET,
    "city",
    "net_manager",
    "purchase_area",
    "type_of_connection",
]


def build(storage: StorageBackend) -> dict[str, pd.DataFrame]:
    """
    Executa o pipeline Gold completo.
    Retorna dict com chaves: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    df = _load_silver(storage)
    feat_df, encoders = _feature_engineering(df)
    feat_cols = _select_feature_cols(feat_df)
    splits = _split(feat_df, feat_cols)
    X_train_s, X_val_s, X_test_s, scaler = _scale(splits)
    _save_gold(splits, X_train_s, X_val_s, X_test_s, scaler, encoders, feat_cols, storage)

    return {
        "X_train": pd.DataFrame(X_train_s, columns=feat_cols),
        "X_val":   pd.DataFrame(X_val_s,   columns=feat_cols),
        "X_test":  pd.DataFrame(X_test_s,  columns=feat_cols),
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


def _feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Aplica label encoding nas colunas categoricas.
    Nao ha features derivadas (zipcode e consume_per_connection foram removidas do notebook).
    """
    df = df.copy()

    # --- Label Encoding para colunas categoricas ---
    encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f"[GOLD] Label encoding: {col} -> {col}_enc ({len(le.classes_)} classes)")

    # Remove linhas com target nulo (seguranca)
    before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    if before - len(df):
        logger.warning(f"[GOLD] Removidos {before - len(df)} registros com target nulo")

    return df, encoders


def _select_feature_cols(df: pd.DataFrame) -> list[str]:
    """Seleciona apenas colunas numericas que nao estao na lista de exclusao."""
    # np.int16 incluido pois 'year' e salvo nesse dtype
    numeric_types = [np.float64, np.int64, np.float32, np.int32, np.int16, float, int]
    feat_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
        and df[c].dtype in numeric_types
    ]
    # Remove colunas com variancia zero (constantes)
    feat_cols = [c for c in feat_cols if df[c].nunique() > 1]

    logger.info(f"[GOLD] Features selecionadas ({len(feat_cols)}): {feat_cols}")
    return feat_cols


def _split(df: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Split estratificado por ano (se disponivel), caso contrario aleatorio."""
    model_df = df[feat_cols + [TARGET]].dropna().copy()
    logger.info(f"[GOLD] Registros para modelagem (sem NaN): {len(model_df):,}")

    X = model_df[feat_cols]
    y = model_df[TARGET]

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


def _scale(splits: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Ajusta o StandardScaler apenas no train e transforma os tres conjuntos."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(splits["X_train"])
    X_val_s   = scaler.transform(splits["X_val"])
    X_test_s  = scaler.transform(splits["X_test"])
    logger.info("[GOLD] StandardScaler ajustado no train e aplicado em val/test")
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
    encoders: dict,
    feat_cols: list[str],
    storage: StorageBackend,
):
    storage.ensure_bucket(GOLD_BUCKET)

    # X escalados (prontos para o modelo)
    for name, arr in [("X_train", X_train_s), ("X_val", X_val_s), ("X_test", X_test_s)]:
        df_out = pd.DataFrame(arr, columns=feat_cols)
        storage.put_object(
            GOLD_BUCKET, f"{GOLD_PREFIX}{name}.csv",
            _df_to_csv_bytes(df_out), "text/csv"
        )
        logger.info(f"[GOLD] Salvo: {GOLD_PREFIX}{name}.csv ({len(df_out):,} rows)")

    # y (targets)
    for name in ["y_train", "y_val", "y_test"]:
        storage.put_object(
            GOLD_BUCKET, f"{GOLD_PREFIX}{name}.csv",
            _df_to_csv_bytes(splits[name]), "text/csv"
        )
        logger.info(f"[GOLD] Salvo: {GOLD_PREFIX}{name}.csv ({len(splits[name]):,} rows)")

    # Scaler serializado
    scaler_buf = BytesIO(pickle.dumps(scaler))
    storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}scaler.pkl", scaler_buf, "application/octet-stream")
    logger.info("[GOLD] Salvo: scaler.pkl")

    # Encoders serializados
    encoders_buf = BytesIO(pickle.dumps(encoders))
    storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}encoders.pkl", encoders_buf, "application/octet-stream")
    logger.info("[GOLD] Salvo: encoders.pkl")

    # Lista de features
    feat_buf = BytesIO(json.dumps(feat_cols).encode("utf-8"))
    storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}feature_cols.json", feat_buf, "application/json")
    logger.info("[GOLD] Salvo: feature_cols.json")

    # Metadados
    metadata = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "target": TARGET,
        "feature_cols": feat_cols,
        "n_features": len(feat_cols),
        "split": {
            "train": len(splits["X_train"]),
            "val":   len(splits["X_val"]),
            "test":  len(splits["X_test"]),
            "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        },
        "scaler": "StandardScaler",
        "seed": SEED,
        "cat_cols_encoded": list(encoders.keys()),
    }
    meta_buf = BytesIO(json.dumps(metadata, indent=2).encode("utf-8"))
    storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}gold_metadata.json", meta_buf, "application/json")
    logger.info("[GOLD] Salvo: gold_metadata.json")
