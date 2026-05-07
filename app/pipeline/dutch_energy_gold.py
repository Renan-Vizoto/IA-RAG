"""
Gold Layer - Dutch Energy Dataset
Le o dataset ja processado do Silver e produz os artefatos prontos para treinamento:

  - Split 70 / 15 / 15  (train / val / test)
  - Target Encoding para colunas categoricas de alta cardinalidade (pos-split)
  - Normalizacao com StandardScaler (fit apenas no train)
  - Salva no gold:
      X_train.csv, X_val.csv, X_test.csv
      y_train.csv, y_val.csv, y_test.csv
      scaler.pkl, target_encoders.pkl, feature_cols.json, gold_metadata.json
  - Gera governance_gold.md (info de modelo preenchida pelo training step)
"""
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

TARGET = "consume_per_conn"
LOG_TARGET = "log_target"
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CAT_COLS_TE = ["city", "purchase_area", "net_manager"]

EXCLUDE_FROM_FEATURES = [
    TARGET,
    "annual_consume",
    "city",
    "net_manager",
    "purchase_area",
    "type_of_connection",
    LOG_TARGET,
]


def build(storage: StorageBackend, force: bool = False) -> dict[str, pd.DataFrame]:
    """
    Executa o pipeline Gold: split + encoding + scaling.
    Retorna dict com chaves: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    meta_obj = f"{GOLD_PREFIX}gold_metadata.json"
    if not force:
        meta = storage.stat_object(GOLD_BUCKET, meta_obj)
        if meta.get("gold_completed") == "true":
            logger.info("[GOLD] Pulo: Artefatos ja gerados.")
            return {}

    df = _load_silver(storage)
    feat_cols = _select_features(df)
    splits, split_stats = _split(df, feat_cols)
    splits, target_encoders = _apply_target_encoding(splits)
    final_feat_cols = [c for c in splits["X_train"].columns if c not in EXCLUDE_FROM_FEATURES]
    X_train_s, X_val_s, X_test_s, scaler = _scale(splits, final_feat_cols)
    _save_gold(splits, X_train_s, X_val_s, X_test_s, scaler, target_encoders,
               final_feat_cols, storage, split_stats)

    return {
        "X_train": pd.DataFrame(X_train_s, columns=final_feat_cols),
        "X_val":   pd.DataFrame(X_val_s,   columns=final_feat_cols),
        "X_test":  pd.DataFrame(X_test_s,  columns=final_feat_cols),
        "y_train": splits["y_train"],
        "y_val":   splits["y_val"],
        "y_test":  splits["y_test"],
    }


_PLACEHOLDER = "## 5. Modelo (preenchido após treinamento)\n[Aguardando treinamento MLflow]"


def update_governance_with_model(storage: StorageBackend, model_info: dict):
    """Atualiza governance_gold.md com informações do modelo treinado."""
    try:
        raw = storage.get_object(GOLD_BUCKET, f"{GOLD_PREFIX}governance_gold.md")
        current_doc = raw.read().decode("utf-8")
    except Exception:
        current_doc = ""

    model_section = f"## 5. Modelo\n{_build_model_section(model_info)}"
    updated = current_doc.replace(_PLACEHOLDER, model_section)

    gov_buf = BytesIO(updated.encode("utf-8"))
    storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}governance_gold.md", gov_buf, "text/markdown")
    logger.info("[GOLD] governance_gold.md atualizado com informações do modelo.")


def _build_model_section(model_info: dict) -> str:
    run_id = model_info.get("run_id", "?")
    algorithm = model_info.get("algorithm", "XGBoost")
    val_metrics  = model_info.get("metrics", {})
    test_metrics = model_info.get("test_metrics", {})
    params = model_info.get("params", {})

    param_rows   = "\n".join(f"| {k} | {v} |" for k, v in params.items())
    val_rows     = "\n".join(f"| {k} | {v} |" for k, v in val_metrics.items())
    test_section = ""
    if test_metrics:
        test_rows = "\n".join(f"| {k} | {v} |" for k, v in test_metrics.items())
        test_section = f"""
**Métricas no Teste (conjunto não visto):**

| Métrica | Valor |
|---------|-------|
{test_rows}"""

    return f"""### Modelo (preenchido após treinamento)
- **Algoritmo**: {algorithm}
- **MLflow Run ID**: {run_id}
- **Experimento**: dutch-energy-training

**Hiperparâmetros:**

| Parâmetro | Valor |
|-----------|-------|
{param_rows}

**Métricas de Validação:**

| Métrica | Valor |
|---------|-------|
{val_rows}
{test_section}"""


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


def _select_features(df: pd.DataFrame) -> list[str]:
    """Seleciona colunas numericas iniciais (excluindo targets e identificadores)."""
    numeric_types = [np.float64, np.int64, np.float32, np.int32, np.int16, float, int]
    feat_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_FEATURES
        and df[c].dtype in numeric_types
        and df[c].nunique() > 1
    ]
    return feat_cols


def _split(df: pd.DataFrame, feat_cols: list[str]) -> tuple[dict, dict]:
    """Split 70/15/15. Retorna (splits, split_stats)."""
    cols_to_keep = feat_cols + CAT_COLS_TE + [LOG_TARGET]
    before_drop = len(df)
    model_df = df[cols_to_keep].dropna(subset=feat_cols + [LOG_TARGET]).copy()
    n_dropped = before_drop - len(model_df)

    # NaNs por coluna que causaram o dropna
    nan_by_col = df[feat_cols + [LOG_TARGET]].isnull().sum()
    nan_by_col = nan_by_col[nan_by_col > 0].to_dict()

    X = model_df.drop(columns=[LOG_TARGET])
    y = model_df[LOG_TARGET]

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
    if n_dropped:
        logger.info(f"[GOLD] Descartados por NaN nas features: {n_dropped:,}")

    split_stats = {
        "silver_rows": before_drop,
        "dropped_nan": n_dropped,
        "nan_by_col": {k: int(v) for k, v in nan_by_col.items()},
        "used_rows": len(model_df),
    }
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }, split_stats


def _apply_target_encoding(splits: dict) -> tuple[dict, dict]:
    """Target encoding nas categoricas de alta cardinalidade (calculado apenas no treino)."""
    X_train = splits["X_train"].copy()
    X_val = splits["X_val"].copy()
    X_test = splits["X_test"].copy()
    y_train = splits["y_train"]

    global_mean = y_train.mean()
    target_encoders = {}

    for col in CAT_COLS_TE:
        if col in X_train.columns:
            te_map = pd.Series(y_train.values, index=X_train[col].values)
            te_map = te_map.groupby(level=0).mean()

            enc_col = f"{col}_te"
            X_train[enc_col] = X_train[col].map(te_map).fillna(global_mean).astype("float32")
            X_val[enc_col]   = X_val[col].map(te_map).fillna(global_mean).astype("float32")
            X_test[enc_col]  = X_test[col].map(te_map).fillna(global_mean).astype("float32")

            target_encoders[col] = {"map": te_map.to_dict(), "global_mean": global_mean}
            logger.info(f"[GOLD] Target encoding: {col} -> {enc_col}")

    X_train = X_train.drop(columns=CAT_COLS_TE)
    X_val   = X_val.drop(columns=CAT_COLS_TE)
    X_test  = X_test.drop(columns=CAT_COLS_TE)

    splits["X_train"] = X_train
    splits["X_val"]   = X_val
    splits["X_test"]  = X_test
    return splits, target_encoders


def _scale(splits: dict, feat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Ajusta StandardScaler apenas no train."""
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


def _build_governance_doc(splits: dict, feat_cols: list[str], ts: str,
                           split_stats: dict = None) -> str:
    n_train = len(splits["X_train"])
    n_val   = len(splits["X_val"])
    n_test  = len(splits["X_test"])
    total   = n_train + n_val + n_test

    feat_list = "\n".join(f"- {f}" for f in feat_cols)

    # Seção de rastreabilidade Silver → Gold
    ss = split_stats or {}
    silver_rows = ss.get("silver_rows", total)
    dropped_nan = ss.get("dropped_nan", silver_rows - total)
    nan_by_col  = ss.get("nan_by_col", {})
    nan_detail  = "\n".join(
        f"  - `{col}`: {cnt:,} NaNs ({cnt/silver_rows*100:.1f}%)"
        for col, cnt in sorted(nan_by_col.items(), key=lambda x: -x[1])
    ) or "  - (não disponível)"

    # Estatísticas do target no treino
    y_tr = splits["y_train"]
    target_stats = (
        f"| Média | {y_tr.mean():.4f} |\n"
        f"| Desvio Padrão | {y_tr.std():.4f} |\n"
        f"| Mínimo | {y_tr.min():.4f} |\n"
        f"| Mediana | {y_tr.median():.4f} |\n"
        f"| Máximo | {y_tr.max():.4f} |"
    )
    cpc_mean = float(np.expm1(y_tr.mean()))
    cpc_median = float(np.expm1(y_tr.median()))

    return f"""# Documento de Governança — Camada Gold
## Dataset: Dutch Energy Electricity Consumption
## Processado em: {ts}

---

## 1. Rastreabilidade de Dados (Silver → Gold)

| Etapa | Registros |
|-------|-----------|
| Recebidos do Silver | {silver_rows:,} |
| Descartados (NaN em features) | {dropped_nan:,} |
| **Utilizados no split** | **{total:,}** |

**Colunas com NaN que causaram descarte:**
{nan_detail}

> `amperage` e `total_capacity` são nulos quando `type_of_connection` é ausente
> ou não segue o padrão `NxM` (ex: conexões trifásicas não padronizadas).

---

## 2. Divisão dos Dados

| Conjunto   | Registros | Percentual | Seed |
|------------|-----------|-----------|------|
| Treino     | {n_train:,} | {n_train/total*100:.1f}% | {SEED} |
| Validação  | {n_val:,}   | {n_val/total*100:.1f}%   | {SEED} |
| Teste      | {n_test:,}  | {n_test/total*100:.1f}%  | {SEED} |
| **Total**  | **{total:,}** | 100% | - |

---

## 3. Pré-processamento Pós-Split

- **Target Encoding**: colunas `city`, `purchase_area`, `net_manager`
  - Médias calculadas **apenas no conjunto de treino** (sem vazamento de dados)
  - Valores ausentes preenchidos com a média global do treino
- **Normalização**: StandardScaler
  - Fit realizado **apenas no conjunto de treino**
  - Transform aplicado em treino, validação e teste

---

## 4. Features do Modelo ({len(feat_cols)} features)

{feat_list}

---

## 5. Variável Alvo

- **Nome**: `log_target` = `log1p(consume_per_conn)`
- **Transformação inversa**: `expm1(prediction)` → kWh/conexão

**Distribuição no conjunto de treino (espaço log):**

| Estatística | Valor |
|-------------|-------|
{target_stats}

**Espaço original (consume_per_conn kWh/conn):**
- Média: ~{cpc_mean:,.1f} kWh/conn
- Mediana: ~{cpc_median:,.1f} kWh/conn

---

## 6. Conjunto de Teste

- {n_test:,} registros reservados para avaliação final
- **Não utilizado durante o treinamento nem na seleção de hiperparâmetros**
- Métricas no teste são reportadas no `mlflow_report.md` após o treinamento

---

## 7. Modelo (preenchido após treinamento)
[Aguardando treinamento MLflow]
"""


def _save_gold(
    splits: dict,
    X_train_s: np.ndarray,
    X_val_s: np.ndarray,
    X_test_s: np.ndarray,
    scaler: StandardScaler,
    target_encoders: dict,
    feat_cols: list[str],
    storage: StorageBackend,
    split_stats: dict = None,
):
    storage.ensure_bucket(GOLD_BUCKET)
    ts = datetime.now(timezone.utc).isoformat()

    for name, arr in [("X_train", X_train_s), ("X_val", X_val_s), ("X_test", X_test_s)]:
        df_out = pd.DataFrame(arr, columns=feat_cols)
        storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}{name}.csv", _df_to_csv_bytes(df_out), "text/csv")

    for name in ["y_train", "y_val", "y_test"]:
        storage.put_object(GOLD_BUCKET, f"{GOLD_PREFIX}{name}.csv", _df_to_csv_bytes(splits[name]), "text/csv")

    storage.put_object(
        GOLD_BUCKET, f"{GOLD_PREFIX}scaler.pkl",
        BytesIO(pickle.dumps(scaler)), "application/octet-stream"
    )
    storage.put_object(
        GOLD_BUCKET, f"{GOLD_PREFIX}target_encoders.pkl",
        BytesIO(pickle.dumps(target_encoders)), "application/octet-stream"
    )
    storage.put_object(
        GOLD_BUCKET, f"{GOLD_PREFIX}feature_cols.json",
        BytesIO(json.dumps(feat_cols).encode("utf-8")), "application/json"
    )

    metadata = {
        "built_at": ts,
        "target": TARGET,
        "log_target": LOG_TARGET,
        "feature_cols": feat_cols,
        "split": {
            "train": len(splits["X_train"]),
            "val":   len(splits["X_val"]),
            "test":  len(splits["X_test"]),
        },
        "seed": SEED,
        "te_cols": list(target_encoders.keys()),
    }
    storage.put_object(
        GOLD_BUCKET,
        f"{GOLD_PREFIX}gold_metadata.json",
        BytesIO(json.dumps(metadata, indent=2).encode("utf-8")),
        "application/json",
        metadata={"gold_completed": "true", "built_at": ts},
    )

    # Documento de governança (modelo preenchido depois pelo training step)
    gov_doc = _build_governance_doc(splits, feat_cols, ts, split_stats)
    storage.put_object(
        GOLD_BUCKET, f"{GOLD_PREFIX}governance_gold.md",
        BytesIO(gov_doc.encode("utf-8")), "text/markdown"
    )

    logger.info("[GOLD] Todos os artefatos salvos com sucesso")
