"""
Dutch Energy Regression - Versao Melhorada
Principais mudancas vs notebook original:
  1. Sem subsampling para XGBoost/LightGBM (usa treino completo)
  2. Target encoding com k-fold cross-validation (sem leakage)
  3. Features de interacao temporal
  4. Hiperparametros mais agressivos para boosting
  5. Analise de feature importance detalhada
"""

import gc
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# --- Caminhos ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ANNUAL_CONSUME_COL = "annual_consume"
TARGET = "consume_per_conn"

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


# --- Carregamento ---
def load_data(data_dir: Path) -> pd.DataFrame:
    elec_files = sorted(set(data_dir.glob("*electr*")))
    if not elec_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {data_dir}")

    frames = []
    for fp in elec_files:
        header = pd.read_csv(fp, nrows=0).columns.tolist()
        cols = [c for c in USE_COLS if c in header]
        chunk = pd.read_csv(fp, usecols=cols, dtype=str, low_memory=False)
        for col in NUM_COLS:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")
        for col in ["net_manager", "purchase_area", "city", "type_of_connection"]:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("category")
        match = re.search(r"(20\d{2})", fp.stem)
        if match:
            chunk["year"] = np.int16(int(match.group(1)))
        frames.append(chunk)

    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    print(f"Carregado: {len(df):,} registros")
    return df


# --- Limpeza ---
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df[ANNUAL_CONSUME_COL].notna() & (df[ANNUAL_CONSUME_COL] > 0)].copy()
    q_high = df[ANNUAL_CONSUME_COL].quantile(0.995)
    df = df[df[ANNUAL_CONSUME_COL] <= q_high].copy()
    print(f"Apos limpeza: {len(df):,} registros")
    return df


# --- Feature Engineering ---
def parse_amperage(conn_str):
    if pd.isna(conn_str) or str(conn_str) == "nan":
        return np.nan
    m = re.match(r"(\d+)x(\d+)", str(conn_str).strip())
    return int(m.group(1)) * int(m.group(2)) if m else np.nan


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Target
    safe_conn = df["num_connections"].replace(0, np.nan)
    df[TARGET] = (df[ANNUAL_CONSUME_COL] / safe_conn).astype("float32")

    q_cpc = df[TARGET].quantile(0.995)
    df = df[df[TARGET] <= q_cpc].copy()
    df = df[df[TARGET].notna()].copy()

    if "type_of_connection" in df.columns:
        df["amperage"] = df["type_of_connection"].apply(parse_amperage).astype("float32")

    df["log_num_connections"] = np.log1p(df["num_connections"]).astype("float32")

    if "perc_of_active_connections" in df.columns:
        df["active_connections"] = (
            df["num_connections"] * df["perc_of_active_connections"] / 100.0
        ).astype("float32")

    if "smartmeter_perc" in df.columns:
        df["smartmeter_connections"] = (
            df["num_connections"] * df["smartmeter_perc"] / 100.0
        ).astype("float32")

    if "amperage" in df.columns:
        df["total_capacity"] = (df["amperage"] * df["num_connections"]).astype("float32")

    if "annual_consume_lowtarif_perc" in df.columns:
        df["hightarif_perc"] = (100.0 - df["annual_consume_lowtarif_perc"]).astype("float32")

    # MELHORIA 1: Features de interacao temporal
    if "year" in df.columns:
        year_min = int(df["year"].min())
        year_max = int(df["year"].max())
        df["year_trend"] = ((df["year"] - year_min) / (year_max - year_min)).astype("float32")

        if "smartmeter_perc" in df.columns:
            df["smart_x_trend"] = (df["smartmeter_perc"] * df["year_trend"]).astype("float32")

        if "total_capacity" in df.columns:
            df["capacity_x_trend"] = (df["total_capacity"] * df["year_trend"]).astype("float32")

    # NOTA: log_annual_consume foi removido — e leakage direto pois
    # TARGET = annual_consume / num_connections, entao o modelo reconstruiria
    # o target algebricamente via log(annual) - log(conn).

    print(f"Features criadas. Shape: {df.shape}")
    return df


# --- Target Encoding com K-Fold (sem leakage) ---
def kfold_target_encode(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list,
    n_splits: int = 5,
    global_mean: float = None,
) -> tuple:
    if global_mean is None:
        global_mean = float(y_train.mean())

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for col in cat_cols:
        if col not in X_train.columns:
            continue

        enc_col = f"{col}_te"
        oof_encoded = pd.Series(np.nan, index=X_train.index, dtype="float32")

        for fold_train_idx, fold_val_idx in kf.split(X_train):
            fold_X = X_train.iloc[fold_train_idx][col]
            fold_y = y_train.iloc[fold_train_idx]

            te_map = fold_y.groupby(fold_X.values).mean()

            fold_val_cats = X_train.iloc[fold_val_idx][col]
            oof_encoded.iloc[fold_val_idx] = fold_val_cats.map(te_map).fillna(global_mean).values

        X_train[enc_col] = oof_encoded.fillna(global_mean).astype("float32")

        full_te_map = y_train.groupby(X_train[col].values).mean()
        X_val[enc_col] = X_val[col].map(full_te_map).fillna(global_mean).astype("float32")
        X_test[enc_col] = X_test[col].map(full_te_map).fillna(global_mean).astype("float32")

    return X_train, X_val, X_test


# --- Split temporal ---
def temporal_split(df: pd.DataFrame, feat_cols: list, cat_cols: list):
    years = sorted(df["year"].unique().tolist())
    test_years = years[-2:]
    val_years = years[-4:-2]
    train_years = [y for y in years if y not in val_years + test_years]

    print(f"Train: {train_years[0]}-{train_years[-1]} | Val: {val_years} | Test: {test_years}")

    log_target = np.log1p(df[TARGET]).astype("float32")
    X_full = df[feat_cols + cat_cols].copy()
    y_full = log_target

    train_mask = df["year"].isin(train_years)
    val_mask = df["year"].isin(val_years)
    test_mask = df["year"].isin(test_years)

    X_train = X_full.loc[train_mask].copy()
    y_train = y_full.loc[train_mask].copy()
    X_val = X_full.loc[val_mask].copy()
    y_val = y_full.loc[val_mask].copy()
    X_test = X_full.loc[test_mask].copy()
    y_test = y_full.loc[test_mask].copy()

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# --- Metricas ---
def to_orig(arr):
    return np.expm1(np.asarray(arr, dtype="float64"))


def evaluate(yt, yp, name):
    yt_o = to_orig(yt)
    yp_o = to_orig(np.clip(yp, 0, None))
    abs_err = np.abs(yt_o - yp_o)
    return {
        "Modelo": name,
        "MAE": mean_absolute_error(yt_o, yp_o),
        "MedAE": median_absolute_error(yt_o, yp_o),
        "RMSE": np.sqrt(mean_squared_error(yt_o, yp_o)),
        "MAPE": mean_absolute_percentage_error(yt_o, yp_o) * 100,
        "WMAPE": abs_err.sum() / np.abs(yt_o).sum() * 100,
        "R2": r2_score(yt_o, yp_o),
    }


def show(m):
    print(
        f"  {m['Modelo']:35s} "
        f"MAE={m['MAE']:.2f}  "
        f"MedAE={m['MedAE']:.2f}  "
        f"RMSE={m['RMSE']:.2f}  "
        f"WMAPE={m['WMAPE']:.1f}%  "
        f"R2={m['R2']:.4f}"
    )


# --- Main ---
def main():
    print("=" * 70)
    print("Dutch Energy Regression - Versao Melhorada")
    print("=" * 70)

    raw = load_data(DATA_DIR)
    clean = clean_data(raw)
    del raw
    gc.collect()

    feat_df = create_features(clean)
    del clean
    gc.collect()

    CAT_COLS = [c for c in ["city", "purchase_area", "net_manager"] if c in feat_df.columns]

    EXCLUDE = {
        TARGET, ANNUAL_CONSUME_COL, "street", "zipcode_from", "zipcode_to",
        "type_of_connection",
    }
    EXCLUDE.update(CAT_COLS)

    NUM_FEAT_COLS = [
        c for c in feat_df.columns
        if c not in EXCLUDE
        and feat_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16]
    ]
    print(f"\nFeatures numericas ({len(NUM_FEAT_COLS)}): {NUM_FEAT_COLS}")
    print(f"Features categoricas para TE: {CAT_COLS}")

    essential = [c for c in NUM_FEAT_COLS if c != "year"]
    model_df = feat_df[NUM_FEAT_COLS + [TARGET] + CAT_COLS].dropna(
        subset=essential + [TARGET]
    ).copy()
    del feat_df
    gc.collect()
    print(f"\nDataset para modelagem: {len(model_df):,} registros")

    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(
        model_df, NUM_FEAT_COLS, CAT_COLS
    )
    del model_df
    gc.collect()

    print("\nAplicando k-fold target encoding...")
    global_mean = float(y_train.mean())
    X_train, X_val, X_test = kfold_target_encode(
        X_train, y_train, X_val, X_test, CAT_COLS, n_splits=5, global_mean=global_mean
    )

    X_train = X_train.drop(columns=CAT_COLS, errors="ignore").astype("float32")
    X_val = X_val.drop(columns=CAT_COLS, errors="ignore").astype("float32")
    X_test = X_test.drop(columns=CAT_COLS, errors="ignore").astype("float32")

    FEAT_COLS = X_train.columns.tolist()
    print(f"Total features apos encoding: {len(FEAT_COLS)}")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train).astype("float32")
    Xv_s = scaler.transform(X_val).astype("float32")
    Xte_s = scaler.transform(X_test).astype("float32")

    results = []

    # --- Baseline ---
    print("\n--- Modelos ---")
    baseline_val = float(np.log1p(np.median(to_orig(y_train.values))))
    p = np.full(len(y_test), baseline_val)
    r = evaluate(y_test.values, p, "Baseline (Mediana)")
    results.append(r)
    show(r)

    # --- Ridge ---
    m_ridge = Ridge(alpha=1.0, random_state=SEED).fit(Xtr_s, y_train)
    p = m_ridge.predict(Xte_s)
    r = evaluate(y_test.values, p, "Ridge")
    results.append(r)
    show(r)

    # --- XGBoost sem subsampling + tuning ---
    print("\n  [XGBoost] Treinando no dataset COMPLETO...")
    m_xgb = XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    ).fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )
    p = m_xgb.predict(X_test)
    r = evaluate(y_test.values, p, "XGBoost (full + tuned)")
    results.append(r)
    show(r)
    print(f"    Melhor iteracao XGBoost: {m_xgb.best_iteration}")

    # --- LightGBM sem subsampling + tuning ---
    print("\n  [LightGBM] Treinando no dataset COMPLETO...")
    m_lgb = LGBMRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.03,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    ).fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(100)],
    )
    p = m_lgb.predict(X_test)
    r = evaluate(y_test.values, p, "LightGBM (full + tuned)")
    results.append(r)
    show(r)

    # --- Ranking ---
    print("\n" + "=" * 70)
    print("RANKING FINAL")
    print("=" * 70)
    res_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    res_df.index += 1
    res_df.index.name = "Rank"
    print(res_df.to_string())

    # --- Delta vs original ---
    print("\n--- Melhoria vs notebook original ---")
    originals = {
        "XGBoost": {"MAE": 68.65, "R2": 0.466},
        "LightGBM": {"MAE": 68.71, "R2": 0.466},
    }
    for r in results:
        for key, orig in originals.items():
            if key in r["Modelo"] and "ORIGINAL" not in r["Modelo"]:
                delta_mae = orig["MAE"] - r["MAE"]
                delta_r2 = r["R2"] - orig["R2"]
                print(
                    f"  {key}: MAE {orig['MAE']:.1f} -> {r['MAE']:.2f} "
                    f"(delta={delta_mae:+.2f})  |  "
                    f"R2 {orig['R2']:.3f} -> {r['R2']:.4f} "
                    f"(delta={delta_r2:+.4f})"
                )

    # --- Feature Importance ---
    print("\n--- Feature Importance (XGBoost) ---")
    imp = pd.Series(m_xgb.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    for feat, val in imp.items():
        bar = "#" * int(val * 300)
        print(f"  {feat:<35s} {val:.4f}  {bar}")

    # Salvar
    res_df.to_csv(RESULTS_DIR / "model_comparison_improved.csv")
    imp.to_csv(RESULTS_DIR / "feature_importance_xgb.csv", header=["importance"])

    # Plot
    _plot_comparison(res_df, RESULTS_DIR)

    print(f"\nResultados salvos em {RESULTS_DIR}")
    return res_df, imp


def _plot_comparison(res_df, results_dir):
    original_rows = pd.DataFrame([
        {"Modelo": "XGBoost (ORIGINAL)", "MAE": 68.65, "RMSE": 128.48, "R2": 0.466},
        {"Modelo": "LightGBM (ORIGINAL)", "MAE": 68.71, "RMSE": 128.49, "R2": 0.466},
    ])

    comp = pd.concat(
        [res_df[["Modelo", "MAE", "RMSE", "R2"]], original_rows],
        ignore_index=True,
    ).sort_values("MAE")

    colors = ["#2ecc71" if "ORIGINAL" not in str(m) else "#e74c3c" for m in comp["Modelo"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Impacto das Melhorias - Original vs Melhorado", fontsize=13)

    for ax, col, lab in zip(axes, ["MAE", "RMSE", "R2"], ["MAE (kWh/conn)", "RMSE (kWh/conn)", "R2"]):
        bars = ax.barh(comp["Modelo"], comp[col], color=colors)
        ax.set_xlabel(lab)
        ax.invert_yaxis()
        for b, v in zip(bars, comp[col]):
            ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=8)

    from matplotlib.patches import Patch
    legend = [
        Patch(color="#2ecc71", label="Melhorado"),
        Patch(color="#e74c3c", label="Original"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = results_dir / "comparison_improved_vs_original.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Plot salvo: {out}")


if __name__ == "__main__":
    main()
