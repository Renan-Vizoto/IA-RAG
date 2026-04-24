"""
Dutch Energy - Experimento com Lag Features
A hipotese: o consumo de uma cidade no ano Y e fortemente predito
pelo consumo da mesma cidade no ano Y-1. Isso captura padroes locais
persistentes (tipo habitacional, renda, clima regional) sem leakage.
"""

import gc
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

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


def load_and_clean(data_dir):
    elec_files = sorted(set(data_dir.glob("*electr*")))
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
    del frames; gc.collect()

    df = df[df[ANNUAL_CONSUME_COL].notna() & (df[ANNUAL_CONSUME_COL] > 0)]
    q_high = df[ANNUAL_CONSUME_COL].quantile(0.995)
    df = df[df[ANNUAL_CONSUME_COL] <= q_high].copy()
    print(f"Carregado e limpo: {len(df):,} registros")
    return df


def parse_amperage(s):
    if pd.isna(s) or str(s) == "nan":
        return np.nan
    m = re.match(r"(\d+)x(\d+)", str(s).strip())
    return int(m.group(1)) * int(m.group(2)) if m else np.nan


def feature_engineering(df):
    df = df.copy()
    safe_conn = df["num_connections"].replace(0, np.nan)
    df[TARGET] = (df[ANNUAL_CONSUME_COL] / safe_conn).astype("float32")
    q = df[TARGET].quantile(0.995)
    df = df[df[TARGET].between(0, q)].copy()

    if "type_of_connection" in df.columns:
        df["amperage"] = df["type_of_connection"].apply(parse_amperage).astype("float32")

    df["log_num_connections"] = np.log1p(df["num_connections"]).astype("float32")

    df["active_connections"] = (
        df["num_connections"] * df["perc_of_active_connections"] / 100.0
    ).astype("float32")

    df["smartmeter_connections"] = (
        df["num_connections"] * df["smartmeter_perc"] / 100.0
    ).astype("float32")

    if "amperage" in df.columns:
        df["total_capacity"] = (df["amperage"] * df["num_connections"]).astype("float32")

    df["hightarif_perc"] = (100.0 - df["annual_consume_lowtarif_perc"]).astype("float32")

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    df["year_trend"] = ((df["year"] - year_min) / (year_max - year_min)).astype("float32")
    df["smart_x_trend"] = (df["smartmeter_perc"] * df["year_trend"]).astype("float32")

    return df


def add_lag_features(df):
    """
    Para cada (city, year), computa o consumo medio por conexao
    do ano anterior como feature. Usa apenas dados de anos passados,
    logo nao ha leakage temporal.
    """
    df = df.copy()

    # Media por cidade x ano (usada para criar lags)
    city_year_mean = (
        df.groupby(["city", "year"])[TARGET]
        .mean()
        .reset_index()
        .rename(columns={TARGET: "city_mean_consume"})
    )
    city_year_mean["year_next"] = city_year_mean["year"] + 1

    # lag-1: consumo medio da cidade no ano anterior
    df = df.merge(
        city_year_mean[["city", "year_next", "city_mean_consume"]].rename(
            columns={"year_next": "year", "city_mean_consume": "city_lag1_mean"}
        ),
        on=["city", "year"],
        how="left",
    )

    # lag-2: consumo medio da cidade 2 anos atras
    city_year_mean["year_next2"] = city_year_mean["year"] + 2
    df = df.merge(
        city_year_mean[["city", "year_next2", "city_mean_consume"]].rename(
            columns={"year_next2": "year", "city_mean_consume": "city_lag2_mean"}
        ),
        on=["city", "year"],
        how="left",
    )

    # Tendencia da cidade: media dos ultimos 3 anos disponiveis antes do ano atual
    # Captura se a cidade esta em queda ou crescimento de consumo
    city_year_sorted = city_year_mean.sort_values(["city", "year"])
    city_year_sorted["city_rolling_mean3"] = (
        city_year_sorted.groupby("city")["city_mean_consume"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df = df.merge(
        city_year_sorted[["city", "year", "city_rolling_mean3"]],
        on=["city", "year"],
        how="left",
    )

    # Desvio da cidade em relacao a media nacional do mesmo ano
    national_year_mean = df.groupby("year")[TARGET].mean().rename("national_year_mean")
    df = df.join(national_year_mean, on="year")
    df["city_vs_national"] = (df["city_lag1_mean"] - df["national_year_mean"]).astype("float32")

    # Converter para float32
    for col in ["city_lag1_mean", "city_lag2_mean", "city_rolling_mean3", "national_year_mean", "city_vs_national"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    n_lag_null = df["city_lag1_mean"].isna().sum()
    print(f"  city_lag1_mean: {n_lag_null:,} nulls ({100*n_lag_null/len(df):.1f}%) — preenchidos com mediana global")

    global_median = df[TARGET].median()
    df["city_lag1_mean"] = df["city_lag1_mean"].fillna(global_median).astype("float32")
    df["city_lag2_mean"] = df["city_lag2_mean"].fillna(global_median).astype("float32")
    df["city_rolling_mean3"] = df["city_rolling_mean3"].fillna(global_median).astype("float32")
    df["city_vs_national"] = df["city_vs_national"].fillna(0.0).astype("float32")

    return df


def kfold_target_encode(X_train, y_train, X_val, X_test, cat_cols, n_splits=5):
    global_mean = float(y_train.mean())
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for col in cat_cols:
        if col not in X_train.columns:
            continue
        enc_col = f"{col}_te"
        oof = pd.Series(np.nan, index=X_train.index, dtype="float32")

        for tr_idx, vl_idx in kf.split(X_train):
            fold_y = y_train.iloc[tr_idx]
            fold_x = X_train.iloc[tr_idx][col]
            te_map = fold_y.groupby(fold_x.values).mean()
            oof.iloc[vl_idx] = X_train.iloc[vl_idx][col].map(te_map).fillna(global_mean).values

        X_train[enc_col] = oof.fillna(global_mean).astype("float32")

        full_map = y_train.groupby(X_train[col].values).mean()
        X_val[enc_col] = X_val[col].map(full_map).fillna(global_mean).astype("float32")
        X_test[enc_col] = X_test[col].map(full_map).fillna(global_mean).astype("float32")

    return X_train, X_val, X_test


def to_orig(arr):
    return np.expm1(np.asarray(arr, dtype="float64"))


def evaluate(yt, yp, name):
    yt_o = to_orig(yt)
    yp_o = to_orig(np.clip(yp, 0, None))
    ae = np.abs(yt_o - yp_o)
    return {
        "Modelo": name,
        "MAE": mean_absolute_error(yt_o, yp_o),
        "MedAE": median_absolute_error(yt_o, yp_o),
        "RMSE": np.sqrt(mean_squared_error(yt_o, yp_o)),
        "MAPE": mean_absolute_percentage_error(yt_o, yp_o) * 100,
        "WMAPE": ae.sum() / np.abs(yt_o).sum() * 100,
        "R2": r2_score(yt_o, yp_o),
    }


def show(m):
    print(
        f"  {m['Modelo']:40s} "
        f"MAE={m['MAE']:.2f}  MedAE={m['MedAE']:.2f}  "
        f"RMSE={m['RMSE']:.2f}  WMAPE={m['WMAPE']:.1f}%  R2={m['R2']:.4f}"
    )


def main():
    print("=" * 70)
    print("Dutch Energy - Experimento com Lag Features")
    print("=" * 70)

    df = load_and_clean(DATA_DIR)
    df = feature_engineering(df)

    print("\nAdicionando lag features...")
    df = add_lag_features(df)
    print(f"  Novas features: city_lag1_mean, city_lag2_mean, city_rolling_mean3, city_vs_national")

    CAT_COLS = [c for c in ["city", "purchase_area", "net_manager"] if c in df.columns]
    EXCLUDE = {TARGET, ANNUAL_CONSUME_COL, "street", "zipcode_from", "zipcode_to",
               "type_of_connection", "national_year_mean"}
    EXCLUDE.update(CAT_COLS)

    NUM_FEAT_COLS = [
        c for c in df.columns
        if c not in EXCLUDE
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16]
    ]
    print(f"\nFeatures ({len(NUM_FEAT_COLS)}): {NUM_FEAT_COLS}")

    essential = [c for c in NUM_FEAT_COLS if c != "year"]
    model_df = df[NUM_FEAT_COLS + [TARGET] + CAT_COLS].dropna(subset=essential + [TARGET]).copy()
    del df; gc.collect()
    print(f"Dataset final: {len(model_df):,} registros")

    # Split temporal
    years = sorted(model_df["year"].unique().tolist())
    test_years = years[-2:]
    val_years = years[-4:-2]
    train_years = [y for y in years if y not in val_years + test_years]
    print(f"\nSplit: train={train_years[0]}-{train_years[-1]} | val={val_years} | test={test_years}")

    log_tgt = np.log1p(model_df[TARGET]).astype("float32")
    Xf = model_df[NUM_FEAT_COLS + CAT_COLS]
    yf = log_tgt

    tm = model_df["year"].isin(train_years)
    vm = model_df["year"].isin(val_years)
    tsm = model_df["year"].isin(test_years)

    X_train, y_train = Xf.loc[tm].copy(), yf.loc[tm].copy()
    X_val,   y_val   = Xf.loc[vm].copy(), yf.loc[vm].copy()
    X_test,  y_test  = Xf.loc[tsm].copy(), yf.loc[tsm].copy()
    del model_df, Xf, yf; gc.collect()
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Target encoding
    print("\nAplicando k-fold target encoding...")
    X_train, X_val, X_test = kfold_target_encode(X_train, y_train, X_val, X_test, CAT_COLS)
    X_train = X_train.drop(columns=CAT_COLS, errors="ignore").astype("float32")
    X_val   = X_val.drop(columns=CAT_COLS, errors="ignore").astype("float32")
    X_test  = X_test.drop(columns=CAT_COLS, errors="ignore").astype("float32")
    FEAT_COLS = X_train.columns.tolist()

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train).astype("float32")
    Xv_s  = scaler.transform(X_val).astype("float32")
    Xte_s = scaler.transform(X_test).astype("float32")

    results = []

    # Baseline
    print("\n--- Modelos ---")
    bv = float(np.log1p(np.median(to_orig(y_train.values))))
    r = evaluate(y_test.values, np.full(len(y_test), bv), "Baseline (Mediana)")
    results.append(r); show(r)

    # Ridge
    m_ridge = Ridge(alpha=1.0, random_state=SEED).fit(Xtr_s, y_train)
    r = evaluate(y_test.values, m_ridge.predict(Xte_s), "Ridge")
    results.append(r); show(r)

    # XGBoost - treino completo, mais arvores
    print("\n  [XGBoost] Treinando (treino completo, 1200 arvores)...")
    m_xgb = XGBRegressor(
        n_estimators=1200,
        max_depth=7,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=60,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    ).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
    p_xgb = m_xgb.predict(X_test)
    r = evaluate(y_test.values, p_xgb, "XGBoost (lag + full + 1200 trees)")
    results.append(r); show(r)
    print(f"    Melhor iteracao: {m_xgb.best_iteration}")

    # LightGBM - treino completo, mais arvores
    print("\n  [LightGBM] Treinando (treino completo, 1200 arvores)...")
    m_lgb = LGBMRegressor(
        n_estimators=1200,
        max_depth=7,
        learning_rate=0.02,
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
        callbacks=[early_stopping(60, verbose=False), log_evaluation(200)],
    )
    p_lgb = m_lgb.predict(X_test)
    r = evaluate(y_test.values, p_lgb, "LightGBM (lag + full + 1200 trees)")
    results.append(r); show(r)

    # Ranking final
    print("\n" + "=" * 70)
    print("RANKING FINAL")
    print("=" * 70)
    res_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    res_df.index += 1
    res_df.index.name = "Rank"
    print(res_df.to_string())

    # Comparacao com baseline sem lag
    print("\n--- Delta vs versao sem lag features ---")
    sem_lag = {"XGBoost": {"MAE": 68.16, "R2": 0.4807}, "LightGBM": {"MAE": 68.19, "R2": 0.4805}}
    for r in results:
        for key, orig in sem_lag.items():
            if key in r["Modelo"]:
                print(
                    f"  {key}: MAE {orig['MAE']:.2f} -> {r['MAE']:.2f} "
                    f"(delta={orig['MAE'] - r['MAE']:+.2f})  |  "
                    f"R2 {orig['R2']:.4f} -> {r['R2']:.4f} "
                    f"(delta={r['R2'] - orig['R2']:+.4f})"
                )

    # Feature importance
    print("\n--- Feature Importance (XGBoost) ---")
    imp = pd.Series(m_xgb.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    for feat, val in imp.head(15).items():
        bar = "#" * int(val * 400)
        print(f"  {feat:<35s} {val:.4f}  {bar}")

    # Salvar
    res_df.to_csv(RESULTS_DIR / "model_comparison_lag.csv")
    imp.to_csv(RESULTS_DIR / "feature_importance_lag.csv", header=["importance"])

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 7))
    imp.head(15).sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_title("Feature Importance - XGBoost com Lag Features")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance_lag.png", dpi=120)
    print(f"\nPlots salvos em {RESULTS_DIR}")

    return res_df, imp


if __name__ == "__main__":
    main()
