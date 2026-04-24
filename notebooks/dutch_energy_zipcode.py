"""
Dutch Energy - Com zipcode_from como feature (target encoding)
Hipotese validada: r=0.77 entre consumo do mesmo CEP em 2015 vs 2019.
MAE baseline com lag de CEP = 59.4 kWh < XGBoost atual (68.2 kWh).
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

# zipcode_from agora incluido
USE_COLS = [
    "net_manager", "purchase_area", "city", "zipcode_from",
    "num_connections", "delivery_perc", "perc_of_active_connections",
    "type_of_connection", "type_conn_perc", "annual_consume",
    "annual_consume_lowtarif_perc", "smartmeter_perc",
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
        for col in ["net_manager", "purchase_area", "city", "type_of_connection", "zipcode_from"]:
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
    print(f"  zipcode_from: {df['zipcode_from'].nunique():,} valores unicos")
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
        f"  {m['Modelo']:45s} "
        f"MAE={m['MAE']:.2f}  MedAE={m['MedAE']:.2f}  "
        f"RMSE={m['RMSE']:.2f}  WMAPE={m['WMAPE']:.1f}%  R2={m['R2']:.4f}"
    )


def main():
    print("=" * 70)
    print("Dutch Energy - Com zipcode_from (feature nova)")
    print("=" * 70)

    df = load_and_clean(DATA_DIR)
    df = feature_engineering(df)

    # Categoricas para target encoding — agora inclui zipcode_from
    CAT_COLS = [c for c in ["zipcode_from", "city", "purchase_area", "net_manager"]
                if c in df.columns]

    EXCLUDE = {TARGET, ANNUAL_CONSUME_COL, "street", "zipcode_to",
               "type_of_connection"}
    EXCLUDE.update(CAT_COLS)

    NUM_FEAT_COLS = [
        c for c in df.columns
        if c not in EXCLUDE
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16]
    ]
    print(f"\nFeatures numericas ({len(NUM_FEAT_COLS)}): {NUM_FEAT_COLS}")
    print(f"Features categoricas para TE ({len(CAT_COLS)}): {CAT_COLS}")

    essential = [c for c in NUM_FEAT_COLS if c != "year"]
    model_df = df[NUM_FEAT_COLS + [TARGET] + CAT_COLS].dropna(
        subset=essential + [TARGET]
    ).copy()
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

    X_train = Xf.loc[model_df["year"].isin(train_years)].copy()
    y_train = yf.loc[model_df["year"].isin(train_years)].copy()
    X_val   = Xf.loc[model_df["year"].isin(val_years)].copy()
    y_val   = yf.loc[model_df["year"].isin(val_years)].copy()
    X_test  = Xf.loc[model_df["year"].isin(test_years)].copy()
    y_test  = yf.loc[model_df["year"].isin(test_years)].copy()
    del model_df, Xf, yf; gc.collect()
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Target encoding com k-fold
    print("\nAplicando k-fold target encoding (inclui zipcode_from)...")
    X_train, X_val, X_test = kfold_target_encode(X_train, y_train, X_val, X_test, CAT_COLS)
    X_train = X_train.drop(columns=CAT_COLS, errors="ignore").astype("float32")
    X_val   = X_val.drop(columns=CAT_COLS, errors="ignore").astype("float32")
    X_test  = X_test.drop(columns=CAT_COLS, errors="ignore").astype("float32")
    FEAT_COLS = X_train.columns.tolist()
    print(f"Total features: {len(FEAT_COLS)}")

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train).astype("float32")
    Xv_s  = scaler.transform(X_val).astype("float32")
    Xte_s = scaler.transform(X_test).astype("float32")

    results = []

    print("\n--- Modelos ---")
    bv = float(np.log1p(np.median(to_orig(y_train.values))))
    r = evaluate(y_test.values, np.full(len(y_test), bv), "Baseline (Mediana)")
    results.append(r); show(r)

    m_ridge = Ridge(alpha=1.0, random_state=SEED).fit(Xtr_s, y_train)
    r = evaluate(y_test.values, m_ridge.predict(Xte_s), "Ridge")
    results.append(r); show(r)

    print("\n  [XGBoost] Treinando...")
    m_xgb = XGBRegressor(
        n_estimators=1500,
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
        early_stopping_rounds=80,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    ).fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
    p_xgb = m_xgb.predict(X_test)
    r = evaluate(y_test.values, p_xgb, "XGBoost (+ zipcode_te)")
    results.append(r); show(r)
    print(f"    Melhor iteracao: {m_xgb.best_iteration}")

    print("\n  [LightGBM] Treinando...")
    m_lgb = LGBMRegressor(
        n_estimators=1500,
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
        callbacks=[early_stopping(80, verbose=False), log_evaluation(200)],
    )
    p_lgb = m_lgb.predict(X_test)
    r = evaluate(y_test.values, p_lgb, "LightGBM (+ zipcode_te)")
    results.append(r); show(r)

    # Ranking
    print("\n" + "=" * 70)
    print("RANKING FINAL")
    print("=" * 70)
    res_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    res_df.index += 1
    res_df.index.name = "Rank"
    print(res_df.to_string())

    print("\n--- Delta vs versao sem zipcode (MAE=68.16, R2=0.481) ---")
    for r in results:
        if "XGBoost" in r["Modelo"] or "LightGBM" in r["Modelo"]:
            print(
                f"  {r['Modelo']}: MAE {68.16:.2f} -> {r['MAE']:.2f} "
                f"(delta={68.16 - r['MAE']:+.2f})  |  "
                f"R2 0.481 -> {r['R2']:.4f} "
                f"(delta={r['R2'] - 0.481:+.4f})"
            )

    # Feature importance
    print("\n--- Feature Importance (XGBoost) ---")
    imp = pd.Series(m_xgb.feature_importances_, index=FEAT_COLS).sort_values(ascending=False)
    for feat, val in imp.head(12).items():
        bar = "#" * int(val * 400)
        print(f"  {feat:<35s} {val:.4f}  {bar}")

    res_df.to_csv(RESULTS_DIR / "model_comparison_zipcode.csv")
    imp.to_csv(RESULTS_DIR / "feature_importance_zipcode.csv", header=["importance"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Feature Importance e Comparacao com/sem zipcode", fontsize=13)

    imp.head(12).sort_values().plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title("Top 12 features - XGBoost (+ zipcode)")
    axes[0].set_xlabel("Importance")

    comparativo = pd.DataFrame([
        {"Versao": "Sem zipcode", "MAE": 68.16, "R2": 0.481},
        {"Versao": "Com zipcode (XGBoost)", "MAE": res_df.loc[res_df["Modelo"].str.contains("XGBoost"), "MAE"].values[0],
         "R2": res_df.loc[res_df["Modelo"].str.contains("XGBoost"), "R2"].values[0]},
        {"Versao": "Com zipcode (LightGBM)", "MAE": res_df.loc[res_df["Modelo"].str.contains("LightGBM"), "MAE"].values[0],
         "R2": res_df.loc[res_df["Modelo"].str.contains("LightGBM"), "R2"].values[0]},
    ])
    colors = ["#e74c3c", "#2ecc71", "#27ae60"]
    axes[1].barh(comparativo["Versao"], comparativo["MAE"], color=colors)
    axes[1].set_xlabel("MAE (kWh/conn) - menor e melhor")
    axes[1].set_title("MAE: sem vs com zipcode_from")
    for i, (v, r2) in enumerate(zip(comparativo["MAE"], comparativo["R2"])):
        axes[1].text(v + 0.2, i, f"{v:.1f}  (R2={r2:.3f})", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison_zipcode.png", dpi=120, bbox_inches="tight")
    print(f"\nPlots salvos em {RESULTS_DIR}")

    return res_df, imp


if __name__ == "__main__":
    main()
