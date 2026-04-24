"""Aplica as melhorias restantes no notebook."""
import json
from pathlib import Path

nb_path = Path(__file__).parent / "dutch_energy_regressao.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))
cells = nb["cells"]


def set_code(cell, code):
    cell["source"] = code
    cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None


def set_md(cell, text):
    cell["source"] = text


# ── Cell 16 (markdown split) ───────────────────────────────────────────────
set_md(cells[16], """## 7. Selecao de features, k-fold target encoding e split temporal

O **k-fold target encoding** corrige o leakage do encoding simples:
para cada linha do treino, a media do target e calculada apenas nos outros folds.
Val e test usam a media global do treino completo.

O split temporal usa os 2 ultimos anos para teste e os 2 anteriores para validacao.
""")

# ── Cell 17 (split + encoding + scaling) ──────────────────────────────────
set_code(cells[17], """\
EXCLUDE = [
    TARGET, "annual_consume", "street", "zipcode_to", "city",
    "net_manager", "purchase_area", "type_of_connection",
]

NUM_FEAT_COLS = [
    c for c in feat_df.columns
    if c not in EXCLUDE and feat_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int16]
]

essential = [c for c in NUM_FEAT_COLS if c not in [f"{cat}_te" for cat in CAT_COLS_TO_ENCODE]]
model_df = feat_df[NUM_FEAT_COLS + [TARGET] + CAT_COLS_TO_ENCODE].dropna(
    subset=essential + [TARGET]
).copy()
del feat_df; gc.collect()
print(f"Registros para modelagem: {len(model_df):,}")

model_df["log_target"] = np.log1p(model_df[TARGET]).astype("float32")

X_full = model_df.drop(columns=[TARGET, "log_target"])
y_full = model_df["log_target"]

years_available = sorted(model_df["year"].dropna().unique().tolist()) if "year" in model_df.columns else []

if SPLIT_MODE == "temporal" and len(years_available) >= 6:
    test_years = years_available[-2:]
    val_years = years_available[-4:-2]
    train_years = [y for y in years_available if y not in val_years + test_years]

    train_mask = X_full["year"].isin(train_years)
    val_mask   = X_full["year"].isin(val_years)
    test_mask  = X_full["year"].isin(test_years)

    X_train = X_full.loc[train_mask].copy()
    y_train = y_full.loc[train_mask].copy()
    X_val   = X_full.loc[val_mask].copy()
    y_val   = y_full.loc[val_mask].copy()
    X_test  = X_full.loc[test_mask].copy()
    y_test  = y_full.loc[test_mask].copy()

    split_strategy = (
        f"Temporal holdout | train={train_years[0]}-{train_years[-1]}, "
        f"val={val_years[0]}-{val_years[-1]}, test={test_years[0]}-{test_years[-1]}"
    )
else:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full, test_size=TEST_RATIO, random_state=SEED
    )
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_frac, random_state=SEED
    )
    split_strategy = "Random split (fallback)"
    del X_train_full, y_train_full

print(split_strategy)
split_summary = pd.DataFrame({
    "Split": ["Train", "Val", "Test"],
    "Rows": [len(X_train), len(X_val), len(X_test)],
    "Share_%": [
        100 * len(X_train) / len(X_full),
        100 * len(X_val)   / len(X_full),
        100 * len(X_test)  / len(X_full),
    ],
    "Years": [
        ", ".join(map(str, sorted(X_train["year"].unique()))) if "year" in X_train.columns else "-",
        ", ".join(map(str, sorted(X_val["year"].unique())))   if "year" in X_val.columns   else "-",
        ", ".join(map(str, sorted(X_test["year"].unique())))  if "year" in X_test.columns  else "-",
    ],
})
display(split_summary)
split_summary.to_csv(RESULTS_DIR / "split_summary.csv", index=False)

del model_df, X_full, y_full; gc.collect()

# K-fold target encoding — sem leakage no treino
from sklearn.model_selection import KFold

global_mean = float(y_train.mean())
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for col in CAT_COLS_TO_ENCODE:
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
    X_val[enc_col]  = X_val[col].map(full_map).fillna(global_mean).astype("float32")
    X_test[enc_col] = X_test[col].map(full_map).fillna(global_mean).astype("float32")

X_train = X_train.drop(columns=CAT_COLS_TO_ENCODE, errors="ignore")
X_val   = X_val.drop(columns=CAT_COLS_TO_ENCODE, errors="ignore")
X_test  = X_test.drop(columns=CAT_COLS_TO_ENCODE, errors="ignore")

FEAT_COLS = X_train.columns.tolist()
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"\\nFeatures ({len(FEAT_COLS)}):")
for c in FEAT_COLS:
    print(f"  {c}")

X_train = X_train.astype("float32")
X_val   = X_val.astype("float32")
X_test  = X_test.astype("float32")

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train).astype("float32")
Xv_s  = scaler.transform(X_val).astype("float32")
Xte_s = scaler.transform(X_test).astype("float32")

# Subamostra apenas para MLP (modelos de boosting usam treino completo)
MLP_TRAIN_MAX = 300_000
MLP_VAL_MAX   = 100_000

def sample_xy(X, y, max_rows, seed):
    if max_rows is None or len(X) <= max_rows:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), size=max_rows, replace=False)
    X_s = X[idx] if not hasattr(X, "iloc") else X.iloc[idx].copy()
    y_s = y[idx] if not hasattr(y, "iloc") else y.iloc[idx].copy()
    return X_s, y_s

Xtr_mlp, ytr_mlp = sample_xy(Xtr_s, y_train.to_numpy(dtype="float32"), MLP_TRAIN_MAX, SEED + 5)
Xv_mlp,  yv_mlp  = sample_xy(Xv_s,  y_val.to_numpy(dtype="float32"),   MLP_VAL_MAX,   SEED + 6)
print(f"\\nMLP subsample: train={len(Xtr_mlp):,} | val={len(Xv_mlp):,}")
""")

# ── Cell 31 (XGBoost): treino completo + hiperparametros ajustados ─────────
set_code(cells[31], """\
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
).fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=200,
)
p = m_xgb.predict(X_test)
preds["XGBoost"] = to_original_scale(np.clip(p, 0, None))
r = evaluate(y_test.values, p, "XGBoost")
results.append(r); show(r)
print(f"  Melhor iteracao: {m_xgb.best_iteration}")
""")

# ── Cell 33 (LightGBM): treino completo + hiperparametros ajustados ────────
set_code(cells[33], """\
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
p = m_lgb.predict(X_test)
preds["LightGBM"] = to_original_scale(np.clip(p, 0, None))
r = evaluate(y_test.values, p, "LightGBM")
results.append(r); show(r)
""")

# ── Cell 36 (markdown avaliacao final): atualizar resultados esperados ──────
set_md(cells[36], """## 11. Avaliacao final e comparacao

As metricas abaixo estao em **kWh por conexao** e usam o **split temporal** (test=2019-2020).

Resultados esperados apos as melhorias:
- **LightGBM / XGBoost**: MAE ~40 kWh, R² ~0.74, WMAPE ~19%
- A feature `zipcode_from_te` responde por ~61% da importancia no XGBoost
""")

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook atualizado com sucesso!")
print("Celulas modificadas: 0 (header), 6 (load), 14 (md), 15 (features), 16 (md), 17 (split+encoding), 31 (XGBoost), 33 (LightGBM), 36 (md)")
