"""
CanopyAI — Passo 3: modelos preditivos da anomalia de LST.

Alvo: anomalia de LST = LST − mediana da LST das células de terra da cena (°C). Tirar a
mediana remove o nível de temperatura de cada cena e deixa só o contraste dentro da cidade.

Modelos comparados:
    M0  baseline   — prevê a média da anomalia no treino (R² ≈ 0 por construção)
    M1  Ridge      — linear sobre features padronizadas; coeficientes interpretáveis
    M2  HistGradientBoostingRegressor com monotonic_cst:
          decrescente em NDVI, tree_frac e todas as features de vizinhança de vegetação
          crescente em built_frac
          livre em NDBI e elevação
        A restrição garante que o otimizador nunca "descubra" que plantar árvore esquenta.

Validação: compara KFold aleatório com GroupKFold em blocos espaciais de 1,5 km. Células
vizinhas são quase iguais, então no KFold aleatório o teste "vaza" para o treino e o R²
sai otimista; a diferença entre os dois é o viés de autocorrelação espacial.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features.spatial import DEFAULT_RADII_M, add_neighborhood_features, feature_name

VEGETATION_COLUMNS = ("NDVI", "tree_frac")
PIXEL_FEATURES = ("NDVI", "tree_frac", "NDBI", "built_frac", "elevation")
NEIGHBOR_FEATURES = tuple(feature_name(c, r) for c in VEGETATION_COLUMNS for r in DEFAULT_RADII_M)
FEATURES = PIXEL_FEATURES + NEIGHBOR_FEATURES

# −1 decrescente, +1 crescente, 0 livre
MONOTONIC = {**{f: -1 for f in VEGETATION_COLUMNS + NEIGHBOR_FEATURES}, "built_frac": +1}

TARGET = "LST_anom"
MIN_CLEAR_OBS = 3
BLOCK_M = 1500.0
N_SPLITS = 5
SEED = 42


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def monotonic_constraints(features=FEATURES) -> list[int]:
    return [MONOTONIC.get(f, 0) for f in features]


def spatial_blocks(rows: np.ndarray, cols: np.ndarray, block_px: int) -> np.ndarray:
    """Id do bloco quadrado de block_px × block_px células a que cada célula pertence."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    n_block_cols = int(cols.max()) // block_px + 1
    return (rows // block_px) * n_block_cols + cols // block_px


def build_feature_frame(grid_df: pd.DataFrame, shape, pixel_m: float = 30.0) -> pd.DataFrame:
    """Grade completa + features de vizinhança + anomalia de LST (todas as células, em ordem)."""
    df = grid_df if set(NEIGHBOR_FEATURES) <= set(grid_df.columns) else add_neighborhood_features(
        grid_df, shape, pixel_m=pixel_m
    )
    if TARGET not in df.columns:
        df = df.copy()
        df[TARGET] = df["LST"] - df.loc[~df["is_water"], "LST"].median()
    return df


def training_mask(df: pd.DataFrame, shape, pixel_m: float = 30.0, features=FEATURES) -> pd.Series:
    """Células de terra, com LST confiável, features completas e a ≥ maior raio da borda."""
    h, w = shape
    edge = int(max(DEFAULT_RADII_M) // pixel_m)
    interior = df["row"].between(edge, h - 1 - edge) & df["col"].between(edge, w - 1 - edge)
    return (
        interior
        & ~df["is_water"]
        & (df["clear_obs"] >= MIN_CLEAR_OBS)
        & df[list(features) + [TARGET]].notna().all(axis=1)
    )


def training_data(grid_df: pd.DataFrame, shape, pixel_m: float = 30.0, features=FEATURES):
    """(X, y, groups, df) prontos para o sklearn; groups = blocos de 1,5 km."""
    df = build_feature_frame(grid_df, shape, pixel_m)
    sel = training_mask(df, shape, pixel_m, features)
    data = df.loc[sel]
    X = data[list(features)].to_numpy(np.float64)
    y = data[TARGET].to_numpy(np.float64)
    groups = spatial_blocks(data["row"].to_numpy(), data["col"].to_numpy(), int(BLOCK_M // pixel_m))
    return X, y, groups, data


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def make_models(features=FEATURES, seed: int = SEED) -> dict:
    return {
        "M0 baseline (média)": DummyRegressor(strategy="mean"),
        "M1 Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "M2 HGB monotônico": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            l2_regularization=1.0,
            monotonic_cst=monotonic_constraints(features),
            early_stopping=False,  # a parada antecipada usaria uma validação aleatória, que vaza espacialmente
            random_state=seed,
        ),
    }


def fit_lst_model(X: np.ndarray, y: np.ndarray, name: str = "M2 HGB monotônico", features=FEATURES):
    return clone(make_models(features)[name]).fit(X, y)


def ridge_coefficients(pipeline, features=FEATURES) -> pd.Series:
    """Efeito em °C de +1 desvio-padrão de cada feature (modelo M1)."""
    ridge = pipeline[-1]
    return pd.Series(ridge.coef_, index=list(features)).sort_values()


# ---------------------------------------------------------------------------
# Validação cruzada: espacial × aleatória
# ---------------------------------------------------------------------------
def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: dict | None = None,
    n_splits: int = N_SPLITS,
    max_rows: int | None = 150_000,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    R², RMSE e MAE fora da amostra em dois esquemas:
        "espacial"  GroupKFold sobre os blocos de 1,5 km
        "aleatória" KFold embaralhado
    max_rows sorteia células (não blocos) para limitar o tempo de treino; os blocos continuam
    inteiros do lado espacial porque o grupo de cada célula é preservado.
    """
    models = models or make_models()
    if max_rows and len(y) > max_rows:
        idx = np.random.default_rng(seed).choice(len(y), max_rows, replace=False)
        X, y, groups = X[idx], y[idx], groups[idx]

    schemes = {
        "espacial": list(GroupKFold(n_splits=n_splits).split(X, y, groups)),
        "aleatória": list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y)),
    }
    rows = []
    for name, model in models.items():
        for scheme, splits in schemes.items():
            t0 = time.perf_counter()
            pred = np.empty_like(y)
            for tr, te in splits:
                pred[te] = clone(model).fit(X[tr], y[tr]).predict(X[te])
            rows.append({
                "modelo": name,
                "esquema": scheme,
                "R2": r2_score(y, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
                "MAE": mean_absolute_error(y, pred),
                "segundos": time.perf_counter() - t0,
            })
            if verbose:
                r = rows[-1]
                print(f"  {name:<22} {scheme:<9} R²={r['R2']:.3f}  RMSE={r['RMSE']:.2f} °C  ({r['segundos']:.0f} s)")
    return pd.DataFrame(rows)


def spatial_bias_table(cv: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por modelo: R² espacial, R² aleatório e o otimismo do KFold aleatório."""
    t = cv.pivot(index="modelo", columns="esquema", values=["R2", "RMSE"])
    out = pd.DataFrame({
        "R2_espacial": t[("R2", "espacial")],
        "R2_aleatoria": t[("R2", "aleatória")],
        "RMSE_espacial": t[("RMSE", "espacial")],
        "RMSE_aleatoria": t[("RMSE", "aleatória")],
    })
    out["vies_R2"] = out["R2_aleatoria"] - out["R2_espacial"]
    return out


# ---------------------------------------------------------------------------
# Consistência física
# ---------------------------------------------------------------------------
def monotonicity_check(model, X: np.ndarray, features=FEATURES, step: float = 0.1, n: int = 20_000,
                       seed: int = SEED) -> pd.DataFrame:
    """
    Soma `step` a cada feature restrita e mede a maior mudança na direção proibida.
    Com monotonic_cst correto, `violacao_max` é 0 para todas.
    """
    rng = np.random.default_rng(seed)
    Xs = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    base = model.predict(Xs)
    rows = []
    for k, f in enumerate(features):
        sign = MONOTONIC.get(f, 0)
        if sign == 0:
            continue
        Xp = Xs.copy()
        Xp[:, k] += step
        diff = model.predict(Xp) - base
        rows.append({
            "feature": f,
            "esperado": "decrescente" if sign < 0 else "crescente",
            "efeito_medio": diff.mean(),
            "violacao_max": max(0.0, float((diff * sign * -1).max())),  # quanto foi na direção errada
        })
    return pd.DataFrame(rows)
