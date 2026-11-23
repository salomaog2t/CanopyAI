"""
CanopyAI — verificação do Passo 2: grade de 30 m, cache e features de vizinhança.

O que este script checa:
  1. Extração: grade alinhada (bordas em 15 mod 30 m), camadas com valores plausíveis.
  2. Cache: a segunda chamada vem do Parquet, sem GEE, com dados idênticos.
  3. Features: média de NDVI e fração arbórea em discos de 90/150/300/600 m.
  4. Poder explicativo sobre a LST: regressão linear com validação cruzada ESPACIAL
     (blocos de 1,5 km, maiores que o maior raio), comparando
        A  pixel isolado (NDVI + tree_frac)
        B  A + vizinhança de um raio
        C  A + todos os raios
        D  pixel com controles (A + NDBI + built_frac + elevação)
        E  D + todos os raios
     A validação espacial evita que a autocorrelação espacial infle o R².

Uso:
  python scripts/verify_step2.py
  python scripts/verify_step2.py --bbox -46.75,-23.65,-46.55,-23.50 --refresh
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from features.spatial import DEFAULT_RADII_M, add_neighborhood_features, feature_name  # noqa: E402
from gee.extract import ExtractParams, extract_grid  # noqa: E402

DEFAULT_BBOX = (-46.75, -23.65, -46.55, -23.50)  # centro expandido de São Paulo
MIN_CLEAR_OBS = 3
BLOCK_M = 1500
N_FOLDS = 5
MIN_GAIN_R2 = 0.02  # ganho mínimo de R² (validação espacial) para dizer que a vizinhança agrega


def load_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    env_bbox = os.getenv("AOI_BBOX", "").strip()

    def env_int(name, default):
        v = os.getenv(name, "").strip()
        return int(v) if v else default

    parser = argparse.ArgumentParser(description="CanopyAI — verificação do Passo 2")
    parser.add_argument("--bbox", type=lambda s: tuple(float(v) for v in s.split(",")),
                        default=tuple(float(v) for v in env_bbox.split(",")) if env_bbox else DEFAULT_BBOX)
    parser.add_argument("--years", type=int, nargs=2,
                        default=(env_int("YEAR_START", 2022), env_int("YEAR_END", 2025)))
    parser.add_argument("--months", type=int, nargs=2,
                        default=(env_int("DRY_SEASON_START_MONTH", 6), env_int("DRY_SEASON_END_MONTH", 9)))
    parser.add_argument("--refresh", action="store_true", help="ignora o cache e baixa de novo do GEE")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checagens
# ---------------------------------------------------------------------------
def check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FALHA'}] {name}: {detail}")
    return ok


def check_grid(grid) -> bool:
    df, g = grid.df, grid.meta["grid"]
    land = ~df["is_water"]
    ok = True
    ok &= check("alinhamento", g["x_min"] % 30 == 15 and g["y_max"] % 30 == 15,
                f"{g['crs']}, origem ({g['x_min']:.0f}, {g['y_max']:.0f}) — bordas em 15 mod 30 m")
    lst_valid = df.loc[land, "LST"].notna().mean()
    ok &= check("cobertura LST", lst_valid > 0.95, f"{lst_valid:.1%} das células de terra com LST")
    lst = df.loc[land, "LST"]
    ok &= check("LST plausível", -10 < lst.quantile(0.01) and lst.quantile(0.99) < 70,
                f"p1={lst.quantile(0.01):.1f} °C, p99={lst.quantile(0.99):.1f} °C")
    fsum = df[["tree_frac", "built_frac", "water_frac"]].sum(axis=1)
    ok &= check("frações WorldCover", (fsum <= 1.0 + 1e-4).all() and df["tree_frac"].between(0, 1).all(),
                f"árvore média={df['tree_frac'].mean():.2f}, construído={df['built_frac'].mean():.2f}, "
                f"água={df['water_frac'].mean():.3f}; soma ≤ 1")
    elev = df["elevation"]
    ok &= check("elevação SRTM", elev.notna().all() and elev.min() > -50,
                f"{elev.min():.0f}–{elev.max():.0f} m")
    ok &= check("observações limpas", df["clear_obs"].median() >= MIN_CLEAR_OBS,
                f"mediana {df['clear_obs'].median():.0f} por célula ({grid.meta['n_scenes']} cenas)")
    return ok


def check_cache(args, first) -> bool:
    t0 = time.perf_counter()
    second = extract_grid(args.bbox, ExtractParams(years=args.years, months=args.months), verbose=False)
    dt = time.perf_counter() - t0
    same = second.df.equals(first.df)
    return check("cache Parquet", second.from_cache and same,
                 f"2ª chamada lida de {second.path.name} em {dt:.2f} s, dados idênticos={same}")


def check_features(feat, shape) -> bool:
    ok = True
    # célula interior: a média em 90 m precisa bater com uma conta direta
    h, w = shape
    nd = feat["NDVI"].to_numpy().reshape(shape)
    water = feat["is_water"].to_numpy().reshape(shape)
    r, c = h // 2, w // 2
    yy, xx = np.mgrid[-3:4, -3:4]
    disk = (yy**2 + xx**2) <= 9
    win = nd[r - 3 : r + 4, c - 3 : c + 4]
    m = disk & ~water[r - 3 : r + 4, c - 3 : c + 4] & np.isfinite(win)
    expected = win[m].mean()
    got = feat[feature_name("NDVI", 90)].to_numpy().reshape(shape)[r, c]
    ok &= check("média focal (conta direta)", water[r, c] or abs(got - expected) < 1e-5,
                f"NDVI_r090 no centro = {got:.4f}, esperado {expected:.4f}")
    # borda: canto superior esquerdo usa só ~1/4 do disco de 600 m e ainda tem valor
    cov = feat["nbr_cov_r600"].to_numpy().reshape(shape)
    ok &= check("bordas", 0.15 < cov[0, 0] < 0.4 or water[0, 0],
                f"cobertura do disco de 600 m no canto = {cov[0, 0]:.2f} (média só com o que existe)")
    ok &= check("água excluída", feat.loc[feat["is_water"], feature_name("NDVI", 300)].isna().all(),
                f"{int(feat['is_water'].sum())} células de água sem feature")
    return ok


# ---------------------------------------------------------------------------
# Poder explicativo
# ---------------------------------------------------------------------------
def spatial_cv_r2(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[float, float]:
    pred = np.empty_like(y)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        beta, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ beta
    resid = y - pred
    r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    return r2, float(np.sqrt((resid**2).mean()))


def explanatory_power(feat, shape, scale_m) -> bool:
    h, w = shape
    n_edge = int(max(DEFAULT_RADII_M) // scale_m)
    interior = feat["row"].between(n_edge, h - 1 - n_edge) & feat["col"].between(n_edge, w - 1 - n_edge)

    pixel = ["NDVI", "tree_frac"]
    controls = pixel + ["NDBI", "built_frac", "elevation"]
    neigh = {r: [feature_name(c, r) for c in pixel] for r in DEFAULT_RADII_M}
    all_neigh = [c for cols in neigh.values() for c in cols]

    used = controls + all_neigh + ["LST"]
    sel = interior & ~feat["is_water"] & (feat["clear_obs"] >= MIN_CLEAR_OBS) & feat[used].notna().all(axis=1)
    data = feat.loc[sel]
    y = data["LST"].to_numpy(np.float64)

    block = int(BLOCK_M // scale_m)
    block_id = (data["row"].to_numpy() // block) * (w // block + 1) + data["col"].to_numpy() // block
    rng = np.random.default_rng(42)
    uniq = np.unique(block_id)
    fold_of_block = dict(zip(uniq, rng.permutation(len(uniq)) % N_FOLDS))
    folds = np.array([fold_of_block[b] for b in block_id])

    print(f"\nAmostra: {len(data):,} células de terra no interior (≥ {max(DEFAULT_RADII_M)} m da borda), "
          f"{len(uniq)} blocos de {BLOCK_M / 1000:.1f} km em {N_FOLDS} folds")

    print("\nCorrelação de Pearson com a LST:")
    for c in pixel:
        row = [f"pixel {data[c].corr(data['LST']):+.3f}"]
        row += [f"{r} m {data[feature_name(c, r)].corr(data['LST']):+.3f}" for r in DEFAULT_RADII_M]
        print(f"  {c:<10} " + " | ".join(row))

    def fit(cols):
        return spatial_cv_r2(data[cols].to_numpy(np.float64), y, folds)

    # (chave, rótulo, colunas, chave do modelo de referência para o ΔR²)
    models = [("A", "A  pixel isolado (NDVI + árvore)", pixel, None)]
    models += [(f"B{r}", f"B  A + vizinhança {r} m", pixel + neigh[r], "A") for r in DEFAULT_RADII_M]
    models += [("C", "C  A + todos os raios", pixel + all_neigh, "A"),
               ("D", "D  pixel + controles (NDBI, construído, elevação)", controls, None),
               ("E", "E  D + todos os raios", controls + all_neigh, "D")]

    print(f"\n{'Modelo':<52}{'R² (CV espacial)':>17}{'RMSE °C':>9}{'ΔR²':>8}")
    res = {}
    for key, label, cols, base in models:
        r2, rmse = fit(cols)
        res[key] = r2
        delta = "" if base is None else f"{r2 - res[base]:+.3f}"
        print(f"{label:<52}{r2:>17.3f}{rmse:>9.2f}{delta:>8}")

    best_r = max(DEFAULT_RADII_M, key=lambda r: res[f"B{r}"])
    print(f"\nMelhor raio isolado: {best_r} m")

    gain_pixel = res["C"] - res["A"]
    gain_ctrl = res["E"] - res["D"]
    ok = check("vizinhança × pixel isolado", gain_pixel >= MIN_GAIN_R2, f"ΔR² = {gain_pixel:+.3f} (mínimo {MIN_GAIN_R2})")
    check("vizinhança × pixel com controles", gain_ctrl >= MIN_GAIN_R2 / 2,
          f"ΔR² = {gain_ctrl:+.3f} (informativo)")
    return ok


def main() -> int:
    args = load_args()
    params = ExtractParams(years=args.years, months=args.months)
    print(f"AOI={args.bbox}  anos={args.years[0]}–{args.years[1]}  meses={args.months[0]}–{args.months[1]}")

    print("\n=== 1. Extração ===")
    t0 = time.perf_counter()
    grid = extract_grid(args.bbox, params, refresh=args.refresh)
    print(f"Grade {grid.shape[0]}×{grid.shape[1]} px ({grid.shape[0] * grid.shape[1]:,} células) "
          f"em {time.perf_counter() - t0:.1f} s")
    ok = check_grid(grid)

    print("\n=== 2. Cache ===")
    ok &= check_cache(args, grid)

    print("\n=== 3. Features de vizinhança ===")
    t0 = time.perf_counter()
    feat = add_neighborhood_features(grid.df, grid.shape, pixel_m=grid.scale_m)
    print(f"{len(DEFAULT_RADII_M) * 2} features em {time.perf_counter() - t0:.1f} s")
    ok &= check_features(feat, grid.shape)

    print("\n=== 4. Poder explicativo sobre a LST ===")
    ok &= explanatory_power(feat, grid.shape, grid.scale_m)

    print("\nResultado:", "Passo 2 verificado." if ok else "há checagens com FALHA — revisar acima.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
