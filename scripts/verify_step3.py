"""
CanopyAI — verificação do Passo 3: modelos de LST e alocação de árvores.

  1. Dados: grade do Passo 2 (lida do cache) + features de vizinhança + anomalia de LST.
  2. Modelos M0/M1/M2 com validação cruzada espacial (blocos de 1,5 km) e aleatória, e o
     viés de autocorrelação espacial (R² aleatório − R² espacial).
  3. Consistência física do M2: nenhuma violação das restrições monotônicas.
  4. Alocação de N árvores (padrão 5.000): Lazy Greedy × aleatória × "mais quente primeiro",
     todas avaliadas do mesmo jeito (intervenção + convolução na grade inteira + M2).

Uso:
  python scripts/verify_step3.py
  python scripts/verify_step3.py --budget 5000 --canopy-area 50 --step 5
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
from gee.extract import ExtractParams, extract_grid  # noqa: E402
from models.lst_model import (  # noqa: E402
    FEATURES,
    build_feature_frame,
    cross_validate_models,
    fit_lst_model,
    make_models,
    monotonicity_check,
    ridge_coefficients,
    spatial_bias_table,
    training_data,
)
from optimize.allocate import (  # noqa: E402
    AllocationProblem,
    InterventionParams,
    hotspot_allocation,
    lazy_greedy,
    random_allocation,
)

DEFAULT_BBOX = (-46.75, -23.65, -46.55, -23.50)
N_RANDOM_SEEDS = 5


def load_args() -> argparse.Namespace:
    load_dotenv(ROOT / ".env")
    env_bbox = os.getenv("AOI_BBOX", "").strip()

    def env_int(name, default):
        v = os.getenv(name, "").strip()
        return int(v) if v else default

    p = argparse.ArgumentParser(description="CanopyAI — verificação do Passo 3")
    p.add_argument("--bbox", type=lambda s: tuple(float(v) for v in s.split(",")),
                   default=tuple(float(v) for v in env_bbox.split(",")) if env_bbox else DEFAULT_BBOX)
    p.add_argument("--years", type=int, nargs=2, default=(env_int("YEAR_START", 2022), env_int("YEAR_END", 2025)))
    p.add_argument("--months", type=int, nargs=2,
                   default=(env_int("DRY_SEASON_START_MONTH", 6), env_int("DRY_SEASON_END_MONTH", 9)))
    p.add_argument("--budget", type=int, default=5000, help="número de árvores a alocar")
    p.add_argument("--canopy-area", type=float, default=50.0, help="área de copa madura por árvore (m²)")
    p.add_argument("--step", type=int, default=5, help="árvores por ação do guloso")
    p.add_argument("--pool", type=int, default=6000, help="candidatas avaliadas exatamente pelo guloso")
    p.add_argument("--cv-rows", type=int, default=150_000, help="células sorteadas para a validação cruzada")
    return p.parse_args()


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FALHA'}] {name}: {detail}")
    return ok


def main() -> int:
    args = load_args()
    ok = True

    print("=== 1. Dados ===")
    grid = extract_grid(args.bbox, ExtractParams(years=args.years, months=args.months))
    t0 = time.perf_counter()
    feat = build_feature_frame(grid.df, grid.shape, grid.scale_m)
    X, y, groups, _ = training_data(feat, grid.shape, grid.scale_m)
    print(f"{len(y):,} células de treino, {len(np.unique(groups))} blocos de 1,5 km, "
          f"{len(FEATURES)} features ({time.perf_counter() - t0:.1f} s)")
    print(f"Anomalia de LST: desvio-padrão {y.std():.2f} °C, p5={np.percentile(y, 5):+.2f}, p95={np.percentile(y, 95):+.2f}")

    print("\n=== 2. Validação cruzada: espacial × aleatória ===")
    cv = cross_validate_models(X, y, groups, make_models(), max_rows=args.cv_rows)
    bias = spatial_bias_table(cv)
    print(f"\n{'Modelo':<22}{'R² espacial':>13}{'R² aleatória':>14}{'viés R²':>9}{'RMSE esp.':>11}{'RMSE alea.':>12}")
    for name, r in bias.iterrows():
        print(f"{name:<22}{r.R2_espacial:>13.3f}{r.R2_aleatoria:>14.3f}{r.vies_R2:>+9.3f}"
              f"{r.RMSE_espacial:>10.2f}°{r.RMSE_aleatoria:>11.2f}°")
    m1, m2 = bias.loc["M1 Ridge"], bias.loc["M2 HGB monotônico"]
    ok &= check("M2 supera M1 (CV espacial)", m2.R2_espacial > m1.R2_espacial,
                f"R² {m2.R2_espacial:.3f} vs {m1.R2_espacial:.3f}")
    ok &= check("M0 sem poder explicativo", abs(bias.loc["M0 baseline (média)"].R2_espacial) < 0.05,
                f"R² = {bias.loc['M0 baseline (média)'].R2_espacial:+.3f}")
    check("viés da CV aleatória", (bias.vies_R2 >= 0).all(),
          "a CV aleatória é otimista em todos os modelos (informativo)" if (bias.vies_R2 >= 0).all()
          else "a CV aleatória não foi otimista em algum modelo (informativo)")

    print("\nM1 Ridge — efeito em °C de +1 desvio-padrão (ajustado em todos os dados):")
    ridge = fit_lst_model(X, y, "M1 Ridge")
    for f, v in ridge_coefficients(ridge).items():
        print(f"  {f:<16}{v:+.3f}")

    print("\n=== 3. Consistência física do M2 ===")
    t0 = time.perf_counter()
    model = fit_lst_model(X, y, "M2 HGB monotônico")
    print(f"M2 ajustado em {len(y):,} células em {time.perf_counter() - t0:.0f} s")
    mono = monotonicity_check(model, X)
    worst = mono["violacao_max"].max()
    for _, r in mono.iterrows():
        print(f"  {r.feature:<16}{r.esperado:<12} efeito médio de +0,1: {r.efeito_medio:+.4f} °C")
    ok &= check("restrições monotônicas", worst <= 1e-9, f"maior violação = {worst:.2e} °C")

    print(f"\n=== 4. Alocação de {args.budget:,} árvores (copa {args.canopy_area:.0f} m²) ===")
    t0 = time.perf_counter()
    problem = AllocationProblem(feat, grid.shape, model, params=InterventionParams(canopy_area_m2=args.canopy_area),
                                pixel_m=grid.scale_m)
    print(f"Problema montado em {time.perf_counter() - t0:.1f} s: {int((problem.capacity > 0).sum()):,} células "
          f"com espaço, capacidade total {int(problem.capacity.sum()):,} árvores, "
          f"NDVI de copa (P95) = {problem.ndvi_tree:.3f}")

    print("\nLazy Greedy:")
    greedy = lazy_greedy(problem, args.budget, step=args.step, pool_size=args.pool)
    print(f"  {greedy.runtime_s:.0f} s, {greedy.n_evaluations:,} avaliações exatas, "
          f"{greedy.fallback_trees} árvores sem avaliação")

    print("\nAvaliando as estratégias na grade inteira (intervenção + convolução + M2)...")
    results = {"Lazy Greedy": problem.evaluate(greedy.trees)}
    results["Mais quente primeiro"] = problem.evaluate(hotspot_allocation(problem, args.budget).trees)
    rand = [problem.evaluate(random_allocation(problem, args.budget, seed=s).trees) for s in range(N_RANDOM_SEEDS)]
    rand_sums = np.array([r["soma_dT"] for r in rand])
    results[f"Aleatória (média de {N_RANDOM_SEEDS})"] = {
        k: float(np.mean([r[k] for r in rand])) for k in rand[0] if k != "mapa_dT"
    }

    print(f"\n{'Estratégia':<24}{'Σ ΔT °C·cél.':>13}{'°C·ha':>8}{'ΔT méd. m°C':>12}{'ΔT máx °C':>11}"
          f"{'cél. ≥0,1°C':>12}{'células':>9}")
    for name, r in results.items():
        print(f"{name:<24}{r['soma_dT']:>13.1f}{r['soma_dT'] * 0.09:>8.1f}{r['dT_medio_mC']:>12.2f}"
              f"{r['dT_max']:>11.3f}{r['celulas_dT_ge_0.1']:>12.0f}{r['celulas_plantadas']:>9.0f}")
    g = results["Lazy Greedy"]["soma_dT"]
    h = results["Mais quente primeiro"]["soma_dT"]
    print(f"\nAleatória: Σ ΔT = {rand_sums.mean():.1f} ± {rand_sums.std(ddof=1):.1f} °C·célula ({N_RANDOM_SEEDS} sorteios)")
    print(f"Ganho do Lazy Greedy: {g / rand_sums.mean():.2f}× a aleatória, {g / h:.2f}× a 'mais quente primeiro'")

    print()
    ok &= check("orçamento", all(int(r["arvores"]) == args.budget for r in results.values()),
                f"{args.budget:,} árvores em todas as estratégias")
    ok &= check("Lazy Greedy > aleatória", g > rand_sums.max(), f"{g:.1f} vs máx. {rand_sums.max():.1f} °C·célula")
    ok &= check("Lazy Greedy > mais quente primeiro", g > h, f"{g:.1f} vs {h:.1f} °C·célula")
    warm = max(results["Lazy Greedy"]["aquecimento_max"], results["Mais quente primeiro"]["aquecimento_max"],
               max(r["aquecimento_max"] for r in rand))
    ok &= check("nenhuma célula aquece", warm <= 1e-9, f"maior aquecimento = {warm:.2e} °C")
    rel = abs(greedy.predicted_gain - g) / max(abs(g), 1e-9)
    ok &= check("atualização local = convolução completa", rel < 0.02,
                f"guloso estimou {greedy.predicted_gain:.1f}, grade inteira deu {g:.1f} (diferença {rel:.2%})")

    print("\nResultado:", "Passo 3 verificado." if ok else "há checagens com FALHA — revisar acima.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
