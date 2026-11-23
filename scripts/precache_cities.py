"""
CanopyAI — pré-carrega as cidades de demonstração do app (Passo 4).

Para cada cidade de app_core/cities.py:
  1. extrai a grade de 30 m do Earth Engine (só se ainda não estiver no cache Parquet);
  2. ajusta o M2 e calcula o plano padrão (5.000 árvores, 50 m²) nos dois objetivos,
     para que a primeira visita ao app seja instantânea.

Uso:
  python scripts/precache_cities.py            # todas as cidades
  python scripts/precache_cities.py --no-plans # só os dados
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app_core.cities import CITIES  # noqa: E402
from app_core.plan import PlanParams, prepare_area, run_plan  # noqa: E402
from gee.extract import extract_grid  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--no-plans", action="store_true")
    args = p.parse_args()

    for city in CITIES:
        print(f"\n=== {city.name} ===")
        t0 = time.perf_counter()
        grid = extract_grid(city.bbox, city.params)
        df = grid.df
        land = ~df["is_water"]
        print(f"  grade {grid.shape[0]}×{grid.shape[1]}, {grid.meta['n_scenes']} cenas, "
              f"LST mediana {df.loc[land, 'LST'].median():.1f} °C, árvore {df['tree_frac'].mean():.0%}, "
              f"população {df['population'].sum():,.0f}")
        if args.no_plans:
            continue
        area = prepare_area(grid)
        print(f"  M2: {area.n_train:,} células, R² no treino {area.r2_in_sample:.2f}")
        for objective in ("cooling", "population"):
            res = run_plan(grid, area, PlanParams(objective=objective))
            m = res["metrics"]
            g, r, h = (m[s]["soma_ponderada" if objective == "population" else "soma_dT"] for s in
                       ("Lazy Greedy", "Random", "Hottest first"))
            src = "cache" if res["from_cache"] else f"{res['runtime_s']:.0f} s"
            print(f"  plano [{objective}]: greedy {g / r:.2f}× aleatória, {g / h:.2f}× mais quente ({src})")
        print(f"  total {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
