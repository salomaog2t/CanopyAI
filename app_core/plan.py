"""
Planting-plan pipeline used by the web app: fit the LST model for an area, run the three
strategies, cache the result on disk and export the plan (CSV / GeoJSON).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from app_core.geo import cell_lonlat, utm_to_lonlat
from gee.extract import DEFAULT_CACHE_DIR, DEMO_DIR, GridData
from models.lst_model import FEATURES, TARGET, build_feature_frame, fit_lst_model, training_data
from optimize.allocate import (
    AllocationProblem,
    InterventionParams,
    hotspot_allocation,
    lazy_greedy,
    random_allocation,
)

PLAN_CACHE_DIR = DEFAULT_CACHE_DIR / "plans"
PLAN_VERSION = 2  # bump when the model or the optimizer changes, to invalidate cached plans

STRATEGIES = ("Lazy Greedy", "Random", "Hottest first")


@dataclass(frozen=True)
class PlanParams:
    budget: int = 5000
    canopy_area_m2: float = 50.0
    objective: str = "cooling"  # "cooling" (Σ ΔLST) or "population" (Σ people·ΔLST)
    street_share: float = 0.15
    step: int = 5
    pool: int = 3000
    explore: int = 3000
    n_random: int = 3


@dataclass
class AreaModel:
    feat: pd.DataFrame
    model: object
    n_train: int
    r2_in_sample: float
    lst_median: float


def prepare_area(grid: GridData) -> AreaModel:
    """Neighbourhood features + monotonic HGB (M2) fitted on this area."""
    feat = build_feature_frame(grid.df, grid.shape, grid.scale_m)
    X, y, _, _ = training_data(feat, grid.shape, grid.scale_m)
    if len(y) < 2000:
        raise ValueError(f"only {len(y)} usable land cells — the area is too small or mostly water/cloud")
    model = fit_lst_model(X, y)
    lst_median = float((feat["LST"] - feat[TARGET]).median())
    return AreaModel(feat, model, len(y), float(r2_score(y, model.predict(X))), lst_median)


def _plan_key(grid: GridData, params: PlanParams) -> str:
    raw = json.dumps({"grid": grid.meta["cache_key"], "params": asdict(params), "v": PLAN_VERSION}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def plan_cache_path(grid: GridData, params: PlanParams):
    """Plano em cache: o local (data/cache/plans) ou o de demo versionado (data/demo/plans)."""
    name = f"plan_{_plan_key(grid, params)}.npz"
    demo = DEMO_DIR / "plans" / name
    return demo if demo.exists() else PLAN_CACHE_DIR / name


def _metrics(ev: dict) -> dict:
    return {k: v for k, v in ev.items() if k != "mapa_dT"}


def run_plan(grid: GridData, area: AreaModel, params: PlanParams, use_cache: bool = True) -> dict:
    path = plan_cache_path(grid, params)
    if use_cache and path.exists():
        with np.load(path, allow_pickle=False) as z:
            result = json.loads(str(z["meta"]))
            for k in ("trees_greedy", "trees_hotspot", "cooling_greedy", "cooling_hotspot", "history"):
                result[k] = z[k]
        result["from_cache"] = True
        return result

    t0 = time.perf_counter()
    if params.objective == "population" and "population" not in area.feat:
        raise ValueError("this area has no WorldPop layer; re-extract it to use the population objective")
    weights = area.feat["population"].to_numpy() if params.objective == "population" else None
    problem = AllocationProblem(
        area.feat, grid.shape, area.model, features=FEATURES, pixel_m=grid.scale_m, weights=weights,
        params=InterventionParams(canopy_area_m2=params.canopy_area_m2, street_share_of_built=params.street_share),
    )
    budget = int(min(params.budget, problem.capacity.sum()))
    greedy = lazy_greedy(problem, budget, step=params.step, pool_size=params.pool, explore=params.explore,
                         verbose=False)
    ev_greedy = problem.evaluate(greedy.trees)
    hotspot = hotspot_allocation(problem, budget).trees
    ev_hot = problem.evaluate(hotspot)
    rnd = [_metrics(problem.evaluate(random_allocation(problem, budget, seed=s).trees)) for s in range(params.n_random)]

    rnd_mean = {k: float(np.mean([r[k] for r in rnd])) for k in rnd[0]}
    rnd_mean["soma_dT_sd"] = float(np.std([r["soma_dT"] for r in rnd], ddof=1)) if len(rnd) > 1 else 0.0
    result = {
        "params": asdict(params),
        "budget_used": budget,
        "capacity_total": int(problem.capacity.sum()),
        "ndvi_tree": problem.ndvi_tree,
        "metrics": {"Lazy Greedy": _metrics(ev_greedy), "Random": rnd_mean, "Hottest first": _metrics(ev_hot)},
        "n_evaluations": greedy.n_evaluations,
        "fallback_trees": greedy.fallback_trees,
        "runtime_s": time.perf_counter() - t0,
    }
    arrays = {
        "trees_greedy": greedy.trees.astype(np.int16),
        "trees_hotspot": hotspot.astype(np.int16),
        "cooling_greedy": ev_greedy["mapa_dT"].astype(np.float32),
        "cooling_hotspot": ev_hot["mapa_dT"].astype(np.float32),
        "history": np.asarray(greedy.history, dtype=np.float64),
    }
    PLAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=json.dumps(result), **arrays)
    tmp.replace(path)
    result.update(arrays)
    result["from_cache"] = False
    return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def plan_table(grid: GridData, area: AreaModel, result: dict, canopy_area_m2: float) -> pd.DataFrame:
    trees = result["trees_greedy"].astype(np.int64)
    sel = np.flatnonzero(trees > 0)
    df = area.feat.iloc[sel]
    lon, lat = cell_lonlat(df, grid.meta)
    cell_area = grid.scale_m**2
    added = trees[sel] * canopy_area_m2 / cell_area
    out = pd.DataFrame({
        "cell_id": sel,
        "lon": np.round(lon, 6),
        "lat": np.round(lat, 6),
        "utm_x": df["x"].to_numpy(),
        "utm_y": df["y"].to_numpy(),
        "trees": trees[sel],
        "canopy_added_m2": trees[sel] * canopy_area_m2,
        "tree_cover_before": np.round(df["tree_frac"].to_numpy(), 3),
        "tree_cover_after": np.round(df["tree_frac"].to_numpy() + added, 3),
        "lst_c": np.round(df["LST"].to_numpy(), 2),
        "lst_anomaly_c": np.round(df[TARGET].to_numpy(), 2),
        "predicted_cooling_c": np.round(result["cooling_greedy"][sel], 3),
        "population": np.round(df["population"].to_numpy(), 1) if "population" in df else np.nan,
    })
    return out.sort_values(["trees", "predicted_cooling_c"], ascending=False).reset_index(drop=True)


def plan_geojson(table: pd.DataFrame, grid: GridData) -> str:
    """One polygon per 30 m cell (corners converted from UTM, so squares stay exact)."""
    g = grid.meta["grid"]
    half = g["scale"] / 2
    features = []
    for rec in table.to_dict("records"):
        xs = np.array([-half, half, half, -half, -half]) + rec["utm_x"]
        ys = np.array([-half, -half, half, half, -half]) + rec["utm_y"]
        lon, lat = utm_to_lonlat(xs, ys, g["crs"])
        props = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()
                 if k not in ("utm_x", "utm_y")}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[round(a, 7), round(b, 7)] for a, b in zip(lon, lat)]]},
            "properties": {k: (v.item() if hasattr(v, "item") else v) for k, v in props.items()},
        })
    return json.dumps({"type": "FeatureCollection", "features": features})
