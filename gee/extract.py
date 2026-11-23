"""
CanopyAI — Passo 2: extração da grade alinhada de 30 m com cache local em Parquet.

Para qualquer bbox (WGS84), baixa do Earth Engine uma grade regular em UTM com as camadas:

    LST, NDVI, NDBI   composto mediano da estação seca (Landsat 8/9 C2 L2)
    elevation         SRTM GL1 (30 m), reamostragem bilinear
    population        pessoas por célula (WorldPop GP 100 m R2025A, 2025, redistribuído por densidade)
    tree_frac         fração de árvore      (ESA WorldCover v200, classe 10) em cada célula
    built_frac        fração construída     (classe 50)
    water_frac        fração de água        (classe 80)
    clear_obs         nº de observações limpas de LST que entraram na mediana

Grade: a zona UTM vem do centro da bbox, e as bordas das células ficam em múltiplos de
30 m + 15 m, o mesmo arranjo da grade nativa do Landsat C2. Assim o LST e o NDVI não são
reamostrados, e duas bboxes sobrepostas na mesma zona caem exatamente nas mesmas células.

Cache: o resultado vai para data/cache/grid_<hash>.parquet. O hash cobre a bbox, os
parâmetros e CACHE_SCHEMA_VERSION. Se o arquivo já existe, ele é lido sem nenhuma chamada
ao Earth Engine (zero EECU).

Uso como biblioteca:
    from gee.extract import extract_grid, ExtractParams
    grid = extract_grid((-46.75, -23.65, -46.55, -23.50))
    lst = grid.array("LST")          # matriz (altura, largura) em °C

Uso pela linha de comando:
    python -m gee.extract --bbox -46.75,-23.65,-46.55,-23.50 [--years 2022 2025] [--months 6 9] [--refresh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gee.landsat import (
    MAX_SCENE_CLOUD,
    PROJECT_ROOT,
    WORLDCOVER_BUILT,
    WORLDCOVER_TREE,
    WORLDCOVER_WATER,
    dry_season_composite,
    init_earth_engine,
    landsat_collection,
    worldcover,
)

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"   # local, fora do git
DEMO_DIR = PROJECT_ROOT / "data" / "demo"              # versionado: cidades de demonstração do app

# Incremente sempre que mudar o processamento no servidor (máscaras, bandas, reamostragem):
# isso muda o hash e invalida o cache antigo, em vez de misturar versões.
CACHE_SCHEMA_VERSION = 3  # v2: + population (WorldPop); v3: WorldPop R2025A, ano 2025

BANDS = ("LST", "NDVI", "NDBI", "elevation", "tree_frac", "built_frac", "water_frac", "clear_obs", "population")
WORLDPOP_ASSET = "WorldPop/GP/100m/R2025A/v1/pop"  # substitui o antigo WorldPop/GP/100m/pop (obsoleto)
WORLDPOP_YEAR = 2025  # mesma janela do composto Landsat (2022–2025); a série vai até 2030 (projeções)
NODATA = -9999.0
TILE_PX = 512                 # 512×512 px × 8 bandas float32 ≈ 8 MB por requisição (limite do GEE: 48 MB)
MAX_WORKERS = 4               # requisições simultâneas ao computePixels
WORLDCOVER_MAX_PIXELS = 64    # células de 10 m agregadas por célula de 30 m (9–16 na prática)
WATER_THRESHOLD = 0.5         # célula com ≥ 50% de água é tratada como água
PARQUET_META_KEY = b"canopyai"


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractParams:
    years: tuple[int, int] = (2022, 2025)
    months: tuple[int, int] = (6, 9)
    max_scene_cloud: int = MAX_SCENE_CLOUD
    scale_m: int = 30

    def __post_init__(self) -> None:
        object.__setattr__(self, "years", tuple(int(v) for v in self.years))
        object.__setattr__(self, "months", tuple(int(v) for v in self.months))
        if self.years[0] > self.years[1]:
            raise ValueError("years: o ano inicial deve ser ≤ ao final.")
        if not all(1 <= m <= 12 for m in self.months):
            raise ValueError("months: os meses devem estar entre 1 e 12.")


@dataclass
class GridData:
    """Grade completa em ordem linha-maior: df tem altura × largura linhas, inclusive células sem dado (NaN)."""

    df: pd.DataFrame
    meta: dict
    path: Path
    from_cache: bool

    @property
    def shape(self) -> tuple[int, int]:
        g = self.meta["grid"]
        return g["height"], g["width"]

    @property
    def scale_m(self) -> float:
        return float(self.meta["grid"]["scale"])

    def array(self, column: str) -> np.ndarray:
        return self.df[column].to_numpy().reshape(self.shape)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def normalize_bbox(bbox) -> tuple[float, float, float, float]:
    values = tuple(round(float(v), 6) for v in bbox)
    if len(values) != 4:
        raise ValueError("bbox precisa de 4 valores: lon_min, lat_min, lon_max, lat_max")
    lon_min, lat_min, lon_max, lat_max = values
    if not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90):
        raise ValueError(f"bbox inválida: {values}")
    return values


def cache_key(bbox, params: ExtractParams) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "bbox": list(normalize_bbox(bbox)),
        "params": asdict(params),
        "bands": list(BANDS),
        "water_threshold": WATER_THRESHOLD,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_path(bbox, params: ExtractParams, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"grid_{cache_key(bbox, params)}.parquet"


def find_cached(bbox, params: ExtractParams, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path | None:
    """Arquivo em cache para a bbox: primeiro o cache local, depois os dados de demo versionados."""
    for d in (Path(cache_dir), DEMO_DIR):
        p = cache_path(bbox, params, d)
        if p.exists():
            return p
    return None


def load_grid(path: Path) -> GridData:
    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[PARQUET_META_KEY])
    if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"{path.name}: versão de schema {meta.get('schema_version')} ≠ {CACHE_SCHEMA_VERSION}")
    return GridData(df=table.to_pandas(), meta=meta, path=Path(path), from_cache=True)


def _save_grid(df: pd.DataFrame, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[PARQUET_META_KEY] = json.dumps(meta).encode()
    table = table.replace_schema_metadata(metadata)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)  # escrita atômica: uma extração interrompida nunca deixa cache corrompido


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------
def utm_epsg(bbox) -> str:
    lon_c = (bbox[0] + bbox[2]) / 2
    lat_c = (bbox[1] + bbox[3]) / 2
    zone = min(max(int((lon_c + 180) // 6) + 1, 1), 60)
    return f"EPSG:{(32600 if lat_c >= 0 else 32700) + zone}"


def _snap_grid(xs: list[float], ys: list[float], crs: str, scale: float) -> dict:
    """Menor grade alinhada (bordas em k·scale + scale/2) que cobre a bbox projetada."""
    half = scale / 2
    x_min = math.floor((min(xs) - half) / scale) * scale + half
    x_max = math.ceil((max(xs) - half) / scale) * scale + half
    y_min = math.floor((min(ys) - half) / scale) * scale + half
    y_max = math.ceil((max(ys) - half) / scale) * scale + half
    return {
        "crs": crs,
        "scale": scale,
        "x_min": x_min,
        "y_max": y_max,
        "width": int(round((x_max - x_min) / scale)),
        "height": int(round((y_max - y_min) / scale)),
    }


def _bbox_geometry(bbox) -> ee.Geometry:
    return ee.Geometry.Rectangle(list(bbox), proj="EPSG:4326", geodesic=False)


def build_image(bbox, params: ExtractParams, crs: str) -> tuple[ee.Image, ee.ImageCollection]:
    """Imagem multibanda (float32, sem máscara: NODATA onde não há dado) pronta para o computePixels."""
    aoi = _bbox_geometry(bbox)
    collection = landsat_collection(aoi, params.years, params.months, params.max_scene_cloud)
    composite, clear_obs = dry_season_composite(collection)

    scale = params.scale_m
    grid_proj = ee.Projection(crs, [scale, 0, scale / 2, 0, -scale, scale / 2])

    wc = worldcover()
    fractions = (
        ee.Image.cat(
            wc.eq(WORLDCOVER_TREE).rename("tree_frac"),
            wc.eq(WORLDCOVER_BUILT).rename("built_frac"),
            wc.eq(WORLDCOVER_WATER).rename("water_frac"),
        )
        .toFloat()
        # média das células de 10 m dentro de cada célula de 30 m = fração de área
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=WORLDCOVER_MAX_PIXELS)
        .reproject(grid_proj)
    )
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation").resample("bilinear")

    # WorldPop: pessoas por célula de ~100 m → densidade (pessoas/m²) × área da célula de 30 m
    pop_tiles = ee.ImageCollection(WORLDPOP_ASSET).filterBounds(aoi).filter(ee.Filter.eq("year", WORLDPOP_YEAR))
    pop_proj = pop_tiles.first().projection()
    population = (
        pop_tiles.mosaic().setDefaultProjection(pop_proj)
        .divide(ee.Image.pixelArea().reproject(pop_proj))
        .multiply(scale * scale)
        .rename("population")
    )

    image = (
        ee.Image.cat(composite, elevation, fractions, clear_obs, population)
        .select(list(BANDS))
        .toFloat()
        .unmask(NODATA, sameFootprint=False)
    )
    return image, collection


def _grid_info(bbox, params: ExtractParams, crs: str, collection: ee.ImageCollection) -> tuple[dict, int]:
    """Uma única chamada getInfo: bbox projetada em UTM + número de cenas."""
    projected = _bbox_geometry(bbox).transform(ee.Projection(crs), 1)
    info = ee.Dictionary({"coords": projected.coordinates(), "n_scenes": collection.size()}).getInfo()
    ring = info["coords"][0]
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    if max(abs(v) for v in xs) <= 180:  # coordenadas voltaram em graus: algo errado na projeção
        raise RuntimeError(f"Falha ao projetar a bbox para {crs}: {ring[:2]}")
    return _snap_grid(xs, ys, crs, float(params.scale_m)), int(info["n_scenes"])


def _fetch_tile(image: ee.Image, grid: dict, r0: int, c0: int, h: int, w: int, retries: int = 4) -> np.ndarray:
    s = grid["scale"]
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {
                "scaleX": s,
                "shearX": 0,
                "translateX": grid["x_min"] + c0 * s,
                "shearY": 0,
                "scaleY": -s,
                "translateY": grid["y_max"] - r0 * s,
            },
            "crsCode": grid["crs"],
        },
    }
    for attempt in range(retries):
        try:
            return ee.data.computePixels(request)
        except ee.EEException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)  # quota / concorrência: espera e tenta de novo
    raise AssertionError("inalcançável")


def _download(image: ee.Image, grid: dict, max_workers: int, verbose: bool) -> dict[str, np.ndarray]:
    H, W = grid["height"], grid["width"]
    arrays = {b: np.full((H, W), np.nan, dtype=np.float32) for b in BANDS}
    tiles = [
        (r0, c0, min(TILE_PX, H - r0), min(TILE_PX, W - c0))
        for r0 in range(0, H, TILE_PX)
        for c0 in range(0, W, TILE_PX)
    ]
    if verbose:
        print(f"Baixando grade {H}×{W} px em {len(tiles)} bloco(s)...")

    def job(tile):
        r0, c0, h, w = tile
        return tile, _fetch_tile(image, grid, r0, c0, h, w)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, ((r0, c0, h, w), data) in enumerate(pool.map(job, tiles), start=1):
            for b in BANDS:
                block = data[b].astype(np.float32)
                block[block == NODATA] = np.nan
                arrays[b][r0 : r0 + h, c0 : c0 + w] = block
            if verbose and len(tiles) > 1:
                print(f"  bloco {i}/{len(tiles)} ok")
    return arrays


def _to_dataframe(arrays: dict[str, np.ndarray], grid: dict) -> pd.DataFrame:
    H, W, s = grid["height"], grid["width"], grid["scale"]
    rows, cols = np.divmod(np.arange(H * W, dtype=np.int32), W)
    df = pd.DataFrame(
        {
            "row": rows,
            "col": cols,
            "x": grid["x_min"] + (cols + 0.5) * s,  # centro da célula, metros UTM
            "y": grid["y_max"] - (rows + 0.5) * s,
        }
    )
    for b in BANDS:
        df[b] = arrays[b].ravel()
    df["clear_obs"] = df["clear_obs"].fillna(0).astype(np.int16)
    df["population"] = df["population"].fillna(0.0)  # fora da cobertura do WorldPop (p.ex. água) = 0
    df["is_water"] = (df["water_frac"] >= WATER_THRESHOLD).to_numpy()
    return df


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def extract_grid(
    bbox,
    params: ExtractParams | None = None,
    *,
    project: str | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    max_workers: int = MAX_WORKERS,
    verbose: bool = True,
) -> GridData:
    """Retorna a grade de 30 m da bbox, usando o cache em Parquet sempre que possível."""
    params = params or ExtractParams()
    bbox = normalize_bbox(bbox)
    path = cache_path(bbox, params, cache_dir)
    found = None if refresh else find_cached(bbox, params, cache_dir)

    if found is not None:
        try:
            grid = load_grid(found)
            if verbose:
                print(f"[cache] {found.name} ({grid.shape[0]}×{grid.shape[1]} px) — nenhuma chamada ao GEE.")
            return grid
        except Exception as exc:  # arquivo corrompido ou de outra versão: baixa de novo
            print(f"[cache] {found.name} inválido ({exc}); baixando novamente.")

    init_earth_engine(project, verbose=verbose)
    crs = utm_epsg(bbox)
    image, collection = build_image(bbox, params, crs)
    grid_def, n_scenes = _grid_info(bbox, params, crs, collection)
    if n_scenes == 0:
        raise RuntimeError("Nenhuma cena Landsat para essa bbox/anos/meses — nada a extrair.")

    t0 = time.perf_counter()
    arrays = _download(image, grid_def, max_workers, verbose)
    df = _to_dataframe(arrays, grid_def)

    meta = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": path.stem.removeprefix("grid_"),
        "bbox": list(bbox),
        "params": asdict(params),
        "grid": grid_def,
        "bands": list(BANDS),
        "water_threshold": WATER_THRESHOLD,
        "n_scenes": n_scenes,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_grid(df, meta, path)
    if verbose:
        mb = path.stat().st_size / 1e6
        print(f"[gee] {n_scenes} cenas, {time.perf_counter() - t0:.1f} s de download → {path.name} ({mb:.1f} MB)")
    return GridData(df=df, meta=meta, path=path, from_cache=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="CanopyAI — extrai a grade de 30 m de uma bbox (com cache)")
    parser.add_argument("--bbox", required=True, type=lambda s: tuple(float(v) for v in s.split(",")),
                        help="lon_min,lat_min,lon_max,lat_max")
    parser.add_argument("--years", type=int, nargs=2, default=ExtractParams.years, metavar=("INICIO", "FIM"))
    parser.add_argument("--months", type=int, nargs=2, default=ExtractParams.months, metavar=("INICIO", "FIM"))
    parser.add_argument("--project", default=None)
    parser.add_argument("--refresh", action="store_true", help="ignora o cache e baixa de novo")
    args = parser.parse_args()

    grid = extract_grid(args.bbox, ExtractParams(years=args.years, months=args.months),
                        project=args.project, refresh=args.refresh)
    print(grid.df.describe().T[["count", "mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
