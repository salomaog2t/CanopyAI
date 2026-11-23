"""
Pré-processamento Landsat 8/9 (Collection 2, Level 2) compartilhado pelos passos do pipeline.

- máscara de nuvem/sombra/cirrus/neve via bits do QA_PIXEL e de saturação via QA_RADSAT
- fatores de escala oficiais do USGS (SR e ST_B10), LST em °C
- NDVI e NDBI
- composto MEDIANO da estação seca + contagem de observações limpas por pixel
"""

from __future__ import annotations

import os
from pathlib import Path

import ee
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_SCENE_CLOUD = 60  # % de nuvem por cena (pré-filtro)

# Bits do QA_PIXEL (Landsat C2 L2): 0 fill, 1 dilated cloud, 2 cirrus,
# 3 cloud, 4 cloud shadow, 5 snow
QA_MASK_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)

# ESA WorldCover v200 (10 m): classes usadas no projeto
WORLDCOVER_ASSET = "ESA/WorldCover/v200"
WORLDCOVER_TREE = 10
WORLDCOVER_BUILT = 50
WORLDCOVER_WATER = 80

_initialized_project: str | None = None


# ---------------------------------------------------------------------------
# Sessão Earth Engine
# ---------------------------------------------------------------------------
def resolve_project(project: str | None = None) -> str:
    """Usa o projeto informado ou GEE_PROJECT_ID do .env."""
    if project:
        return project
    load_dotenv(PROJECT_ROOT / ".env")
    project = os.getenv("GEE_PROJECT_ID", "").strip()
    if not project:
        raise RuntimeError("Defina GEE_PROJECT_ID no arquivo .env (ou passe project=...).")
    return project


def init_earth_engine(project: str | None = None, verbose: bool = True) -> str:
    """Inicializa o Earth Engine uma única vez por processo."""
    global _initialized_project
    project = resolve_project(project)
    if _initialized_project == project:
        return project
    try:
        ee.Initialize(project=project)
    except Exception:
        print("Credenciais do Earth Engine não encontradas — abrindo fluxo de autenticação...")
        ee.Authenticate()
        ee.Initialize(project=project)
    _initialized_project = project
    if verbose:
        print(f"[ok] Earth Engine inicializado no projeto '{project}'.")
    return project


# ---------------------------------------------------------------------------
# Landsat
# ---------------------------------------------------------------------------
def mask_and_scale(img: ee.Image) -> ee.Image:
    """Aplica máscara QA_PIXEL/QA_RADSAT e fatores de escala do USGS C2 L2."""
    qa = img.select("QA_PIXEL")
    clear = qa.bitwiseAnd(QA_MASK_BITS).eq(0)
    unsaturated = img.select("QA_RADSAT").eq(0)

    optical = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    lst_c = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("LST")

    return (
        img.addBands(optical, None, True)
        .addBands(lst_c)
        .updateMask(clear)
        .updateMask(unsaturated)
        .copyProperties(img, ["system:time_start"])
    )


def add_indices(img: ee.Image) -> ee.Image:
    # Landsat 8/9 OLI: B4 = red, B5 = NIR, B6 = SWIR1
    ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    ndbi = img.normalizedDifference(["SR_B6", "SR_B5"]).rename("NDBI")
    return img.addBands([ndvi, ndbi])


def month_filter(start: int, end: int) -> ee.Filter:
    if start <= end:
        return ee.Filter.calendarRange(start, end, "month")
    # estação que cruza a virada do ano (ex.: nov–fev)
    return ee.Filter.Or(
        ee.Filter.calendarRange(start, 12, "month"),
        ee.Filter.calendarRange(1, end, "month"),
    )


def landsat_collection(
    aoi: ee.Geometry,
    years: tuple[int, int],
    months: tuple[int, int],
    max_scene_cloud: int = MAX_SCENE_CLOUD,
) -> ee.ImageCollection:
    """Coleção Landsat 8+9 filtrada, mascarada, escalada e com NDVI/NDBI/LST."""
    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(aoi)
        .filter(ee.Filter.calendarRange(years[0], years[1], "year"))
        .filter(month_filter(*months))
        .filter(ee.Filter.lt("CLOUD_COVER", max_scene_cloud))
        .map(mask_and_scale)
        .map(add_indices)
    )


def dry_season_composite(collection: ee.ImageCollection) -> tuple[ee.Image, ee.Image]:
    """Mediana de NDVI/NDBI/LST e número de observações limpas por pixel (sem clip nem máscara de água)."""
    composite = collection.select(["NDVI", "NDBI", "LST"]).median()
    clear_obs = collection.select("LST").count().rename("clear_obs")
    return composite, clear_obs


def worldcover() -> ee.Image:
    """Mapa de classes do ESA WorldCover v200 (imagem global única, 10 m)."""
    return ee.ImageCollection(WORLDCOVER_ASSET).first().select("Map")
