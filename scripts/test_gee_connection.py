"""
CanopyAI — Passo 1: conexão com o Google Earth Engine + checagem de sanidade física.

O que este script faz:
  1. Autentica/inicializa o Earth Engine com o projeto definido em GEE_PROJECT_ID (.env).
  2. Monta um composto MEDIANO da estação seca com Landsat 8 + 9 (Collection 2, Level 2):
       - máscara de nuvem/sombra/cirrus/neve via bits do QA_PIXEL
       - máscara de saturação radiométrica via QA_RADSAT
       - fatores de escala oficiais do USGS (SR e ST_B10)
  3. Calcula NDVI e NDBI e converte ST_B10 para LST em °C.
  4. Remove água permanente usando o ESA WorldCover v200 (classe 80).
  5. Amostra pixels e calcula correlações de Pearson e Spearman:
       NDVI x LST  -> esperado NEGATIVO (vegetação resfria)
       NDBI x LST  -> esperado POSITIVO (superfície construída aquece)
     Se os sinais baterem com a física, o pipeline está pronto para o Passo 2.

Uso:
  python scripts/test_gee_connection.py
  python scripts/test_gee_connection.py --bbox -46.75,-23.65,-46.55,-23.50 --months 6 9 --years 2022 2025
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ee
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Padrões (sobrescritos por .env ou argumentos de linha de comando)
# ---------------------------------------------------------------------------
DEFAULT_BBOX = (-46.75, -23.65, -46.55, -23.50)  # centro expandido de São Paulo
DEFAULT_MONTHS = (6, 9)                            # jun–set (estação seca)
DEFAULT_YEARS = (2022, 2025)                       # Landsat 9 operacional desde 2022
MAX_SCENE_CLOUD = 60                               # % de nuvem por cena (pré-filtro)
SAMPLE_PIXELS = 5000
SCALE_M = 30
WORLDCOVER_WATER = 80

# Bits do QA_PIXEL (Landsat C2 L2): 0 fill, 1 dilated cloud, 2 cirrus,
# 3 cloud, 4 cloud shadow, 5 snow
QA_MASK_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
def _env_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


def load_config() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    env_bbox = os.getenv("AOI_BBOX", "").strip()
    bbox_default = tuple(float(v) for v in env_bbox.split(",")) if env_bbox else DEFAULT_BBOX
    months_default = (
        _env_int("DRY_SEASON_START_MONTH") or DEFAULT_MONTHS[0],
        _env_int("DRY_SEASON_END_MONTH") or DEFAULT_MONTHS[1],
    )
    years_default = (
        _env_int("YEAR_START") or DEFAULT_YEARS[0],
        _env_int("YEAR_END") or DEFAULT_YEARS[1],
    )

    parser = argparse.ArgumentParser(description="CanopyAI — teste de conexão GEE + sanidade NDVI/NDBI x LST")
    parser.add_argument("--project", default=os.getenv("GEE_PROJECT_ID", "").strip())
    parser.add_argument(
        "--bbox",
        type=lambda s: tuple(float(v) for v in s.split(",")),
        default=bbox_default,
        help="lon_min,lat_min,lon_max,lat_max",
    )
    parser.add_argument("--months", type=int, nargs=2, default=months_default, metavar=("INICIO", "FIM"))
    parser.add_argument("--years", type=int, nargs=2, default=years_default, metavar=("INICIO", "FIM"))
    parser.add_argument("--samples", type=int, default=SAMPLE_PIXELS)
    args = parser.parse_args()

    if not args.project:
        sys.exit("ERRO: defina GEE_PROJECT_ID no arquivo .env (ou use --project).")
    if len(args.bbox) != 4:
        sys.exit("ERRO: bbox precisa de 4 valores: lon_min,lat_min,lon_max,lat_max")
    for m in args.months:
        if not 1 <= m <= 12:
            sys.exit("ERRO: meses devem estar entre 1 e 12.")
    return args


def init_earth_engine(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:
        print("Credenciais do Earth Engine não encontradas — abrindo fluxo de autenticação...")
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"[ok] Earth Engine inicializado no projeto '{project}'.")


# ---------------------------------------------------------------------------
# Pré-processamento Landsat
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


def build_dry_season_composite(aoi: ee.Geometry, years: tuple[int, int], months: tuple[int, int]):
    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(aoi)
        .filter(ee.Filter.calendarRange(years[0], years[1], "year"))
        .filter(month_filter(*months))
        .filter(ee.Filter.lt("CLOUD_COVER", MAX_SCENE_CLOUD))
        .map(mask_and_scale)
        .map(add_indices)
    )

    bands = ["NDVI", "NDBI", "LST"]
    composite = collection.select(bands).median().clip(aoi)
    clear_obs = collection.select("LST").count().rename("clear_obs").clip(aoi)

    # Máscara de água permanente (ESA WorldCover 10 m, classe 80)
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    not_water = worldcover.neq(WORLDCOVER_WATER)
    composite = composite.updateMask(not_water)

    return collection, composite, clear_obs


# ---------------------------------------------------------------------------
# Checagem de sanidade
# ---------------------------------------------------------------------------
def sanity_check(aoi, collection, composite, clear_obs, n_samples: int) -> dict:
    samples = composite.sample(
        region=aoi, scale=SCALE_M, numPixels=n_samples, seed=42, geometries=False, dropNulls=True
    )

    def corr(x: str, y: str) -> ee.Dictionary:
        return ee.Dictionary(
            {
                "pearson": samples.reduceColumns(ee.Reducer.pearsonsCorrelation(), [x, y]).get("correlation"),
                "spearman": samples.reduceColumns(ee.Reducer.spearmansCorrelation(), [x, y]).get("correlation"),
            }
        )

    lst_stats = composite.select("LST").reduceRegion(
        reducer=ee.Reducer.percentile([5, 50, 95]).combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=aoi,
        scale=SCALE_M,
        maxPixels=1e9,
        bestEffort=True,
    )
    obs_stats = clear_obs.reduceRegion(
        reducer=ee.Reducer.median(), geometry=aoi, scale=SCALE_M, maxPixels=1e9, bestEffort=True
    )

    # Uma única chamada getInfo() para minimizar idas e voltas ao servidor
    return ee.Dictionary(
        {
            "n_scenes": collection.size(),
            "n_samples": samples.size(),
            "lst": lst_stats,
            "median_clear_obs": obs_stats.get("clear_obs"),
            "ndvi_lst": corr("NDVI", "LST"),
            "ndbi_lst": corr("NDBI", "LST"),
        }
    ).getInfo()


def report(r: dict) -> bool:
    print("\n=== Composto da estação seca ===")
    print(f"Cenas Landsat 8/9 usadas:        {r['n_scenes']}")
    print(f"Observações limpas (mediana/px): {r['median_clear_obs']}")
    lst = r["lst"]
    if r["n_scenes"] == 0 or lst.get("LST_p50") is None:
        print("\nNenhum pixel válido — revise bbox, anos e meses.")
        return False
    print(
        f"LST (°C)  p5={lst['LST_p5']:.1f}  p50={lst['LST_p50']:.1f}  "
        f"p95={lst['LST_p95']:.1f}  média={lst['LST_mean']:.1f}"
    )
    print(f"Pixels amostrados (sem água):    {r['n_samples']}")

    print("\n=== Correlações de sanidade ===")
    checks = [
        ("NDVI x LST", r["ndvi_lst"], lambda v: v < 0, "negativa"),
        ("NDBI x LST", r["ndbi_lst"], lambda v: v > 0, "positiva"),
    ]
    all_ok = True
    for name, c, expected, label in checks:
        ok = None not in (c["pearson"], c["spearman"]) and expected(c["pearson"]) and expected(c["spearman"])
        all_ok &= ok
        status = "PASS" if ok else "FALHA"
        fmt = lambda v: "n/a" if v is None else f"{v:+.3f}"  # noqa: E731
        print(f"[{status}] {name}: Pearson r={fmt(c['pearson'])} | Spearman ρ={fmt(c['spearman'])}  (esperado: {label})")

    if r["median_clear_obs"] is not None and r["median_clear_obs"] < 3:
        print("\nAVISO: poucas observações limpas por pixel; considere ampliar a janela de anos.")

    print("\nResultado:", "física consistente — pronto para o Passo 2." if all_ok else "revisar pré-processamento.")
    return all_ok


def main() -> int:
    args = load_config()
    init_earth_engine(args.project)

    aoi = ee.Geometry.Rectangle(list(args.bbox), proj="EPSG:4326", geodesic=False)
    print(
        f"AOI={args.bbox}  anos={args.years[0]}–{args.years[1]}  "
        f"meses={args.months[0]}–{args.months[1]}  amostras={args.samples}"
    )

    collection, composite, clear_obs = build_dry_season_composite(aoi, tuple(args.years), tuple(args.months))
    result = sanity_check(aoi, collection, composite, clear_obs, args.samples)
    return 0 if report(result) else 1


if __name__ == "__main__":
    sys.exit(main())
