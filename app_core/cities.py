"""Preloaded demo areas (cached on disk by scripts/precache_cities.py) and custom-area limits."""

from __future__ import annotations

from dataclasses import dataclass

from gee.extract import ExtractParams, cache_path

# Custom areas: above this size the Earth Engine request and the optimizer get slow, and a
# free-tier quota can be eaten quickly. ~100 km² ≈ 110k cells ≈ 20 s download.
MAX_CUSTOM_AREA_KM2 = 100.0


@dataclass(frozen=True)
class City:
    name: str
    bbox: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    months: tuple[int, int]
    years: tuple[int, int] = (2022, 2025)
    note: str = ""

    @property
    def params(self) -> ExtractParams:
        return ExtractParams(years=self.years, months=self.months)

    @property
    def is_cached(self) -> bool:
        return cache_path(self.bbox, self.params).exists()


CITIES = [
    City("São Paulo — Pinheiros & Ibirapuera", (-46.70, -23.60, -46.62, -23.53), (6, 9),
         note="Dense high-rise core with Ibirapuera Park; dry-season (Jun–Sep) composite."),
    City("Rio de Janeiro — North Zone", (-43.30, -22.92, -43.22, -22.85), (6, 9),
         note="One of Rio's hottest districts, next to the Tijuca forest; Jun–Sep composite."),
    City("Madrid — City Centre", (-3.74, 40.38, -3.65, 40.45), (6, 8),
         note="Compact European core with Retiro Park; summer (Jun–Aug) composite."),
    City("Phoenix — Downtown", (-112.12, 33.42, -112.03, 33.49), (6, 8),
         note="Hot-desert city with sparse canopy; summer (Jun–Aug) composite."),
]
CITY_BY_NAME = {c.name: c for c in CITIES}
