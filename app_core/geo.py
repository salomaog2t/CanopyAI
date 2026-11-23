"""
Geographic helpers for the web app, with no heavy dependencies (no pyproj/GDAL).

- WGS84 <-> UTM conversion (USGS / Snyder 1987 series; sub-metre accuracy inside a zone)
- Resampling the UTM grid to a Web-Mercator-regular raster for Leaflet image overlays
- Simple numpy colormaps
"""

from __future__ import annotations

import math

import numpy as np

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_K0 = 0.9996
_R_EARTH = 6371008.8


def parse_utm_epsg(crs: str) -> tuple[int, bool]:
    code = int(crs.split(":")[1])
    if 32601 <= code <= 32660:
        return code - 32600, False
    if 32701 <= code <= 32760:
        return code - 32700, True
    raise ValueError(f"not a WGS84 UTM EPSG code: {crs}")


def _lon0(zone: int) -> float:
    return math.radians((zone - 1) * 6 - 180 + 3)


def lonlat_to_utm(lon, lat, crs: str):
    zone, south = parse_utm_epsg(crs)
    phi = np.radians(np.asarray(lat, dtype=np.float64))
    lam = np.radians(np.asarray(lon, dtype=np.float64))
    e4, e6 = _E2**2, _E2**3
    n = _A / np.sqrt(1 - _E2 * np.sin(phi) ** 2)
    t = np.tan(phi) ** 2
    c = _EP2 * np.cos(phi) ** 2
    a = np.cos(phi) * (lam - _lon0(zone))
    m = _A * (
        (1 - _E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
        - (3 * _E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * np.sin(2 * phi)
        + (15 * e4 / 256 + 45 * e6 / 1024) * np.sin(4 * phi)
        - (35 * e6 / 3072) * np.sin(6 * phi)
    )
    x = _K0 * n * (a + (1 - t + c) * a**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * _EP2) * a**5 / 120) + 500000.0
    y = _K0 * (m + n * np.tan(phi) * (
        a**2 / 2 + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
        + (61 - 58 * t + t**2 + 600 * c - 330 * _EP2) * a**6 / 720
    ))
    if south:
        y = y + 10000000.0
    return x, y


def utm_to_lonlat(x, y, crs: str):
    zone, south = parse_utm_epsg(crs)
    x = np.asarray(x, dtype=np.float64) - 500000.0
    y = np.asarray(y, dtype=np.float64) - (10000000.0 if south else 0.0)
    e4, e6 = _E2**2, _E2**3
    m = y / _K0
    mu = m / (_A * (1 - _E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256))
    e1 = (1 - math.sqrt(1 - _E2)) / (1 + math.sqrt(1 - _E2))
    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * np.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * np.sin(4 * mu)
        + (151 * e1**3 / 96) * np.sin(6 * mu)
        + (1097 * e1**4 / 512) * np.sin(8 * mu)
    )
    s1 = np.sin(phi1)
    n1 = _A / np.sqrt(1 - _E2 * s1**2)
    t1 = np.tan(phi1) ** 2
    c1 = _EP2 * np.cos(phi1) ** 2
    r1 = _A * (1 - _E2) / (1 - _E2 * s1**2) ** 1.5
    d = x / (n1 * _K0)
    phi = phi1 - (n1 * np.tan(phi1) / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * _EP2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * _EP2 - 3 * c1**2) * d**6 / 720
    )
    lam = _lon0(zone) + (
        d - (1 + 2 * t1 + c1) * d**3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * _EP2 + 24 * t1**2) * d**5 / 120
    ) / np.cos(phi1)
    return np.degrees(lam), np.degrees(phi)


def bbox_area_km2(bbox) -> float:
    lon_min, lat_min, lon_max, lat_max = bbox
    dlam = math.radians(lon_max - lon_min)
    return _R_EARTH**2 * dlam * (math.sin(math.radians(lat_max)) - math.sin(math.radians(lat_min))) / 1e6


def grid_latlon_bounds(grid_meta: dict) -> tuple[float, float, float, float]:
    """(lon_min, lat_min, lon_max, lat_max) enclosing the whole UTM grid."""
    g = grid_meta["grid"]
    x0, y1 = g["x_min"], g["y_max"]
    x1, y0 = x0 + g["width"] * g["scale"], y1 - g["height"] * g["scale"]
    xs = np.array([x0, x1, x1, x0, (x0 + x1) / 2, (x0 + x1) / 2, x0, x1])
    ys = np.array([y0, y0, y1, y1, y0, y1, (y0 + y1) / 2, (y0 + y1) / 2])
    lon, lat = utm_to_lonlat(xs, ys, g["crs"])
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def cell_lonlat(df, grid_meta: dict):
    return utm_to_lonlat(df["x"].to_numpy(), df["y"].to_numpy(), grid_meta["grid"]["crs"])


# ---------------------------------------------------------------------------
# Rasters for Leaflet overlays
# ---------------------------------------------------------------------------
def _merc_y(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _inv_merc_y(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


def to_mercator_raster(values: np.ndarray, grid_meta: dict, max_px: int = 900):
    """
    Nearest-neighbour resample of a (H, W) UTM grid onto a raster that is regular in Web Mercator,
    which is how Leaflet stretches an ImageOverlay. Returns (raster, [[lat_s, lon_w], [lat_n, lon_e]]).
    """
    g = grid_meta["grid"]
    lon_min, lat_min, lon_max, lat_max = grid_latlon_bounds(grid_meta)
    h, w = values.shape
    scale = min(1.0, max_px / max(h, w)) * 1.15  # slight oversampling keeps 30 m cells crisp
    out_w, out_h = int(w * scale) + 1, int(h * scale) + 1
    lons = np.linspace(lon_min, lon_max, out_w)
    lats = _inv_merc_y(np.linspace(_merc_y(lat_max), _merc_y(lat_min), out_h))  # top row first
    lon_g, lat_g = np.meshgrid(lons, lats)
    x, y = lonlat_to_utm(lon_g, lat_g, g["crs"])
    col = np.floor((x - g["x_min"]) / g["scale"]).astype(np.int64)
    row = np.floor((g["y_max"] - y) / g["scale"]).astype(np.int64)
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    out = np.full((out_h, out_w), np.nan, dtype=np.float32)
    out[inside] = values[row[inside], col[inside]]
    return out, [[lat_min, lon_min], [lat_max, lon_max]]


def colorize(values: np.ndarray, vmin: float, vmax: float, stops: list[str], alpha: float = 0.8,
             transparent_below: float | None = None) -> np.ndarray:
    """Linear colormap through hex `stops`; NaN (and values < transparent_below) become transparent."""
    rgb = np.array([[int(s[i : i + 2], 16) for i in (1, 3, 5)] for s in stops], dtype=np.float64)
    t = np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0, 1) * (len(stops) - 1)
    t = np.nan_to_num(t)
    i0 = np.minimum(np.floor(t).astype(int), len(stops) - 2)
    frac = (t - i0)[..., None]
    color = rgb[i0] * (1 - frac) + rgb[i0 + 1] * frac
    a = np.full(values.shape, alpha * 255)
    hidden = ~np.isfinite(values)
    if transparent_below is not None:
        hidden |= np.nan_to_num(values, nan=-np.inf) < transparent_below
    a[hidden] = 0
    return np.dstack([color, a]).astype(np.uint8)
