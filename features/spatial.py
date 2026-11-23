"""
CanopyAI — Passo 2: features de vizinhança sobre a grade de 30 m.

Para cada célula, calcula a média de NDVI e de fração arbórea dentro de um disco de raio
90, 150, 300 e 600 m, com convolução normalizada (scipy.ndimage):

    média = Σ k·v·m / Σ k·m

onde k é o disco, v o valor e m a máscara de células válidas. Assim:
  - Bordas: o que fica fora da grade conta como ausente (não como zero), e a média usa só
    as células que existem. A cobertura (Σ k·m / Σ k) é devolvida para o usuário decidir
    o que fazer com células perto da borda.
  - Água: células com water_frac ≥ limiar ficam fora do numerador e do denominador, para
    que um rio ou represa não puxe para baixo a "vegetação ao redor". As próprias células
    de água recebem NaN.
  - Lacunas (NaN, p.ex. sem observação Landsat) também são ignoradas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage

DEFAULT_RADII_M = (90, 150, 300, 600)
DEFAULT_COLUMNS = ("NDVI", "tree_frac")
DEFAULT_WATER_THRESHOLD = 0.5
DEFAULT_MIN_COVERAGE = 0.2  # abaixo disso (≤ 20% do disco com dado válido) a média vira NaN


def feature_name(column: str, radius_m: int) -> str:
    return f"{column}_r{int(radius_m):03d}"


def coverage_name(radius_m: int) -> str:
    return f"nbr_cov_r{int(radius_m):03d}"


def disk_kernel(radius_m: float, pixel_m: float = 30.0) -> np.ndarray:
    """Disco binário: células cujo centro está a ≤ radius_m do centro da célula alvo."""
    r = radius_m / pixel_m
    n = int(np.floor(r))
    yy, xx = np.mgrid[-n : n + 1, -n : n + 1]
    return (xx**2 + yy**2 <= r**2 + 1e-9).astype(np.float64)


def focal_mean(values: np.ndarray, valid: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Média focal normalizada; devolve (média, cobertura de células válidas no disco)."""
    mask = valid & np.isfinite(values)
    num = ndimage.convolve(np.where(mask, values, 0.0).astype(np.float64), kernel, mode="constant", cval=0.0)
    den = ndimage.convolve(mask.astype(np.float64), kernel, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, np.nan)
    return mean, den / kernel.sum()


def add_neighborhood_features(
    df: pd.DataFrame,
    shape: tuple[int, int],
    *,
    pixel_m: float = 30.0,
    radii_m=DEFAULT_RADII_M,
    columns=DEFAULT_COLUMNS,
    water_col: str = "water_frac",
    water_threshold: float = DEFAULT_WATER_THRESHOLD,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> pd.DataFrame:
    """
    Devolve uma cópia de df com as colunas <coluna>_rNNN e nbr_cov_rNNN.

    df precisa ser a grade completa em ordem linha-maior (como sai de gee.extract), com
    altura × largura linhas. nbr_cov_rNNN é a fração do disco ocupada por células de terra
    dentro da grade: fica < 1 perto da borda da bbox e de corpos d'água.
    """
    h, w = shape
    if len(df) != h * w:
        raise ValueError(f"df tem {len(df)} linhas, mas a grade {h}×{w} pede {h * w}.")
    if {"row", "col"} <= set(df.columns):
        r, c = df["row"].to_numpy(), df["col"].to_numpy()
        if not (np.array_equal(r, np.repeat(np.arange(h), w)) and np.array_equal(c, np.tile(np.arange(w), h))):
            raise ValueError("df não está em ordem linha-maior (row, col); não reordene a grade antes.")

    water = df[water_col].to_numpy().reshape(shape)
    land = np.isfinite(water) & (water < water_threshold)

    out = df.copy()
    for radius in radii_m:
        kernel = disk_kernel(radius, pixel_m)
        _, coverage = focal_mean(np.ones(shape), land, kernel)
        too_sparse = (coverage < min_coverage) | ~land
        out[coverage_name(radius)] = coverage.ravel().astype(np.float32)
        for column in columns:
            values = df[column].to_numpy(dtype=np.float64).reshape(shape)
            mean, _ = focal_mean(values, land, kernel)
            mean[too_sparse] = np.nan
            out[feature_name(column, radius)] = mean.ravel().astype(np.float32)
    return out
