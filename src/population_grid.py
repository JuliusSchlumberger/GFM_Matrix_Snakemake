"""WorldPop people per pixel on a finer grid; numpy/rasterio only, so hydromt-sfincs-dev imports it too."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

EARTH_RADIUS_KM = 6371.0088


def cell_area_km2(lat_top: float, lat_bottom: float, dlon_deg: float) -> float:
    """Area (km2) of a lon/lat cell on the sphere."""
    return (EARTH_RADIUS_KM ** 2 * np.radians(dlon_deg)
            * abs(np.sin(np.radians(lat_top)) - np.sin(np.radians(lat_bottom))))


def population_per_pixel(pop_path: str | Path, shape: tuple, transform, crs,
                         open_raster: Callable = rasterio.open) -> tuple[np.ndarray, np.ndarray]:
    """Return (people per pixel, people per km2 of the pixel's population cell) on a finer grid.

    Each pixel takes count x pixel area / cell area of the population cell its centre lies in, so
    a fully covered cell sums to its count: production's equal share (rasters.average_pool_to_grid
    divides by all fine pixels of a cell). Pixel area from `transform` (metres or degrees at the
    pixel's latitude). NaN-free; zero outside settled cells. `open_raster` lets callers on a
    network share retry transient errors.
    """
    h, w = shape
    bounds = transform_bounds(crs, "EPSG:4326", *rasterio.transform.array_bounds(h, w, transform), densify_pts=21)
    with open_raster(pop_path) as src:
        pad = 2 * abs(src.transform.a)
        # Affine arithmetic, not windows.from_bounds: its GDAL transformer crashes hydromt-sfincs-dev on Windows.
        c0, r0 = ~src.transform * (bounds[0] - pad, bounds[3] + pad)
        c1, r1 = ~src.transform * (bounds[2] + pad, bounds[1] - pad)
        win = Window(c0, r0, c1 - c0, r1 - r0).round_offsets().round_lengths()
        count = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
        nodata = src.nodata
        wtr = src.window_transform(win)
    count[(count == nodata) | ~np.isfinite(count) | (count < 0)] = 0.0
    lat_top = wtr.f + wtr.e * np.arange(count.shape[0])
    area = np.array([cell_area_km2(t, t + wtr.e, abs(wtr.a)) for t in lat_top])[:, None] * np.ones(count.shape[1])
    density = np.divide(count, area, out=np.zeros_like(count), where=area > 0)
    out_density = np.zeros(shape, dtype=np.float64)
    reproject(density, out_density, src_transform=wtr, src_crs="EPSG:4326", dst_transform=transform, dst_crs=crs,
              resampling=Resampling.nearest)
    if rasterio.crs.CRS.from_user_input(crs).is_geographic:
        lat = transform.f + transform.e * (np.arange(h) + 0.5)
        pixel_km2 = np.array([cell_area_km2(t - transform.e / 2, t + transform.e / 2, abs(transform.a))
                              for t in lat])[:, None]
    else:
        pixel_km2 = abs(transform.a * transform.e - transform.b * transform.d) / 1e6
    return out_density * pixel_km2, out_density
