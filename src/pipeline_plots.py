"""Functions for plotting merged flood model rasters with contextual coastlines."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import shapely.geometry
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from config_utils import retry_transient_io

_KM_PER_DEG = 111.32
_DELTADTM_LAND_CODE = 0  # inputs/DeltaDTM_masks/deltadtm_mask.vrt convention: 0=land, 1=ocean, 2=lake, 3=river


def land_polygons_from_deltadtm_mask(data_catalog_root: str | Path, bounds: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Vectorize DeltaDTM mask land cells (value 0) within `bounds` (EPSG:4326).

    Cosmetic plot background only: never raises; returns an empty GeoDataFrame if
    the VRT is unreadable or the window has no land.
    """
    mask_vrt = Path(data_catalog_root) / "inputs" / "DeltaDTM_masks" / "deltadtm_mask.vrt"
    try:
        with retry_transient_io(rasterio.open, mask_vrt) as src:
            window = from_bounds(*bounds, transform=src.transform)
            arr = src.read(1, window=window, boundless=True, fill_value=255)
            transform = src.window_transform(window)
    except Exception as exc:
        print(f"WARNING: land_polygons_from_deltadtm_mask read failed ({type(exc).__name__}: {exc}) - "
              f"plotting without the coastline background layer", flush=True)
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    land = (arr == _DELTADTM_LAND_CODE).astype(np.uint8)
    if not land.any():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    shapes = rasterio.features.shapes(land, mask=land.astype(bool), transform=transform)
    geoms = [shapely.geometry.shape(geom) for geom, _val in shapes]
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")


def cached_land_polygons(data_catalog_root: str | Path, bounds: tuple[float, float, float, float],
                         cache_dir: str | Path) -> gpd.GeoDataFrame:
    """Return `land_polygons_from_deltadtm_mask`, pickled on disk per (bounds, mask VRT).

    Key covers exact bounds and the VRT's path, size and mtime. Written via temp file
    + atomic rename so concurrent plot jobs never read a partial file. Empty results
    are not cached.
    """
    import hashlib
    import os
    import pickle

    mask_vrt = Path(data_catalog_root) / "inputs" / "DeltaDTM_masks" / "deltadtm_mask.vrt"
    try:
        st = mask_vrt.stat()
        stamp = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    key_src = f"{mask_vrt.resolve()}|{stamp}|{tuple(float(b).hex() for b in bounds)}"
    key = hashlib.sha256(key_src.encode()).hexdigest()[:24]
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"land_polygons_{key}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached_key, gdf = pickle.load(f)
            if cached_key == key_src:
                return gdf
        except Exception as exc:  # corrupt/unreadable entry: recompute and overwrite
            print(f"WARNING: ignoring unreadable land-polygon cache {cache_path} ({type(exc).__name__}: {exc})", flush=True)

    gdf = land_polygons_from_deltadtm_mask(data_catalog_root, bounds)
    if len(gdf):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                pickle.dump((key_src, gdf), f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
        except OSError as exc:  # caching is an optimisation only - never fail the plot over it
            print(f"WARNING: could not write land-polygon cache {cache_path} ({exc})", flush=True)
    return gdf


def pixel_area_km2_grid(transform, width: int, height: int, row_offset: int = 0) -> np.ndarray:
    """Compute a (height, width) per-cell area grid in km² for an EPSG:4326 window.

    Approximation: 1° latitude ≈ 111.32 km, 1° longitude ≈ cos(lat) × 111.32 km,
    evaluated per row. `row_offset` is the window's row within the full raster, so
    block-wise reads get their true latitude.
    """
    row_centres = transform.f + (row_offset + np.arange(height) + 0.5) * transform.e
    pixel_height_km = abs(transform.e) * _KM_PER_DEG
    pixel_width_km = np.cos(np.radians(row_centres)) * transform.a * _KM_PER_DEG
    return np.broadcast_to((pixel_width_km * pixel_height_km)[:, None], (height, width))


def compute_flood_area_km2(raster_path: str | Path, threshold_m: float) -> float:
    """Compute the total area (km²) of cells with flood depth >= `threshold_m` (metres).

    Raster must be EPSG:4326. Blocks are reduced in parallel threads but summed in
    window order, so the result is bit-identical to a serial reduction.
    """
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    with retry_transient_io(rasterio.open, raster_path) as src:
        t = src.transform
        nodata = src.nodata
        windows = [window for _, window in src.block_windows(1)]

    # One dataset handle per thread: rasterio handles must not be shared across
    # threads. Partials are summed in window order (float addition is not associative).
    local, handles, handles_lock = threading.local(), [], threading.Lock()

    def window_sum(window) -> float | None:
        ds = getattr(local, "ds", None)
        if ds is None:
            ds = local.ds = retry_transient_io(rasterio.open, raster_path)
            with handles_lock:
                handles.append(ds)
        data = ds.read(1, window=window)
        valid = (data >= threshold_m)
        if nodata is not None:
            valid &= data != nodata
        valid &= ~np.isnan(data)
        if not valid.any():
            return None
        pixel_area_km2 = pixel_area_km2_grid(
            t, window.width, window.height, row_offset=window.row_off,
        )
        return float((valid * pixel_area_km2).sum())

    try:
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            partials = list(pool.map(window_sum, windows))
    finally:
        for ds in handles:
            ds.close()

    total_km2 = 0.0
    for part in partials:
        if part is not None:
            total_km2 += part
    return total_km2


def plot_raster_with_coastlines(
    raster_path: str | Path,
    coastlines: gpd.GeoDataFrame,
    output_path: str | Path,
    title: str,
    label: str,
    cmap: str,
    resolution_arcsec: float,
    mask_value: float | None = None,
    oom_tiles: gpd.GeoDataFrame | None = None,
    annotation: str | None = None,
    vmax_m: float = 10.0,
    figsize: tuple[float, float] = (10, 10),
    dpi: int = 200,
) -> None:
    """Plot a raster over a land-polygon background and save it as an image.

    Reads the raster average-downsampled to `resolution_arcsec` (1 arcsec = 1/3600°;
    no-op if native is coarser). Cells equal to `mask_value` are transparent;
    `oom_tiles` (tiles skipped with OutOfMemoryError) are drawn as a grey overlay;
    `vmax_m` caps the colour scale in metres.
    """
    with retry_transient_io(rasterio.open, raster_path) as src:
        native_deg = src.transform.a
        target_deg = resolution_arcsec / 3600.0
        scale = max(1.0, target_deg / native_deg)
        out_shape = (max(1, round(src.height / scale)), max(1, round(src.width / scale)))
        data = src.read(1, out_shape=out_shape, resampling=Resampling.average)
        nodata = src.nodata
        bounds = src.bounds

    masked = data
    if nodata is not None:
        masked = np.ma.masked_equal(masked, nodata)
    if mask_value is not None:
        masked = np.ma.masked_equal(masked, mask_value)

    fig, ax = plt.subplots(figsize=figsize)
    coastlines.plot(ax=ax, color="whitesmoke", edgecolor="whitesmoke", linewidth=0.5, zorder=0)

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    image = ax.imshow(masked, extent=extent, cmap=cmap, origin="upper", zorder=1, vmin=0, vmax=min(float(masked.max()), vmax_m))

    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if oom_tiles is not None and not oom_tiles.empty:
        oom_tiles.plot(ax=ax, color="grey", edgecolor="grey", linewidth=0.5, alpha=0.3, zorder=2)
    if annotation is not None:
        ax.text(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="lightgrey"))

    fig.colorbar(image, ax=ax, label=label, shrink=0.7)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
