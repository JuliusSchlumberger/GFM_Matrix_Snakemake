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
    """Build a land-polygon background layer from the project's DeltaDTM land/ocean mask.

    Vectorizes land cells (value 0) from `inputs/DeltaDTM_masks/deltadtm_mask.vrt`
    (EPSG:4326), windowed to `bounds`, replacing the retired external OSM
    `land_polygons` dataset this project no longer uses (2026-10-02). Purely cosmetic
    context for `plot_raster_with_coastlines`, never load-bearing: never raises, and
    returns an empty GeoDataFrame if the VRT is missing or the window has no land cells.

    Parameters
    ----------
    data_catalog_root : str or Path
        Root directory containing `inputs/DeltaDTM_masks/deltadtm_mask.vrt`.
    bounds : tuple of float
        (minx, miny, maxx, maxy) window to vectorize, in the mask's CRS (EPSG:4326).

    Returns
    -------
    geopandas.GeoDataFrame
        Land polygons in EPSG:4326, empty if the mask could not be read or has no
        land cells within `bounds`.
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
    """Return `land_polygons_from_deltadtm_mask`, cached on disk per (bounds, mask VRT).

    Every plot_merged_results job of one study plots the same mosaic extent, so
    without caching all of them vectorize the identical land polygons from the
    full-resolution mask - ~half of each plot job's runtime (thailand_bangkok,
    2026-10-09: 45 s of 94 s). The cache is a pickle of the exact GeoDataFrame
    (coordinates and row order preserved), so the rendered figure is pixel-identical
    to an uncached run. The key covers the exact bounds and the mask VRT's path, size
    and mtime. Written to a temp name and atomically renamed, so concurrent plot jobs
    never read a partial file. An empty result is never cached.

    Parameters
    ----------
    data_catalog_root : str or Path
        Root directory containing `inputs/DeltaDTM_masks/deltadtm_mask.vrt`.
    bounds : tuple of float
        (minx, miny, maxx, maxy) window to vectorize, in the mask's CRS (EPSG:4326).
    cache_dir : str or Path
        Directory to read/write the cached pickle in; created if missing.

    Returns
    -------
    geopandas.GeoDataFrame
        Land polygons in EPSG:4326 (see `land_polygons_from_deltadtm_mask`).
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
    """Compute a per-cell area (km²) grid for a window of an EPSG:4326 raster.

    Uses the standard approximation (1° latitude ≈ 111.32 km; 1° longitude ≈
    cos(lat) × 111.32 km), varying **per row** since a pixel's real east-west
    ground size shrinks toward the poles even where its degree-width does not -
    so pixel size must always be read from the raster's own transform, never
    assumed/hardcoded. Shared by `compute_flood_area_km2` and `validation.py`
    (area-weighted metrics/exposure-difference disaggregation) - one formula, not two.

    Parameters
    ----------
    transform : affine.Affine
        Raster affine transform (degrees), giving pixel size and row latitude.
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    row_offset : int, optional
        Row offset of the window within the full raster (0 for a read covering the
        whole raster), needed so a block-wise caller's window gets its true latitude,
        not one computed as if the block's own row 0 were the raster's row 0.

    Returns
    -------
    numpy.ndarray
        (height, width) array of per-cell area in km².
    """
    row_centres = transform.f + (row_offset + np.arange(height) + 0.5) * transform.e
    pixel_height_km = abs(transform.e) * _KM_PER_DEG
    pixel_width_km = np.cos(np.radians(row_centres)) * transform.a * _KM_PER_DEG
    return np.broadcast_to((pixel_width_km * pixel_height_km)[:, None], (height, width))


def compute_flood_area_km2(raster_path: str | Path, threshold_m: float) -> float:
    """Compute the total area (km²) of cells with flood depth >= threshold_m.

    Reads the raster block-by-block in parallel threads to keep memory use bounded
    for large merged rasters; per-window partial sums are added to the total
    sequentially in the original window order so the result is bit-identical to a
    serial reduction (float addition is not associative). Pixel area comes from
    `pixel_area_km2_grid` (see there for the formula/rationale).

    Parameters
    ----------
    raster_path : str or Path
        Path to a single-band flood-depth raster (EPSG:4326).
    threshold_m : float
        Minimum depth (metres) to count as flooded.

    Returns
    -------
    float
        Total flooded area in km².
    """
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    with retry_transient_io(rasterio.open, raster_path) as src:
        t = src.transform
        nodata = src.nodata
        windows = [window for _, window in src.block_windows(1)]

    # Windows are read and reduced in parallel threads (GDAL and numpy release
    # the GIL; one dataset handle per thread - rasterio handles must not be
    # shared across threads), but the per-window partial sums are added to
    # the total sequentially IN THE ORIGINAL WINDOW ORDER, exactly as the
    # former serial loop did - the result is bit-identical to it (float
    # addition is not associative, so the order is what must not change).
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
            partials = list(pool.map(window_sum, windows))  # map() preserves input order
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
    """Plot a raster with a land-polygon background and save it as an image.

    The raster is read downsampled to `resolution_arcsec` arc-seconds per pixel
    (1 arcsec = 1/3600°), so the output resolution is consistent regardless of the
    combined area size; has no effect if the native resolution is already coarser.
    Since flooding only occurs on land, ocean pixels carry 0 or nodata and are
    already transparent via `mask_value=0` - no explicit ocean masking is needed.
    Saves the figure to `output_path` and closes it.

    Parameters
    ----------
    raster_path : str or Path
        Path to the raster to plot (single band).
    coastlines : geopandas.GeoDataFrame
        Land polygons drawn as a whitesmoke background for geographic context.
        Should already be (roughly) limited to the raster's area.
    output_path : str or Path
        Where to save the plot image (e.g. a `.png` file).
    title : str
        Plot title.
    label : str
        Colorbar label.
    cmap : str
        Matplotlib colormap name for the raster.
    resolution_arcsec : float
        Target pixel size in arc-seconds; the raster is downsampled so that each
        output pixel covers this many arc-seconds.
    mask_value : float, optional
        If given, cells equal to this value are masked (transparent), so the
        coastline background is visible through them.
    oom_tiles : geopandas.GeoDataFrame, optional
        If given, tile polygons that were skipped due to OutOfMemoryError are
        drawn as a semi-transparent grey overlay.
    annotation : str, optional
        If given, drawn as a text box in the lower-left corner of the plot.
    vmax_m : float, optional
        Colour-scale cap (m); depths above this clip to the same top colour
        (postprocessing.plots.waterdepth_vmax_m). Default is 10.0.
    figsize : tuple of float, optional
        Figure size in inches (postprocessing.plots.merged_map_figsize).
        Default is (10, 10).
    dpi : int, optional
        Output image resolution in dots per inch. Default is 200.
    """
    with retry_transient_io(rasterio.open, raster_path) as src:
        native_deg = src.transform.a          # native pixel width in degrees
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
