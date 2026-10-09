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
    """Land-polygon background for `plot_raster_with_coastlines`, vectorized
    from the project's own `inputs/DeltaDTM_masks/deltadtm_mask.vrt`
    (EPSG:4326, land=0) instead of the retired external OSM `land_polygons`
    dataset this project no longer uses (2026-10-02). Windowed to `bounds`
    only - cheap, same pattern as `sfincs_tiles/build_sfincs_tile.py`'s
    `_ocean_polygon_wgs84` (that one vectorizes ocean cells from a single
    tile's own mask.tif; this one vectorizes land cells from the global
    mask VRT, windowed to an arbitrary plot bbox).

    Returns an empty GeoDataFrame (never raises) if the VRT is missing or
    the window has no land cells - this is a purely cosmetic background
    layer (see plot_merged_results.py's own comment), never load-bearing.
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
    """`land_polygons_from_deltadtm_mask`, computed once per (plot bounds,
    mask VRT) and reused from `cache_dir` afterwards.

    Every plot_merged_results job of one study plots the same mosaic extent,
    so all of them vectorized the identical land polygons from the full-
    resolution mask - ~half of each plot job's runtime (thailand_bangkok,
    2026-10-09: 45 s of 94 s). The cache is a pickle of the exact
    GeoDataFrame (float64 coordinates and row order preserved), so the
    rendered figure is pixel-identical to an uncached run. The key covers the
    exact bounds and the mask VRT's path, size and mtime - a changed mask or
    extent gets its own entry. Written to a temp name and atomically renamed,
    so concurrent plot jobs never read a partial file (at worst two of them
    compute the same entry once). An empty result - including the
    read-failure fallback - is never cached.
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
    """Per-cell area (km²) grid for a `(height, width)` window of an EPSG:4326 raster.

    Standard approximation (1° latitude ≈ 111.32 km; 1° longitude ≈ cos(lat)
    × 111.32 km), varying **per row** since a pixel's real east-west ground
    size shrinks toward the poles even where its degree-width does not (a
    pixel's degree-width is not even reliably latitude-invariant to begin
    with - so this must always be read from the raster's own transform,
    never assumed/hardcoded).

    `row_offset`: the window's row offset within the FULL raster (0 for a
    read covering the whole raster) - required so a block-wise caller's
    window gets its true latitude, not one computed as if the block's own
    row 0 were the raster's row 0.

    Shared by `compute_flood_area_km2` (sums this to a scalar) and
    `validation.py` (needs the full per-cell grid for area-weighted
    metrics/exposure-difference disaggregation) - one formula, not two.
    """
    row_centres = transform.f + (row_offset + np.arange(height) + 0.5) * transform.e
    pixel_height_km = abs(transform.e) * _KM_PER_DEG
    pixel_width_km = np.cos(np.radians(row_centres)) * transform.a * _KM_PER_DEG
    return np.broadcast_to((pixel_width_km * pixel_height_km)[:, None], (height, width))


def compute_flood_area_km2(raster_path: str | Path, threshold_m: float) -> float:
    """Return the total area (km²) of cells with flood depth >= threshold_m.

    Reads the raster block-by-block to keep memory use bounded for large
    merged rasters. Pixel area comes from `pixel_area_km2_grid` (see there
    for the formula/rationale).

    Args:
        raster_path: Path to a single-band flood-depth raster (EPSG:4326).
        threshold_m: Minimum depth (metres) to count as flooded.

    Returns:
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
    """Plot a raster with land polygons for context and save it as an image.

    The raster is read downsampled to `resolution_arcsec` arc-seconds per
    pixel (1 arcsec = 1/3600°), so the output resolution is consistent
    regardless of the combined area size.  Since flooding only occurs on land,
    ocean pixels carry 0 or nodata and are already transparent via
    `mask_value=0` — no explicit ocean masking is needed.

    Args:
        raster_path: Path to the raster to plot (single band).
        coastlines: GeoDataFrame of OSM land polygons drawn as a whitesmoke
            background for geographic context. Should already be (roughly)
            limited to the raster's area.
        output_path: Where to save the plot image (e.g. a `.png` file).
        title: Plot title.
        label: Colorbar label.
        cmap: Matplotlib colormap name for the raster.
        resolution_arcsec: Target pixel size in arc-seconds.  The raster is
            downsampled so that each output pixel covers this many arc-seconds.
            Has no effect if the raster's native resolution is already coarser.
        mask_value: If given, cells equal to this value are masked
            (transparent), so the coastline background is visible through
            them.
        oom_tiles: If given, tile polygons that were skipped due to
            OutOfMemoryError are drawn as a semi-transparent grey overlay.
        vmax_m: Colour-scale cap (m); depths above this clip to the same top
            colour (postprocessing.plots.waterdepth_vmax_m).
        figsize: Figure size in inches (postprocessing.plots.merged_map_figsize).
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
