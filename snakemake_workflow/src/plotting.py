"""Functions for plotting merged flood model rasters with contextual coastlines."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling


def compute_flood_area_km2(raster_path: str | Path, threshold_m: float) -> float:
    """Return the total area (km²) of cells with flood depth >= threshold_m.

    Reads the raster block-by-block to keep memory use bounded for large
    merged rasters.  Pixel area is computed per latitude row using the
    standard approximation (1° latitude ≈ 111.32 km; 1° longitude ≈
    cos(lat) × 111.32 km).

    Args:
        raster_path: Path to a single-band flood-depth raster (EPSG:4326).
        threshold_m: Minimum depth (metres) to count as flooded.

    Returns:
        Total flooded area in km².
    """
    _KM_PER_DEG = 111.32
    total_km2 = 0.0
    with rasterio.open(raster_path) as src:
        t = src.transform
        nodata = src.nodata
        pixel_height_km = abs(t.e) * _KM_PER_DEG
        pixel_width_deg = t.a
        for _, window in src.block_windows(1):
            data = src.read(1, window=window)
            valid = (data >= threshold_m)
            if nodata is not None:
                valid &= data != nodata
            valid &= ~np.isnan(data)
            if not valid.any():
                continue
            row_centres = t.f + (window.row_off + np.arange(window.height) + 0.5) * t.e
            pixel_width_km = np.cos(np.radians(row_centres)) * pixel_width_deg * _KM_PER_DEG
            pixel_area_km2 = pixel_width_km * pixel_height_km
            total_km2 += float((valid.sum(axis=1) * pixel_area_km2).sum())
    return total_km2


def plot_overlap_correlation(
    overlap_pairs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
    waterlevel_name: str,
) -> None:
    """2-D heatmap of paired flood depths from overlapping tiles with Pearson r.

    All tile pairs are pooled into a single hexbin density plot so the colour
    encodes the number of co-occurring flood cells rather than individual points
    (avoids over-plotting for the large number of cells in overlap zones).
    The y = x diagonal marks perfect inter-tile agreement.

    Args:
        overlap_pairs: Mapping ``(tile_i_name, tile_j_name)`` → ``(xi, xj)``
            as returned by :func:`merge_tile_rasters`.
        output_path: Where to save the PNG.
        waterlevel_name: Scenario label used in the plot title.
    """
    if not overlap_pairs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No overlapping flood cells found.",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return

    all_xi = np.concatenate([xi for xi, _ in overlap_pairs.values()])
    all_xj = np.concatenate([xj for _, xj in overlap_pairs.values()])
    n_total = len(all_xi)
    n_pairs = len(overlap_pairs)

    r = float(np.corrcoef(all_xi, all_xj)[0, 1])

    fig, ax = plt.subplots(figsize=(7, 7))

    vmax = float(max(all_xi.max(), all_xj.max()))
    hb = ax.hexbin(all_xi, all_xj, gridsize=60, cmap="YlOrRd", mincnt=1,
                   extent=(0, vmax, 0, vmax))
    fig.colorbar(hb, ax=ax, label="Number of cells", shrink=0.75)

    ax.plot([0, vmax], [0, vmax], "k--", lw=1.5, label="y = x  (perfect agreement)")
    ax.legend(fontsize=8, loc="upper left")

    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Flood depth — tile A (m)")
    ax.set_ylabel("Flood depth — tile B (m)")
    ax.set_title(
        f"Cross-tile flood depth correlation ({waterlevel_name})\n"
        f"Pearson r = {r:.4f},   r² = {r ** 2:.4f},   "
        f"n = {n_total:,}  ({n_pairs} tile pair{'s' if n_pairs != 1 else ''})"
    )

    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


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
    """
    with rasterio.open(raster_path) as src:
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

    fig, ax = plt.subplots(figsize=(10, 10))
    coastlines.plot(ax=ax, color="whitesmoke", edgecolor="whitesmoke", linewidth=0.5, zorder=0)

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    image = ax.imshow(masked, extent=extent, cmap=cmap, origin="upper", zorder=1, vmin=0, vmax=min(float(masked.max()), 10))

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
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
