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


def plot_overlap_continent_diagnostics(
    mins: np.ndarray,
    maxs: np.ndarray,
    threshold_m: float,
    output_path: str | Path,
    continent_name: str,
    waterlevel_name: str,
    n_chunks: int,
    dpi: int = 130,
) -> None:
    """Two-subplot per-continent overlap-agreement diagnostic.

    Left: hexbin density of (min depth, max depth) across all tiles
    overlapping each cell, pooled from every chunk in this continent, with
    Pearson r and a y = x "perfect agreement" line.
    Right: pie chart classifying every sampled cell as confirmed-flood
    (min >= threshold), confirmed-no-flood (max < threshold), or ambiguous
    (min < threshold <= max, i.e. tiles disagree on flood status) —
    mutually exclusive and exhaustive over the sampled cells.

    Args:
        mins, maxs: Per-cell min/max depth across overlapping tiles, pooled
            across all of this continent's chunks (see
            merge_tile_rasters_chunk / plot_overlap_continent_diagnostics.py).
        threshold_m: Minimum depth (m) counted as "flooded"
            (postprocessing.flood_area_threshold_m).
        output_path: Where to save the PNG.
        continent_name: Continent label used in the plot title.
        waterlevel_name: Scenario label used in the plot title.
        n_chunks: Number of chunks pooled into this continent's sample, for
            the title annotation.
    """
    fig, (ax_hex, ax_pie) = plt.subplots(1, 2, figsize=(13, 6.5), dpi=dpi)

    if len(mins) == 0:
        for ax in (ax_hex, ax_pie):
            ax.text(0.5, 0.5, "No overlapping flood cells found.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11)
            ax.set_axis_off()
        fig.suptitle(f"{continent_name} — overlap diagnostics ({waterlevel_name})")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    n_total = len(mins)
    r = float(np.corrcoef(mins, maxs)[0, 1]) if n_total > 1 else float("nan")

    # ── Left: min/max hexbin ────────────────────────────────────────────────
    vmax = float(max(mins.max(), maxs.max()))
    hb = ax_hex.hexbin(mins, maxs, gridsize=60, cmap="YlOrRd", mincnt=1,
                       extent=(0, vmax, 0, vmax))
    fig.colorbar(hb, ax=ax_hex, label="Number of cells", shrink=0.75)
    ax_hex.plot([0, vmax], [0, vmax], "k--", lw=1.5, label="y = x  (perfect agreement)")
    ax_hex.legend(fontsize=8, loc="upper left")
    ax_hex.set_xlim(0, vmax)
    ax_hex.set_ylim(0, vmax)
    ax_hex.set_aspect("equal", adjustable="box")
    ax_hex.set_xlabel("Min depth across overlapping tiles (m)")
    ax_hex.set_ylabel("Max depth across overlapping tiles (m)")
    ax_hex.set_title(f"Pearson r = {r:.4f}   (n = {n_total:,})", fontsize=10)

    # ── Right: flood-agreement pie chart ────────────────────────────────────
    confirmed_flood = int((mins >= threshold_m).sum())
    confirmed_no_flood = int((maxs < threshold_m).sum())
    ambiguous = int(((mins < threshold_m) & (maxs >= threshold_m)).sum())
    counts = [confirmed_flood, confirmed_no_flood, ambiguous]
    labels = [
        f"Confirmed flood\n({confirmed_flood:,})",
        f"Confirmed no-flood\n({confirmed_no_flood:,})",
        f"Ambiguous\n({ambiguous:,})",
    ]
    colors = ["#d62728", "#1f77b4", "#7f7f7f"]
    nonzero = [(c, l, col) for c, l, col in zip(counts, labels, colors) if c > 0]
    ax_pie.pie(
        [c for c, _, _ in nonzero],
        labels=[l for _, l, _ in nonzero],
        colors=[col for _, _, col in nonzero],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    ax_pie.set_title(f"Flood-status agreement (threshold = {threshold_m:.2f} m)", fontsize=10)

    fig.suptitle(
        f"{continent_name} — overlap diagnostics ({waterlevel_name})\n"
        f"pooled from {n_chunks} chunk{'s' if n_chunks != 1 else ''}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
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
    dpi: int = 130,
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
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
