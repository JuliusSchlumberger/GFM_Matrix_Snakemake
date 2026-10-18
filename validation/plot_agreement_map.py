"""Plot agreement maps from validate_country.py's per-region category rasters.

Rasters: uint8, 0=dry, 1=agree, 2=under, 3=over, nodata=255. Per benchmark, the
largest region is the main map and the others go to a sub-region grid; benchmarks
with comparison_return_periods get one panel per RP in one file. Panels carry
HR/FAR/CSI/bias and no titles. The background is the same permanent-water mask
the evaluation domain excludes.

Usage:
    python validation/plot_agreement_map.py \\
        --config snakemake_workflow/config/config.yml --country GBR
"""

import argparse
import math
import string
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from map_style import (  # noqa: E402
    LAND_COLOR as _LAND_COLOR,
    LAND_LABEL as _LAND_LABEL,
    WATER_COLOR as _WATER_COLOR,
    WATER_LABEL as _WATER_LABEL,
    draw_panel_letter as _draw_panel_letter,
)
import validation as v  # noqa: E402
from validate_country import _find_benchmark_keys  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CAT_NODATA_DEFAULT = 255


def _load_region_metrics(metrics_csv: Path, benchmark_key: str | None = None) -> dict[str, dict[str, float]]:
    # benchmark_key is required when benchmarks share a country: each writes its own
    # "ALL" row to the same CSV.
    if not metrics_csv.exists():
        return {}
    df = pd.read_csv(metrics_csv)
    if "HR" not in df.columns or df.empty:
        return {}
    if benchmark_key is not None and "benchmark_key" in df.columns:
        df = df[df["benchmark_key"] == benchmark_key]
    cols = ["HR", "FAR", "CSI", "bias", "CSI_tol"]
    return {
        str(row["region"]): {c: row[c] for c in cols if c in df.columns}
        for _, row in df.iterrows()
    }


def _format_metrics_lines(metrics: dict[str, float] | None) -> list[str]:
    def _fmt(x) -> str:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        return f"{x:.2f}"

    if not metrics:
        return []
    return [
        f"HR={_fmt(metrics.get('HR'))}  FAR={_fmt(metrics.get('FAR'))}",
        f"CSI={_fmt(metrics.get('CSI'))}  bias={_fmt(metrics.get('bias'))}",
    ]


def _draw_annotation_box(ax: plt.Axes, lines: list[str]) -> None:
    if not lines:
        return
    ax.text(
        0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5, pad=4.0),
    )


def _draw_background(
    ax: plt.Axes,
    bbox: list[float],
    transform: Affine,
    shape: tuple[int, int],
    gfm_catalog,
    permanent_water_source: str | None,
    permanent_water_codes: list[int] | None,
) -> None:
    if gfm_catalog is None:
        return
    water_mask = v.read_permanent_water_mask(
        gfm_catalog, permanent_water_source, permanent_water_codes, bbox, transform, shape,
    )
    bg = np.where(water_mask, 1, 0).astype("uint8")
    extent = (bbox[0], bbox[2], bbox[1], bbox[3])
    ax.imshow(
        bg, extent=extent, cmap=ListedColormap([_LAND_COLOR, _WATER_COLOR]),
        vmin=0, vmax=1, origin="upper", zorder=0,
    )


def _draw_agreement_panel(
    ax: plt.Axes,
    raster_path: str | Path,
    colors: dict[str, str],
    gfm_catalog,
    permanent_water_source: str | None,
    permanent_water_codes: list[int] | None,
) -> tuple[float, float, float, float]:
    with retry_transient_io(rasterio.open, raster_path) as src:
        data = src.read(1)
        bounds = src.bounds
        transform = src.transform
        nodata = src.nodata if src.nodata is not None else _CAT_NODATA_DEFAULT

    masked = np.ma.masked_equal(data, nodata)
    cmap = ListedColormap(["none", colors["agree"], colors["under"], colors["over"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    _draw_background(
        ax, [bounds.left, bounds.bottom, bounds.right, bounds.top], transform, data.shape,
        gfm_catalog, permanent_water_source, permanent_water_codes,
    )
    ax.imshow(masked, extent=extent, cmap=cmap, norm=norm, origin="upper", zorder=1)
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    # Long tick labels on narrow close-up panels overlap otherwise.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.tick_params(axis="x", rotation=30)
    return bounds.left, bounds.right, bounds.bottom, bounds.top


_DEFAULT_LEGEND_LABELS = {
    "agree": "Agree (both wet)",
    "under": "Model under-predicts",
    "over": "Model over-predicts",
}


def _legend_handles(colors: dict[str, str], labels: dict[str, str]) -> list[Patch]:
    return [
        Patch(facecolor=colors["agree"], label=labels["agree"]),
        Patch(facecolor=colors["under"], label=labels["under"]),
        Patch(facecolor=colors["over"], label=labels["over"]),
        Patch(facecolor=_LAND_COLOR, edgecolor="black", linewidth=0.3, label=_LAND_LABEL),
        Patch(facecolor=_WATER_COLOR, edgecolor="black", linewidth=0.3, label=_WATER_LABEL),
    ]


def _sorted_rps(rp_labels) -> list[str]:
    return sorted(rp_labels, key=lambda s: int(s[2:]))


def plot_agreement_map(
    rp_rasters: dict[str, Path],
    gfm_catalog,
    permanent_water_source: str | None,
    permanent_water_codes: list[int] | None,
    output_path: str | Path,
    colors: dict[str, str],
    figsize: tuple[float, float],
    dpi: int,
    legend_labels: dict[str, str] | None = None,
    rp_metrics: dict[str, dict[str, float]] | None = None,
) -> None:
    """Plot one region's agreement raster, one panel per RP (keys of `rp_rasters`).

    `figsize` is per panel, in inches. `gfm_catalog=None` skips the land/water background.
    """
    if not rp_rasters:
        raise ValueError("rp_rasters is empty - nothing to plot.")
    labels = legend_labels or _DEFAULT_LEGEND_LABELS
    rp_metrics = rp_metrics or {}
    rp_order = _sorted_rps(rp_rasters)
    n = len(rp_order)
    letters = list(string.ascii_lowercase[:n]) if n > 1 else [None]

    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]), sharey=True, squeeze=False)
    for i, rp_label in enumerate(rp_order):
        ax = axes[0][i]
        _draw_agreement_panel(
            ax, rp_rasters[rp_label], colors, gfm_catalog, permanent_water_source, permanent_water_codes,
        )
        ax.set_xlabel("Longitude")
        if i == 0:
            ax.set_ylabel("Latitude")
        else:
            ax.tick_params(labelleft=False)
        _draw_panel_letter(ax, letters[i])
        _draw_annotation_box(ax, _format_metrics_lines(rp_metrics.get(rp_label)))

    fig.subplots_adjust(wspace=0.04, bottom=0.17, top=0.92)  # room for rotated x-ticks
    fig.legend(
        handles=_legend_handles(colors, labels), loc="lower center", ncol=5, fontsize=9,
        bbox_to_anchor=(0.5, 0.01), framealpha=0.9,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_subregions_grid(
    rp_region_rasters: dict[str, dict[str, Path]],
    gfm_catalog,
    permanent_water_source: str | None,
    permanent_water_codes: list[int] | None,
    output_path: str | Path,
    colors: dict[str, str],
    dpi: int,
    legend_labels: dict[str, str] | None = None,
    rp_region_metrics: dict[str, dict[str, dict[str, float]]] | None = None,
    panel_size: float = 5.0,
) -> None:
    """Plot sub-regions as a grid: one row per region, one column per RP.

    `rp_region_rasters` is RP -> {region: raster path}. `panel_size` is in inches.
    Writes nothing if there are no sub-regions.
    """
    rp_region_metrics = rp_region_metrics or {}
    labels = legend_labels or _DEFAULT_LEGEND_LABELS
    rp_order = _sorted_rps(rp_region_rasters)
    if not rp_order:
        return

    region_names: list[str] = []
    for rp_label in rp_order:
        for region in rp_region_rasters[rp_label]:
            if region not in region_names:
                region_names.append(region)
    if not region_names:
        return

    n_regions, n_rps = len(region_names), len(rp_order)
    total_panels = n_regions * n_rps
    use_letters = total_panels > 1
    letters = string.ascii_lowercase

    fig, axes = plt.subplots(
        n_regions, n_rps, figsize=(panel_size * n_rps, panel_size * n_regions),
        sharey="row", squeeze=False,
    )
    panel_i = 0
    for ri, region in enumerate(region_names):
        for ci, rp_label in enumerate(rp_order):
            ax = axes[ri][ci]
            raster_path = rp_region_rasters[rp_label].get(region)
            if raster_path is None:
                ax.axis("off")
                panel_i += 1
                continue
            _draw_agreement_panel(
                ax, raster_path, colors, gfm_catalog, permanent_water_source, permanent_water_codes,
            )
            ax.set_xlabel("Longitude")
            if ci == 0:
                ax.set_ylabel("Latitude")
            else:
                ax.tick_params(labelleft=False)
            letter = letters[panel_i] if use_letters and panel_i < len(letters) else None
            _draw_panel_letter(ax, letter)
            metrics = rp_region_metrics.get(rp_label, {}).get(region)
            _draw_annotation_box(ax, _format_metrics_lines(metrics))
            panel_i += 1

    # Margins fixed in inches (bottom: xlabel+rotated ticks+legend, top: panel letter),
    # so they do not shrink as rows are added.
    fig_height_in = panel_size * n_regions
    bottom_frac = min(0.24, 1.05 / fig_height_in)
    top_frac = 1 - min(0.12, 0.3 / fig_height_in)
    fig.subplots_adjust(wspace=0.04, hspace=0.3, bottom=bottom_frac, top=top_frac)
    fig.legend(
        handles=_legend_handles(colors, labels), loc="lower center", ncol=5, fontsize=9,
        bbox_to_anchor=(0.5, 0.0), framealpha=0.9,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_tile_coverage(
    benchmark_gdf: gpd.GeoDataFrame,
    tile_gdf: gpd.GeoDataFrame,
    region_bbox: list[float],
    output_path: str | Path,
    gfm_catalog,
    permanent_water_source: str | None,
    permanent_water_codes: list[int] | None,
    dpi: int,
    background_resolution_m: float = 200.0,
) -> None:
    """Plot the production tiles that intersect the benchmark within one region.

    `region_bbox` is [minx, miny, maxx, maxy] in EPSG:4326; `background_resolution_m`
    is the land/water background grid in meters. Tile IDs are labelled only if <= 60.
    """
    bminx, bminy, bmaxx, bmaxy = region_bbox
    bench = benchmark_gdf.cx[bminx:bmaxx, bminy:bmaxy]
    if bench.empty:
        print(f"    plot_tile_coverage: no benchmark geometry in region bbox {region_bbox} - skipping.")
        return
    # sjoin, not union_all(): some benchmark geometries are invalid and make
    # union_all() raise a GEOS TopologyException.
    tile_idx = gpd.sjoin(tile_gdf, bench[["geometry"]], predicate="intersects", how="inner").index.unique()
    hits = tile_gdf.loc[tile_idx].copy()

    from shapely.geometry import box as shapely_box
    view_box = shapely_box(bminx, bminy, bmaxx, bmaxy)

    fig, ax = plt.subplots(figsize=(11, 11))

    res_deg = background_resolution_m / 111320.0
    w = max(1, int((bmaxx - bminx) / res_deg))
    h = max(1, int((bmaxy - bminy) / res_deg))
    bg_transform = Affine(res_deg, 0, bminx, 0, -res_deg, bmaxy)
    _draw_background(ax, region_bbox, bg_transform, (h, w), gfm_catalog, permanent_water_source, permanent_water_codes)

    bench.plot(ax=ax, color="#4daf4a", alpha=0.6, linewidth=0, zorder=1)
    hits.boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=2)

    n_labeled = 0
    if len(hits) <= 60:
        for _, row in hits.iterrows():
            visible_part = row.geometry.intersection(view_box)
            if visible_part.is_empty:
                continue
            c = visible_part.centroid
            ax.annotate(
                str(int(row["tile_id"])), (c.x, c.y), ha="center", va="center", fontsize=6,
                color="black", zorder=3,
            )
            n_labeled += 1

    ax.set_xlim(bminx, bmaxx)
    ax.set_ylim(bminy, bmaxy)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(handles=[
        Patch(facecolor="#4daf4a", alpha=0.6, label="benchmark extent"),
        Patch(facecolor="none", edgecolor="black", label="tile footprint"),
        Patch(facecolor=_LAND_COLOR, label=_LAND_LABEL),
        Patch(facecolor=_WATER_COLOR, label=_WATER_LABEL),
    ], loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_country(cfg: dict, country_iso: str, metric: str = "agreement") -> None:
    """Write every agreement, sub-region and tile-coverage plot for one country.

    Rasters are found by globbing the output dir, so call once after all RPs are scored.
    Benchmarks with several configured RPs get an RP-less filename. Missing rasters
    print a NOTE instead of raising.
    """
    val_cfg = cfg["validation"]
    slr = val_cfg["waterlevel_name"]
    out_dir = Path(val_cfg["output_dir"]) / country_iso
    plots_cfg = val_cfg["plots"]
    legend_labels = _DEFAULT_LEGEND_LABELS
    variable = "extent"

    gfm_catalog = None
    try:
        gfm_catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    except Exception as e:
        print(f"  WARNING: could not build the GFM data catalog for the land/water background ({e}) - plotting without it.")

    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])

    relevant = [
        (key, v.load_benchmark_spec(bench_catalog, key))
        for key in _find_benchmark_keys(bench_catalog, country_iso)
    ]
    relevant = [(key, spec) for key, spec in relevant if spec.variable == variable]
    if not relevant:
        print(f"  NOTE: no {variable} benchmark found for {country_iso} - nothing to plot.")
        return
    multi_benchmark = len(relevant) > 1

    def _pixel_count(path: Path) -> int:
        with retry_transient_io(rasterio.open, path) as src:
            return src.width * src.height

    for benchmark_key, spec in relevant:
        rp_ints = spec.comparison_return_periods if spec.comparison_return_periods \
            else [int(val_cfg["return_period"][2:])]
        rp_labels = [f"RP{rp}" for rp in rp_ints]
        own_region_names = set(spec.regions.keys()) if spec.regions else {country_iso}

        rp_region_rasters: dict[str, dict[str, Path]] = {}
        rp_region_metrics: dict[str, dict[str, dict[str, float]]] = {}
        for rp_label in rp_labels:
            prefix, suffix = f"{metric}_{country_iso}_", f"_{rp_label}_{slr}.tif"
            found = {
                p.name[len(prefix):-len(suffix)]: p
                for p in sorted(out_dir.glob(f"{prefix}*{suffix}"))
            }
            own = {r: p for r, p in found.items() if r in own_region_names}
            if not own:
                continue
            rp_region_rasters[rp_label] = own
            metrics_csv = out_dir / f"metrics_{country_iso}_{rp_label}_{slr}.csv"
            rp_region_metrics[rp_label] = _load_region_metrics(
                metrics_csv, benchmark_key=benchmark_key,
            )

        if not rp_region_rasters:
            print(f"  NOTE: no {metric} rasters found for {benchmark_key} at any of {rp_labels} - skipping.")
            continue

        file_tag = f"{country_iso}_{benchmark_key}" if multi_benchmark else country_iso
        combined_rp_file = len(rp_labels) > 1  # configured, not on-disk, count: stable filename
        rp_suffix = "" if combined_rp_file else f"_{rp_labels[0]}"

        first_rp = next(iter(rp_region_rasters))
        main_region = max(rp_region_rasters[first_rp], key=lambda r: _pixel_count(rp_region_rasters[first_rp][r]))

        main_rp_rasters = {rp: rasters[main_region] for rp, rasters in rp_region_rasters.items() if main_region in rasters}
        main_rp_metrics = {rp: rp_region_metrics.get(rp, {}).get(main_region) for rp in main_rp_rasters}

        out_path = out_dir / f"{metric}_{file_tag}{rp_suffix}_{slr}.png"
        plot_agreement_map(
            main_rp_rasters, gfm_catalog, val_cfg["permanent_water_source"], val_cfg["permanent_water_codes"],
            out_path, colors=plots_cfg["agreement_colors"], figsize=tuple(plots_cfg["figsize"]),
            dpi=int(plots_cfg["dpi"]), legend_labels=legend_labels, rp_metrics=main_rp_metrics,
        )
        print(f"Written: {out_path}")

        sub_rp_rasters = {
            rp: {r: p for r, p in rasters.items() if r != main_region}
            for rp, rasters in rp_region_rasters.items()
        }
        sub_rp_rasters = {rp: d for rp, d in sub_rp_rasters.items() if d}
        if sub_rp_rasters:
            sub_rp_metrics = {
                rp: {r: rp_region_metrics.get(rp, {}).get(r) for r in d}
                for rp, d in sub_rp_rasters.items()
            }
            subregions_path = out_dir / f"{metric}_{file_tag}_subregions{rp_suffix}_{slr}.png"
            plot_subregions_grid(
                sub_rp_rasters, gfm_catalog, val_cfg["permanent_water_source"], val_cfg["permanent_water_codes"],
                subregions_path, colors=plots_cfg["agreement_colors"], dpi=int(plots_cfg["dpi"]),
                legend_labels=legend_labels, rp_region_metrics=sub_rp_metrics,
            )
            print(f"Written: {subregions_path}")

    tile_gdf = None

    for benchmark_key, spec in relevant:
        if tile_gdf is None:
            tile_gdf = retry_transient_io(gpd.read_file, cfg["tile_grid"]["path"])

        region_bbox = (spec.regions or {}).get(v.primary_region_name(spec.regions)) if spec.regions else None
        if region_bbox is not None:
            bench_full = v.load_benchmark_full(bench_catalog, spec)
            if bench_full.crs is None or bench_full.crs.to_epsg() != 4326:
                bench_full = bench_full.to_crs(4326)
            if not bench_full.empty:
                tile_coverage_path = out_dir / f"tile_coverage_{country_iso}_{benchmark_key}.png"
                plot_tile_coverage(
                    bench_full, tile_gdf, region_bbox, tile_coverage_path,
                    gfm_catalog, val_cfg["permanent_water_source"], val_cfg["permanent_water_codes"],
                    dpi=int(plots_cfg["dpi"]),
                )
                print(f"Written: {tile_coverage_path}")


def main() -> None:
    """Write one country's plots under {validation.output_dir}/{ISO}/."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--country", required=True, help="ISO-3 country code, e.g. GBR")
    parser.add_argument(
        "--metric", choices=["agreement"], default="agreement",
        help="'agreement' (default, only choice) plots the extent comparison's category rasters "
             "(validate_country.py's metrics_*.csv).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    plot_country(cfg, args.country.upper(), metric=args.metric)


if __name__ == "__main__":
    main()
