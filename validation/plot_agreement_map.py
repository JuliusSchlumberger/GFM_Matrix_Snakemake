"""Three-colour agreement map from validate_country.py's category raster(s).

Reads the uint8 category raster(s) (0=dry, 1=agree, 2=under, 3=over, nodata=255)
written by validate_country.py's `_write_agreement_raster` - one per named
`regions:` entry (see src/validation.py::region_for_point), e.g. Spain's
`mainland` and `canary_islands` - already priority-reduced (over > under > agree
> dry) to `validation.plots.resolution_m`.

Every region raster found for a country is grouped by which BENCHMARK's own
`meta.regions` claims it - a country can have more than one benchmark sharing
its `country_iso` (GBR: Wales' `wales_nrw_floodzone_seas` + Scotland's
`scotland_sepa_coastal_m`, distinguished only by `geogunit_ids`), and each
gets its own independent main map + sub-regions grid, never mixed into one
plot. Within each benchmark's own group, the largest region (by raster
extent) is the main map; every OTHER region (a nested sub-area, e.g. Wales'
severn_estuary/menai_strait) is rendered separately by `plot_subregions_grid`
- the main map never shows sub-area insets, only the one whole-territory
domain.

For a benchmark with `meta.comparison_return_periods` set (Norway, Wales,
Scotland - each `[100, 250]`), both RPs are drawn as side-by-side subplots in
ONE figure sharing a y-axis, not two separate files - the output filename
drops the `{RP}` segment for these benchmarks
(`{metric}_{country}[_{benchmark_key}]_{SLR}.png`). A benchmark with only one
RP available (either because it has no `comparison_return_periods`, or
because only one RP has been scored so far) gets a single panel. `plot_subregions_grid`
follows the same rule, one row per sub-region and one column per RP (columns
of the same row share a y-axis). Every panel beyond the first is lettered
`(a)..(d)`, drawn just OUTSIDE its own axes (above the top-left corner, never
inside the data area and never an `ax.set_title`) - a bare letter only, no
region name and no RP label anywhere on a panel; reading order (RP ascending
for the main map, row-major for the sub-region grid) is what ties a letter to
what it shows.

Every panel carries its own HR/FAR/CSI/bias, rounded to 2 decimals, in an
in-axes annotation box (`_load_region_metrics`, reads
`metrics_{country}_{RP}_{SLR}.csv`, filtered to that panel's own
benchmark_key) instead of a title - no plot in this module has a title or
suptitle. CSI_tol is deliberately NOT included here (it lives in the metrics
CSV as a diagnostic, not on a quick-glance map).

The permanent-water mask (`validation.permanent_water_source`/`_codes`) is
drawn as an explicit two-colour background behind every spatial panel (land
grey, permanent water blue-grey) - the same mask the evaluation domain
itself excludes, so a cell over permanent water never shows as
agree/under/over in the first place (outside the domain, written as `dry`).

`--metric depth_agreement` (default: `agreement`) plots the depth-band
comparison's own category rasters instead (validate_country_depth_bands' -
same 0/1/2/3/255 codes and priority-reduction, different underlying
classification rule and legend wording - see validate_country.py).

Usage:
    python snakemake_workflow/validation/plot_agreement_map.py \\
        --config snakemake_workflow/config/config.yml --country ESP [--metric depth_agreement]
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
    """Per-region {HR, FAR, CSI, bias, CSI_tol}, from validate_country.py's
    own metrics CSV - region -> metrics dict, empty if the CSV doesn't exist
    yet or has no `HR` column (depth-band CSVs use `pct_agree`/`pct_under`/
    `pct_over`/`depth_EB` instead - not read by this helper).

    `benchmark_key`, when given, filters to that benchmark's own rows before
    building the region->metrics dict - REQUIRED whenever more than one
    benchmark shares a country_iso (GBR: Wales + Scotland both write a
    `region == "ALL"` blended row to the same metrics CSV; without this
    filter the dict comprehension below would silently collapse both into
    whichever one iterates last).
    """
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
    """Lines for the in-panel metrics box: HR/FAR/CSI/bias ONLY - no panel
    letter, no region name, no RP label (those are drawn separately, see
    `_draw_panel_letter`), no CSI_tol (a diagnostic for the metrics CSV, not
    for a quick-glance map annotation). Rounded to 2 decimals, not 3 -
    applies pipeline-wide, every country goes through this one function.
    NaN/missing values render as 'n/a' rather than being silently dropped,
    so a reader sees that a number was expected and isn't available, not
    that this panel has no metrics at all.
    """
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
    """Top-left in-axes text box - HR/FAR/CSI/bias only (`_format_metrics_lines`).
    This module never sets `ax.set_title`/`fig.suptitle`."""
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
    """Flat two-colour land/water base (land grey, permanent water
    blue-grey) from the SAME permanent-water source the evaluation domain
    itself is built from (validation.permanent_water_source) - drawn behind
    every spatial panel in this module (agreement maps, sub-region grids,
    tile coverage), so permanent water is always visually distinguishable
    from land, not just implicitly blank. `gfm_catalog=None` skips this
    entirely (resilience - see plot_country's own try/except around
    building it).
    """
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
    """Draw one region's category raster onto `ax`, over the shared land/water
    background. Returns its (left, right, bottom, top) bounds."""
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
    # A close-up native-resolution panel (a sub-region, a few tenths of a
    # degree across) gets MANY more default tick marks than a whole-country
    # one - at a narrow panel width those 6-character "-3.850"-style labels
    # run into each other and into the xlabel below. Capping tick count and
    # rotating keeps every panel readable regardless of its real-world extent.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    ax.tick_params(axis="x", rotation=30)
    return bounds.left, bounds.right, bounds.bottom, bounds.top


_DEFAULT_LEGEND_LABELS = {
    "agree": "Agree (both wet)",
    "under": "Model under-predicts",
    "over": "Model over-predicts",
}
_DEPTH_BAND_LEGEND_LABELS = {
    "agree": "Model depth within benchmark band",
    "under": "Model under-predicts depth",
    "over": "Model over-predicts depth",
}


def _legend_handles(colors: dict[str, str], labels: dict[str, str]) -> list[Patch]:
    """The 3 category colors are config-driven (`validation.plots.agreement_colors`
    - a user can override them), but happen to equal `map_style`'s own fixed
    AGREE/A_ONLY/B_ONLY palette by default; the land/water entries always use
    `map_style`'s shared colors/labels directly, same as every other
    agreement/inundation map in the repo (`sfincs_tiles/*.py`)."""
    return [
        Patch(facecolor=colors["agree"], label=labels["agree"]),
        Patch(facecolor=colors["under"], label=labels["under"]),
        Patch(facecolor=colors["over"], label=labels["over"]),
        Patch(facecolor=_LAND_COLOR, edgecolor="black", linewidth=0.3, label=_LAND_LABEL),
        Patch(facecolor=_WATER_COLOR, edgecolor="black", linewidth=0.3, label=_WATER_LABEL),
    ]


def _sorted_rps(rp_labels) -> list[str]:
    """['RP100', 'RP250'] in numeric order, not lexical - matters once a
    native RP list stops being all the same digit-width."""
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
    """One benchmark's own whole-territory (main) region, one panel per RP.

    `rp_rasters` maps RP label ("RP100") -> that RP's own category raster
    path for the SAME region (the caller has already picked the main
    region - this function draws no insets and shows no other region).
    One panel if `rp_rasters` has one entry (no letter needed - e.g. every
    benchmark without `comparison_return_periods`, or a multi-RP benchmark
    where only one RP has been scored so far); two panels side by side,
    sharing one y-axis, lettered `(a)`/`(b)` just outside each axes'
    top-left corner (`_draw_panel_letter` - never inside the panel, never an
    RP label), if it has two. No title or suptitle anywhere - each panel's
    own HR/FAR/CSI/bias (from `rp_metrics`) is drawn as an in-axes
    annotation box instead (`_draw_annotation_box`).
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
            ax.tick_params(labelleft=False)  # sharey=True already syncs the range - just hide the repeat labels
        _draw_panel_letter(ax, letters[i])
        _draw_annotation_box(ax, _format_metrics_lines(rp_metrics.get(rp_label)))

    fig.subplots_adjust(wspace=0.04, bottom=0.17, top=0.92)  # a bit more than a bare
    # horizontal-tick panel would need - _draw_agreement_panel rotates its own x-ticks
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
    """Companion to `plot_agreement_map`: every NON-main region of one
    benchmark, as its own full-size panel - one row per region, one column
    per RP (1 or 2). `rp_region_rasters` maps RP label -> {region_name:
    raster_path} for that benchmark's own sub-regions only (the caller has
    already excluded the main region). Writes nothing if there are no
    sub-regions at all. No title or suptitle, and no region name or RP label
    anywhere on a panel - reading order (row-major) is what ties a lettered
    panel to what it shows, not text repeated on it. Panels are lettered
    `(a)..(d)` just outside each axes' own top-left corner
    (`_draw_panel_letter`) when there is more than one; each carries its own
    HR/FAR/CSI/bias in an in-axes annotation box (`_draw_annotation_box`,
    no CSI_tol). Columns of the same row (same region, different RP) share
    one y-axis.
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
                ax.tick_params(labelleft=False)  # sharey="row" already syncs the range per row
            letter = letters[panel_i] if use_letters and panel_i < len(letters) else None
            _draw_panel_letter(ax, letter)
            metrics = rp_region_metrics.get(rp_label, {}).get(region)
            _draw_annotation_box(ax, _format_metrics_lines(metrics))
            panel_i += 1

    # Fixed ABSOLUTE margins (converted to the figure-fraction subplots_adjust
    # needs), not a flat fraction of the whole figure - a flat fraction (e.g.
    # 0.08 regardless of n_regions) reserves LESS real room as the figure
    # grows taller with more rows, which is backwards: every row needs the
    # same xlabel+ticks+legend height regardless of how many OTHER rows
    # exist above it. ~0.85in at the bottom (xlabel+ticks+legend) and
    # ~0.3in at the top (the top row's own panel letter) keeps both clear of
    # the axes regardless of n_regions.
    fig_height_in = panel_size * n_regions
    bottom_frac = min(0.24, 1.05 / fig_height_in)  # a bit more than plot_agreement_map's own
    # margin - rotated x-tick labels (_draw_agreement_panel) take more vertical room than
    # horizontal ones
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
    """Which production tiles intersect a benchmark's own real extent, for
    one region (the "primary" one - v.primary_region_name - a country with
    several small/far regions like France's overseas territories would
    otherwise need one of these per region for little benefit; the main
    territory is what this diagnostic is really for). No title; the
    benchmark extent/tile footprints are drawn over the same land/water
    background every other panel in this module uses (land grey, permanent
    water blue-grey), synthesized on an ad hoc grid at
    `background_resolution_m` within `region_bbox` (this function has no
    production raster of its own to read a transform from, unlike
    `_draw_agreement_panel`).
    """
    bminx, bminy, bmaxx, bmaxy = region_bbox
    bench = benchmark_gdf.cx[bminx:bmaxx, bminy:bmaxy]
    if bench.empty:
        print(f"    plot_tile_coverage: no benchmark geometry in region bbox {region_bbox} - skipping.")
        return
    # Spatial join, not union_all() + intersects(): some real benchmark files have
    # individually-invalid geometries (e.g. France's TRI/extent shapefiles - a known
    # degenerate high-vertex-count record this session already hit elsewhere) that
    # raise a GEOS TopologyException under union_all() specifically, even though a
    # per-row intersects() test against each one works fine - sjoin never needs a
    # single combined geometry.
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


def plot_unit_csi_dots(
    units_csv: str | Path,
    output_path: str | Path,
    title: str,
    dpi: int,
) -> None:
    """One dot per evaluation unit (validate_country.py's own `unit_rows` -
    a TRI zone for a "perimeter" study area, a connected flood component
    for a "segments" one - see units_{benchmark_key}_{RP}_{SLR}.csv's own
    header / _record_unit's docstring), at that unit's centroid, coloured
    by its own CSI. Partial-coverage benchmarks only (Spain/France) - this
    is the per-unit detail a blended region/country number can't show: a
    handful of well-matched zones averaging out a handful of badly-matched
    ones into one deceptively middling CSI.
    """
    df = pd.read_csv(units_csv)
    if df.empty:
        print(f"    plot_unit_csi_dots: {units_csv} has no rows - skipping.")
        return

    fig, ax = plt.subplots(figsize=(11, 9))
    sc = ax.scatter(
        df["lon"], df["lat"], c=df["csi"], cmap="RdYlGn", vmin=0, vmax=1,
        s=80, edgecolors="black", linewidths=0.6, zorder=2,
    )
    for _, row in df.iterrows():
        if pd.notna(row["csi"]):
            ax.annotate(
                f"{row['csi']:.2f}", (row["lon"], row["lat"]), fontsize=7,
                ha="left", va="bottom", xytext=(3, 3), textcoords="offset points",
            )
    fig.colorbar(sc, ax=ax, label="CSI", fraction=0.046)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    _draw_annotation_box(ax, [title, f"{len(df)} evaluation unit(s), coloured by CSI"])
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_country(cfg: dict, country_iso: str, metric: str = "agreement") -> None:
    """Every plot this module produces for one country.

    For each benchmark matching `metric` ("agreement" -> `variable: extent`,
    "depth_agreement" -> `variable: depth`): gathers that benchmark's own
    region rasters across every RP it should be shown at
    (`meta.comparison_return_periods` if set, else just
    `cfg["validation"]["return_period"]`), and plots its main region
    (`plot_agreement_map`) and sub-regions, if any (`plot_subregions_grid`)
    - one or two RP panels each, per those functions' own docstrings. A
    benchmark with more than one RP configured gets an RP-less output
    filename (`{metric}_{country}[_{benchmark_key}]_{SLR}.png` /
    `..._subregions_{SLR}.png`), since one file now covers every RP it has
    data for; every other benchmark keeps the single-RP filename
    (`..._{RP}_{SLR}.png`). A country with more than one matching benchmark
    (GBR: Wales + Scotland) gets one file set per benchmark, named with
    `_{benchmark_key}`; a country with only one has no benchmark_key
    segment, unchanged from before per-benchmark grouping existed.

    For `metric == "agreement"` only, also plots tile coverage (every
    `GeoDataFrame` extent benchmark) and CSI dots (partial-coverage
    benchmarks with a `units_*.csv` on disk), both at the single global RP
    only - neither is RP-swept.

    Safe to call once per country regardless of how many RPs a sweep like
    `run_multi_rp_summary.py` has scored - every RP's own rasters are
    discovered by globbing disk, not passed in, so this only needs to run
    once per country AFTER all relevant RPs have been scored, not once per
    RP. Returns quietly (prints a NOTE) wherever no benchmark/raster is
    found for this country/metric - never raises.
    """
    val_cfg = cfg["validation"]
    slr = val_cfg["waterlevel_name"]
    out_dir = Path(val_cfg["output_dir"]) / country_iso
    plots_cfg = val_cfg["plots"]
    legend_labels = _DEFAULT_LEGEND_LABELS if metric == "agreement" else _DEPTH_BAND_LEGEND_LABELS
    variable = "extent" if metric == "agreement" else "depth"

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
        rp_ints = spec.comparison_return_periods if (metric == "agreement" and spec.comparison_return_periods) \
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
                metrics_csv, benchmark_key=benchmark_key if metric == "agreement" else None,
            )

        if not rp_region_rasters:
            print(f"  NOTE: no {metric} rasters found for {benchmark_key} at any of {rp_labels} - skipping.")
            continue

        file_tag = f"{country_iso}_{benchmark_key}" if multi_benchmark else country_iso
        combined_rp_file = len(rp_labels) > 1  # configured RP count, not how many are on disk yet -
        # keeps the filename stable across runs rather than renaming itself once a later RP's data appears.
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

    if metric != "agreement":
        return  # tile coverage + CSI dots are extent-path (partial-coverage) diagnostics only

    tile_gdf = None  # lazy - only loaded if a benchmark actually needs it

    for benchmark_key, spec in relevant:
        rp = val_cfg["return_period"]
        if spec.data_type != "GeoDataFrame":
            continue  # tile coverage needs real benchmark geometry - skips Denmark's RasterDataset
            # (load_benchmark_full is vector-only); CSI dots (below) are skipped separately,
            # per-benchmark, by the units_*.csv existence check - national-coverage benchmarks
            # never write one (_record_unit is a partial-coverage-only concept), so this loop
            # naturally produces tile-coverage-only (no CSI dots) for Norway/Finland/Wales/
            # Scotland/New Brunswick, and both for Spain/France.

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

        units_csv = out_dir / f"units_{benchmark_key}_{rp}_{slr}.csv"
        if units_csv.exists():
            dots_path = out_dir / f"csi_dots_{country_iso}_{benchmark_key}_{rp}_{slr}.png"
            plot_unit_csi_dots(
                units_csv, dots_path,
                title=f"{country_iso} — {benchmark_key} — per-unit CSI ({rp}, {slr})",
                dpi=int(plots_cfg["dpi"]),
            )
            print(f"Written: {dots_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--country", required=True, help="ISO-3 country code, e.g. ESP")
    parser.add_argument(
        "--metric", choices=["agreement", "depth_agreement"], default="agreement",
        help="'agreement' (default) plots the extent comparison's category rasters "
             "(validate_country.py's metrics_*.csv); 'depth_agreement' plots the "
             "depth-band comparison's instead (depth_metrics_*.csv).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    plot_country(cfg, args.country.upper(), metric=args.metric)


if __name__ == "__main__":
    main()
