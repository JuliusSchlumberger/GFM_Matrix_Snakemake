"""Three-colour agreement map from validate_country.py's category raster(s).

Reads the uint8 category raster(s) (0=dry, 1=agree, 2=under, 3=over, nodata=255)
written by validate_country.py's `_write_agreement_raster` - one per named
`regions:` entry (see src/validation.py::region_for_point), e.g. Spain's
`mainland` and `canary_islands` - already priority-reduced (over > under > agree
> dry) to `validation.plots.resolution_m`. The largest region (by raster extent)
is drawn as the main map; every other region gets its own small inset panel
rather than being dropped or forced into one shared, mostly-empty mosaic
spanning the gap between e.g. mainland Spain and the Canary Islands (see
docs/flood_extent_validation_caveats.md §2.1/§3). A country with only one
region (no `regions:` configured) plots exactly as before, with no insets.

`--metric depth_agreement` (default: `agreement`) plots the depth-band
comparison's own category rasters instead (validate_country_depth_bands' -
same 0/1/2/3/255 codes and priority-reduction, different underlying
classification rule and legend wording - see validate_country.py).

Usage:
    python snakemake_workflow/validation/plot_agreement_map.py \\
        --config snakemake_workflow/config/config.yml --country ESP [--metric depth_agreement]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
import validation as v  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CAT_NODATA_DEFAULT = 255

# Inset panel layout (figure-fraction coordinates) - a grid anchored at the
# bottom-left, wrapping onto additional rows once a row is full, small enough
# to read as "context", not compete with the main map. At 0.16 size + 0.02
# margin, up to 5 insets fit in a single row (e.g. France's 5 overseas
# territories) before wrapping - a single unbounded row (the original design,
# fine for Spain's 1 inset) would run the 4th/5th panel off the right edge of
# the figure (>1.0 in figure-fraction coordinates) once a country has more
# than ~4 secondary regions.
_INSET_SIZE = 0.16
_INSET_MARGIN = 0.02


def _draw_agreement_panel(
    ax: plt.Axes,
    raster_path: str | Path,
    colors: dict[str, str],
    gfm_catalog,
    permanent_water_source: str | None,
    permanent_water_codes: list[int] | None,
    label: str | None = None,
) -> tuple[float, float, float, float]:
    """Draw one region's category raster onto `ax`. Returns its (left, right, bottom, top) bounds."""
    with retry_transient_io(rasterio.open, raster_path) as src:
        data = src.read(1)
        bounds = src.bounds
        transform = src.transform
        nodata = src.nodata if src.nodata is not None else _CAT_NODATA_DEFAULT

    masked = np.ma.masked_equal(data, nodata)
    cmap = ListedColormap(["none", colors["agree"], colors["under"], colors["over"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    if gfm_catalog is not None:
        # Land background from the SAME permanent-water source the evaluation
        # domain itself is built from (validation.permanent_water_source -
        # DeltaDTM's own land/ocean/lake/river mask since 2026-09, see caveats
        # doc §1.3 - via validation.permanent_water_mask/read_permanent_water_mask),
        # in place of a separate `land_polygons` vector source (2026-09) - one
        # fewer dataset dependency, and it's never out of sync with what the
        # domain actually excludes as water.
        water_mask = v.read_permanent_water_mask(
            gfm_catalog, permanent_water_source, permanent_water_codes,
            [bounds.left, bounds.bottom, bounds.right, bounds.top], transform, data.shape,
        )
        land = np.ma.masked_where(water_mask, np.ones(data.shape, dtype="uint8"))
        ax.imshow(
            land, extent=extent, cmap=ListedColormap(["whitesmoke"]),
            vmin=0, vmax=1, origin="upper", zorder=0,
        )

    ax.imshow(masked, extent=extent, cmap=cmap, norm=norm, origin="upper", zorder=1)
    ax.set_xlim(bounds.left, bounds.right)
    ax.set_ylim(bounds.bottom, bounds.top)
    if label:
        ax.text(
            0.03, 0.97, label, transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.0),
        )
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


def plot_agreement_map(
    region_rasters: dict[str, Path],
    gfm_catalog,
    permanent_water_source: str | None,
    permanent_water_codes: list[int] | None,
    output_path: str | Path,
    title: str,
    colors: dict[str, str],
    figsize: tuple[float, float],
    dpi: int,
    legend_labels: dict[str, str] | None = None,
) -> None:
    """Plot every region's 4-category (dry/agree/under/over) raster with a discrete legend.

    Category codes are 0=dry, 1=agree, 2=under, 3=over (validate_country.py's
    `_CAT_*` constants) - dry is drawn fully transparent so the land-use
    background shows through instead of a fourth colour competing with the
    three the plan actually asks for. `region_rasters` maps region name ->
    its own agreement raster path; the region with the largest raster extent
    (pixel count - all regions share the same `plots.resolution_m`) becomes
    the main panel, the rest become small inset panels in a bottom-left grid
    (wrapping onto additional rows as needed - see _INSET_SIZE/_INSET_MARGIN).
    `gfm_catalog=None` skips the land-use background entirely (resilience -
    see main()'s own try/except around building it).
    """
    if not region_rasters:
        raise ValueError("region_rasters is empty - nothing to plot.")
    labels = legend_labels or _DEFAULT_LEGEND_LABELS

    def _pixel_count(path: Path) -> int:
        with retry_transient_io(rasterio.open, path) as src:
            return src.width * src.height

    main_region = max(region_rasters, key=lambda r: _pixel_count(region_rasters[r]))
    other_regions = [r for r in region_rasters if r != main_region]

    fig, ax = plt.subplots(figsize=figsize)
    _draw_agreement_panel(
        ax, region_rasters[main_region], colors, gfm_catalog, permanent_water_source, permanent_water_codes,
        label=main_region if other_regions else None,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend_handles = [
        Patch(facecolor=colors["agree"], label=labels["agree"]),
        Patch(facecolor=colors["under"], label=labels["under"]),
        Patch(facecolor=colors["over"], label=labels["over"]),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    max_cols = max(1, int((1.0 - _INSET_MARGIN) // (_INSET_SIZE + _INSET_MARGIN)))
    for i, region in enumerate(other_regions):
        col, row = i % max_cols, i // max_cols
        left = _INSET_MARGIN + col * (_INSET_SIZE + _INSET_MARGIN)
        bottom = _INSET_MARGIN + row * (_INSET_SIZE + _INSET_MARGIN)
        inset_ax = fig.add_axes([left, bottom, _INSET_SIZE, _INSET_SIZE])
        _draw_agreement_panel(
            inset_ax, region_rasters[region], colors, gfm_catalog, permanent_water_source, permanent_water_codes,
            label=region,
        )
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(0.8)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


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
    val_cfg = cfg["validation"]
    country_iso = args.country.upper()
    rp, slr = val_cfg["return_period"], val_cfg["waterlevel_name"]
    metric = args.metric

    out_dir = Path(val_cfg["output_dir"]) / country_iso
    prefix, suffix = f"{metric}_{country_iso}_", f"_{rp}_{slr}.tif"
    region_rasters = {
        p.name[len(prefix):-len(suffix)]: p
        for p in sorted(out_dir.glob(f"{prefix}*{suffix}"))
    }
    if not region_rasters:
        print(f"ERROR: no {prefix}*{suffix} files found in {out_dir} - run validate_country.py --country {country_iso} first.")
        sys.exit(1)
    print(f"Found {len(region_rasters)} region raster(s): {sorted(region_rasters)}")

    # Land background from validation.permanent_water_source (DeltaDTM's own
    # land/ocean/lake/river mask since 2026-09, see caveats doc §1.3),
    # masked to non-permanent-water - the SAME source/mask the evaluation domain
    # itself is built from (2026-09, replaces a separate `land_polygons` vector
    # source that was frequently missing/broken on disk - see caveats doc). Each
    # panel reads its own bbox lazily inside _draw_agreement_panel; only the
    # catalog itself is built once here.
    gfm_catalog = None
    try:
        gfm_catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    except Exception as e:
        print(f"  WARNING: could not build the GFM data catalog for the land background ({e}) - plotting without it.")

    plots_cfg = val_cfg["plots"]
    title_kind = "model vs. benchmark agreement" if metric == "agreement" else "model vs. benchmark depth-band agreement"
    out_path = out_dir / f"{metric}_{country_iso}_{rp}_{slr}.png"
    plot_agreement_map(
        region_rasters, gfm_catalog, val_cfg["permanent_water_source"], val_cfg["permanent_water_codes"], out_path,
        title=f"{country_iso} — {title_kind} ({rp}, {slr})",
        colors=plots_cfg["agreement_colors"],
        figsize=tuple(plots_cfg["figsize"]),
        dpi=int(plots_cfg["dpi"]),
        legend_labels=_DEFAULT_LEGEND_LABELS if metric == "agreement" else _DEPTH_BAND_LEGEND_LABELS,
    )
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
