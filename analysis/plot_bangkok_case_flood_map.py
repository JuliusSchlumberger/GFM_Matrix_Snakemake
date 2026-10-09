"""Bangkok / Chao Phraya case study: flood-depth map at RP=100, one panel
per SLR scenario, case polygon outlined on every panel. Reads the merged
per-(RP,SLR) VRTs the Snakemake postprocess target already writes
(merged_results/waterdepth_RP100_{SLR}.vrt), windowed to the case
polygon's own bbox (+ a small buffer for context) - adapts
src/plotting.py::plot_raster_with_coastlines's own windowed-read/colorbar
idiom (that function saves one whole figure per call, not a subplot, so
its internals are reproduced here instead of calling it directly).

Background: the project's own `land_use` catalog entry (Copernicus Global
Land Cover 100m, data_catalog_gfm.yml) rather than a flat land/water fill -
Marjolijn asked for real land-use context. The raw layer has 23 classes
(see inputs/Copernicus/lu_to_roughness_lookup.csv for the full code->name
table) - too many to read as background texture, so _CATEGORY_LUT collapses
them to 5: forest, built-up, cultivated land, bare/sparse land, water
(Marjolijn's own grouping). No legend - purely visual context underneath
the flood-depth overlay.

Usage:
    python plot_bangkok_case_flood_map.py \\
        --config snakemake_workflow/config/bangkok_chao_phraya_materialized.yml \\
        --case-polygon-gpkg P:/.../Marjolijn_Thailand/Bangkok_tiles.gpkg \\
        --case-polygon-layer domain_tiles_globalgpkg \\
        --return-period RP100 \\
        --outdir P:/.../bangkok_chao_phraya/figures
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, merged_slr_scenarios, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUFFER_DEG = 0.15  # context margin around the case polygon's own bbox

# Copernicus Land Cover 100m class codes (inputs/Copernicus/
# lu_to_roughness_lookup.csv) collapsed to 5 broad categories for
# background texture - code 0 ("No data") is left unmapped -> masked out.
_CATEGORY_NAMES = ["Forest", "Built-up", "Cultivated land", "Bare / sparse land", "Water"]
# Representative-but-distinguishable: forest green, brick-red urban, warm
# cropland gold, a cooler/grayer taupe for bare land (so it doesn't read as
# "pale cropland"), and a steeper blue than the waterdepth Blues colormap's
# own pale low end so "Water" land-use doesn't get lost next to dry/shallow
# flooded cells.
_CATEGORY_COLORS = ["#2d6a4f", "#b23a2e", "#d9a441", "#a69484", "#3d7ea6"]
_CODE_TO_CATEGORY = {
    # Forest - every closed/open forest variant.
    111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0,
    121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0,
    50: 1,   # Built-up
    40: 2,   # Cropland
    20: 3, 30: 3, 60: 3, 100: 3,  # Shrubland / grassland / bare-sparse / moss-lichen
    80: 4, 90: 4, 95: 4, 200: 4, 70: 4,  # Water bodies / wetland / mangroves / open sea / snow-ice
}


def _read_land_use_categories(catalog_path: Path, root: str, bounds: tuple[float, float, float, float]) -> np.ndarray:
    """Windowed read of the land_use raster, remapped from its 23 raw
    Copernicus class codes to the 5 _CATEGORY_NAMES indices via
    _CODE_TO_CATEGORY (masked where unmapped, i.e. code 0/"No data")."""
    catalog = get_data_catalog(catalog_path, root=root)
    land_use_path = catalog.get_source("land_use").path
    with retry_transient_io(rasterio.open, land_use_path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        data = src.read(1, window=window, boundless=True, fill_value=src.nodata or 255)

    lut = np.full(256, -1, dtype=np.int8)
    for code, category in _CODE_TO_CATEGORY.items():
        lut[code] = category
    category_arr = lut[data]
    return np.ma.masked_equal(category_arr, -1)


def main() -> None:
    _default_cfg = str(_REPO_ROOT / "snakemake_workflow" / "config" / "bangkok_chao_phraya_materialized.yml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--case-polygon-gpkg", required=True)
    parser.add_argument("--case-polygon-layer", default="domain_tiles_globalgpkg")
    parser.add_argument("--return-period", default="RP100")
    parser.add_argument("--outdir", default=None, help="default: visualization.output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bc = cfg["boundary_conditions"]
    slr_order = merged_slr_scenarios(bc, cfg["adaptation"])
    slr_order = sorted(slr_order, key=lambda s: int(s.split("_")[1]))

    plots_cfg = cfg["postprocessing"]["plots"]
    cmap = plots_cfg.get("waterdepth_cmap", "Blues")
    vmax_m = float(plots_cfg.get("waterdepth_vmax_m", 10.0))

    merged_dir = Path(cfg["postprocessing"]["merged_outputs"])
    out_dir = Path(args.outdir) if args.outdir else Path(cfg["visualization"]["output_dir"])
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    case_poly = gpd.read_file(args.case_polygon_gpkg, layer=args.case_polygon_layer)
    minx, miny, maxx, maxy = case_poly.total_bounds
    bounds = (minx - _BUFFER_DEG, miny - _BUFFER_DEG, maxx + _BUFFER_DEG, maxy + _BUFFER_DEG)

    land_use = _read_land_use_categories(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], cfg["paths"]["root"], bounds)
    land_use_cmap = ListedColormap(_CATEGORY_COLORS)

    fig, axes = plt.subplots(1, len(slr_order), figsize=(4.5 * len(slr_order), 5.5), sharex=True, sharey=True)
    if len(slr_order) == 1:
        axes = [axes]

    image = None
    for ax, slr in zip(axes, slr_order):
        vrt_path = merged_dir / f"waterdepth_{args.return_period}_{slr}.vrt"
        if not vrt_path.exists():
            ax.set_title(f"{slr}\n(missing)")
            continue
        with retry_transient_io(rasterio.open, vrt_path) as src:
            window = from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=window, boundless=True, fill_value=src.nodata or 0.0)
            nodata = src.nodata
            win_transform = src.window_transform(window)

        # Mask BOTH true nodata and zero-depth (dry) cells - not just nodata -
        # so the land-use background actually shows through dry land, not
        # just outside the raster's own data extent. Dry land is the vast
        # majority of any panel; leaving it opaque at Blues(0) (near-white,
        # but NOT transparent) previously hid the land-use layer underneath
        # almost everywhere.
        dry_or_nodata = (data <= 0) if nodata is None else ((data == nodata) | (data <= 0))
        masked = np.ma.masked_array(data, mask=dry_or_nodata)
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])

        ax.imshow(land_use, extent=extent, cmap=land_use_cmap, origin="upper", zorder=0,
                  vmin=-0.5, vmax=len(_CATEGORY_NAMES) - 0.5, alpha=0.85)
        image = ax.imshow(masked, extent=extent, cmap=cmap, origin="upper", zorder=1,
                           vmin=0, vmax=vmax_m)
        case_poly.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=2)

        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_title(slr, fontsize=10)
        ax.set_xlabel("Longitude")

    axes[0].set_ylabel("Latitude")
    fig.suptitle(f"Bangkok / Chao Phraya case study: flood depth at {args.return_period}", y=0.98)

    # Explicit margins + dedicated axes for both the colorbar (vertical,
    # right of the last panel) and the land-use legend (below all panels) -
    # not fig.colorbar(..., ax=axes) + tight_layout(), which placed the
    # colorbar ON TOP of the panels instead of in reserved space (tight_layout
    # doesn't know about axes/legends added after the subplots are laid out).
    fig.subplots_adjust(left=0.05, right=0.90, top=0.88, bottom=0.2, wspace=0.08)
    if image is not None:
        fig.colorbar(image, ax=axes[-1], label="Water depth (m)", orientation="vertical",
                     fraction=0.08, pad=0.03)

    legend_handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=0.5, label=name)
        for name, color in zip(_CATEGORY_NAMES, _CATEGORY_COLORS)
    ]
    fig.legend(handles=legend_handles, title="Land use", loc="lower center",
               bbox_to_anchor=(0.46, 0.0), ncol=len(_CATEGORY_NAMES), fontsize=9, title_fontsize=9, frameon=False)

    out_path = out_dir / f"bangkok_case_flood_map_{args.return_period}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
