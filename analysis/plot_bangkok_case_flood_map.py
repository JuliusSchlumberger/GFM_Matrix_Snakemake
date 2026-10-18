"""Plot Bangkok / Chao Phraya flood depth at one RP, one panel per SLR scenario.

Reads the merged waterdepth_{RP}_{SLR}.vrt windowed to the case polygon bbox, over a
Copernicus land-use background collapsed from 23 classes to 5.

Usage:
    python plot_bangkok_case_flood_map.py \\
        --config snakemake_workflow/config/bangkok_chao_phraya_materialized.yml \\
        --case-polygon-gpkg P:/.../bangkok_chao_phraya/bangkok_chao_phraya_domain.gpkg \\
        --case-polygon-layer bangkok_tile \\
        --return-period RP100 \\
        --out-dir P:/.../bangkok_chao_phraya/figures
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
_BUFFER_DEG = 0.15

# Copernicus Land Cover 100m codes (inputs/Copernicus/lu_to_roughness_lookup.csv)
# collapsed to 5 categories; unmapped codes (0 = "No data") are masked.
_CATEGORY_NAMES = ["Forest", "Built-up", "Cultivated land", "Bare / sparse land", "Water"]
# Water is darker than the low end of the depth colormap so it stays distinguishable.
_CATEGORY_COLORS = ["#2d6a4f", "#b23a2e", "#d9a441", "#a69484", "#3d7ea6"]
_CODE_TO_CATEGORY = {
    # Closed/open forest
    111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0,
    121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0,
    50: 1,   # Built-up
    40: 2,   # Cropland
    20: 3, 30: 3, 60: 3, 100: 3,  # Shrubland / grassland / bare-sparse / moss-lichen
    80: 4, 90: 4, 95: 4, 200: 4, 70: 4,  # Water bodies / wetland / mangroves / open sea / snow-ice
}


def _read_land_use_categories(catalog_path: Path, root: str, bounds: tuple[float, float, float, float]) -> np.ndarray:
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
    parser.add_argument("--case-polygon-layer", default="bangkok_tile")
    parser.add_argument("--return-period", default="RP100")
    parser.add_argument("--out-dir", default=None, help="default: visualization.output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bc = cfg["boundary_conditions"]
    slr_order = merged_slr_scenarios(bc, cfg["adaptation"])
    slr_order = sorted(slr_order, key=lambda s: int(s.split("_")[1]))

    plots_cfg = cfg["postprocessing"]["plots"]
    cmap = plots_cfg["waterdepth_cmap"]
    vmax_m = float(plots_cfg["waterdepth_vmax_m"])

    merged_dir = Path(cfg["postprocessing"]["merged_outputs"])
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["visualization"]["output_dir"])
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

        # Mask dry cells too, else Blues(0) hides the land-use background.
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

    # Explicit margins: tight_layout ignores the colorbar/legend added afterwards.
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
