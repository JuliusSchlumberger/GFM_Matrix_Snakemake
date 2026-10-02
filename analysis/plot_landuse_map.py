"""Land-use input map for a tile-grid region - the Copernicus Global Land
Service Land Cover 100 m classification (`land_use` catalog source) that
`compute_friction.py` converts to Manning's n roughness for every tile,
clipped to the region's own tile grid and shown at native resolution.

No production script plotted this input before (confirmed via codebase
review) - every prior land-use figure was a one-off diagnostic. Generic over
`tile_grid.path`, so it works for any region's config, not just one project.

Usage:
    python analysis/plot_landuse_map.py --config <config.yml> [--out <path>]
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from map_style import WATER_COLOR, WATER_LABEL, draw_caption_box  # noqa: E402
from tiles import load_tile_grid  # noqa: E402

# Coarse Copernicus CGLS-LC100 classes (code, name, color) - the 11 forest
# subtypes (111-126), all mapped to the SAME Manning's n in
# lu_to_roughness_lookup.csv, are merged into one "Forest" entry so the
# legend stays readable; every other code keeps its own class.
_CLASSES = [
    (0, "No data", "#ffffff"),
    (20, "Shrubland", "#dcbe76"),
    (30, "Grassland", "#ffcc66"),
    (40, "Cropland", "#e8e81a"),
    (50, "Built-up", "#c41c1c"),
    (60, "Bare / sparse vegetation", "#b2b2b2"),
    (70, "Snow and ice", "#f0f0ff"),
    (80, "Permanent water bodies", WATER_COLOR),
    (90, "Herbaceous wetland", "#6ba8a9"),
    (95, "Mangroves", "#00cf75"),
    (100, "Moss and lichen", "#b0cc66"),
    (111, "Forest", "#196d12"),  # also catches 112-126, see _remap below
    (200, "Open sea", WATER_COLOR),
]
_FOREST_CODES = {111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126}


def _remap(codes: np.ndarray) -> np.ndarray:
    out = codes.copy()
    out[np.isin(out, list(_FOREST_CODES))] = 111
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="output PNG path (default: visualization.output_dir/landuse_map.png)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tile_grid = load_tile_grid(cfg["tile_grid"]["path"])
    minx, miny, maxx, maxy = tile_grid.total_bounds
    pad = 0.1
    bbox = [minx - pad, miny - pad, maxx + pad, maxy + pad]

    catalog = get_data_catalog(cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    da = catalog.get_rasterdataset("land_use", bbox=bbox)
    values = _remap(da.values.astype(np.int32))

    codes = [c for c, _, _ in _CLASSES]
    present = sorted(set(np.unique(values)) & set(codes))
    present_classes = [(c, n, col) for c, n, col in _CLASSES if c in present]

    code_to_idx = {c: i for i, (c, _, _) in enumerate(present_classes)}
    idx_arr = np.vectorize(lambda v: code_to_idx.get(int(v), -1))(values)

    cmap = ListedColormap([col for _, _, col in present_classes])
    bounds = np.arange(-0.5, len(present_classes) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    ext = [float(da.x.min()), float(da.x.max()), float(da.y.min()), float(da.y.max())]
    fig, ax = plt.subplots(figsize=(10, 10 * (maxy - miny) / max(maxx - minx, 1e-6)))
    ax.imshow(np.ma.masked_equal(idx_arr, -1), extent=ext, origin="upper" if da.y[0] > da.y[-1] else "lower",
              cmap=cmap, norm=norm, interpolation="nearest")
    tile_grid.boundary.plot(ax=ax, color="black", linewidth=0.8)
    for _, row in tile_grid.iterrows():
        c = row.geometry.centroid
        ax.annotate(str(int(row.tile_id)), (c.x, c.y), fontsize=7, ha="center", va="center", color="black")

    legend_handles = [Patch(facecolor=col, edgecolor="black", linewidth=0.3, label=name)
                       for _, name, col in present_classes]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=7, ncol=2,
              frameon=True, facecolor="white", edgecolor="grey")
    draw_caption_box(ax, "Copernicus Global Land Cover 100 m (2019) - tile outlines labelled by tile_id", loc="upper left")
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    out_path = Path(args.out) if args.out else Path(cfg.get("visualization", {}).get("output_dir", f"{cfg['paths']['root']}/figures")) / "landuse_map.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
