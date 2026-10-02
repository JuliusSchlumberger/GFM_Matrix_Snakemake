"""Map of the real COAST-RP boundary stations forcing a tile-grid region, for
one (return_period, waterlevel_name) scenario - the per-tile
`boundaries_{RP}_{SLR}.gpkg` files `extract_boundaries.py` writes during
preprocessing (post ocean-connectivity filtering - exactly what each tile's
simulation actually used, not the unfiltered candidate pool).

No production script plotted this before (confirmed via codebase review) -
the only prior version was a one-off Martinique diagnostic
(tests/martinique_diagnostics/06_make_figures.py fig1). Generic over
`tile_grid.path`/`simulation.model_outputs`, so it works for any region.

Usage:
    python analysis/plot_coastrp_stations_map.py --config <config.yml> \\
        [--return-period RP100] [--waterlevel-name SLR_0] [--out <path>]
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402
from map_style import LAND_COLOR, LAND_LABEL, WATER_COLOR, WATER_LABEL, draw_caption_box  # noqa: E402
from rasters import decode_waterlevel_cm  # noqa: E402
from tiles import load_tile_grid  # noqa: E402


def _land_background(mask_vrt: Path, bbox: list[float]):
    with rasterio.open(mask_vrt) as src:
        b = src.bounds
        cb = (max(bbox[0], b.left), max(bbox[1], b.bottom), min(bbox[2], b.right), min(bbox[3], b.top))
        window = from_bounds(*cb, transform=src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=window)
        transform = src.window_transform(window)
    ext = [transform.c, transform.c + arr.shape[1] * transform.a,
           transform.f + arr.shape[0] * transform.e, transform.f]
    return np.where(arr == 0, 1.0, np.nan), ext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--return-period", default="RP100")
    parser.add_argument("--waterlevel-name", default="SLR_0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tile_grid = load_tile_grid(cfg["tile_grid"]["path"])
    model_outputs = Path(cfg["simulation"]["model_outputs"])
    mask_vrt = Path(cfg["paths"]["root"]) / "inputs" / "DeltaDTM_masks" / "deltadtm_mask.vrt"

    frames = []
    missing = []
    for tid in sorted(tile_grid["tile_id"].astype(int)):
        bpath = model_outputs / str(tid) / "inputs" / f"boundaries_{args.return_period}_{args.waterlevel_name}.gpkg"
        if not bpath.exists():
            missing.append(tid)
            continue
        gdf = gpd.read_file(bpath)
        if gdf.empty:
            continue
        wl_col = [c for c in gdf.columns if c != "geometry"][0]
        gdf = gdf.assign(waterlevel_m=decode_waterlevel_cm(gdf[wl_col].to_numpy()), tile_id=tid)
        frames.append(gdf[["tile_id", "waterlevel_m", "geometry"]])
    if missing:
        print(f"skipped {len(missing)} tile(s) with no boundaries file yet: {missing}")
    if not frames:
        raise SystemExit(f"no boundary station files found for {args.return_period}/{args.waterlevel_name}")
    stations = gpd.GeoDataFrame(__import__("pandas").concat(frames, ignore_index=True), crs=tile_grid.crs)
    stations = stations.drop_duplicates(subset=["geometry"])

    minx, miny, maxx, maxy = tile_grid.total_bounds
    pad = 0.5
    bbox = [minx - pad, miny - pad, maxx + pad, maxy + pad]
    land, land_ext = _land_background(mask_vrt, bbox)

    fig, ax = plt.subplots(figsize=(10, 10 * (maxy - miny) / max(maxx - minx, 1e-6)))
    ax.imshow(land, extent=land_ext, origin="upper", cmap=ListedColormap([LAND_COLOR]), interpolation="nearest", zorder=0)
    ax.set_facecolor(WATER_COLOR)
    tile_grid.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=2)
    sc = ax.scatter(stations.geometry.x, stations.geometry.y, c=stations["waterlevel_m"],
                     cmap="viridis", s=28, edgecolor="white", linewidth=0.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(f"COAST-RP {args.return_period} / {args.waterlevel_name} water level (m)", fontsize=8)

    ax.legend(handles=[
        Patch(facecolor=LAND_COLOR, edgecolor="black", linewidth=0.3, label=LAND_LABEL),
        Patch(facecolor=WATER_COLOR, edgecolor="black", linewidth=0.3, label=WATER_LABEL),
    ], loc="lower left", fontsize=7, frameon=True, facecolor="white", edgecolor="grey")
    draw_caption_box(ax, f"{len(stations)} boundary station(s) actually used, "
                          f"{args.return_period}/{args.waterlevel_name}", loc="upper left")
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    out_path = (Path(args.out) if args.out else
                Path(cfg.get("visualization", {}).get("output_dir", f"{cfg['paths']['root']}/figures")) /
                f"coastrp_stations_{args.return_period}_{args.waterlevel_name}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
