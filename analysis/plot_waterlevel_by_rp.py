"""Water-level-vs-return-period curve for a tile-grid region's real COAST-RP
boundary forcing, one SLR scenario, built from the per-tile
`boundaries_{RP}_{SLR}.gpkg` files `extract_boundaries.py` writes during
preprocessing. Shows the median and min/max range across every station
actually used, across all `boundary_conditions.return_periods`.

A statistical chart, not a spatial map - keeps a normal title, matching the
rest of the codebase's existing split between spatial land/water maps (no
titles, src/map_style.py) and plain statistical charts (titles kept, e.g.
sfincs_tiles/plot_validation_results.py's histograms/scatter plots).

No production script plotted this before (confirmed via codebase review).
Generic over `tile_grid.path`/`simulation.model_outputs`.

Usage:
    python analysis/plot_waterlevel_by_rp.py --config <config.yml> \\
        [--waterlevel-name SLR_0] [--out <path>]
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402
from rasters import decode_waterlevel_cm  # noqa: E402
from tiles import load_tile_grid  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--waterlevel-name", default="SLR_0")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tile_grid = load_tile_grid(cfg["tile_grid"]["path"])
    model_outputs = Path(cfg["simulation"]["model_outputs"])
    return_periods = cfg["boundary_conditions"]["return_periods"]

    rp_vals, rp_med, rp_lo, rp_hi, rp_n = [], [], [], [], []
    for rp in return_periods:
        rp_label = f"RP{rp}"
        levels = []
        for tid in sorted(tile_grid["tile_id"].astype(int)):
            bpath = model_outputs / str(tid) / "inputs" / f"boundaries_{rp_label}_{args.waterlevel_name}.gpkg"
            if not bpath.exists():
                continue
            gdf = gpd.read_file(bpath)
            if gdf.empty:
                continue
            wl_col = [c for c in gdf.columns if c != "geometry"][0]
            levels.append(decode_waterlevel_cm(gdf[wl_col].to_numpy()))
        if not levels:
            print(f"no boundary data yet for {rp_label}/{args.waterlevel_name} - skipping")
            continue
        levels = np.concatenate(levels)
        rp_vals.append(rp)
        rp_med.append(float(np.median(levels)))
        rp_lo.append(float(levels.min()))
        rp_hi.append(float(levels.max()))
        rp_n.append(levels.size)

    if not rp_vals:
        raise SystemExit("no boundary station data found for any return period - run preprocessing first")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(rp_vals, rp_lo, rp_hi, color="#4292c6", alpha=0.25, label="min-max across stations")
    ax.plot(rp_vals, rp_med, color="#08519c", marker="o", lw=1.8, label="median across stations")
    for x, y, n in zip(rp_vals, rp_med, rp_n):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 8),
                    fontsize=7, ha="center", color="#4a4a55")
    ax.set_xscale("log")
    ax.set_xticks(rp_vals)
    ax.set_xticklabels([str(rp) for rp in rp_vals])
    ax.set_xlabel("return period (yr)")
    ax.set_ylabel("COAST-RP water level (m)")
    ax.set_title(f"Boundary forcing water level vs. return period ({args.waterlevel_name})", loc="left")
    ax.legend(fontsize=8, loc="upper left", frameon=True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    out_path = (Path(args.out) if args.out else
                Path(cfg.get("visualization", {}).get("output_dir", f"{cfg['paths']['root']}/figures")) /
                f"waterlevel_by_rp_{args.waterlevel_name}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
