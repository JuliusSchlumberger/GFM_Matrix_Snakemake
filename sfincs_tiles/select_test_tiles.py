"""Select a globally diverse ~10% sample of GFM tiles for a broader SFINCS
test batch, beyond the 3 hand-picked tiles used so far.

Eligibility (all real, checked against actual per-tile files, not assumed):
  - hop_distance == 0 (real ocean boundary - see sfincs_tiles' own plan doc:
    there is no SFINCS equivalent of the eikonal model's hop>=1 hinterland
    neighbour-seeding, so hop>=1 tiles have nothing physically meaningful
    to force a SFINCS boundary with).
  - mask.tif exists and has <= MAX_CELLS cells (native EPSG:4326 grid, not
    the later UTM SFINCS grid - a cheap proxy available before any SFINCS
    build work happens, checked via a header-only read first so oversized
    tiles never pay for a full array read).
  - river-mask (mask==3) fraction <= MAX_RIVER_FRAC - build_elevation.py has
    no river bathymetry/discharge handling (see its own module docstring),
    so a tile with meaningful river coverage isn't usable yet regardless of
    size.
  - at least one of the tile's own already-selected boundaries_RP100_SLR_0
    points has a real COAST-HG station within MAX_MATCH_DIST_DEG (same
    backstop distance as build_boundary_forcing.py's own MAX_MATCH_DIST_DEG)
    - otherwise build_boundary_forcing.py would drop every point and the
    tile can't be forced at all.

From the eligible pool, samples ~10% of ALL tiles (not just the eligible
pool) via spatially stratified sampling on a coarse global lon/lat grid, so
the selection spans multiple continents/regions rather than clustering
wherever the eligible pool happens to be densest.

Usage:
    python select_test_tiles.py
    python select_test_tiles.py --sample-frac 0.10 --max-cells 20000000 --max-river-frac 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from scipy.spatial import cKDTree

MAX_CELLS_DEFAULT = 20_000_000
MAX_RIVER_FRAC_DEFAULT = 0.01
MAX_MATCH_DIST_DEG = 0.5  # same backstop as build_boundary_forcing.py's own MAX_MATCH_DIST_DEG
SAMPLE_FRAC_DEFAULT = 0.10
GRID_DEG = 15.0  # coarse global stratification cell size for diversity sampling
RIVER_CODE = 3
OCEAN_CODE = 1


def build_coast_hg_tree(coast_hg_nc_path: Path) -> cKDTree:
    with xr.open_dataset(coast_hg_nc_path) as ds:
        lon = ds.station_x_coordinate.values
        lat = ds.station_y_coordinate.values
    return cKDTree(np.column_stack([lon, lat]))


def evaluate_tile(tile_id: int, root: Path, coast_hg_tree: cKDTree, max_cells: int, max_river_frac: float) -> dict | None:
    """Returns a metadata dict if this tile passes every eligibility check,
    else None (with the reason left in the caller's own reject-count log -
    kept cheap: header-only size check before any full-array read)."""
    tile_dir = root / "model_outputs" / str(tile_id) / "inputs"
    mask_path = tile_dir / "mask.tif"
    if not mask_path.exists():
        return None

    with rasterio.open(mask_path) as src:
        n_cells = src.width * src.height
        if n_cells > max_cells:
            return None
        mask = src.read(1)
        bounds = src.bounds

    river_frac = float(np.mean(mask == RIVER_CODE))
    ocean_frac = float(np.mean(mask == OCEAN_CODE))
    if river_frac > max_river_frac or ocean_frac <= 0.0:
        return None

    boundaries_path = tile_dir / "boundaries_RP100_SLR_0.gpkg"
    if not boundaries_path.exists():
        return None
    boundaries_gdf = gpd.read_file(boundaries_path)
    if boundaries_gdf.empty:
        return None
    pts = np.column_stack([boundaries_gdf.geometry.x.to_numpy(), boundaries_gdf.geometry.y.to_numpy()])
    dist, _ = coast_hg_tree.query(pts)
    if dist.min() > MAX_MATCH_DIST_DEG:
        return None

    return {
        "tile_id": tile_id,
        "n_cells": n_cells,
        "river_frac": river_frac,
        "ocean_frac": ocean_frac,
        "n_boundary_points": len(boundaries_gdf),
        "min_coast_hg_dist_deg": float(dist.min()),
        "lon": (bounds.left + bounds.right) / 2.0,
        "lat": (bounds.bottom + bounds.top) / 2.0,
    }


def stratified_sample(df: pd.DataFrame, n_target: int, grid_deg: float, seed: int = 0) -> pd.DataFrame:
    """Proportional-to-occupied-bin sampling on a coarse lon/lat grid, so the
    selection spans whatever spread of regions the eligible pool actually
    has rather than concentrating whichever region has the most tiles."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_bin"] = (
        (np.floor(df["lon"] / grid_deg).astype(int)).astype(str) + "_" +
        (np.floor(df["lat"] / grid_deg).astype(int)).astype(str)
    )
    bins = df["_bin"].unique()
    n_bins = len(bins)
    picks = []
    remaining = n_target
    bin_order = rng.permutation(bins)
    for i, b in enumerate(bin_order):
        bin_rows = df[df["_bin"] == b]
        bins_left = n_bins - i
        share = max(1, round(remaining / bins_left))
        take = min(share, len(bin_rows))
        picks.append(bin_rows.sample(n=take, random_state=int(rng.integers(1e9))))
        remaining -= take
        if remaining <= 0:
            break
    result = pd.concat(picks).drop(columns="_bin") if picks else df.iloc[0:0].drop(columns="_bin")
    return result.sort_values("tile_id").reset_index(drop=True)


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--max-cells", type=int, default=MAX_CELLS_DEFAULT)
    parser.add_argument("--max-river-frac", type=float, default=MAX_RIVER_FRAC_DEFAULT)
    parser.add_argument("--sample-frac", type=float, default=SAMPLE_FRAC_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gfm_config import read_root

    root = read_root(Path(args.config))
    tiles_gdf = gpd.read_file(root / "processed_inputs" / "mask" / "domain_tiles_global.gpkg")
    n_all = len(tiles_gdf)
    hop0 = tiles_gdf[tiles_gdf["hop_distance"] == 0]
    print(f"{n_all} tiles total, {len(hop0)} with hop_distance == 0")

    coast_hg_tree = build_coast_hg_tree(root / "inputs" / "COAST_HG" / "COAST-HG_RP100.nc")

    records = []
    n_checked = 0
    for tile_id in hop0["tile_id"]:
        n_checked += 1
        rec = evaluate_tile(int(tile_id), root, coast_hg_tree, args.max_cells, args.max_river_frac)
        if rec is not None:
            records.append(rec)
        if n_checked % 200 == 0:
            print(f"  checked {n_checked}/{len(hop0)}, {len(records)} eligible so far")

    eligible_df = pd.DataFrame(records)
    print(f"\n{len(eligible_df)} of {len(hop0)} hop_distance==0 tiles eligible "
          f"(<= {args.max_cells:,} cells, <= {args.max_river_frac:.1%} river, real COAST-HG match)")

    out_dir = root / "validation_sfincs"
    out_dir.mkdir(parents=True, exist_ok=True)
    eligible_df.to_csv(out_dir / "test_tile_pool.csv", index=False)
    print(f"Wrote {out_dir / 'test_tile_pool.csv'} (full eligible pool, for audit)")

    n_target = round(args.sample_frac * n_all)
    n_target = min(n_target, len(eligible_df))
    selected_df = stratified_sample(eligible_df, n_target, GRID_DEG, args.seed)
    print(f"\nSelected {len(selected_df)} tiles ({args.sample_frac:.0%} of {n_all} total tiles), "
          f"spatially stratified across a {GRID_DEG} deg grid")

    selected_csv = out_dir / "test_tile_selection.csv"
    selected_df.to_csv(selected_csv, index=False)
    print(f"Wrote {selected_csv}")

    selected_txt = out_dir / "test_tile_selection_ids.txt"
    selected_txt.write_text("\n".join(str(t) for t in selected_df["tile_id"]) + "\n")
    print(f"Wrote {selected_txt}")


if __name__ == "__main__":
    main()
