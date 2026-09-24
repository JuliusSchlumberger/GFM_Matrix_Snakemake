"""TODO (2026-09-24, user direction, currently BLOCKED - see below): this
two-set design is scheduled to be replaced with a SINGLE ~520-tile
selection - no A/B split, no latitude restriction at all (drop Set A's own
|lat| 5-55 deg band entirely - take from the full range, poles included),
smallest-by-area, antimeridian-excluded, spatially stratified, with a NEW
floor of ocean_frac >= 0.05 applied to every selected tile (not just a
Set-B-only filter as before - every tile in the single selection must
clear it). The old two-set design's own asymmetry (only Set A had a
latitude restriction; Set B had none) pulled 56% of Set B's 260 tiles (127
tiles, 96 of those above the Arctic/Antarctic Circle) outside that band,
visible directly on the resulting tile-locations map - dropping the
latitude restriction entirely (rather than re-applying it inconsistently)
was the user's own direction once that asymmetry was pointed out. The
replacement script (working name select_validation_tiles.py) should also
generate, from the selected sample: a global Equal Earth tile-locations map
(light grey land, no figure title), a histogram of the selected sample's
size/ocean-fraction distribution against the eligible pool, and an Excel
workbook with the summary statistics tables. BLOCKED until model_outputs/
exists again (deliberately deleted 2026-09-24 to force a clean production
preprocessing rebuild after the GEBCO-based mask fix) - this script's own
evaluate_tile eligibility check reads each candidate tile's mask.tif/
boundaries gpkg from model_outputs/{tile_id}/inputs/. See sfincs_tiles/
SFINCS_VALIDATION_METHODOLOGY.md's own "Known follow-up work" section.

Select two new, non-overlapping ~260-tile sets for the clean rebuild of
the eikonal-vs-SFINCS validation batch (2026-09), reusing
select_test_tiles.py's own proven eligibility logic (evaluate_tile/
build_coast_hg_tree/stratified_sample) rather than re-deriving it - see
that module's own docstring for the full eligibility rationale (hop_distance
==0, mask.tif size cap, <=1% river, real COAST-HG match).

Works directly off model_outputs/{tile_id}/inputs/ as it currently exists on
disk (not a fresh catalog re-derivation) - by design, per this session's own
plan: selection is meant to be cheap and immediate, not block on
re-preprocessing the whole globe first.

Set A: globally scattered, hop_distance==0, eligible, smallest-by-area,
non-polar (|lat| 5-55 deg) - same criteria as this morning's original
258-tile selection.

Set B: same eligibility, explicitly targeting small boundary-distance-proxy
tiles (SFINCS's own create_active/create_boundary places its water-level
boundary at the perimeter of the tile's own UTM bounding box - see
build_sfincs_tile.py's own mask/boundary comments - so the real predictor of
a good/bad SFINCS boundary placement is how far the tile's own bbox edge
sits from its nearest real land, computable straight from mask.tif with no
SFINCS build needed), restricted to ocean_frac in [0.05, 0.50] - an earlier
version targeted [0.10, 0.20] but only 46 of 1294 eligible tiles globally
fall in that band (confirmed 2026-09-23), nowhere near enough for a
260-tile set. A histogram of the remaining pool's ocean_frac (same date)
showed no natural low-ocean cluster - it thins out smoothly - so per user
direction the band was widened to [0.05, 0.50] (336 tiles in the remaining
pool, close to the 260 target while still meaningfully "not ocean-
dominated", and wide enough to leave room to also prefer short boundary
distance within the band). Applied BEFORE the (expensive, one
distance-transform-per-tile) boundary-distance proxy computation, not
after, to avoid wasting time on tiles that would be filtered out anyway.
Takes a wider band by that proxy (bottom ~40th percentile) rather than the
strict best-260, then spatially stratifies within that band, so Set B isn't
clustered wherever the proxy happens to be smallest.

New: antimeridian exclusion - real, confirmed failure mode this session
(tiles straddling +-180 deg break hydromt_sfincs's own water_level.create()
with a GEOS topology exception - see build_sfincs_tile.py's own
_classify_water_level_create_error). No existing geometric check for this
anywhere in the repo (confirmed via search) - a tile whose own geometry
bounds span an implausibly wide longitude range (only possible for a real
coastal tile if its polygon wraps the dateline, since real GFM tiles are
never anywhere near this wide) is flagged and dropped from both sets.

Usage:
    python select_new_tile_sets.py
    python select_new_tile_sets.py --n-set-a 260 --n-set-b 260
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_test_tiles import (  # noqa: E402
    MAX_CELLS_DEFAULT,
    MAX_RIVER_FRAC_DEFAULT,
    GRID_DEG,
    build_coast_hg_tree,
    evaluate_tile,
    stratified_sample,
)
from gfm_config import read_root  # noqa: E402

LAND_CODE = 0
N_SET_A_DEFAULT = 260
N_SET_B_DEFAULT = 260
LAT_MIN_DEFAULT = 5.0
LAT_MAX_DEFAULT = 55.0
OCEAN_FRAC_MIN_DEFAULT = 0.05
OCEAN_FRAC_MAX_DEFAULT = 0.50
SET_B_DISTANCE_PERCENTILE_DEFAULT = 70.0  # bottom-70th-percentile-by-proxy-distance band - the
# stricter 40th only yields 171 tiles from the (ocean_frac-filtered, ~428-tile) pool, short of
# the 260 target (confirmed 2026-09-23); 70th comfortably clears 260 while still excluding the
# worst (highest-distance) 30% of that pool
ANTIMERIDIAN_BBOX_WIDTH_DEG_DEFAULT = 60.0  # real GFM tiles are never anywhere near this wide


def is_antimeridian_tile(geom, max_width_deg: float) -> bool:
    """True if this tile's own geometry bounds span an implausibly wide
    longitude range - the real signature of a polygon that wraps the dateline
    (naive min/max longitude across a dateline-crossing feature spans nearly
    360 deg, vastly wider than any real GFM tile)."""
    minx, _, maxx, _ = geom.bounds
    return (maxx - minx) > max_width_deg


def boundary_distance_proxy_km(mask_path: Path) -> float | None:
    """Median distance (km) from the tile's own raster bbox edge to the
    nearest real land cell in mask.tif - a cheap, no-SFINCS-build-needed
    proxy for where SFINCS's own create_active/create_boundary would end up
    placing its water-level boundary (see build_sfincs_tile.py's own mask
    step: create_active(include_polygon=tile_gdf) sets the WHOLE tile bbox
    active, and create_boundary places boundary cells on that active
    domain's own perimeter - so the bbox-edge-to-land distance IS
    essentially the real boundary distance, before any SFINCS model exists
    to measure it directly)."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
    real_land = mask == LAND_CODE
    if not real_land.any():
        return None
    # real per-axis metre resolution at this tile's own latitude, via the raster's own transform
    px_w_deg = abs(transform.a)
    px_h_deg = abs(transform.e)
    lat_center = transform.f + transform.e * (mask.shape[0] / 2.0)
    px_w_m = px_w_deg * 111320.0 * np.cos(np.radians(lat_center))
    px_h_m = px_h_deg * 110540.0
    dist_to_land = ndimage.distance_transform_edt(~real_land, sampling=(px_h_m, px_w_m))

    edge = np.zeros_like(real_land, dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    d_edge = dist_to_land[edge & ~real_land]
    if d_edge.size == 0:
        return None
    return float(np.median(d_edge)) / 1000.0


def stratified_sample_exact(df: pd.DataFrame, n_target: int, grid_deg: float, seed: int) -> pd.DataFrame:
    """Like select_test_tiles.stratified_sample, but round-robins one tile at
    a time across bins until n_target is met or the pool is exhausted,
    instead of a single capped pass. select_test_tiles.stratified_sample's
    per-bin "share" is fixed at visit time and never revisited, so when bin
    sizes are uneven it can undershoot even when n_target == the pool size
    (confirmed 2026-09-23 for Set B: asked for all 171 of a 171-tile pool,
    got only 131 - a few bins visited early got capped below their real
    size, and that leftover capacity was never reclaimed). Used for Set B,
    where the distance-filtered pool is often close in size to the target;
    Set A keeps the original stratified_sample since its pool is
    deliberately oversampled 3x, which sidesteps the bug in practice."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_bin"] = (
        (np.floor(df["lon"] / grid_deg).astype(int)).astype(str) + "_" +
        (np.floor(df["lat"] / grid_deg).astype(int)).astype(str)
    )
    bin_pools = {b: list(g.index) for b, g in df.groupby("_bin")}
    for b in bin_pools:
        rng.shuffle(bin_pools[b])
    bins = list(bin_pools.keys())
    rng.shuffle(bins)

    picked = []
    while len(picked) < n_target and any(bin_pools.values()):
        for b in bins:
            if len(picked) >= n_target:
                break
            if bin_pools[b]:
                picked.append(bin_pools[b].pop())
    return df.loc[picked].drop(columns="_bin").sort_values("tile_id").reset_index(drop=True)


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--max-cells", type=int, default=MAX_CELLS_DEFAULT)
    parser.add_argument("--max-river-frac", type=float, default=MAX_RIVER_FRAC_DEFAULT)
    parser.add_argument("--n-set-a", type=int, default=N_SET_A_DEFAULT)
    parser.add_argument("--n-set-b", type=int, default=N_SET_B_DEFAULT)
    parser.add_argument("--lat-min", type=float, default=LAT_MIN_DEFAULT)
    parser.add_argument("--lat-max", type=float, default=LAT_MAX_DEFAULT)
    parser.add_argument("--ocean-frac-min", type=float, default=OCEAN_FRAC_MIN_DEFAULT)
    parser.add_argument("--ocean-frac-max", type=float, default=OCEAN_FRAC_MAX_DEFAULT)
    parser.add_argument("--set-b-distance-percentile", type=float, default=SET_B_DISTANCE_PERCENTILE_DEFAULT)
    parser.add_argument("--antimeridian-width-deg", type=float, default=ANTIMERIDIAN_BBOX_WIDTH_DEG_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    tiles_gdf = gpd.read_file(root / "processed_inputs" / "mask" / "domain_tiles_global.gpkg")
    n_all = len(tiles_gdf)
    hop0 = tiles_gdf[tiles_gdf["hop_distance"] == 0].copy()
    print(f"{n_all} tiles total, {len(hop0)} with hop_distance == 0")

    hop0["antimeridian"] = hop0.geometry.apply(lambda g: is_antimeridian_tile(g, args.antimeridian_width_deg))
    n_antimeridian = int(hop0["antimeridian"].sum())
    hop0 = hop0[~hop0["antimeridian"]].copy()
    print(f"{n_antimeridian} antimeridian-crossing tile(s) excluded, {len(hop0)} remain")

    # real area (km2, latitude-corrected) and centroid lat, straight from the tile's own geometry -
    # no raster read needed for this part
    bounds = hop0.geometry.bounds
    centroid = hop0.geometry.centroid
    hop0["lat_centroid"] = centroid.y.to_numpy()
    lon_span_deg = bounds["maxx"] - bounds["minx"]
    lat_span_deg = bounds["maxy"] - bounds["miny"]
    lat_mid = (bounds["maxy"] + bounds["miny"]) / 2.0
    w_km = lon_span_deg * 111.32 * np.cos(np.radians(lat_mid))
    h_km = lat_span_deg * 110.54
    hop0["area_km2_bbox"] = (w_km * h_km).to_numpy()

    coast_hg_tree = build_coast_hg_tree(root / "inputs" / "COAST_HG" / "COAST-HG_RP100.nc")

    records = []
    n_checked = 0
    for tile_id in hop0["tile_id"]:
        n_checked += 1
        rec = evaluate_tile(int(tile_id), root, coast_hg_tree, args.max_cells, args.max_river_frac)
        if rec is not None:
            records.append(rec)
        if n_checked % 200 == 0:
            print(f"  checked {n_checked}/{len(hop0)}, {len(records)} eligible so far", flush=True)

    eligible_df = pd.DataFrame(records)
    eligible_df = eligible_df.merge(
        hop0[["tile_id", "lat_centroid", "area_km2_bbox"]], on="tile_id", how="left",
    )
    print(f"\n{len(eligible_df)} of {len(hop0)} hop_distance==0, non-antimeridian tiles eligible "
          f"(<= {args.max_cells:,} cells, <= {args.max_river_frac:.1%} river, real COAST-HG match)")

    out_dir = root / "validation_sfincs_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    eligible_df.to_csv(out_dir / "eligible_tile_pool.csv", index=False)
    print(f"Wrote {out_dir / 'eligible_tile_pool.csv'} (full eligible pool, for audit)")

    # -- Set A: smallest-by-area, non-polar, globally scattered --
    set_a_pool = eligible_df[
        (eligible_df["lat_centroid"].abs() >= args.lat_min) & (eligible_df["lat_centroid"].abs() <= args.lat_max)
    ].copy()
    set_a_pool = set_a_pool.sort_values("area_km2_bbox").head(max(len(set_a_pool), args.n_set_a * 3))
    set_a = stratified_sample(set_a_pool, min(args.n_set_a, len(set_a_pool)), GRID_DEG, args.seed)
    set_a["set"] = "A"
    print(f"\nSet A: {len(set_a)} tiles (smallest-by-area, |lat| {args.lat_min}-{args.lat_max}, stratified)")

    # -- Set B: ocean_frac in [ocean-frac-min, ocean-frac-max] (see module docstring for why
    # this band, not the stricter original 10-20%), then small boundary-distance proxy within
    # that band, wide band + stratified, non-overlapping with A --
    remaining_pool = eligible_df[~eligible_df["tile_id"].isin(set_a["tile_id"])].copy()
    ocean_ok = remaining_pool[
        (remaining_pool["ocean_frac"] >= args.ocean_frac_min) & (remaining_pool["ocean_frac"] <= args.ocean_frac_max)
    ].copy()
    print(f"  {len(ocean_ok)} of {len(remaining_pool)} remaining tiles have ocean_frac in "
          f"[{args.ocean_frac_min:.0%}, {args.ocean_frac_max:.0%}]")

    proxy_dists = []
    n_checked = 0
    for tile_id in ocean_ok["tile_id"]:
        n_checked += 1
        mask_path = root / "model_outputs" / str(tile_id) / "inputs" / "mask.tif"
        proxy_dists.append(boundary_distance_proxy_km(mask_path))
        if n_checked % 200 == 0:
            print(f"  boundary-distance proxy: checked {n_checked}/{len(ocean_ok)}", flush=True)
    ocean_ok["boundary_dist_proxy_km"] = proxy_dists
    ocean_ok = ocean_ok.dropna(subset=["boundary_dist_proxy_km"])

    cutoff = np.percentile(ocean_ok["boundary_dist_proxy_km"], args.set_b_distance_percentile)
    set_b_pool = ocean_ok[ocean_ok["boundary_dist_proxy_km"] <= cutoff].copy()
    print(f"  {len(set_b_pool)} tiles within the bottom {args.set_b_distance_percentile:.0f}th percentile "
          f"of boundary-distance proxy (<= {cutoff:.2f} km)")
    set_b = stratified_sample_exact(set_b_pool, min(args.n_set_b, len(set_b_pool)), GRID_DEG, args.seed + 1)
    set_b["set"] = "B"
    print(f"\nSet B: {len(set_b)} tiles (ocean_frac {args.ocean_frac_min:.0%}-{args.ocean_frac_max:.0%}, "
          f"low boundary-distance proxy, stratified)")

    combined = pd.concat([set_a, set_b], ignore_index=True)
    assert combined["tile_id"].is_unique, "Set A and Set B overlap - selection logic bug"
    combined.to_csv(out_dir / "tile_selection_metadata.csv", index=False)
    (out_dir / "set_a_tile_ids.txt").write_text("\n".join(str(t) for t in set_a["tile_id"]) + "\n")
    (out_dir / "set_b_tile_ids.txt").write_text("\n".join(str(t) for t in set_b["tile_id"]) + "\n")
    print(f"\nWrote {out_dir / 'tile_selection_metadata.csv'}, "
          f"{out_dir / 'set_a_tile_ids.txt'} ({len(set_a)}), {out_dir / 'set_b_tile_ids.txt'} ({len(set_b)})")


if __name__ == "__main__":
    main()
