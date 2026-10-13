"""Overlap-corrected re-derivation of compute_calibration_result.py's
per-tile calibration_result.json - friction-sweep tiles overlap each other
(production tile grid, deliberate boundary-condition buffer), so naively
pooling matched_km2/model_only_km2/sfincs_only_km2 across tiles (calibration_
results.sweep_tables) counts any (mis)agreement in an overlap zone once PER
overlapping tile, not once per real location - confirmed 2026-10 via
tile_selection.gpkg: summed individual tile footprints overcount the real
(union) area by 1.237x.

Fix: a Voronoi partition of the selected tile set (each tile's "core" =
the portion of its domain closer to ITS OWN center than to any other
selected tile's center) - same structural idea as src/merge.py's own
provenance-raster resolution of overlapping PRODUCTION tiles (one winning
tile per real-world cell), just applied to the comparison mask instead of
merged depth values. Every pixel counts toward exactly one tile's
contribution to the pooled metric.

Reuses compute_calibration_result.py's own per-pixel comparison logic
UNCHANGED (_depth_stats, _sfincs_active_on_subgrid, _compare) - the ONLY
change is intersecting `score` (SFINCS-active real land) with the new
`core_mask` before any counting happens, so every downstream count
(domain stats, bathtub/eikonal flooded_km2, vs_sfincs confusion counts)
is automatically restricted to the de-duplicated core.

Writes to a SEPARATE base directory (e.g. calibration_roughness_SFINCS_dedup),
mirroring the {tile_id}/outputs/calibration_result.json layout the original
study uses - NOT overwriting calibration_roughness_SFINCS - so the existing,
unmodified calibration_results.py/compute_friction_sweep_metrics.py can read
it back with zero changes, just pointed at the new --base-dir-name.

Needs hydromt-sfincs-dev (reads sfincs.msk through SfincsModel, same as
compute_calibration_result.py).

Usage:
    python compute_calibration_result_dedup.py --config <config.yml> \\
        --base-dir-name calibration_roughness_SFINCS \\
        --out-dir-name calibration_roughness_SFINCS_dedup \\
        [--tile-id 1122] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import (  # noqa: E402
    FRICTION_SCALE_FACTORS, WATERLEVEL_NAME, bathtub_path, decode_waterdepth_cm, eikonal_path,
)
from compute_calibration_result import LAND_CODE, OCEAN_CODE, _compare, _depth_stats, _sfincs_active_on_subgrid  # noqa: E402
from flood_agreement import WET_THRESHOLD_M, prune_to_ocean_connected  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

_PARTITION_CRS = "EPSG:6933"  # equal-area - fair distance comparison regardless of latitude


def build_tile_tree(tile_selection_gpkg: Path) -> tuple[cKDTree, np.ndarray]:
    """KDTree over every SELECTED tile's centroid (in EPSG:6933), + the
    parallel tile_id array (tree index i -> tile_ids[i]). Built from ALL
    selected tiles (not just ones with a usable result) so a tile missing
    its own eikonal/bathtub output still correctly claims its own Voronoi
    territory from its overlapping neighbours, rather than silently ceding
    it to them.
    """
    gdf = gpd.read_file(tile_selection_gpkg)
    sel = gdf[gdf["category"] == "selected"].to_crs(_PARTITION_CRS)
    centers = np.column_stack([sel.geometry.centroid.x.to_numpy(), sel.geometry.centroid.y.to_numpy()])
    tile_ids = sel["tile_id"].astype(str).to_numpy()
    return cKDTree(centers), tile_ids


def _core_mask(shape: tuple, transform, crs, this_tile_id: str, tree: cKDTree, tile_ids: np.ndarray) -> np.ndarray:
    """True where this tile's own center is the NEAREST of every selected
    tile's center - i.e. this tile's Voronoi core. Vectorized: builds every
    pixel's own real-world center coordinate from the raster's affine
    transform directly (no per-pixel rasterio.transform.xy call), reprojects
    the whole grid to the partition CRS in one batch, then one vectorized
    KDTree query for the whole tile.
    """
    h, w = shape
    rows, cols = np.indices((h, w))
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    transformer = Transformer.from_crs(crs, _PARTITION_CRS, always_xy=True)
    px, py = transformer.transform(xs.ravel(), ys.ravel())
    _, idx = tree.query(np.column_stack([px, py]), k=1)
    nearest_tile_id = tile_ids[idx].reshape(h, w)
    return nearest_tile_id == this_tile_id


def compute_result_dedup(
    tile_dir: Path, fsfs: list[float], flooding: dict, river_code: int,
    tree: cKDTree, tile_ids: np.ndarray,
) -> dict:
    """Same as compute_calibration_result.compute_result, except `score`
    (SFINCS-active real land) is additionally intersected with this tile's
    own Voronoi core mask before any counting - see module docstring."""
    this_tile_id = tile_dir.name
    sfincs_dir = tile_dir / "sfincs_model"
    out_dir = tile_dir / "outputs"
    with retry_transient_io(rasterio.open, sfincs_dir / "hmax_subgrid.tif") as src:
        sfincs_depth = src.read(1).astype(np.float32)
        nodata, transform, crs, shape = src.nodata, src.transform, src.crs, src.shape
    if nodata is not None and not np.isnan(nodata):
        sfincs_depth[sfincs_depth == nodata] = np.nan

    mog = np.empty(shape, dtype=np.float32)
    with retry_transient_io(rasterio.open, tile_dir / "inputs" / "mask.tif") as src:
        reproject(source=rasterio.band(src, 1), destination=mog, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest)
    land = mog == LAND_CODE
    active, counts, msk, coarse_tr = _sfincs_active_on_subgrid(sfincs_dir, shape, transform)

    core = _core_mask(shape, transform, crs, this_tile_id, tree, tile_ids)
    score = land & active & core  # <- the one real change vs compute_result()

    cell_km2 = abs(transform.a * transform.e - transform.b * transform.d) / 1e6
    ref = {"score": score, "weight": np.full(shape, cell_km2, dtype=np.float64), "sfincs_depth": sfincs_depth,
           "sfincs_wet": score & np.isfinite(sfincs_depth) & (sfincs_depth > WET_THRESHOLD_M)}

    domain = {"land_km2": float(land.sum() * cell_km2), "sfincs_active_land_km2": float(score.sum() * cell_km2),
              "sfincs_inactive_land_km2": float((land & ~active).sum() * cell_km2),
              "core_fraction_of_active_land": float(score.sum() / max(1, (land & active).sum())), **counts}
    forcing = sfincs_dir / "matched_boundary_points.gpkg"
    domain["n_forcing_points"] = int(len(gpd.read_file(forcing))) if forcing.exists() else None

    result = {
        "tile_id": this_tile_id, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "dedup": True,
        "settings": {"wet_threshold_m": WET_THRESHOLD_M, "subgrid_res_m": float(abs(transform.a)),
                     "friction_scale_factors": list(fsfs), "flooding": flooding},
        "domain": domain,
        "sfincs": {"flooded_km2": float(ref["sfincs_wet"].sum() * cell_km2),
                   "depth": _depth_stats(sfincs_depth, ref["sfincs_wet"])},
        "bathtub": None, "eikonal": {},
    }

    bp = bathtub_path(out_dir)
    if bp.exists():
        depth, _, _, bshape = decode_waterdepth_cm(bp)
        if bshape == shape:
            wet = prune_to_ocean_connected(land & np.isfinite(depth) & (depth > WET_THRESHOLD_M), mog,
                                           ocean_code=OCEAN_CODE, land_code=LAND_CODE, river_code=river_code) & score
            boundaries = gpd.read_file(tile_dir / "inputs" / "boundaries_RP100_SLR_0.gpkg")
            result["bathtub"] = {"level_m": float(boundaries[WATERLEVEL_NAME].max()) / 100.0,
                                 "flooded_km2": float(wet.sum() * cell_km2), "depth": _depth_stats(depth, wet),
                                 "vs_sfincs": _compare(depth, wet, ref)}

    diag_path = out_dir / "eikonal_solver_diagnostics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    for fsf in fsfs:
        p = eikonal_path(out_dir, fsf)
        if not p.exists():
            continue
        depth, _, _, eshape = decode_waterdepth_cm(p)
        if eshape != shape:
            continue
        wet = score & np.isfinite(depth) & (depth > WET_THRESHOLD_M)
        key = f"fsf_{fsf:g}"
        result["eikonal"][key] = {"flooded_km2": float(wet.sum() * cell_km2), "depth": _depth_stats(depth, wet),
                                  "vs_sfincs": _compare(depth, wet, ref), "solver": diag.get(key)}
    return result


def _process_one(args_tuple) -> tuple[str, str | None]:
    """One tile, for the process pool - returns (tile_id, error_or_None)."""
    tile_dir, out_tile_dir, fsfs, flooding, river_code, tree, tile_ids = args_tuple
    tile_id = tile_dir.name
    try:
        result = compute_result_dedup(tile_dir, fsfs, flooding, river_code, tree, tile_ids)
    except Exception as exc:
        return tile_id, f"{type(exc).__name__}: {exc}"
    out_dir = out_tile_dir / "outputs"
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((out_dir / "calibration_result.json").write_text, json.dumps(result, indent=1), encoding="utf-8")
    return tile_id, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--out-dir-name", default="calibration_roughness_SFINCS_dedup")
    parser.add_argument("--tile-id", default=None, help="single tile only (default: every tile in tile_ids.txt)")
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", default=list(FRICTION_SCALE_FACTORS))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    out_base_dir = root / args.out_dir_name
    retry_transient_io(out_base_dir.mkdir, parents=True, exist_ok=True)

    print("Building Voronoi tile-center tree from tile_selection.gpkg...", flush=True)
    tree, tile_ids = build_tile_tree(base_dir / "tile_selection.gpkg")
    print(f"  {len(tile_ids)} selected tile(s) in the partition.", flush=True)

    if args.tile_id:
        tile_list = [args.tile_id]
    else:
        tile_list = (base_dir / "tile_ids.txt").read_text().split()
    print(f"{len(tile_list)} tile(s) to process -> {out_base_dir}", flush=True)

    flooding = config["simulation"]["flooding"]
    river_code = int(config["tile_generation"]["river_code"])

    jobs = [
        (base_dir / tid, out_base_dir / tid, args.friction_scale_factors, flooding, river_code, tree, tile_ids)
        for tid in tile_list
        if (base_dir / tid / "sfincs_model" / "hmax_subgrid.tif").exists()
    ]
    print(f"{len(jobs)} tile(s) have the required source files.", flush=True)

    n_ok = 0
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            tile_id, err = fut.result()
            if err:
                failed.append((tile_id, err))
                print(f"  [FAILED] tile {tile_id}: {err}", flush=True)
            else:
                n_ok += 1
            if i % 50 == 0 or i == len(jobs):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta_min = (len(jobs) - i) / rate / 60 if rate else float("nan")
                print(f"  {i}/{len(jobs)} done ({n_ok} ok, {len(failed)} failed) - "
                      f"{elapsed:.0f}s elapsed, ETA {eta_min:.1f} min", flush=True)

    print(f"\nDone: {n_ok}/{len(jobs)} tile(s) written to {out_base_dir}, {len(failed)} failed.")
    if failed:
        print("Failed tiles:", ", ".join(t for t, _ in failed[:40]) + (" ..." if len(failed) > 40 else ""))


if __name__ == "__main__":
    main()
