"""Per-delta flooded-extent CSV across both delta scenarios (ssp126/ssp245)
and all 5 SLR magnitudes, RP100 - flooded area in km2 AND as a percentage of
each delta's own REAL outline area (not its bounding box).

Deltas come from `inputs/DeltasGlobal/all_deltas.shp` (EPSG:4326, 49
polygons, fields `Id`/`Area` only - same file
`preparation/build_delta_tile_subset.py` already uses to select tiles, same
hardcoded path convention). Delta `Id=44` (Volga) is dropped - confirmed
zero tile coverage in both scenarios.

Tile<->delta attribution is NOT persisted anywhere upstream (the tile
selection step's own sjoin, against each delta's buffered BBOX for
simulation-coverage purposes, is never saved) - this script redoes that
join itself, against each delta's REAL polygon, via the shared
`deltas_global_tiles.gpkg` tile grid both scenarios' materialized configs
point at.

Production tiles genuinely OVERLAP each other (confirmed directly: tile
grid core polygons for tiles covering the Amazon delta sum to 1.65x their
own union area) - a first version of this script naively summed each
covering tile's own contribution independently and got 120-380% "coverage"
for every delta as a result. Fixed by applying the SAME per-cell MAXIMUM
combine rule `src.merge.merge_tile_rasters_chunk` uses in production (see
that function's own docstring for why max, not sum/mean) - each delta's
covering tiles are merged onto one grid (chunk_bounds = the delta's own
bbox) BEFORE any area/flood accounting happens, so every real cell is
counted exactly once regardless of how many tiles' footprints reach it.

Merging happens IN MEMORY, not via that production function directly - a
single delta's own bbox comfortably fits in memory (confirmed: even the
largest, Amazon, is ~287M pixels at 30m resolution, ~1.1GB as float64),
unlike the full 5deg production postprocessing chunks that function is
built to stream to disk in bounded-memory blocks. Skipping the
write-compressed-GeoTIFF-then-reread round trip that function would
otherwise impose per (delta, SLR) was the single biggest speedup found
live on this run (see `merge_in_memory`/`_tile_geometry` below) - this is
NOT a safe change to make in `src/merge.py` itself, since a dense,
continent-spanning production chunk genuinely can exceed available memory
if loaded whole; it's only safe here because a delta's bbox is reliably
small. Each tile's own window-intersection geometry (which output pixels
it covers) is also computed ONCE per delta and reused across all 5 SLR
magnitudes, since every SLR raster for the same tile shares an identical
pixel grid (production's own guarantee - see src/merge.py's module
docstring: "all tiles share the same origin and pixel size").

Flooded extent uses depth > 0 (any real flood depth, matching production's
own binary flood classification in `flood_model.py` and
`validation.primary_threshold_m`'s own "any real depth" reasoning) - NOT the
0.1m exposure threshold, which is a separate, not-yet-implemented
extension (see this script's own module docstring in the project plan: the
exposure pipeline needs a full merge/chunk prerequisite run first).

`delta_name_inferred`: best-effort names matched by hand against each
delta's real centroid during planning (cross-checked against the one
already-known case - Id=44 is confirmed Volga - as a sanity check on the
method) - NOT from any authoritative source in this repo (none exists).
Treat as a convenience label, never a join key; `delta_id` is authoritative.

`pct_flooded_of_modeled_area` is computed against `delta_area_km2_modeled`
(the portion of the delta's real outline actually covered by tiles THIS
scenario modeled), not the shapefile's full `Area` - a delta whose outline
extends beyond the simulated tile footprint would otherwise read an
artificially low percentage. `pct_delta_covered_by_model` makes that gap
visible per row.

Usage:
    python compute_delta_flood_extent.py
    python compute_delta_flood_extent.py --scenarios ssp126
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402
from merge import AQUEDUCT_NODATA, _make_chunk_transform, _open_overlapping_tiles, _tile_id_from_path  # noqa: E402
from plotting import pixel_area_km2_grid  # noqa: E402
from validation import tri_domain_mask  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
DELTAS_PATH = Path(r"P:\11212688-004-global-floodmaps\modelling\inputs\DeltasGlobal\all_deltas.shp")
RETURN_PERIOD = "RP100"
SLR_SCENARIOS = ["SLR_0", "SLR_500", "SLR_1000", "SLR_1500", "SLR_2000"]
SCENARIOS_DEFAULT = ["ssp126", "ssp245"]
VOLGA_ID = 44  # confirmed zero tile coverage in both scenarios - excluded

# Best-effort Id -> name, matched by centroid location during planning - see
# module docstring. NOT authoritative; cross-checked against Id=44 (Volga,
# independently confirmed via deltas_ssp126.yml's own comment) as the one
# available sanity check on the method.
DELTA_NAME_INFERRED = {
    1: "Amazon", 2: "Amur", 3: "Fitzroy (Australia)", 4: "Colorado (Mexico/US)",
    5: "Congo", 6: "Chao Phraya", 7: "Danube", 8: "Dnieper", 9: "Ebro",
    10: "Fly", 11: "Ganges-Brahmaputra", 12: "Godavari", 13: "Grijalva-Usumacinta",
    14: "Han", 15: "Indus", 16: "Irrawaddy", 17: "Krishna", 18: "Lena",
    19: "Limpopo", 20: "Mackenzie", 21: "Magdalena", 22: "Mahakam", 23: "Mahanadi",
    24: "Mekong", 25: "Mississippi", 26: "Moulouya", 27: "Niger", 28: "Nile",
    29: "Orinoco", 30: "Parana-Rio de la Plata", 31: "Pearl", 32: "Po",
    33: "Red River", 34: "Rio Grande", 35: "Rhine-Meuse", 36: "Rhone",
    37: "Sebou", 38: "Senegal", 39: "Sao Francisco", 40: "Tana", 41: "Shatt al-Arab",
    42: "Tone", 43: "Vistula", 44: "Volga", 45: "Volta", 46: "Yangtze",
    47: "Yellow River", 48: "Yukon-Kuskokwim", 49: "Zambezi",
}


def build_tile_delta_join(tile_grid: gpd.GeoDataFrame, deltas: gpd.GeoDataFrame) -> dict[int, list[int]]:
    """{delta Id: [tile_id, ...]} - tiles whose geometry actually intersects
    the delta's REAL polygon (not the wider simulation-selection buffer)."""
    joined = gpd.sjoin(tile_grid[["tile_id", "geometry"]], deltas[["Id", "geometry"]],
                        how="inner", predicate="intersects")
    return joined.groupby("Id")["tile_id"].apply(list).to_dict()


def _tile_geometry(tile_rasters: list[Path], bbox: tuple[float, float, float, float]):
    """Computes the shared output grid (transform/width/height) and each
    surviving tile's own window-intersection geometry ONCE, from a
    reference set of tile rasters (SLR_0) - reused across every other SLR
    magnitude for this delta (see module docstring for why that's safe).
    Returns (out_transform, out_w, out_h, [(tile_id, row_off, col_off,
    n_rows, n_cols, src_r0, src_c0), ...]) - a tile whose footprint doesn't
    actually reach the delta's bbox is silently dropped here (same
    `_open_overlapping_tiles` behaviour production's own merge relies on).
    """
    with rasterio.open(tile_rasters[0]) as ref:
        ref_transform = ref.transform
    out_transform, out_w, out_h = _make_chunk_transform(bbox, ref_transform)
    metas = _open_overlapping_tiles(tile_rasters, out_transform, out_w, out_h)
    geometry = [
        (_tile_id_from_path(tm.path), tm.row_off, tm.col_off, tm.n_rows, tm.n_cols, tm.src_r0, tm.src_c0)
        for tm in metas
    ]
    for tm in metas:
        tm.src.close()
    return out_transform, out_w, out_h, geometry


def merge_in_memory(model_outputs: Path, slr: str, out_w: int, out_h: int, geometry: list[tuple]) -> np.ndarray | None:
    """In-memory equivalent of merge.merge_tile_rasters_chunk's per-cell
    MAXIMUM combine (same production rule - see module docstring and that
    function's own docstring for why max, not sum/mean) - skips the
    GeoTIFF write+re-read round trip entirely. NaN wherever no tile's
    footprint reaches a cell. Returns None if no tile actually has this
    SLR's output on disk (should only happen for a genuinely incomplete run)."""
    best = np.full((out_h, out_w), -np.inf, dtype="float64")
    covered = np.zeros((out_h, out_w), dtype=bool)
    any_found = False
    for tile_id, row_off, col_off, n_rows, n_cols, src_r0, src_c0 in geometry:
        path = model_outputs / str(tile_id) / "results" / f"waterdepth_{RETURN_PERIOD}_{slr}.tif"
        if not path.exists():
            print(f"    WARNING: {path} missing (had SLR_0 but not {slr}) - excluded from this merge", flush=True)
            continue
        any_found = True
        with rasterio.open(path) as src:
            patch = src.read(1, window=Window(src_c0, src_r0, n_cols, n_rows)).astype("float64")
        valid = patch < AQUEDUCT_NODATA
        sl = (slice(row_off, row_off + n_rows), slice(col_off, col_off + n_cols))
        sub_best = best[sl]
        wins = valid & (patch > sub_best)
        sub_best[wins] = patch[wins]
        best[sl] = sub_best
        covered[sl] |= valid
    if not any_found:
        return None
    return np.where(covered, best, np.nan).astype("float32")


def score_delta(delta_row, candidate_tiles: list[int], model_outputs: Path) -> dict:
    """Per-(scenario, delta): merges every covering tile's own waterdepth
    raster onto one common grid (chunk_bounds = the delta's own bbox),
    in memory, once per SLR magnitude - each real cell is counted exactly
    once regardless of how many tiles' footprints reach it.
    `n_tiles`/`delta_area_km2_modeled` come from the SLR_0 merge (the same
    set of tiles, hence the same footprint, covers every SLR magnitude)."""
    delta_id = int(delta_row["Id"])
    delta_gdf = gpd.GeoDataFrame({"Id": [delta_id]}, geometry=[delta_row.geometry], crs="EPSG:4326")
    bbox = tuple(delta_row.geometry.bounds)

    slr0_rasters = [
        model_outputs / str(tid) / "results" / f"waterdepth_{RETURN_PERIOD}_SLR_0.tif"
        for tid in candidate_tiles
    ]
    slr0_rasters = [p for p in slr0_rasters if p.exists()]
    if not slr0_rasters:
        zero = {slr: 0.0 for slr in SLR_SCENARIOS}
        return {"n_tiles": 0, "delta_area_km2_modeled": 0.0,
                "delta_area_km2_shapefile": float(delta_row["Area"]),
                "pct_delta_covered_by_model": 0.0, "flooded_km2_by_slr": zero}

    out_transform, out_w, out_h, geometry = _tile_geometry(slr0_rasters, bbox)

    per_slr_flooded_km2: dict[str, float] = {}
    n_tiles = len(geometry)
    modeled_km2 = 0.0
    domain = None
    area_grid = None
    for slr in SLR_SCENARIOS:
        depth = merge_in_memory(model_outputs, slr, out_w, out_h, geometry)
        if depth is None:
            per_slr_flooded_km2[slr] = 0.0
            continue

        if domain is None:  # SLR-independent (same footprint/grid every SLR) - build once
            # NOT model_domain_mask (depth != nodata) - merge_in_memory's own nodata
            # sentinel is NaN, and NaN != NaN is always True in numpy, so an equality
            # check would silently treat every uncovered cell as "covered".
            covered = ~np.isnan(depth)
            domain = tri_domain_mask(delta_gdf, out_transform, "EPSG:4326", (out_h, out_w)) & covered
            area_grid = pixel_area_km2_grid(out_transform, out_w, out_h)
            modeled_km2 = float((domain * area_grid).sum())

        wet = (depth > 0) & domain  # depth>0 is already False wherever depth is NaN - no separate NaN guard needed
        per_slr_flooded_km2[slr] = float((wet * area_grid).sum())

    shapefile_km2 = float(delta_row["Area"])
    pct_covered = 100.0 * modeled_km2 / shapefile_km2 if shapefile_km2 > 0 else float("nan")

    return {
        "n_tiles": n_tiles,
        "delta_area_km2_modeled": modeled_km2,
        "delta_area_km2_shapefile": shapefile_km2,
        "pct_delta_covered_by_model": pct_covered,
        "flooded_km2_by_slr": per_slr_flooded_km2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", default=str(_REPO_ROOT / "snakemake_workflow" / "config"))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS_DEFAULT)
    parser.add_argument("--out", default=None, help="default: {root}/deltas_floodmaps/delta_flood_extent.csv")
    args = parser.parse_args()

    deltas = retry_transient_io(gpd.read_file, DELTAS_PATH)
    deltas = deltas[deltas["Id"] != VOLGA_ID].reset_index(drop=True)
    centroids = deltas.geometry.centroid
    deltas["centroid_lon"] = centroids.x
    deltas["centroid_lat"] = centroids.y
    print(f"{len(deltas)} delta(s) (Id={VOLGA_ID} Volga excluded - zero tile coverage)")

    first_cfg = load_config(Path(args.config_dir) / f"deltas_{args.scenarios[0]}_materialized.yml")
    tile_grid = retry_transient_io(gpd.read_file, Path(first_cfg["tile_grid"]["path"]))
    if tile_grid.crs != deltas.crs:
        tile_grid = tile_grid.to_crs(deltas.crs)
    tiles_by_delta = build_tile_delta_join(tile_grid, deltas)
    n_unmatched = sum(1 for _, row in deltas.iterrows() if int(row["Id"]) not in tiles_by_delta)
    if n_unmatched:
        print(f"WARNING: {n_unmatched} delta(s) matched zero tiles in the shared tile grid")

    root = Path(first_cfg["paths"]["root"])
    out_path = Path(args.out) if args.out else root / "deltas_floodmaps" / "delta_flood_extent.csv"

    rows = []
    t_start = time.time()
    n_deltas = len(deltas)
    for scenario in args.scenarios:
        cfg = load_config(Path(args.config_dir) / f"deltas_{scenario}_materialized.yml")
        model_outputs = Path(cfg["simulation"]["model_outputs"])
        print(f"\n=== {scenario} (model_outputs={model_outputs}) ===", flush=True)

        for i, (_, delta_row) in enumerate(deltas.iterrows(), start=1):
            delta_id = int(delta_row["Id"])
            delta_name = DELTA_NAME_INFERRED.get(delta_id, f"delta_{delta_id}")
            candidate_tiles = tiles_by_delta.get(delta_id, [])
            print(f"  [{i}/{n_deltas}] delta {delta_id:2d} ({delta_name}): "
                  f"{len(candidate_tiles)} candidate tile(s) - merging...", flush=True)
            t0 = time.time()
            result = score_delta(delta_row, candidate_tiles, model_outputs)
            elapsed = time.time() - t0
            modeled_km2 = result["delta_area_km2_modeled"]

            print(f"    -> {result['n_tiles']} tile(s), "
                  f"modeled {modeled_km2:.1f}/{result['delta_area_km2_shapefile']:.1f} km2 "
                  f"({result['pct_delta_covered_by_model']:.1f}% covered) - {elapsed:.1f}s "
                  f"[{(time.time() - t_start) / 60:.1f}min elapsed]", flush=True)

            for slr in SLR_SCENARIOS:
                flooded_km2 = result["flooded_km2_by_slr"][slr]
                pct_flooded = 100.0 * flooded_km2 / modeled_km2 if modeled_km2 > 0 else float("nan")
                rows.append({
                    "delta_id": delta_id,
                    "delta_name_inferred": delta_name,
                    "centroid_lon": round(float(delta_row["centroid_lon"]), 4),
                    "centroid_lat": round(float(delta_row["centroid_lat"]), 4),
                    "ssp_scenario": scenario,
                    "slr_scenario": slr,
                    "return_period": RETURN_PERIOD,
                    "n_tiles": result["n_tiles"],
                    "flooded_area_km2": round(flooded_km2, 4),
                    "delta_area_km2_modeled": round(modeled_km2, 4),
                    "delta_area_km2_shapefile": round(result["delta_area_km2_shapefile"], 4),
                    "pct_delta_covered_by_model": round(result["pct_delta_covered_by_model"], 2),
                    "pct_flooded_of_modeled_area": round(pct_flooded, 4),
                })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retry_transient_io(df.to_csv, out_path, index=False)
    print(f"\nWrote {len(df)} row(s) to {out_path}")

    low_coverage = df[["delta_id", "delta_name_inferred", "ssp_scenario", "pct_delta_covered_by_model"]].drop_duplicates()
    low_coverage = low_coverage[low_coverage["pct_delta_covered_by_model"] < 90]
    if not low_coverage.empty:
        print(f"\nWARNING: {len(low_coverage)} (delta, scenario) pair(s) below 90% modeled coverage:")
        print(low_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
