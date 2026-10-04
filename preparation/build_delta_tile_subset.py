"""Select the production tile_grid chunks relevant to the 49 global deltas
(inputs/DeltasGlobal/all_deltas.shp) for the RCP2.6/RCP4.5 (ssp126/ssp245)
delta flood-hazard-map deliverable.

Filters the EXISTING global tile grid (tile_grid.path - a frozen-geometry,
scenario-independent output already computed for the whole world, see
run_preparation.py's tile_generation step) down to just the chunks
intersecting ANY delta's bounding box, buffered by a configurable margin
(default 0.5 deg, per direct user confirmation 2026-10) - no tile geometry
is recomputed here, this is a pure selection step over already-correct
chunk geometries (hop_distance/overlap/shave decisions already made),
matching the same restriction pattern used for the Wales/Scotland
(gbr_wales_scotland_friction9_tiles.gpkg) and Thailand
(thailand_bangkok_tiles.gpkg) isolated runs this session.

Buffers each delta's own bounding BOX (not its exact polygon outline) -
matches the user's own framing ("a bbox around each delta with a bit of
buffer"), simpler and more predictable than buffering an often-irregular
polygon boundary.

Standalone - not wired into run_preparation.py's ALL_STEPS (this is
specific to the deltas deliverable, not a general pipeline stage). Shared,
scenario-independent output: both the ssp126 and ssp245 materialized
configs point `tile_grid.path` at the SAME file this script writes.

Usage:
    python preparation/build_delta_tile_subset.py
    python preparation/build_delta_tile_subset.py --buffer-deg 0.25
"""

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DELTAS_PATH = Path(r"P:\11212688-004-global-floodmaps\modelling\inputs\DeltasGlobal\all_deltas.shp")
DEFAULT_BUFFER_DEG = 0.5


def build_delta_tile_subset(
    config_path: str | Path, buffer_deg: float = DEFAULT_BUFFER_DEG, out_path: Path | None = None,
) -> gpd.GeoDataFrame:
    cfg = load_config(config_path)
    tile_grid_path = Path(cfg["tile_grid"]["path"])

    deltas = retry_transient_io(gpd.read_file, DELTAS_PATH)
    tile_grid = retry_transient_io(gpd.read_file, tile_grid_path)
    if tile_grid.crs != deltas.crs:
        tile_grid = tile_grid.to_crs(deltas.crs)

    buffered_boxes = []
    for geom in deltas.geometry:
        minx, miny, maxx, maxy = geom.bounds
        buffered_boxes.append(box(minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg))
    deltas_buffered = gpd.GeoDataFrame({"delta_id": deltas["Id"]}, geometry=buffered_boxes, crs=deltas.crs)

    joined = gpd.sjoin(tile_grid, deltas_buffered, how="inner", predicate="intersects")
    selected_ids = sorted(joined["tile_id"].unique())
    selected = tile_grid[tile_grid["tile_id"].isin(selected_ids)].reset_index(drop=True)

    matched_delta_ids = set(joined["delta_id"].unique())
    missing = sorted(set(deltas["Id"]) - matched_delta_ids)
    print(f"{len(deltas)} deltas, buffer={buffer_deg} deg")
    print(f"{len(selected)} tile(s) selected out of {len(tile_grid)} global tiles")
    if missing:
        print(f"WARNING: {len(missing)} delta(s) matched ZERO tiles: {missing}")
    else:
        print("Every delta matched at least one tile.")

    if out_path is not None:
        retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
        retry_transient_io(selected.to_file, out_path, driver="GPKG")
        print(f"Wrote {out_path}")

    return selected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--buffer-deg", type=float, default=DEFAULT_BUFFER_DEG)
    parser.add_argument("--out", default=None, help="defaults to <tile_grid.path's dir>/deltas_global_tiles.gpkg")
    args = parser.parse_args()

    cfg = load_config(args.config)
    default_out = Path(cfg["tile_grid"]["path"]).parent / "deltas_global_tiles.gpkg"
    out_path = Path(args.out) if args.out else default_out

    build_delta_tile_subset(args.config, buffer_deg=args.buffer_deg, out_path=out_path)
