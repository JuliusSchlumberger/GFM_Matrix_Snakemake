"""Select global tile_grid tiles intersecting the 49 deltas' buffered bboxes.

For the ssp126/ssp245 delta deliverable (all_deltas.shp); both scenario configs
share the output. Pure selection, no tile geometry recomputed. Buffers each
delta's bbox (degrees), not its polygon.

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
DEFAULT_BUFFER_DEG = 0.5


def build_delta_tile_subset(
    config_path: str | Path, buffer_deg: float = DEFAULT_BUFFER_DEG, out_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Return (and optionally write) tiles intersecting any buffered delta bbox."""
    cfg = load_config(config_path)
    tile_grid_path = Path(cfg["tile_grid"]["path"])
    deltas_path = Path(cfg["paths"]["deltas_shapefile"])

    deltas = retry_transient_io(gpd.read_file, deltas_path)
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
