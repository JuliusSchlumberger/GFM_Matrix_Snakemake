"""Filter the tile_grid to tile_ids listed one per line in a text file.

Usage:
    python preparation/build_tile_subset_by_id.py <tile_ids_file> <out_path>
"""

import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def build_tile_subset_by_id(tile_ids_file: Path, out_path: Path, config_path: str | Path | None = None) -> gpd.GeoDataFrame:
    """Write and return the tile_grid rows whose tile_id is in tile_ids_file."""
    config_path = config_path or REPO / "snakemake_workflow" / "config" / "config.yml"
    cfg = load_config(config_path)
    tile_grid_path = Path(cfg["tile_grid"]["path"])

    tile_ids = [int(line.strip()) for line in tile_ids_file.read_text().splitlines() if line.strip()]
    tile_grid = retry_transient_io(gpd.read_file, tile_grid_path)
    selected = tile_grid[tile_grid["tile_id"].isin(tile_ids)].reset_index(drop=True)

    missing = sorted(set(tile_ids) - set(selected["tile_id"]))
    print(f"{len(tile_ids)} tile_id(s) requested, {len(selected)} found in {tile_grid_path}")
    if missing:
        print(f"WARNING: {len(missing)} tile_id(s) not found: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
    retry_transient_io(selected.to_file, out_path, driver="GPKG")
    print(f"Wrote {out_path}")
    return selected


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tile_ids_file")
    parser.add_argument("out_path")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    build_tile_subset_by_id(Path(args.tile_ids_file), Path(args.out_path), args.config)
