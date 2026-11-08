"""Merge england_local_input's fine tiles into one native-CRS waterdepth raster per scenario.

The tiles' native grid is England's own LIDAR source CRS (EPSG:27700), not the
global EPSG:4326 chunk grid merge_chunk assumes - merge_tile_rasters_native keeps
them at full native resolution rather than reprojecting (TODO.md). Not
Snakemake-tracked: rerun manually after re-running the eikonal solve.

Usage:
    python merge_england_local_tiles.py --config <england_local_input_materialized.yml> \
        --return-periods RP100 RP250 --waterlevel-name SLR_0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402
from merge import merge_tile_rasters_native  # noqa: E402


def run(config: dict, return_periods: list[str], waterlevel_name: str) -> list[Path]:
    model_outputs = Path(config["simulation"]["model_outputs"])
    tile_ids = sorted(int(p.name) for p in model_outputs.iterdir() if p.name.isdigit())
    block_size = config["postprocessing"]["block_size"]
    raster_config = config["raster_format"]

    out_dir = Path(config["postprocessing"]["merged_outputs"]) / "local_native"
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    written = []
    for rp in return_periods:
        tile_rasters = [
            model_outputs / str(tid) / "results" / f"waterdepth_{rp}_{waterlevel_name}.tif"
            for tid in tile_ids
        ]
        missing = [p for p in tile_rasters if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{rp}: missing waterdepth output(s): {missing}")

        wd_path = out_dir / f"waterdepth_{rp}_{waterlevel_name}.tif"
        prov_path = out_dir / f"provenance_{rp}_{waterlevel_name}.tif"
        merge_tile_rasters_native(tile_rasters, wd_path, prov_path, block_size, raster_config)
        print(f"{rp}: merged {len(tile_rasters)} tiles -> {wd_path}")
        written.append(wd_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--return-periods", nargs="+", default=["RP100", "RP250"])
    parser.add_argument("--waterlevel-name", default="SLR_0")
    args = parser.parse_args()
    run(load_config(args.config), args.return_periods, args.waterlevel_name)


if __name__ == "__main__":
    main()
