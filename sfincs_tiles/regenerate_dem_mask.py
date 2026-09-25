"""Regenerate dem.tif and mask.tif for one tile using the current production
DEM-extraction logic (src/rasters.py's extract_dem/extract_dem_mask),
instead of trusting whatever copy exists in model_outputs/{tile_id}/inputs/,
which may have been built by an older version of that logic.

extract_dem/extract_dem_mask are plain functions with no Snakemake coupling
(only the thin wrapper scripts snakemake_workflow/scripts/extract_dem.py /
extract_dem_mask.py use the snakemake.input/output indirection) - this
calls them directly, using model_bbox.json (already copied into this
tile's own inputs/ by run_one_tile_v2.sh) for the tile's bbox, and the same
data catalog / dem_gap_fill config production itself uses. Overwrites the
copied dem.tif/mask.tif in place with what today's Snakemake rules would
produce for the exact same tile.

Needs the "gfm" env (config_utils.py needs hydromt 0.9.3, same constraint
as run_eikonal_on_sfincs_subgrid.py) - NOT hydromt-sfincs-dev.

Usage:
    python regenerate_dem_mask.py --tile-id 1454 --base-dir-name validation_sfincs_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from rasters import extract_dem, extract_dem_mask, save_raster  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retry_io import retry_transient_io  # noqa: E402


def regenerate_dem_mask(tile_id: str, root: Path, base_dir_name: str, config: dict) -> None:
    tile_inputs_dir = root / base_dir_name / tile_id / "inputs"
    with open(tile_inputs_dir / "model_bbox.json") as f:
        bbox = json.load(f)

    # hydromt_data_catalog is relative to the repo root (not auto-expanded by
    # load_config - only literal {root}/{code_root} placeholders are), same
    # join pattern as build_tile_manifest.py/prepare_boundary_conditions.py.
    repo_root = Path(__file__).resolve().parent.parent
    catalog_path = repo_root / config["paths"]["hydromt_data_catalog"]
    data_catalog = get_data_catalog(catalog_path, root=root)

    gap_fill_cfg = config["simulation"]["dem_gap_fill"]
    geoid_offset_raster = Path(config["vertical_datum_correction"]["offset_raster_path"])

    dem = retry_transient_io(
        extract_dem, data_catalog, "deltadtm", bbox, mask_source="deltadtm_mask",
        geoid_offset_raster=geoid_offset_raster,
        min_hard_fill_component_size=gap_fill_cfg["min_hard_fill_component_size"],
        interp_max_search_distance=gap_fill_cfg["interp_max_search_distance"],
        interp_smoothing_iterations=gap_fill_cfg["interp_smoothing_iterations"],
        land_fill_value_m=gap_fill_cfg["land_fill_value_m"],
        gebco_source="gebco",
    )
    mask = retry_transient_io(
        extract_dem_mask, data_catalog, "deltadtm_mask", list(dem.raster.bounds), dem,
        gebco_source="gebco",
    )

    raster_config = config["raster_format"]
    save_raster(dem, tile_inputs_dir / "dem.tif", raster_config, dtype="int16")
    save_raster(mask, tile_inputs_dir / "mask.tif", raster_config)
    print(f"Regenerated {tile_inputs_dir / 'dem.tif'} and {tile_inputs_dir / 'mask.tif'} "
          f"using current extract_dem/extract_dem_mask logic")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    root = Path(config["paths"]["root"])
    regenerate_dem_mask(args.tile_id, root, args.base_dir_name, config)


if __name__ == "__main__":
    main()
