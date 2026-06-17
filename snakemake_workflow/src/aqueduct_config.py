"""Functions for writing per-tile, per-scenario Aqueduct TOML configuration files."""

from pathlib import Path
from typing import Any

import toml


def build_aqueduct_config(
    dem_filename: str,
    mask_filename: str,
    friction_filename: str,
    boundaries_filename: str,
    waterdepth_filename: str,
    waterlevel_name: str,
    flooding_config: dict[str, Any],
    results_dir: str = "../results",
) -> dict[str, Any]:
    """Build the dictionary structure for an Aqueduct TOML configuration.

    The resulting TOML is written to `model_outputs/{tile_id}/inputs/`. All
    `input` filenames are resolved relative to that directory (`input_dir = "."`),
    while `results_dir` points to `model_outputs/{tile_id}/results/` so that
    `output.waterdepth` is written there (see `core/src/config.jl`).

    Args:
        dem_filename: Filename of the tile's DEM raster (in `inputs/`).
        mask_filename: Filename of the tile's DEM-validity mask raster (in `inputs/`).
        friction_filename: Filename of the tile's friction raster (in `inputs/`).
        boundaries_filename: Filename of the tile's boundary points GeoPackage (in `inputs/`).
        waterdepth_filename: Filename of the simulation output raster (in `results/`),
            named so it can be linked back to `waterlevel_name`.
        waterlevel_name: Name of the sea level rise scenario, used as the
            `waterlevels.name` value and matching the corresponding column
            in `boundaries_filename`.
        flooding_config: The workflow's `flooding` configuration section, with
            keys `resolution`, `knn` and `debug`.
        results_dir: Path to the results directory, relative to `inputs/`.

    Returns:
        A dictionary matching the structure expected by `core/src/config.jl`.
    """
    return {
        "input_dir": ".",
        "results_dir": results_dir,
        "input": {
            "dem": dem_filename,
            "mask": mask_filename,
            "friction": friction_filename,
            "boundaries": boundaries_filename,
        },
        "output": {
            "waterdepth": waterdepth_filename,
        },
        "flooding": {
            "resolution": flooding_config["resolution"],
            "debug": flooding_config["debug"],
        },
        "waterlevels": {
            "knn": flooding_config["knn"],
            "name": waterlevel_name,
        },
    }


def write_aqueduct_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Write an Aqueduct configuration dictionary to a TOML file."""
    with open(output_path, "w") as f:
        toml.dump(config, f)
