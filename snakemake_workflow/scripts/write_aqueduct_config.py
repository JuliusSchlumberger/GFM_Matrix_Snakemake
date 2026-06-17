"""Write the Aqueduct TOML configuration for a single tile and SLR scenario."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqueduct_config import build_aqueduct_config, write_aqueduct_config  # noqa: E402

waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821

config = build_aqueduct_config(
    dem_filename=Path(snakemake.input.dem).name,  # noqa: F821
    mask_filename=Path(snakemake.input.mask).name,  # noqa: F821
    friction_filename=Path(snakemake.input.friction).name,  # noqa: F821
    boundaries_filename=Path(snakemake.input.boundaries).name,  # noqa: F821
    waterdepth_filename=f"waterdepth_{waterlevel_name}.tif",
    waterlevel_name=waterlevel_name,
    flooding_config=snakemake.config["simulation"]["flooding"],  # noqa: F821
)

write_aqueduct_config(config, snakemake.output.toml)  # noqa: F821
