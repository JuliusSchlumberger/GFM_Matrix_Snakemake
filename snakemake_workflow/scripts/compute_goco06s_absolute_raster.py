"""Compute and cache GOCO06s's own absolute geoid-height raster (one-time)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vertical_datum import write_single_geoid_raster  # noqa: E402

write_single_geoid_raster(
    gfc_path=snakemake.input.goco06s_gfc,  # noqa: F821
    out_path=snakemake.output.absolute_raster,  # noqa: F821
)
