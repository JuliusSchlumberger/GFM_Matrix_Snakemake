"""Compute and cache the global EGM2008 -> GOCO06s geoid-offset raster (one-time)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vertical_datum import write_geoid_offset_raster  # noqa: E402

write_geoid_offset_raster(
    goco06s_gfc=snakemake.input.goco06s_gfc,  # noqa: F821
    egm2008_gfc=snakemake.input.egm2008_gfc,  # noqa: F821
    out_path=snakemake.output.offset_raster,  # noqa: F821
)
