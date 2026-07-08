"""Resolve and cache the population raster and per-pixel geogunit IDs (on population's grid), once per spatial chunk."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import get_data_catalog  # noqa: E402
from exposure import prepare_exposure_grid_chunk  # noqa: E402

data_catalog = get_data_catalog(snakemake.params.data_catalog)  # noqa: F821

prepare_exposure_grid_chunk(
    reference_path=snakemake.input.reference,  # noqa: F821
    data_catalog=data_catalog,
    population_source=snakemake.params.population_source,  # noqa: F821
    geogunit_source=snakemake.params.geogunit_source,  # noqa: F821
    population_output_path=snakemake.output.population,  # noqa: F821
    geogunit_output_path=snakemake.output.geogunit,  # noqa: F821
)
