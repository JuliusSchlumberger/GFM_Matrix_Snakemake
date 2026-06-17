"""Shared helpers for accessing the HydroMT data catalogs used by the workflow."""

from pathlib import Path

import hydromt
from hydromt.log import setuplog

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Catalog for the main Snakemake workflow (DEM, land use, coastlines, …)
_CATALOG = _CONFIG_DIR / "data_catalog_gfm.yml"

# Catalog for standalone preprocessing scripts (MDT, SLR, COAST-RP, …)
# Loaded on top of _CATALOG so that deltadtm / land_use etc. are also available
# in select_tiles.py, merge_tiles.py and prepare_boundary_conditions.py without
# duplicating entries across both files.
_PREPROCESSING_CATALOG = _CONFIG_DIR / "preprocessing_data.yml"


def get_data_catalog(logger_name: str = "snakemake_workflow") -> hydromt.DataCatalog:
    """Return a HydroMT DataCatalog loaded from config/data_catalog_gfm.yml.

    Used by the Snakemake preprocessing rules (extract_dem, compute_friction, …).
    The catalog path is resolved relative to this file so the function works
    regardless of where Snakemake is invoked from.
    """
    logger = setuplog(logger_name, log_level=10)
    return hydromt.DataCatalog(data_libs=str(_CATALOG), logger=logger)


def get_preprocessing_catalog(logger_name: str = "snakemake_workflow") -> hydromt.DataCatalog:
    """Return a HydroMT DataCatalog combining both catalog files.

    Loads data_catalog_gfm.yml first, then preprocessing_data.yml on top,
    giving access to all datasets needed by the standalone preprocessing
    scripts (select_tiles.py, merge_tiles.py, prepare_boundary_conditions.py):
      - deltadtm, land_use, … from data_catalog_gfm.yml
      - coast_rp, mdt_hybrid_cnes_cls22_cmems2020,
        ipcc_ar6_slr_projections, … from preprocessing_data.yml
    """
    logger = setuplog(logger_name, log_level=10)
    return hydromt.DataCatalog(
        data_libs=[str(_CATALOG), str(_PREPROCESSING_CATALOG)],
        logger=logger,
    )
