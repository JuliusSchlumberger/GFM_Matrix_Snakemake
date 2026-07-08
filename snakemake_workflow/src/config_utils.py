"""Shared config-parsing and data-catalog helpers used across the workflow."""

from pathlib import Path

import hydromt
from hydromt.log import setuplog


def get_data_catalog(catalog_path: str | Path, logger_name: str = "snakemake_workflow") -> hydromt.DataCatalog:
    """Return a HydroMT DataCatalog loaded from `catalog_path`.

    `catalog_path` comes from the caller — a rule's `params.data_catalog`
    (sourced from `config["paths"]["hydromt_data_catalog"]`) for
    Snakemake-driven scripts, or a path read directly from config.yml by a
    standalone script. Nothing is hardcoded here.
    """
    logger = setuplog(logger_name, log_level=10)
    return hydromt.DataCatalog(data_libs=str(catalog_path), logger=logger)


def merged_slr_scenarios(bc_cfg: dict, adapt_cfg: dict) -> list[str]:
    """Union of boundary_conditions.slr_scenarios and adaptation.slr_intensities.

    Deduplicated and sorted by numeric SLR value (ascending) — e.g.
    ``["SLR_0", "SLR_200", "SLR_250", "SLR_400", "SLR_500", ...]``.

    The sort matters beyond cosmetics: several callers pass the parallel mm
    values positionally into ``np.interp``/``pchip_interpolate`` to
    interpolate EAI across SLR levels, and both require a strictly
    increasing x-sequence — ``np.interp`` silently returns nonsense (per its
    own docs) and ``pchip_interpolate`` raises on unsorted input. A plain
    ``list(dict.fromkeys(slr_scenarios + slr_intensities))`` does not sort,
    since ``adaptation.slr_intensities`` (e.g. SLR_250, SLR_500) are declared
    separately from ``boundary_conditions.slr_scenarios`` and get appended
    after it.
    """
    names = list(dict.fromkeys(bc_cfg["slr_scenarios"] + adapt_cfg.get("slr_intensities", [])))
    return sorted(names, key=lambda s: int(s.split("_")[1]))
