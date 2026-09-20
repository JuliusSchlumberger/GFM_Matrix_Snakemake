"""Minimal, hydromt-version-independent config/catalog helpers for
sfincs_tiles/'s own scripts.

Deliberately does NOT use src/config_utils.py (get_data_catalog/load_config)
- confirmed (2026-09, live test) that importing config_utils at all fails
under hydromt-sfincs-dev's own hydromt 1.4.1 (`ImportError: cannot import
name 'setuplog' from 'hydromt.log'` - removed in hydromt's 1.x rewrite;
config_utils.py was written against/tested with the main pipeline's hydromt
0.9.3). This is the reason every sfincs_tiles/ script can run entirely
under ONE environment (hydromt-sfincs-dev) instead of needing to shell out
to a second (gfm_python_preprocessing) env via `conda run` - these two tiny
functions are all sfincs_tiles/ actually needs from config_utils.py's own
much larger surface (root-path resolution, and a couple of static catalog
path lookups - never any real hydromt DataCatalog machinery: driver
parsing, reprojection, filtering, etc.).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def read_root(config_path: Path) -> Path:
    """paths.root, honouring a sibling config_local.yml override - same
    behaviour as config_utils.load_config, just without needing hydromt at
    all. The base config.yml's own paths.root is a dev-machine placeholder
    ("D:/GFM") - the real, machine-local value only exists in
    config_local.yml, which every other consumer of this config (the
    Snakefile, config_utils.load_config) already auto-merges on top.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = cfg["paths"]["root"]

    local_path = config_path.parent / "config_local.yml"
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
        local_root = (local_cfg.get("paths") or {}).get("root")
        if local_root:
            root = local_root

    return Path(root)


def resolve_catalog_path(catalog_path: Path, root: Path, source_name: str) -> Path:
    """The real, on-disk path for one data_catalog_gfm.yml entry - a plain
    YAML read + `root / entry["path"]` join, not hydromt.DataCatalog.

    Mirrors what config_utils.get_data_catalog(...).get_source(name).path
    does for every entry actually used by sfincs_tiles/ (gebco,
    mdt_cnes_cls22) - both are plain relative `path:` entries with no
    driver-specific parsing needed to resolve where the real file is, so
    this simple join is exactly equivalent for this use case (NOT a
    general-purpose catalog reader - anything needing real hydromt
    DataCatalog behaviour, e.g. reading a source's own meta/rename/driver
    options, still belongs in the main pipeline's own environment).
    """
    with open(catalog_path, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    if source_name not in catalog:
        raise KeyError(f"{source_name!r} not found in {catalog_path}")
    return Path(root) / catalog[source_name]["path"]
