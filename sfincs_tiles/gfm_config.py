"""Minimal, hydromt-version-independent config/catalog helpers for
sfincs_tiles/'s own scripts.

Does not use src/config_utils.py: importing it fails under
hydromt-sfincs-dev's hydromt 1.4.1 (`ImportError: cannot import name
'setuplog' from 'hydromt.log'`). Provides only root-path resolution and
static catalog path lookups, not full hydromt DataCatalog machinery.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def read_root(config_path: Path) -> Path:
    """Returns paths.root from config_path, overridden by a sibling
    config_local.yml if one exists."""
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
    """Resolves the on-disk path for a data_catalog_gfm.yml entry by joining
    root with the entry's `path` field. Handles only plain relative path
    entries, not full hydromt.DataCatalog driver parsing.
    """
    with open(catalog_path, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    if source_name not in catalog:
        raise KeyError(f"{source_name!r} not found in {catalog_path}")
    return Path(root) / catalog[source_name]["path"]
