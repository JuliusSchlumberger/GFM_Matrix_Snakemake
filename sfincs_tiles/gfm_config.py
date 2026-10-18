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
    """Resolve the data root path from a config file, honoring a local override.

    Parameters
    ----------
    config_path : Path
        Path to the YAML config file containing a ``paths.root`` entry.

    Returns
    -------
    Path
        `paths.root` from `config_path`, overridden by ``paths.root`` in a
        sibling ``config_local.yml`` if one exists.
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
    """Resolve the on-disk path for a data-catalog entry.

    Handles only plain relative ``path`` entries, not full
    hydromt.DataCatalog driver parsing.

    Parameters
    ----------
    catalog_path : Path
        Path to the YAML data catalog (e.g. ``data_catalog_gfm.yml``).
    root : Path
        Data root directory to join the entry's relative path against.
    source_name : str
        Key of the catalog entry to resolve.

    Returns
    -------
    Path
        `root` joined with the entry's ``path`` field.

    Raises
    ------
    KeyError
        If `source_name` is not present in the catalog.
    """
    with open(catalog_path, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    if source_name not in catalog:
        raise KeyError(f"{source_name!r} not found in {catalog_path}")
    return Path(root) / catalog[source_name]["path"]
