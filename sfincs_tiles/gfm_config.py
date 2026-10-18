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
    root = (cfg.get("paths") or {}).get("root")

    local_path = config_path.parent / "config_local.yml"
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            local_cfg = yaml.safe_load(f) or {}
        local_root = (local_cfg.get("paths") or {}).get("root")
        if local_root:
            root = local_root

    if not root:
        raise ValueError(f"{config_path}: paths.root is not set - create snakemake_workflow/config/config_local.yml (copy config_local.yml.example) or pass a config that sets it (config_hpc.yml / a materialized scenario config)")
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


_HYDRAX_ENV = Path(__file__).resolve().parent / "hydrax_env.sh"


def hydrax_paths() -> dict[str, str]:
    """Return the Hydrax paths defined in ``sfincs_tiles/hydrax_env.sh``.

    That file is the single definition shared by the shell drivers (which `source` it) and the
    Python job generators (which call this). Same semantics as bash's ``${NAME:-default}``: an
    environment variable that is set and non-empty overrides the file's default.

    Returns
    -------
    dict of str to str
        ``GFM_DATA_ROOT``, ``GFM_CODE_ROOT``, ``GFM_PYTHON`` and ``GFM_SFINCS_PYTHON``.
    """
    import os
    import re

    out = {}
    for line in _HYDRAX_ENV.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r'(\w+)="\$\{(\w+):-(.*)\}"', line.strip())
        if m and m.group(1) == m.group(2):
            out[m.group(1)] = os.environ.get(m.group(1)) or m.group(3)
    return out
