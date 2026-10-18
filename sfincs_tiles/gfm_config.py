"""Hydromt-version-independent config/catalog helpers for sfincs_tiles/.

src/config_utils.py fails to import under hydromt-sfincs-dev's hydromt 1.4.1
(`cannot import name 'setuplog' from 'hydromt.log'`).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def read_root(config_path: Path) -> Path:
    """Return ``paths.root`` from `config_path`, overridden by a sibling config_local.yml."""
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
    """Return `root` / the catalog entry's relative ``path`` (no hydromt driver parsing)."""
    with open(catalog_path, encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    if source_name not in catalog:
        raise KeyError(f"{source_name!r} not found in {catalog_path}")
    return Path(root) / catalog[source_name]["path"]


_HYDRAX_ENV = Path(__file__).resolve().parent / "hydrax_env.sh"


def hydrax_paths() -> dict[str, str]:
    """Return the Hydrax paths defined in ``sfincs_tiles/hydrax_env.sh``.

    As bash ``${NAME:-default}``: a set, non-empty environment variable overrides the default.
    """
    import os
    import re

    out = {}
    for line in _HYDRAX_ENV.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r'(\w+)="\$\{(\w+):-(.*)\}"', line.strip())
        if m and m.group(1) == m.group(2):
            out[m.group(1)] = os.environ.get(m.group(1)) or m.group(3)
    return out
