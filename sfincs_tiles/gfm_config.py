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


_SITE_ENV = Path(__file__).resolve().parent / "site_env.sh"
SITE_ENV_MISSING = (
    "sfincs_tiles/site_env.sh not found: copy sfincs_tiles/site_env.example.sh to "
    "sfincs_tiles/site_env.sh and fill in your site values"
)


def site_paths() -> dict[str, str]:
    """Return the site values defined in ``sfincs_tiles/site_env.sh``.

    As bash ``${NAME:-default}``: a set, non-empty environment variable overrides the default.
    Raises FileNotFoundError when the file is missing.
    """
    import os
    import re

    if not _SITE_ENV.exists():
        raise FileNotFoundError(SITE_ENV_MISSING)
    out = {}
    for line in _SITE_ENV.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r'(\w+)="\$\{(\w+):-(.*)\}"', line.strip())
        if m and m.group(1) == m.group(2):
            out[m.group(1)] = os.environ.get(m.group(1)) or m.group(3)
    return out


def check_site_roots(site: dict[str, str], hpc_paths: dict) -> None:
    """Raise ValueError unless site_env.sh's roots equal config_hpc.yml's paths.root / code_root.

    The shell drivers read the former, the Python generators the latter; both are
    embedded in the same job chain.
    """
    pairs = [("GFM_DATA_ROOT", "root"), ("GFM_CODE_ROOT", "code_root")]
    bad = [f"{env}={site.get(env)!r} vs paths.{key}={hpc_paths.get(key)!r}"
           for env, key in pairs if str(site.get(env, "")).rstrip("/") != str(hpc_paths.get(key, "")).rstrip("/")]
    if bad:
        raise ValueError("sfincs_tiles/site_env.sh and snakemake_workflow/config/config_hpc.yml disagree: "
                         + "; ".join(bad) + " - make them equal")
