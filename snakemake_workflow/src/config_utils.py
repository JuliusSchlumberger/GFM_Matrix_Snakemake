"""Shared config-parsing and data-catalog helpers used across the workflow."""

from pathlib import Path
from typing import Any

import hydromt
import yaml
from hydromt.log import setuplog


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (override's values win).

    Matches Snakemake's own multi-`configfile:` merge semantics (used by the
    root Snakefile to layer config_local.yml on top of config.yml) - a
    shallow `{**base, **override}` would silently drop every OTHER key in a
    nested dict (e.g. overriding just `paths.root` would wipe out
    `paths.code_root`/`paths.hydromt_data_catalog` if they weren't also
    repeated in the override file).
    """
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _expand_paths(obj: Any, substitutions: dict[str, str]) -> Any:
    """Recursively substitute {key} placeholders in all string config values.

    Identical logic to the root Snakefile's own `_expand_paths` (kept here
    too, rather than imported from there, since standalone scripts don't
    load the Snakefile).
    """
    if isinstance(obj, str):
        for key, val in substitutions.items():
            obj = obj.replace(f"{{{key}}}", val)
        return obj
    if isinstance(obj, dict):
        return {k: _expand_paths(v, substitutions) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_paths(v, substitutions) for v in obj]
    return obj


def load_config(config_path: str | Path, extra_override: str | Path | None = None) -> dict:
    """Load config.yml the same way the root Snakefile does, for standalone
    (non-Snakemake) scripts.

    Every entry-point script under preparation/, analysis/, and
    run_pipeline.py should load its config through this function instead of
    a bare `yaml.safe_load(open(...))` - otherwise a machine-local override
    in a git-ignored `config_local.yml` sibling (honored by `snakemake`
    itself, via the Snakefile's own `configfile:` layering) is silently
    ignored by that script, and every `{root}`/`{code_root}` placeholder in
    every value it reads has to be expanded by hand at every use site
    instead of once, up front.

    Looks for `config_local.yml` next to `config_path` (same directory,
    e.g. `snakemake_workflow/config/config_local.yml` next to
    `snakemake_workflow/config/config.yml`) and, if present, deep-merges it
    on top of the base config before expanding `{root}`/`{code_root}`/
    `{aqueduct_root}` throughout the WHOLE merged result - so every value any
    script reads back out is already a real, absolute path; no per-script
    `_expand()` helper or manual `.replace("{root}", ...)` call is needed
    anywhere.

    Args:
        config_path: Path to the base config.yml.
        extra_override: Optional path to a second git-ignored override file,
            merged on top of config_local.yml (if any) before expansion -
            e.g. `config_hpc.yml`, which points `paths.root`/`aqueduct_root`
            at a Linux HPC mount instead of the local machine's paths.
            Ignored if it doesn't exist.

    Returns:
        The merged, fully path-expanded config dict.
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    local_path = config_path.parent / "config_local.yml"
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            local_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, local_config)

    if extra_override is not None and Path(extra_override).exists():
        with open(extra_override, encoding="utf-8") as f:
            override_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, override_config)

    root = config["paths"].get("root", "")
    code_root = config["paths"].get("code_root", root)
    aqueduct_root = config["paths"].get("aqueduct_root", code_root)
    subs = {"root": root, "code_root": code_root, "aqueduct_root": aqueduct_root}
    # processed_inputs_dir itself needs root/code_root/aqueduct_root expanded
    # first (it's declared as "{root}/processed_inputs") before it can be
    # used as a substitution value in turn - mirrors the root Snakefile's own
    # two-pass expansion (see its _PATH_SUBS comment) so standalone scripts
    # resolve "{processed_inputs_dir}"-referencing values identically.
    if "processed_inputs_dir" in config["paths"]:
        subs["processed_inputs_dir"] = _expand_paths(config["paths"]["processed_inputs_dir"], subs)
    return _expand_paths(config, subs)


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
