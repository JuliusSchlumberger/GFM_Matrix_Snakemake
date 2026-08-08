"""Shared config-parsing and data-catalog helpers used across the workflow."""

import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypeVar

import hydromt
import yaml
from hydromt.log import setuplog

_T = TypeVar("_T")

# rasterio.errors.RasterioIOError IS an OSError subclass, so plain OSError
# already covers it - but geopandas' vector backends are not: neither
# fiona.errors.DriverError nor pyogrio.errors.DataSourceError derive from
# OSError (confirmed against this env's installed versions), even though
# both are raised for the exact same "transient P:\ blip" condition this
# helper exists to retry (e.g. fiona.errors.DriverError: ...: No such file
# or directory). Without these, every gpd.read_file() call wrapped with
# retry_transient_io would silently never retry - the exception would just
# fall straight through on the first attempt. Imported defensively in case
# either backend is ever absent from a given environment.
_TRANSIENT_IO_EXCEPTION_TYPES: list[type[BaseException]] = [FileNotFoundError, OSError]
try:
    from fiona.errors import DriverError as _FionaDriverError
    _TRANSIENT_IO_EXCEPTION_TYPES.append(_FionaDriverError)
except ImportError:
    pass
try:
    from pyogrio.errors import DataSourceError as _PyogrioDataSourceError
    _TRANSIENT_IO_EXCEPTION_TYPES.append(_PyogrioDataSourceError)
except ImportError:
    pass
_TRANSIENT_IO_EXCEPTIONS = tuple(_TRANSIENT_IO_EXCEPTION_TYPES)


def retry_transient_io(fn: Callable[..., _T], *args, retries: int = 4, delay_s: float = 5.0, **kwargs) -> _T:
    """Call `fn(*args, **kwargs)`, retrying on a transient I/O error before giving up.

    Several independent read paths in this pipeline (rasterio, fiona/
    geopandas, xarray/netCDF4) have hit the same failure mode: a momentary
    SMB hiccup on the P:\\ network share makes a file that both exists and
    is reachable moments later raise an I/O error right now (see
    _TRANSIENT_IO_EXCEPTIONS above for exactly which exception types this
    catches, and why OSError alone isn't enough). None of these libraries
    retry that themselves, so one blip anywhere in a multi-hour Snakemake
    run previously aborted the whole invocation. A genuinely missing/
    corrupted file fails the same way on every attempt and still raises
    after retries are exhausted - this only adds a few seconds of delay in
    that case, in exchange for shrugging off the far more common transient
    blip. Mirrors tiles.py's `_open_mask_tile` (kept separate there since
    it's rasterio-specific and predates this more general helper).
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except _TRANSIENT_IO_EXCEPTIONS as e:
            last_err = e
            if attempt < retries:
                print(
                    f"[retry {attempt}/{retries - 1}] {getattr(fn, '__name__', fn)} failed: {e} "
                    f"- retrying in {delay_s:.0f}s",
                    flush=True,
                )
                time.sleep(delay_s)
    raise last_err


def path_ready(path: str | Path) -> bool:
    """Check whether `path` exists, retrying transient I/O errors first.

    `os.path.exists()` catches `OSError` internally and just returns
    `False` - so a momentary P:\\ blip (see `retry_transient_io` above)
    looks IDENTICAL to a genuinely missing file, and no retry is ever
    triggered. Used to check whether a lower-hop neighbour tile's output
    already exists before seeding a hinterland tile from it
    (`run_aqueduct.py` / `run_aqueduct_cli.py`'s hop>=1 branch) - on a
    flaky network mount, silently treating a transient blip as "no
    neighbour output yet" would write a wrongly-confident real-zero result
    instead of the correct flooded one. `os.stat` DOES raise, so it can go
    through the same retry path as every other read in this pipeline.
    """
    try:
        retry_transient_io(os.stat, path)
        return True
    except _TRANSIENT_IO_EXCEPTIONS:
        return False


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


def get_data_catalog(
    catalog_path: str | Path,
    logger_name: str = "snakemake_workflow",
    root: str | Path | None = None,
) -> hydromt.DataCatalog:
    """Return a HydroMT DataCatalog loaded from `catalog_path`.

    `catalog_path` comes from the caller — a rule's `params.data_catalog`
    (sourced from `config["paths"]["hydromt_data_catalog"]`) for
    Snakemake-driven scripts, or a path read directly from config.yml by a
    standalone script. Nothing is hardcoded here.

    `root`, if given, overrides the catalog file's own `root:`/`meta.root:`
    entry for resolving every source's relative `path:` (data_catalog_gfm.yml
    hardcodes its own machine-specific root, entirely separate from
    config.yml's `paths.root` - passing `config["paths"]["root"]` here, which
    IS resolved through config_local.yml, keeps the two in sync instead of
    needing the catalog file edited per machine). `DataCatalog.__init__`
    itself has no such override hook (its `data_libs` loader always falls
    back to the file's own root), so this goes through `from_yml` directly.
    """
    logger = setuplog(logger_name, log_level=10)
    catalog = hydromt.DataCatalog(logger=logger)
    catalog.from_yml(str(catalog_path), root=str(root) if root is not None else None)

    # Wrap the catalog's own read methods (not get_source/__getitem__, which
    # only return metadata/an adapter - no I/O happens there) with
    # retry_transient_io: these three are where every catalog-driven script
    # in the project actually triggers a file read, and P:\'s transient SMB
    # drops have already taken down runs at this exact point (see
    # tiles.py's _open_mask_tile for the same issue in raw rasterio calls
    # that don't go through the catalog). Rebinding on this INSTANCE only -
    # not monkey-patching hydromt/rasterio/etc. globally - keeps this safe
    # and contained to catalogs obtained through this function.
    for _method_name in ("get_rasterdataset", "get_dataframe", "get_geodataframe"):
        _orig = getattr(catalog, _method_name)
        setattr(catalog, _method_name, partial(retry_transient_io, _orig))
    return catalog


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
