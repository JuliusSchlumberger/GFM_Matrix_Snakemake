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
    """Call ``fn(*args, **kwargs)``, retrying on a transient I/O error before giving up.

    A momentary SMB hiccup on the P:\\ network share can make a file that both
    exists and is reachable moments later raise an I/O error right now; none
    of rasterio/fiona/geopandas/xarray retry that themselves, so one blip
    previously aborted a whole multi-hour Snakemake run. A genuinely missing/
    corrupted file still raises after retries are exhausted, at the cost of a
    few seconds of delay. Mirrors ``tiles.py``'s ``_open_mask_tile``.

    Parameters
    ----------
    fn : Callable[..., _T]
        Function to call.
    *args
        Positional arguments passed through to `fn`.
    retries : int, optional
        Maximum number of attempts, by default 4.
    delay_s : float, optional
        Seconds to sleep between attempts, by default 5.0.
    **kwargs
        Keyword arguments passed through to `fn`.

    Returns
    -------
    _T
        Whatever `fn` returns.

    Raises
    ------
    BaseException
        The last exception from `fn`, if every attempt failed with one of
        the transient I/O exception types (see `_TRANSIENT_IO_EXCEPTIONS`).
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


def atomic_write(output_path: str | Path, write_fn: Callable[[Any], None], mode: str = "w", **open_kwargs) -> None:
    """Write a file atomically via a temp file + ``os.replace``.

    Ensures a process killed mid-write (SLURM time limit, node failure/
    preemption) never leaves a truncated file at `output_path` - the
    plain-file counterpart of ``rasters._atomic_raster_write``. Resumability
    of batch pipelines depends on "does this output file exist" actually
    meaning "is it complete". The temp file is removed if the write fails.

    Parameters
    ----------
    output_path : str or Path
        Final destination path.
    write_fn : Callable[[Any], None]
        Called as ``write_fn(f)`` with the open temp-file handle to perform
        the actual write, e.g. ``lambda f: json.dump(data, f)``.
    mode : str, optional
        Mode for the temp file's ``open()`` call, by default ``"w"`` (use
        ``"wb"`` for pickle).
    **open_kwargs
        Extra keyword arguments passed to the temp file's ``open()`` call
        (e.g. ``encoding="utf-8"``).
    """
    output_path = Path(output_path)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp{os.getpid()}")
    try:
        with retry_transient_io(open, tmp_path, mode, **open_kwargs) as f:
            write_fn(f)
        retry_transient_io(os.replace, tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def path_ready(path: str | Path) -> bool:
    """Check whether `path` exists, retrying transient I/O errors first.

    ``os.path.exists()`` catches ``OSError`` internally and just returns
    ``False``, so a momentary P:\\ blip looks identical to a genuinely
    missing file and no retry is ever triggered. Used to check whether a
    lower-hop neighbour tile's output already exists before seeding a
    hinterland tile from it (``run_aqueduct.py``'s hop>=1 branch); silently
    treating a blip as "no neighbour output yet" there would write a
    wrongly-confident real-zero result instead of the correct flooded one.
    Goes through ``os.stat``, which does raise, so it can use the same
    retry path (`retry_transient_io`) as every other read in this pipeline.

    Parameters
    ----------
    path : str or Path
        Path to check.

    Returns
    -------
    bool
        True if the path exists, False if it is missing (after retries).
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

    The single implementation: the root Snakefile imports this function too,
    so Snakemake and every standalone script expand paths identically.
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
    """Load config.yml the same way the root Snakefile does, for standalone scripts.

    Every entry-point script under preparation/, analysis/, and
    run_pipeline.py should load its config through this function instead of
    a bare ``yaml.safe_load(open(...))``, so a machine-local override in a
    git-ignored ``config_local.yml`` sibling (honored by ``snakemake`` itself
    via the Snakefile's own ``configfile:`` layering) is not silently
    ignored, and every ``{root}``/``{code_root}`` placeholder is expanded
    once up front instead of by hand at every use site.

    Looks for ``config_local.yml`` next to `config_path` and, if present,
    deep-merges it on top of the base config, then merges `extra_override`
    (if given and it exists) on top of that, before expanding
    ``{root}``/``{code_root}``/``{aqueduct_root}``/``{processed_inputs_dir}``
    throughout the whole merged result - so every value read back out is
    already a real, absolute path.

    Parameters
    ----------
    config_path : str or Path
        Path to the base config.yml.
    extra_override : str, Path, or None, optional
        Path to a second git-ignored override file, merged on top of
        config_local.yml (if any) before expansion - e.g. ``config_hpc.yml``,
        which points ``paths.root``/``aqueduct_root`` at a Linux HPC mount
        instead of the local machine's paths. Ignored if it doesn't exist.

    Returns
    -------
    dict
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


def materialize_config(
    base_config_path: str | Path,
    overrides: list[str | Path | dict],
    output_path: str | Path | None = None,
) -> dict:
    """Deep-merge config.yml + config_local.yml (if present) + `overrides`.

    Builds a real, standalone scenario config file, e.g.
    ``snakemake_workflow/scripts/materialize_configs.py``'s per-scenario configs. Each entry
    in `overrides` is either a path to a YAML file (merged in list order,
    missing ones silently skipped - same convention as `load_config`'s
    `extra_override`) or a plain dict, merged in directly - letting a caller
    layer named scenario files followed by a computed dict (e.g. run-specific
    output-path isolation) in one call.

    Deliberately does NOT path-expand the result, unlike `load_config`:
    the output is meant to itself be a valid `load_config()` input later
    (every consumer of a materialized scenario config already calls
    `load_config` on it, which re-runs expansion). Expanding here too would
    be redundant and would bake this machine's ``{root}`` into a file that
    may be read on a different machine via its own override files.

    Parameters
    ----------
    base_config_path : str or Path
        Path to the base config.yml.
    overrides : list of str, Path, or dict
        Override layers applied in order on top of the base config (and
        config_local.yml, if present). A path to a missing file is skipped.
    output_path : str, Path, or None, optional
        If given, the merged config is also written there as YAML via
        `atomic_write`.

    Returns
    -------
    dict
        The merged (not path-expanded) config dict.
    """
    base_config_path = Path(base_config_path)
    with open(base_config_path, encoding="utf-8") as f:
        merged = yaml.safe_load(f)

    local_path = base_config_path.parent / "config_local.yml"
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            merged = _deep_merge(merged, yaml.safe_load(f) or {})

    for override in overrides:
        if isinstance(override, dict):
            merged = _deep_merge(merged, override)
            continue
        override_path = Path(override)
        if override_path.exists():
            with open(override_path, encoding="utf-8") as f:
                merged = _deep_merge(merged, yaml.safe_load(f) or {})

    if output_path is not None:
        atomic_write(output_path, lambda f: yaml.safe_dump(merged, f, sort_keys=False), encoding="utf-8")

    return merged


def get_data_catalog(
    catalog_path: str | Path,
    logger_name: str = "snakemake_workflow",
    root: str | Path | None = None,
) -> hydromt.DataCatalog:
    """Return a HydroMT DataCatalog loaded from `catalog_path`.

    `root`, if given, overrides the catalog file's own ``root:``/``meta.root:``
    entry for resolving every source's relative ``path:`` (data_catalog_gfm.yml
    hardcodes its own machine-specific root, separate from config.yml's
    ``paths.root``; passing ``config["paths"]["root"]`` here keeps the two in
    sync instead of needing the catalog file edited per machine).
    ``DataCatalog.__init__`` has no such override hook, so this goes through
    ``from_yml`` directly. The returned catalog has ``get_rasterdataset``,
    ``get_dataframe``, and ``get_geodataframe`` wrapped with
    `retry_transient_io` on this instance only - these are where every
    catalog-driven script actually triggers a file read.

    Parameters
    ----------
    catalog_path : str or Path
        Path to the HydroMT data catalog YAML file. Typically a rule's
        ``params.data_catalog`` (sourced from
        ``config["paths"]["hydromt_data_catalog"]``), or read directly from
        config.yml by a standalone script.
    logger_name : str, optional
        Name for the HydroMT logger, by default ``"snakemake_workflow"``.
    root : str, Path, or None, optional
        Root path overriding the catalog file's own root for resolving
        relative source paths. If None, the catalog file's own root is used.

    Returns
    -------
    hydromt.DataCatalog
        The loaded catalog, with retry-wrapped read methods.
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

    Deduplicated and sorted by numeric SLR value (ascending), e.g.
    ``["SLR_0", "SLR_200", "SLR_250", "SLR_400", "SLR_500", ...]``.

    The sort matters beyond cosmetics: several callers pass the parallel mm
    values positionally into ``np.interp``/``pchip_interpolate`` to
    interpolate EAI across SLR levels, and both require a strictly
    increasing x-sequence - ``np.interp`` silently returns nonsense (per its
    own docs) and ``pchip_interpolate`` raises on unsorted input. A plain
    ``list(dict.fromkeys(...))`` would not sort, since ``slr_intensities``
    is declared separately from ``slr_scenarios`` and appended after it.

    Parameters
    ----------
    bc_cfg : dict
        The ``boundary_conditions`` config section; must have a
        ``slr_scenarios`` key (list of ``"SLR_<mm>"`` names).
    adapt_cfg : dict
        The ``adaptation`` config section; its ``slr_intensities`` key
        (same name format), if present, is unioned in.

    Returns
    -------
    list of str
        Deduplicated ``"SLR_<mm>"`` names, sorted ascending by mm value.
    """
    names = list(dict.fromkeys(bc_cfg["slr_scenarios"] + adapt_cfg.get("slr_intensities", [])))
    return sorted(names, key=lambda s: int(s.split("_")[1]))
