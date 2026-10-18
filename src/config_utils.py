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

# RasterioIOError is an OSError, but fiona DriverError and pyogrio
# DataSourceError are not, though both are raised for transient P:\ blips.
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
    """Call ``fn(*args, **kwargs)``, retrying on transient I/O errors.

    SMB blips on the P:\\ share raise errors for files that exist; none of
    rasterio/fiona/geopandas/xarray retry. A truly missing file still raises
    after `retries` attempts (`delay_s` seconds apart).
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


def atomic_write(
    output_path: str | Path,
    write_fn: Callable[[Any], None],
    mode: str = "w",
    opener: Callable[..., Any] = open,
    **open_kwargs,
) -> None:
    """Write a file atomically via a temp file + ``os.replace``.

    A killed job never leaves a truncated file that resumable pipelines would
    treat as complete. ``write_fn(f)`` receives the handle returned by
    ``opener(tmp_path, mode, **open_kwargs)`` (a context manager; ``open`` by
    default, e.g. ``rasterio.open`` for rasters). The PID suffix of the temp
    file keeps concurrent writers off each other's temp file.
    """
    output_path = Path(output_path)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp{os.getpid()}")
    try:
        with retry_transient_io(opener, tmp_path, mode, **open_kwargs) as f:
            write_fn(f)
        retry_transient_io(os.replace, tmp_path, output_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def path_ready(path: str | Path) -> bool:
    """Check whether `path` exists, retrying transient I/O errors first.

    ``os.path.exists()`` swallows OSError, so a P:\\ blip would read as
    missing; for hop>=1 neighbour seeding that writes a false all-dry result.
    """
    try:
        retry_transient_io(os.stat, path)
        return True
    except _TRANSIENT_IO_EXCEPTIONS:
        return False


def _deep_merge(base: dict, override: dict) -> dict:
    # Deep merge, matching Snakemake's multi-configfile semantics.
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _expand_paths(obj: Any, substitutions: dict[str, str]) -> Any:
    # Also imported by the root Snakefile, so both expand paths identically.
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

    Layers: config.yml < sibling config_local.yml < `extra_override` (skipped
    if missing; HPC callers use `load_hpc_config`). Then expands {root}, {code_root} and
    {processed_inputs_dir} everywhere.
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

    root = config["paths"].get("root")
    if not root:
        raise ValueError(f"{config_path}: paths.root is not set - create snakemake_workflow/config/config_local.yml (copy config_local.yml.example) or pass a config that sets it (config_hpc.yml / a materialized scenario config)")
    code_root = config["paths"].get("code_root", root)
    subs = {"root": root, "code_root": code_root}
    # Two-pass, as the Snakefile's _PATH_SUBS: processed_inputs_dir contains {root}.
    if "processed_inputs_dir" in config["paths"]:
        subs["processed_inputs_dir"] = _expand_paths(config["paths"]["processed_inputs_dir"], subs)
    return _expand_paths(config, subs)


HPC_SITE_CONFIG_MISSING = (
    "snakemake_workflow/config/config_hpc.yml not found: copy snakemake_workflow/config/config_hpc.example.yml "
    "to snakemake_workflow/config/config_hpc.yml and fill in your site values"
)


def has_hpc_site_values(config: dict) -> bool:
    """Return whether `config` carries the HPC site values (hpc.sbatch.env_activate_cmd set).

    config.yml leaves them empty; config_hpc.yml sets them, and so does every
    resolved_config.yml a generator writes from the layered config.
    """
    return bool(((config.get("hpc") or {}).get("sbatch") or {}).get("env_activate_cmd"))


def load_hpc_config(config_path: str | Path) -> dict:
    """Load `config_path` as the HPC jobs see it: `load_config` + sibling config_hpc.yml.

    A config without a sibling config_hpc.yml must already carry the site values,
    as the resolved_config.yml that chained on-node generators read; otherwise
    raises FileNotFoundError naming the example file to copy.
    """
    config_path = Path(config_path)
    site_path = config_path.parent / "config_hpc.yml"
    if site_path.exists():
        return load_config(config_path, extra_override=site_path)
    config = load_config(config_path)
    if not has_hpc_site_values(config):
        raise FileNotFoundError(f"{HPC_SITE_CONFIG_MISSING} ({config_path} has no hpc site values either)")
    return config


def with_hpc_site_values(hpc_cfg: dict, config_dir: str | Path) -> dict:
    """Layer the `hpc` block of `config_dir`/config_hpc.yml over `hpc_cfg`.

    For the Snakefile's `config["hpc"]`, which has no site layer. Without the
    file, `hpc_cfg` must already carry the site values (GFM_CONFIG_PATH set to
    a resolved_config.yml); otherwise raises FileNotFoundError.
    """
    site_path = Path(config_dir) / "config_hpc.yml"
    if site_path.exists():
        with open(site_path, encoding="utf-8") as f:
            hpc_cfg = _deep_merge(hpc_cfg, (yaml.safe_load(f) or {}).get("hpc") or {})
    if not has_hpc_site_values({"hpc": hpc_cfg}):
        raise FileNotFoundError(HPC_SITE_CONFIG_MISSING)
    return hpc_cfg


def materialize_config(
    base_config_path: str | Path,
    overrides: list[str | Path | dict],
    output_path: str | Path | None = None,
) -> dict:
    """Deep-merge config.yml + config_local.yml (if present) + `overrides`.

    `overrides` are YAML paths (missing ones skipped) or dicts, applied in
    order. Not path-expanded: the output is a `load_config` input and must not
    bake in this machine's {root}. Written via `atomic_write` if `output_path`.
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

    `root` overrides the catalog's own root for relative source paths (pass
    config paths.root). The three get_* read methods are wrapped with
    `retry_transient_io` on this instance only.
    """
    logger = setuplog(logger_name, log_level=10)
    catalog = hydromt.DataCatalog(logger=logger)
    catalog.from_yml(str(catalog_path), root=str(root) if root is not None else None)

    for _method_name in ("get_rasterdataset", "get_dataframe", "get_geodataframe"):
        _orig = getattr(catalog, _method_name)
        setattr(catalog, _method_name, partial(retry_transient_io, _orig))
    return catalog


def merged_waterlevel_names(bc_cfg: dict, adapt_cfg: dict) -> list[str]:
    """Union of boundary_conditions.waterlevel_names and adaptation.slr_intensities.

    Names are "SLR_<mm>", sorted ascending by mm: callers feed them to
    np.interp/pchip_interpolate, which need strictly increasing x.
    """
    names = list(dict.fromkeys(bc_cfg["waterlevel_names"] + adapt_cfg.get("slr_intensities", [])))
    return sorted(names, key=lambda s: int(s.split("_")[1]))


def return_period_label(rp: int) -> str:
    """Return the "RP100" form of a return period given in years (a config list entry).

    "RP100" (the Snakemake `return_period` wildcard) is what every output CSV's
    `return_period` column holds; config lists hold plain ints.
    """
    return f"RP{int(rp)}"


def return_period_years(label: str) -> int:
    """Return the integer years of an "RP100"-form return period, for numeric axes and sorting.

    Raises
    ------
    ValueError
        If `label` is not of the form "RP<int>".
    """
    s = str(label)
    if not (s.startswith("RP") and s[2:].isdigit()):
        raise ValueError(f'return period {label!r} is not of the form "RP<years>"')
    return int(s[2:])
