"""Run the flood-model variants for one calibration tile on SFINCS's 30 m subgrid.

Phases (src/flood_variants.py names the variants; every phase also runs both bathtubs):
  sweep     calibration tiles: straight_line and eikonal_uniform over UNIFORM_RATES_M_PER_KM,
            eikonal_landcover over FRICTION_SCALE_FACTORS
  duration  calibration tiles: both duration variants over DURATION_ALPHAS, at the parameter
            variant_selection.json picked for their base variant
  selected  every tile: each variant at its run in variant_selection.json
A tile's group is calibration_split.split_group. Seeds, coastline and solver settings are
production's (flood_model, config simulation.flooding); the forcing is the SFINCS run's, the
duration correction takes its boundary hydrographs. No static (two-component) level: the
SFINCS runs have no separate SLR part. Existing outputs are skipped.

Usage:
    python run_calibration_models.py --config <resolved_config.yml> --tile-id 1122 --phase sweep
    python run_calibration_models.py --config <resolved_config.yml> --tile-id 1122 --run eikonal_uniform:0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from event_volume import DurationCorrection  # noqa: E402
from flood_model import flood_depth_dense, idw_seed_values_at  # noqa: E402
from flood_variants import VARIANTS, run_key, variant_flooding  # noqa: E402
from rasters import WATERDEPTH_NODATA_INT16, decode_waterlevel_cm, encode_waterdepth_cm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import (  # noqa: E402
    DURATION_ALPHAS, FRICTION_SCALE_FACTORS, RETURN_PERIOD, RUNS_NAME, SELECTION_NAME, UNIFORM_RATES_M_PER_KM,
    WATERLEVEL_NAME, run_path,
)
from calibration_split import split_group  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)
LAND_CODE, OCEAN_CODE = 0, 1
NODATA_CODE = 4  # subgrid cells the native lon/lat mask.tif does not cover


def build_inputs_from_sfincs_subgrid(
    sfincs_dir: Path, native_mask_path: Path,
    friction_scale_factor: float, default_friction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine, str]:
    """Load (dem, mask, friction, transform, crs) on SFINCS's subgrid UTM grid.

    DEM NaN -> 99.0 m dry sentinel (rasters.DEM_NODATA_M). The lon/lat mask is reprojected
    nearest; the UTM rectangle overshoots it at the corners (meridian convergence), and
    those cells get NODATA_CODE. Friction = manning_subgrid.tif / 100 * friction_scale_factor,
    NaN filled with default_friction * 100 before scaling.
    """
    dep_path = sfincs_dir / "subgrid" / "dep_subgrid.tif"
    man_path = sfincs_dir / "subgrid" / "manning_subgrid.tif"

    with retry_transient_io(rasterio.open, dep_path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        crs = src.crs
        shape = src.shape
    dem = np.where(np.isnan(dem), np.float32(99.0), dem)

    with retry_transient_io(rasterio.open, man_path) as src:
        manning_n = src.read(1).astype(np.float32)
    manning_n = np.where(np.isnan(manning_n), np.float32(default_friction * 100.0), manning_n)
    friction = (manning_n / np.float32(100.0)) * np.float32(friction_scale_factor)

    with retry_transient_io(rasterio.open, native_mask_path) as src:
        mask_f = np.empty(shape, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=mask_f,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.nearest,
        )
    valid_code = np.isin(mask_f, [0, 1, 2, 3])
    mask = np.where(valid_code, mask_f, np.float32(NODATA_CODE)).astype(np.int8)

    return dem, mask, friction, transform, crs


def _write_waterdepth(waterdepth: np.ndarray, transform, crs, output_path: Path) -> None:
    encoded = encode_waterdepth_cm(waterdepth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "dtype": "int16", "count": 1,
        "height": waterdepth.shape[0], "width": waterdepth.shape[1],
        "transform": transform, "crs": crs, "nodata": WATERDEPTH_NODATA_INT16,
        "compress": "zstd", "predictor": 1, "tiled": True, "blockxsize": 512, "blockysize": 512,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(encoded, indexes=1)


def subgrid_coastline_mask(mask: np.ndarray, ocean_code: int, river_code: int | None) -> np.ndarray:
    """Mark coastline cells on the rotated subgrid, as flood_model.coastline_mask does.

    Ocean within 1 px of land (or river) in an ocean body reaching the domain edge:
    the array edge or the NODATA_CODE corners (the tile edge on this grid).
    """
    ocean = mask == ocean_code
    components, _ = ndimage.label(ocean, structure=_STRUCTURE_8)
    edge = np.zeros_like(ocean)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    edge |= ndimage.binary_dilation(mask == NODATA_CODE, structure=_STRUCTURE_8)
    edge_labels = np.unique(components[edge & ocean])
    edge_connected_ocean = np.isin(components, edge_labels[edge_labels != 0])
    landlike = mask == LAND_CODE
    if river_code is not None:
        landlike |= mask == river_code
    return ndimage.binary_dilation(landlike, structure=_STRUCTURE_8) & edge_connected_ocean


def solver_inputs(dem: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn NODATA_CODE cells into dry land at 99 m for `flood_depth_dense`.

    flood_depth_dense zeroes every non-land cell, so the no-data frame would be a 0 m
    channel carrying water along the border (tile 631: Gran Bajo de San Julian flooded
    past its 20-30 m rim). Coastline and seeds still use the original mask.
    """
    nodata = mask == NODATA_CODE
    return (np.where(nodata, np.float32(99.0), dem).astype(dem.dtype),
            np.where(nodata, np.int8(LAND_CODE), mask).astype(mask.dtype))


def production_seeds(mask, transform, crs, boundaries: gpd.GeoDataFrame, knn: int, river_code: int,
                     idw_power: float):
    """Return (rows, cols, level_m) eikonal seeds on the subgrid coastline, or None if none.

    Each cell gets the inverse-distance average (weights distance^-idw_power) of its `knn` nearest
    connected stations (flood_model.idw_seed_values_at, as production), computed in lon/lat.
    """
    coast = subgrid_coastline_mask(mask, OCEAN_CODE, river_code)
    rows, cols = np.nonzero(coast)
    if len(rows) == 0:
        return None
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(np.asarray(xs), np.asarray(ys))
    st = boundaries.to_crs("EPSG:4326")
    values = decode_waterlevel_cm(st[WATERLEVEL_NAME].to_numpy())
    seeds = idw_seed_values_at(np.column_stack([lon, lat]), rows, cols,
                               np.column_stack([st.geometry.x, st.geometry.y]), values, knn, mask, OCEAN_CODE,
                               idw_power)
    return rows.astype(np.int64), cols.astype(np.int64), seeds.astype(np.float32)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


HYDRO_WINDOW_HR = (40.0, 110.0)  # build_sfincs_tile.TRUNCATE_WINDOW_HR_DEFAULT: the simulated window
PHASES = ("sweep", "duration", "selected")
BATHTUBS = ("bathtub", "bathtub_connected")


def phase_runs(phase: str, group: str, selection: dict | None) -> list[tuple[str, float | None, float | None]]:
    """Return the (variant, parameter, alpha) runs of `phase` on a tile of `group`."""
    runs = [(name, None, None) for name in BATHTUBS]
    if phase == "sweep" and group == "calibration":
        runs += [(v, r, None) for v in ("straight_line", "eikonal_uniform") for r in UNIFORM_RATES_M_PER_KM]
        runs += [("eikonal_landcover", f, None) for f in FRICTION_SCALE_FACTORS]
    elif phase == "duration" and group == "calibration":
        for name in ("eikonal_uniform_duration", "eikonal_landcover_duration"):
            parameter = selection["variants"][name.removesuffix("_duration")]["parameter"]
            runs += [(name, parameter, a) for a in DURATION_ALPHAS]
    elif phase == "selected":
        runs += [(name, v["parameter"], v["alpha"]) for name, v in selection["variants"].items()
                 if name not in BATHTUBS]
    return runs


def parse_run(spec: str) -> tuple[str, float | None, float | None]:
    """Parse VARIANT[:PARAMETER[:ALPHA]]."""
    name, *rest = spec.split(":")
    if name not in VARIANTS:
        raise SystemExit(f"--run {spec}: unknown variant {name!r} (one of {', '.join(VARIANTS)})")
    values = [float(x) if x else None for x in rest] + [None, None]
    return name, values[0], values[1]


def sfincs_duration_correction(sfincs_dir: Path, crs, duration_config: dict) -> DurationCorrection:
    """Build correction B from the SFINCS run's own boundary hydrographs (matched points)."""
    hydro = pd.read_csv(sfincs_dir / "corrected_hydrographs.csv")
    hydro = hydro[(hydro.elapsed_hr >= HYDRO_WINDOW_HR[0]) & (hydro.elapsed_hr <= HYDRO_WINDOW_HR[1])]
    points = gpd.read_file(sfincs_dir / "matched_boundary_points.gpkg").to_crs("EPSG:4326")
    series = np.vstack([hydro[str(i)].to_numpy(dtype=np.float64) for i in range(len(points))])
    return DurationCorrection(
        np.column_stack([points.geometry.x, points.geometry.y]), series,
        float(np.median(np.diff(hydro.elapsed_hr))) * 3600.0,
        t_ref_s=float(duration_config["t_ref_h"]) * 3600.0, alpha=float(duration_config["alpha"]),
        max_multiplier=float(duration_config["max_multiplier"]), crs=crs)


def main() -> None:
    """Run one phase (or the --run list) for one tile, skipping existing outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", required=True, help="config.yml or the base dir's resolved_config.yml")
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--phase", choices=PHASES, default="sweep")
    parser.add_argument("--run", action="append", default=None, metavar="VARIANT[:PARAMETER[:ALPHA]]",
                        help="run these instead of the phase (repeatable)")
    parser.add_argument("--overwrite", action="store_true", help="redo outputs that already exist")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root = read_root(Path(args.config))
    flooding = config["simulation"]["flooding"]
    oc = flooding["obstacle_coupling"]
    river_code = int(config["tile_generation"]["river_code"])
    base_dir = root / args.base_dir_name
    tile_dir = base_dir / args.tile_id
    out_dir = tile_dir / "outputs"
    sfincs_dir = tile_dir / "sfincs_model"
    group = split_group(args.tile_id)

    if args.run:
        runs = [parse_run(spec) for spec in args.run]
    else:
        selection_path = base_dir / SELECTION_NAME
        selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
        if args.phase != "sweep" and selection is None:
            raise SystemExit(f"phase {args.phase} needs {selection_path} (compute_variant_metrics.py)")
        runs = phase_runs(args.phase, group, selection)
    todo = [r for r in runs if args.overwrite or not run_path(out_dir, run_key(*r)).exists()]
    if not todo:
        print(f"tile {args.tile_id} ({group}): {len(runs)} run(s) of phase {args.phase} already done")
        return

    boundaries = retry_transient_io(gpd.read_file, tile_dir / "inputs" / f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg")
    if boundaries.empty:
        raise SystemExit(f"tile {args.tile_id}: no station in boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg")
    dem, mask, base_friction, transform, crs = build_inputs_from_sfincs_subgrid(
        sfincs_dir, tile_dir / "inputs" / "mask.tif", friction_scale_factor=1.0,
        default_friction=float(flooding["default_friction"]))
    cell_m = float(np.hypot(transform.a, transform.d))
    seeds = production_seeds(mask, transform, crs, boundaries, int(flooding["knn"]), river_code,
                             float(flooding["idw_power"]))
    if seeds is None:
        raise SystemExit(f"tile {args.tile_id}: no coastline cell on the subgrid - nothing to seed")
    seed_rows, seed_cols, seed_values = seeds
    print(f"tile {args.tile_id} ({group}): {len(todo)} run(s), {len(seed_rows)} coastline seed cell(s), levels "
          f"{seed_values.min():.2f}-{seed_values.max():.2f} m", flush=True)

    solver_dem, solver_mask = solver_inputs(dem, mask)
    runs_path = out_dir / RUNS_NAME
    all_runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    for name, parameter, alpha in todo:
        spec = VARIANTS[name]
        v_flooding = variant_flooding(flooding, name, parameter, alpha)
        rate_m_per_m = float(v_flooding["uniform_attenuation_m_per_km"]) / 1000.0
        if spec["friction_source"] == "uniform":
            friction = np.full(dem.shape, rate_m_per_m * cell_m, dtype=np.float32)
        else:
            friction = base_friction * np.float32(float(v_flooding["friction_scale_factor"]) * cell_m)
        duration = (sfincs_duration_correction(sfincs_dir, crs, v_flooding["duration_correction"])
                    if spec["duration"] else None)
        t0 = time.monotonic()
        waterdepth, diagnostics = flood_depth_dense(
            solver_dem, solver_mask, friction, transform,
            seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
            resolution=cell_m, k=int(flooding["knn"]), idw_power=float(flooding["idw_power"]),
            ocean_code=OCEAN_CODE, river_code=river_code,
            obstacle_coupling=bool(oc["enabled"]),
            max_outer_iterations=int(oc["max_outer_iterations"]),
            outer_convergence_pct=float(oc["outer_convergence_pct"]),
            max_rounds=int(flooding["max_rounds"]),
            waterlevel_epsilon_m=float(flooding["waterlevel_epsilon_m"]),
            duration_correction=duration, model=spec["model"], attenuation_m_per_m=rate_m_per_m,
        )
        key = run_key(name, parameter, alpha)
        _write_waterdepth(waterdepth, transform, crs, run_path(out_dir, key))
        runtime = time.monotonic() - t0
        all_runs[key] = {"variant": name, "parameter": parameter, "alpha": alpha,
                         "solver": _jsonable(diagnostics), "runtime_s": round(runtime, 1)}
        runs_path.write_text(json.dumps(all_runs, indent=1))
        print(f"tile {args.tile_id}: {key} - {int((waterdepth > 0).sum())} flooded cell(s), {runtime:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()
