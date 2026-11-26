"""Compute per-tile candidate drivers of model-vs-SFINCS disagreement, per whole tile and owned area.

Run under the gfm env:
    python compute_tile_drivers.py --base-dir-name calibration_roughness_SFINCS [--workers 4]
Writes tile_drivers.csv (whole tile, for the raw analysis) and tile_drivers_owned.csv (the land
each tile owns among overlapping selected tiles, sfincs_grid.core_mask, as the dedup scores).
Tiles already in both files are kept (their inputs do not change between runs) unless a file
lacks a driver column, then all are recomputed. With --shard K/N only that share of the missing
tiles is computed, into part files (tile_drivers.partKofN.csv); a run without --shard merges the
parts into the main files. read_drivers reads main file and parts together.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import shard  # noqa: E402
from calibration_results import RESULT_NAME  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402
from sfincs_grid import core_mask, load_footprints  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from event_volume import spill_level  # noqa: E402
from config_utils import atomic_write  # noqa: E402
from flood_model import LAND_CODE, open_sea_mask  # noqa: E402
from process_pool import available_cpus  # noqa: E402
from rasters import decode_friction_int16  # noqa: E402

DRIVERS_NAME = "tile_drivers.csv"
DRIVERS_OWNED_NAME = "tile_drivers_owned.csv"
OCEAN_CODE = 1
LAKE_CODE = 2
RIVER_CODE = 3
DEM_SCALE = 100.0  # dem.tif is int16 cm
BOUNDARY_LEVEL_COLUMN = "SLR_0"  # boundaries_RP100_SLR_0.gpkg, cm
ENCLOSED_DEPTH_M = 0.2  # a cell is behind a sill when its spill level is this far above its ground
PEAK_HOLD_FRACTION = 0.8  # of the hydrograph's own range, from its minimum
DRIVER_COLUMNS = ["coast_km", "bathtub_zone_km2", "reach_km", "friction_m_per_km", "river_pct", "lake_pct",
                  "overtop_coast_pct", "enclosed_pct", "surge_range_m", "peak_level_spread_m", "peak_hold_h"]
NO_GROUND_M = 99.0  # DEM nodata: high ground that never floods
_S8 = np.ones((3, 3), dtype=bool)


def tile_fields(dem_m: np.ndarray, mask: np.ndarray, peak_levels_m: np.ndarray) -> dict[str, np.ndarray]:
    """Return the tile's cell classes the area drivers count, from connectivity over the whole tile.

    zone: land below the median boundary peak level, 8-connected to the open sea (the connected
    bathtub). enclosed: zone cells whose spill level (lowest level connecting them to the open
    sea, priority flood) is at least ENCLOSED_DEPTH_M above their ground, i.e. behind a sill,
    dike or ridge. coast: land touching the open sea; front: coast below the peak level.
    """
    land = mask == LAND_CODE
    sea = open_sea_mask(mask.astype(np.int8), OCEAN_CODE)
    coast = land & ndimage.binary_dilation(sea, _S8)
    peak_m = np.median(peak_levels_m)
    low = land & (dem_m < peak_m)
    labels, _ = ndimage.label(low | sea, _S8)
    seaward = np.unique(labels[sea])
    zone = low & np.isin(labels, seaward[seaward > 0])
    enclosed = np.zeros_like(zone)
    if zone.any():
        ground = np.where(np.isfinite(dem_m), dem_m, NO_GROUND_M).astype(np.float64)
        enclosed = zone & (spill_level(ground, sea) > ground + ENCLOSED_DEPTH_M)
    return {"coast": coast, "front": coast & (dem_m < peak_m), "zone": zone, "enclosed": enclosed,
            "valid": np.isfinite(dem_m) | (mask != LAND_CODE), "lake": mask == LAKE_CODE, "river": mask == RIVER_CODE}


def area_drivers(fields: dict[str, np.ndarray], friction_per_m: np.ndarray, cell_dx_m: float, cell_dy_m: float,
                 counted: np.ndarray | None = None) -> dict[str, float]:
    """Return the area drivers over the `counted` cells (default: the whole tile).

    reach_km: connected-bathtub area per km of coastline. friction_m_per_km: median unscaled
    land-cover attenuation over the bathtub zone. river_pct / lake_pct: river / lake cells per
    valid cell. overtop_coast_pct: coastline below the peak level. enclosed_pct: share of the
    bathtub zone behind a sill.
    """
    f = fields if counted is None else {k: v & counted for k, v in fields.items()}
    coast_km = f["coast"].sum() * np.sqrt(cell_dx_m * cell_dy_m) / 1000.0
    zone, n_zone, n_valid = f["zone"], int(f["zone"].sum()), int(f["valid"].sum())
    zone_km2 = n_zone * cell_dx_m * cell_dy_m / 1e6
    return {
        "coast_km": float(coast_km),
        "bathtub_zone_km2": float(zone_km2),
        "reach_km": float(zone_km2 / coast_km) if coast_km > 0 else np.nan,
        "friction_m_per_km": float(np.median(friction_per_m[zone]) * 1000.0) if n_zone else np.nan,
        "river_pct": float(100.0 * f["river"].sum() / n_valid) if n_valid else np.nan,
        "lake_pct": float(100.0 * f["lake"].sum() / n_valid) if n_valid else np.nan,
        "overtop_coast_pct": float(100.0 * f["front"].sum() / f["coast"].sum()) if f["coast"].any() else np.nan,
        "enclosed_pct": float(100.0 * f["enclosed"].sum() / n_zone) if n_zone else np.nan,
    }


def forcing_drivers(peak_levels_m: np.ndarray, hydrographs_m: np.ndarray, hydrograph_dt_s: float) -> dict[str, float]:
    """Return the drivers of the tile's SFINCS forcing (its boundary points, owned area or not).

    surge_range_m: median over boundary points of the hydrograph's max - min (tide plus surge).
    peak_level_spread_m: max - min of the boundary peak levels. peak_hold_h: median over
    boundary points of the longest unbroken time the hydrograph stays above
    min + PEAK_HOLD_FRACTION x (max - min) (separate high tides do not add up; relative, so a
    microtidal sea is not near its peak all record long).
    """
    hold = [_longest_run(h >= np.nanmin(h) + PEAK_HOLD_FRACTION * np.ptp(h[np.isfinite(h)])) for h in hydrographs_m.T]
    return {"surge_range_m": float(np.median(np.nanmax(hydrographs_m, axis=0) - np.nanmin(hydrographs_m, axis=0))),
            "peak_level_spread_m": float(np.ptp(peak_levels_m)),
            "peak_hold_h": float(np.median(hold) * hydrograph_dt_s / 3600.0)}


def tile_drivers(dem_m: np.ndarray, mask: np.ndarray, friction_per_m: np.ndarray, peak_levels_m: np.ndarray,
                 hydrographs_m: np.ndarray, hydrograph_dt_s: float, cell_dx_m: float, cell_dy_m: float,
                 owned: np.ndarray | None = None) -> dict[str, float]:
    """Return one tile's drivers over the `owned` cells (default: the whole tile)."""
    fields = tile_fields(dem_m, mask, peak_levels_m)
    return {**area_drivers(fields, friction_per_m, cell_dx_m, cell_dy_m, owned),
            **forcing_drivers(peak_levels_m, hydrographs_m, hydrograph_dt_s)}


def _longest_run(flags: np.ndarray) -> int:
    padded = np.concatenate([[0], flags.astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[::2]).max()) if edges.size else 0


def _read_tile(tile_dir: Path) -> tuple[dict, object, object]:
    with retry_transient_io(rasterio.open, tile_dir / "inputs" / "dem.tif") as src:
        dem = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        dem /= DEM_SCALE
        lat = (src.bounds.top + src.bounds.bottom) / 2.0
        dx = abs(src.res[0]) * 111320.0 * np.cos(np.radians(lat))
        dy = abs(src.res[1]) * 110540.0
        transform, crs = src.transform, src.crs
    with retry_transient_io(rasterio.open, tile_dir / "inputs" / "mask.tif") as src:
        mask = src.read(1)
    with retry_transient_io(rasterio.open, tile_dir / "inputs" / "friction.tif") as src:
        friction = decode_friction_int16(src.read(1))
    levels = gpd.read_file(tile_dir / "inputs" / "boundaries_RP100_SLR_0.gpkg")[BOUNDARY_LEVEL_COLUMN]
    hydro = pd.read_csv(tile_dir / "sfincs_model" / "corrected_hydrographs.csv", index_col=0)  # index: elapsed_hr
    inputs = {"dem_m": dem, "mask": mask, "friction_per_m": friction,
              "peak_levels_m": levels.to_numpy(dtype=float) / DEM_SCALE,
              "hydrographs_m": hydro.to_numpy(dtype=float),
              "hydrograph_dt_s": float(np.diff(hydro.index.to_numpy(dtype=float)).mean() * 3600.0),
              "cell_dx_m": dx, "cell_dy_m": dy}
    return inputs, transform, crs


def _one(tile_dir: Path, footprints: gpd.GeoDataFrame) -> tuple[dict, dict]:
    tid = {"tile_id": int(tile_dir.name)}
    try:
        inputs, transform, crs = _read_tile(tile_dir)
        owned = core_mask(inputs["dem_m"].shape, transform, crs, tile_dir.name, footprints)
        fields = tile_fields(inputs["dem_m"], inputs["mask"], inputs["peak_levels_m"])
        forcing = forcing_drivers(inputs["peak_levels_m"], inputs["hydrographs_m"], inputs["hydrograph_dt_s"])
        area = partial(area_drivers, fields, inputs["friction_per_m"], inputs["cell_dx_m"], inputs["cell_dy_m"])
        return {**tid, **area(), **forcing}, {**tid, "owned_share": float(owned.mean()), **area(owned), **forcing}
    except (OSError, ValueError, KeyError) as e:
        print(f"tile {tile_dir.name}: drivers failed ({e!r})", flush=True)
        return tid, tid


def _part_paths(base: Path, name: str) -> list[Path]:
    return sorted(base.glob(f"{Path(name).stem}.part*of*.csv"))


def _complete(df: pd.DataFrame) -> pd.DataFrame:
    # Rows of the current driver set (a file written before a driver was added has none).
    if not set(DRIVER_COLUMNS) <= set(df.columns):
        return df.iloc[:0].reindex(columns=["tile_id", *DRIVER_COLUMNS])
    return df.dropna(subset=DRIVER_COLUMNS, how="all")


def read_drivers(base: Path, name: str = DRIVERS_NAME) -> pd.DataFrame:
    """Return the drivers of `name` in `base`: the main file and any shard part files, one row per tile."""
    frames = [_complete(pd.read_csv(p)) for p in [base / name, *_part_paths(base, name)] if p.exists()]
    if not frames:
        return pd.DataFrame(columns=["tile_id", *DRIVER_COLUMNS])
    return pd.concat(frames, ignore_index=True).drop_duplicates("tile_id").sort_values("tile_id")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    retry_transient_io(atomic_write, path, lambda f: df.to_csv(f, index=False), newline="")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--workers", type=int, default=None, help="default: every core this process may use")
    parser.add_argument("--shard", default=None, metavar="K/N", help="compute only every N-th missing tile from the K-th")
    args = parser.parse_args()
    args.workers = args.workers or available_cpus()

    base = read_root(Path(args.config)) / args.base_dir_name
    names = (DRIVERS_NAME, DRIVERS_OWNED_NAME)
    done = [read_drivers(base, n) for n in names]
    kept = set(done[0]["tile_id"].astype(int)) & set(done[1]["tile_id"].astype(int))
    done = [d[d["tile_id"].astype(int).isin(kept)] for d in done]
    scored = sorted({int(p.parent.parent.name) for p in base.glob(f"*/outputs/{RESULT_NAME}")} - kept)
    todo = [base / str(t) for t in shard(scored, args.shard)]
    print(f"{len(kept)} tile(s) already computed, {len(todo)} to compute"
          + (f" (shard {args.shard} of {len(scored)} missing)" if args.shard else ""), flush=True)
    footprints = load_footprints(base / "tile_selection.gpkg")
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(partial(_one, footprints=footprints), todo, chunksize=4))
    n_failed = 0
    for k, (name, old) in enumerate(zip(names, done)):
        new = pd.DataFrame([r[k] for r in rows]) if rows else old.iloc[:0]
        if args.shard:
            kk, nn = args.shard.split("/")
            path, out = base / f"{Path(name).stem}.part{kk}of{nn}.csv", new
        else:
            path, out = base / name, pd.concat([old, new], ignore_index=True).sort_values("tile_id")
        _write_csv(out, path)
        n_failed = max(n_failed, int(new.reindex(columns=DRIVER_COLUMNS).isna().all(axis=1).sum()) if len(new) else 0)
        print(f"Wrote {path} ({len(out)} tile(s))")
    if n_failed:
        print(f"{n_failed} tile(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
