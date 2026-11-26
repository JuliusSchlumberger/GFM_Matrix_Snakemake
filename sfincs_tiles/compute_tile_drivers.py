"""Compute per-tile candidate drivers of model-vs-SFINCS disagreement into tile_drivers.csv.

Run under the gfm env:
    python compute_tile_drivers.py --base-dir-name calibration_roughness_SFINCS [--workers 4]
Tiles already in tile_drivers.csv are kept (their inputs do not change between runs) unless the
file lacks a driver column, then all are recomputed.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_results import RESULT_NAME  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flood_model import LAND_CODE, open_sea_mask  # noqa: E402
from rasters import decode_friction_int16  # noqa: E402

DRIVERS_NAME = "tile_drivers.csv"
OCEAN_CODE = 1
LAKE_CODE = 2
DEM_SCALE = 100.0  # dem.tif is int16 cm
BOUNDARY_LEVEL_COLUMN = "SLR_0"  # boundaries_RP100_SLR_0.gpkg, cm
WEIR_COEFFICIENT = 1.7  # broad-crested weir, q = C h^1.5 (m2/s per m of crest)
DRIVER_COLUMNS = ["coast_km", "bathtub_zone_km2", "reach_km", "friction_m_per_km", "surge_range_m",
                  "peak_level_spread_m", "lake_pct", "overtop_coast_pct", "storage_to_inflow_ratio", "abs_lat_deg"]
_S8 = np.ones((3, 3), dtype=bool)


def tile_drivers(dem_m: np.ndarray, mask: np.ndarray, friction_per_m: np.ndarray, peak_levels_m: np.ndarray,
                 hydrographs_m: np.ndarray, hydrograph_dt_s: float, cell_dx_m: float, cell_dy_m: float,
                 ) -> dict[str, float]:
    """Return one tile's drivers.

    reach_km: area of land below the median boundary peak level and 8-connected to the open
    sea, per km of coastline (land cells touching the open sea), i.e. the connected bathtub's
    mean inland reach. friction_m_per_km: median unscaled land-cover attenuation over that land.
    surge_range_m: median over boundary points of the forcing hydrograph's max - min.
    peak_level_spread_m: max - min of the boundary peak levels along the tile.
    lake_pct: lake cells per valid tile cell. overtop_coast_pct: coastline below the peak level.
    storage_to_inflow_ratio: water volume of that connected land at the peak level over the
    volume a broad-crested weir along the overtoppable coastline (crest at its median ground
    level) passes during the median hydrograph. The weir inflow is an upper bound (whole crest
    at full head), so the ratio ranks tiles; it is not a fill criterion.
    """
    land = mask == LAND_CODE
    sea = open_sea_mask(mask.astype(np.int8), OCEAN_CODE)
    coast_land = land & ndimage.binary_dilation(sea, _S8)
    cell_m = np.sqrt(cell_dx_m * cell_dy_m)
    coast_km = coast_land.sum() * cell_m / 1000.0
    peak_m = np.median(peak_levels_m)
    low = land & (dem_m < peak_m)
    labels, _ = ndimage.label(low | sea, _S8)
    seaward = np.unique(labels[sea])
    zone = low & np.isin(labels, seaward[seaward > 0])
    zone_km2 = zone.sum() * cell_dx_m * cell_dy_m / 1e6
    front = coast_land & (dem_m < peak_m)
    storage_m3 = np.nansum(peak_m - dem_m[zone]) * cell_dx_m * cell_dy_m
    if front.any():
        crest_m = np.nanmedian(dem_m[front])
        over_m = np.clip(hydrographs_m - crest_m, 0.0, None)
        inflow_m3 = np.median(np.nansum(WEIR_COEFFICIENT * over_m**1.5, axis=0)) * hydrograph_dt_s * front.sum() * cell_m
    else:
        inflow_m3 = 0.0
    valid = np.isfinite(dem_m) | (mask != LAND_CODE)
    return {
        "coast_km": float(coast_km),
        "bathtub_zone_km2": float(zone_km2),
        "reach_km": float(zone_km2 / coast_km) if coast_km > 0 else np.nan,
        "friction_m_per_km": float(np.median(friction_per_m[zone]) * 1000.0) if zone.any() else np.nan,
        "surge_range_m": float(np.median(np.nanmax(hydrographs_m, axis=0) - np.nanmin(hydrographs_m, axis=0))),
        "peak_level_spread_m": float(np.ptp(peak_levels_m)),
        "lake_pct": float(100.0 * (mask == LAKE_CODE).sum() / valid.sum()),
        "overtop_coast_pct": float(100.0 * front.sum() / coast_land.sum()) if coast_land.any() else np.nan,
        "storage_to_inflow_ratio": float(storage_m3 / inflow_m3) if inflow_m3 > 0 else np.nan,
    }


def _read_tile(tile_dir: Path) -> tuple[dict, float]:
    with retry_transient_io(rasterio.open, tile_dir / "inputs" / "dem.tif") as src:
        dem = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        dem /= DEM_SCALE
        lat = (src.bounds.top + src.bounds.bottom) / 2.0
        dx = abs(src.res[0]) * 111320.0 * np.cos(np.radians(lat))
        dy = abs(src.res[1]) * 110540.0
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
    return inputs, lat


def _one(tile_dir: Path) -> dict:
    try:
        inputs, lat = _read_tile(tile_dir)
        return {"tile_id": int(tile_dir.name), "abs_lat_deg": abs(lat), **tile_drivers(**inputs)}
    except (OSError, ValueError, KeyError) as e:
        print(f"tile {tile_dir.name}: drivers failed ({e!r})", flush=True)
        return {"tile_id": int(tile_dir.name)}


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    base = read_root(Path(args.config)) / args.base_dir_name
    out_path = base / DRIVERS_NAME
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame({"tile_id": []})
    if not set(DRIVER_COLUMNS) <= set(done.columns):  # written before a driver was added: recompute all
        done = done.iloc[:0]
    done = done.dropna(subset=DRIVER_COLUMNS, how="all")
    todo = sorted({p.parent.parent for p in base.glob(f"*/outputs/{RESULT_NAME}")} - {base / str(int(t)) for t in done["tile_id"]},
                  key=lambda p: int(p.name))
    print(f"{len(done)} tile(s) already in {DRIVERS_NAME}, {len(todo)} to compute", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(_one, todo, chunksize=4))
    meta = pd.read_csv(base / "tile_selection_metadata.csv")[["tile_id", "river_frac"]]
    new = pd.DataFrame(rows).merge(meta, on="tile_id", how="left") if rows else done.iloc[:0]
    out = pd.concat([done, new], ignore_index=True).sort_values("tile_id")
    retry_transient_io(out.to_csv, out_path, index=False)
    n_failed = int(new.reindex(columns=DRIVER_COLUMNS).isna().all(axis=1).sum()) if len(new) else 0
    print(f"Wrote {out_path} ({len(out)} tile(s), {n_failed} failed)")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
