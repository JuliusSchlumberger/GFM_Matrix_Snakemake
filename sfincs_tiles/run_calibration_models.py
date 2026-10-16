"""Bathtub + the eikonal friction sweep for one tile of the final SFINCS
calibration (calibration_roughness_SFINCS), on SFINCS's own 30 m subgrid -
the same grid SFINCS's hmax_subgrid.tif is compared on.

Compared with the eikonal runner of the earlier calibration studies:
  - solver settings come from the main config's simulation.flooding block
    (max_rounds, waterlevel_epsilon_m, obstacle_coupling.enabled /
    max_outer_iterations / outer_convergence_pct, knn) - nothing hard-coded;
  - production coastline + production station weighting: the coastline is
    flood_model.coastline_mask's rule (ocean within 1 px of land/river, in the
    ocean body connected to the domain edge - on this rotated grid "edge" also
    means the no-data corners outside the lon/lat tile), and each coastline
    cell gets the inverse-distance-squared average of its knn nearest
    CONNECTED stations (flood_model.idw_seed_values_at - the same function
    production uses, fed the cells' lon/lat);
  - inputs and seeds are built once, then every friction scale factor is
    solved in the same process;
  - outputs are always tagged by friction only:
      outputs/eikonal_on_subgrid_waterdepth_RP100_SLR_0_fsf<X>.tif
      outputs/bathtub_waterdepth_RP100_SLR_0.tif
      outputs/eikonal_solver_diagnostics.json   ({"fsf_<X>": {solver diagnostics, runtime_s}})

Usage:
    python run_calibration_models.py --config <resolved_config.yml> --base-dir-name calibration_roughness_SFINCS --tile-id 1122
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from flood_model import flood_depth_dense, idw_seed_values_at  # noqa: E402
from rasters import WATERDEPTH_NODATA_INT16, decode_waterlevel_cm, encode_waterdepth_cm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import (  # noqa: E402
    FRICTION_SCALE_FACTORS, RETURN_PERIOD, WATERLEVEL_NAME, bathtub_path, eikonal_path,
)
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)
LAND_CODE, OCEAN_CODE = 0, 1
NODATA_CODE = 4  # subgrid cells the native lon/lat mask.tif does not cover - see build_inputs_from_sfincs_subgrid


def build_inputs_from_sfincs_subgrid(
    sfincs_dir: Path, native_mask_path: Path,
    friction_scale_factor: float, default_friction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine, str]:
    """(dem, mask, friction, transform, crs) on SFINCS's own subgrid UTM grid.

    dem: dep_subgrid.tif, NaN cells filled with rasters.DEM_NODATA_M's 99.0m
        dry sentinel.
    mask: native mask.tif reprojected onto this grid (nearest-neighbour).
        A tile's native mask.tif is a lon/lat rectangle, which becomes a
        curved shape in UTM (meridian convergence); SFINCS's own UTM
        rectangle overshoots that curve at the corners, so cells with no
        real coverage there get NODATA_CODE rather than land or ocean.
    friction: manning_subgrid.tif / 100 * friction_scale_factor, NaN cells
        filled with default_friction * 100 before scaling.
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


def compute_bathtub_depth(dem: np.ndarray, mask: np.ndarray, max_waterlevel_m: float) -> np.ndarray:
    """Naive "bathtub" baseline: every land cell below `max_waterlevel_m`
    (a single scalar - the tile's highest real boundary station value,
    not spatially interpolated) floods to exactly that level. No
    connectivity, friction, or propagation.
    """
    depth = np.maximum(np.float32(max_waterlevel_m) - dem, np.float32(0.0))
    return np.where(mask == LAND_CODE, depth, np.float32(0.0)).astype(np.float32)


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
    """flood_model.coastline_mask on the rotated subgrid: ocean cells within 1 px
    of land (or river) in an ocean body that reaches the domain edge - the
    array edge, or the no-data corners where the rotated UTM rectangle sticks
    out of the lon/lat tile (that boundary IS the tile edge on this grid)."""
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
    """(dem, mask) as handed to flood_depth_dense: NODATA_CODE cells become
    dry land at the 99m sentinel. flood_depth_dense's effective_dem zeroes
    every non-land cell, so left as-is the no-data frame around the rotated
    grid is a 0m channel touching the ocean that carries water along the
    domain border into any below-level land beside it (tile 631: the
    Gran Bajo de San Julian, cut off from the sea by a 20-30m rim, flooded
    via the frame). The original mask is still what subgrid_coastline_mask
    and the seeds use - there the frame marks the domain edge."""
    nodata = mask == NODATA_CODE
    return (np.where(nodata, np.float32(99.0), dem).astype(dem.dtype),
            np.where(nodata, np.int8(LAND_CODE), mask).astype(mask.dtype))


def production_seeds(mask, transform, crs, boundaries: gpd.GeoDataFrame, knn: int, river_code: int):
    """(rows, cols, values) on the subgrid's coastline, production weighting."""
    coast = subgrid_coastline_mask(mask, OCEAN_CODE, river_code)
    rows, cols = np.nonzero(coast)
    if len(rows) == 0:
        return None
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(np.asarray(xs), np.asarray(ys))
    st = boundaries.to_crs("EPSG:4326")
    values = decode_waterlevel_cm(st[WATERLEVEL_NAME].to_numpy())
    seeds = idw_seed_values_at(np.column_stack([lon, lat]), rows, cols,
                               np.column_stack([st.geometry.x, st.geometry.y]), values, knn, mask, OCEAN_CODE)
    return rows.astype(np.int64), cols.astype(np.int64), seeds.astype(np.float32)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", required=True, help="config.yml or the base dir's resolved_config.yml")
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", default=list(FRICTION_SCALE_FACTORS))
    parser.add_argument("--models", nargs="+", choices=["bathtub", "eikonal"], default=["bathtub", "eikonal"])
    parser.add_argument("--overwrite", action="store_true", help="redo outputs that already exist")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root = read_root(Path(args.config))  # paths.root, or a sibling config_local.yml's
    flooding = config["simulation"]["flooding"]
    oc = flooding.get("obstacle_coupling", {})
    river_code = int(config["tile_generation"]["river_code"])
    tile_dir = root / args.base_dir_name / args.tile_id
    out_dir = tile_dir / "outputs"
    sfincs_dir = tile_dir / "sfincs_model"
    native_mask_path = tile_dir / "inputs" / "mask.tif"
    boundaries = retry_transient_io(gpd.read_file, tile_dir / "inputs" / f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg")
    if boundaries.empty:
        raise SystemExit(f"tile {args.tile_id}: no COAST-RP station in boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg")

    dem, mask, base_friction, transform, crs = build_inputs_from_sfincs_subgrid(
        sfincs_dir, native_mask_path, friction_scale_factor=1.0, default_friction=float(flooding["default_friction"]))

    if "bathtub" in args.models and (args.overwrite or not bathtub_path(out_dir).exists()):
        level = float(decode_waterlevel_cm(boundaries[WATERLEVEL_NAME].to_numpy()).max())
        _write_waterdepth(compute_bathtub_depth(dem, mask, level), transform, crs, bathtub_path(out_dir))
        print(f"tile {args.tile_id}: bathtub at {level:.3f} m -> {bathtub_path(out_dir).name}", flush=True)

    if "eikonal" not in args.models:
        return
    todo = [f for f in args.friction_scale_factors if args.overwrite or not eikonal_path(out_dir, f).exists()]
    if not todo:
        print(f"tile {args.tile_id}: eikonal sweep already done")
        return
    seeds = production_seeds(mask, transform, crs, boundaries, int(flooding["knn"]), river_code)
    if seeds is None:
        raise SystemExit(f"tile {args.tile_id}: no coastline cell on the subgrid - nothing to seed")
    seed_rows, seed_cols, seed_values = seeds
    print(f"tile {args.tile_id}: {len(seed_rows)} coastline seed cell(s), levels "
          f"{seed_values.min():.2f}-{seed_values.max():.2f} m (knn={flooding['knn']}, connected stations)", flush=True)

    solver_dem, solver_mask = solver_inputs(dem, mask)
    diag_path = out_dir / "eikonal_solver_diagnostics.json"
    all_diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    for fsf in todo:
        t0 = time.monotonic()
        waterdepth, diagnostics = flood_depth_dense(
            solver_dem, solver_mask, base_friction * np.float32(fsf), transform,
            seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
            ocean_code=OCEAN_CODE, river_code=river_code,
            obstacle_coupling=bool(oc.get("enabled", False)),
            max_outer_iterations=int(oc.get("max_outer_iterations", 3)),
            outer_convergence_pct=float(oc.get("outer_convergence_pct", 0.01)),
            max_rounds=int(flooding["max_rounds"]),
            waterlevel_epsilon_m=float(flooding["waterlevel_epsilon_m"]),
        )
        _write_waterdepth(waterdepth, transform, crs, eikonal_path(out_dir, fsf))
        runtime = time.monotonic() - t0
        all_diag[f"fsf_{fsf:g}"] = {**_jsonable(diagnostics), "runtime_s": round(runtime, 1)}
        diag_path.write_text(json.dumps(all_diag, indent=1))
        print(f"tile {args.tile_id}: eikonal fsf {fsf:g} - {int((waterdepth > 0).sum())} flooded cell(s), "
              f"{runtime:.0f} s -> {eikonal_path(out_dir, fsf).name}", flush=True)


if __name__ == "__main__":
    main()
