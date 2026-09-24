"""Run the production eikonal flood solver directly on SFINCS's own subgrid
inputs (dep_subgrid.tif / manning_subgrid.tif, real UTM metres) instead of
the eikonal model's own separately-built lon/lat DeltaDTM grid - isolating
the PHYSICS/NUMERICS disagreement between the two models from every
grid/projection/elevation-source difference between their normally-separate
input pipelines.

Real, concrete benefits of running on this shared grid (see conversation):
  - SFINCS's subgrid IS a true UTM metre grid - genuinely isotropic (every
    step the same real distance in every direction) - so the eikonal
    solver's own known anisotropy bug (no real-distance scaling on a
    lon/lat grid, investigated and reverted earlier this session) simply
    doesn't apply here; no fix needed for this comparison.
  - The subgrid's own resolution (30m at the current 120m/4 settings)
    matches simulation.flooding.friction_scale_factor=30's own calibration
    target - no recalibration needed.
  - Reusing flood_model.py's own effective_dem() (flattens ANY non-land
    cell - ocean, lake, AND river - to 0m) for free fixes the exact
    "deep lake bathymetry contaminates a coarse cell's own hypsometric
    table" issue found investigating tile 37's depth disagreement, on the
    eikonal side at least.

Boundary seeding uses the eikonal solver's own DIRECT seed_rows/seed_cols/
seed_values path (the same one hop>=1 hinterland tiles already use in
production) - NOT the boundaries=/coastline_mask+haversine-IDW path, since
_idw_seed_values hard-codes a haversine (lon/lat) distance metric that
would be wrong on this projected UTM grid. Seeding is computed here with a
PLANAR Euclidean IDW instead (build_boundary_forcing.idw_interpolate_to_grid,
already built for exactly this - SFINCS's own zsini uses the identical
approach), from the SAME boundaries_RP100_SLR_0.gpkg values already used
for every other comparison this session (deliberately still the old,
pre-MDT-fix boundary forcing, for consistency with the existing SFINCS
results being compared against).

No eikonal solver code changes needed at all - flood_depth_dense/
effective_dem/coastline_mask are all pure array functions with no lon/lat
assumption; only _idw_seed_values (not used here) is lon/lat-specific.

Usage:
    python run_eikonal_on_sfincs_subgrid.py --tile-id 37
    python run_eikonal_on_sfincs_subgrid.py --config <resolved_config.yml> --tile-id 37
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.warp import Resampling, reproject
from scipy import ndimage

_STRUCTURE_8 = np.ones((3, 3), dtype=bool)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from flood_model import flood_depth_dense  # noqa: E402
from rasters import WATERDEPTH_NODATA_INT16, encode_waterdepth_cm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_boundary_forcing import idw_interpolate_to_grid  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"
LAND_CODE = 0
OCEAN_CODE = 1
RIVER_CODE = 3
NODATA_CODE = 4  # native mask.tif has no real coverage here - see build_inputs_from_sfincs_subgrid
FRICTION_SCALE_FACTOR_DEFAULT = 30.0  # matches simulation.flooding.friction_scale_factor in config.yml
DEFAULT_FRICTION = 0.002  # matches simulation.flooding.default_friction in config.yml
MAX_ROUNDS_DEFAULT = 200
WATERLEVEL_EPSILON_M_DEFAULT = 0.03


def build_inputs_from_sfincs_subgrid(
    sfincs_dir: Path, native_mask_path: Path,
    friction_scale_factor: float = FRICTION_SCALE_FACTOR_DEFAULT,
    default_friction: float = DEFAULT_FRICTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, rasterio.Affine, str]:
    """(dem, mask, friction, transform, crs) on SFINCS's own subgrid UTM grid.

    dem: dep_subgrid.tif, NaN cells (outside the real subgrid footprint,
        e.g. tile buffer padding) filled with DEM_NODATA_M's own 99.0m
        convention (definitely-dry sentinel - matches rasters.DEM_NODATA_M).
    mask: native mask.tif reprojected onto this grid (nearest-neighbour,
        the same pre-reprojection-workaround pattern build_sfincs_tile.py
        already uses). Cells with no real coverage (value outside
        {0,1,2,3}) get their own NODATA_CODE rather than being folded into
        land or ocean.

        NOTE this is NOT a leftover DeltaDTM nodata signal - mask.tif
        itself (confirmed directly, e.g. tile 1757) contains ONLY real
        {0,1,2,3} values; extract_dem_mask() (src/rasters.py) already
        resolves DeltaDTM's own raw 255 nodata sentinel to land(0) well
        upstream of this file. The gaps reprojecting here to a real
        "nothing here" value are a genuine geometry artefact: a tile's
        native mask.tif is a rectangle in lon/lat, but reprojected to UTM
        it becomes a CURVED shape (meridian convergence - e.g. tile
        1757's own north edge, spanning 3 degrees of longitude, sits
        ~4.4km further north at its east end than its west end).
        SFINCS's create_from_region(..., crs="utm") builds an
        axis-aligned UTM rectangle that must fully contain that curve, so
        it overshoots at the opposite corners into territory the tile's
        own native mask.tif never covered at all - genuinely outside the
        tile, not unclassified deep ocean within it. Confirmed real,
        substantial impact on wide/high-latitude tiles (212/253 tiles
        affected by >5 km2 of this, ~56,800 km2 total across the batch;
        negligible on typical ~1x1 degree tiles). A blanket "nodata ->
        land" fallback flooded these corners as spurious dry land; a
        blanket "nodata -> ocean" fallback would be an equally unjustified
        guess the other way. Simplest and most honest: give it its own
        code, excluded by construction from every existing
        `mask == LAND_CODE` reporting/bathtub restriction elsewhere in
        this module, without asserting a classification this data was
        never able to support. effective_dem() (flood_model.py) already
        flattens ANY non-LAND_CODE cell to 0m, so NODATA_CODE gets that
        same treatment automatically - it behaves like open water for
        propagation purposes without being counted as real ocean for
        coastline/seeding logic (which checks mask == OCEAN_CODE
        specifically, not "not land").
    friction: manning_subgrid.tif / 100 * friction_scale_factor, NaN cells
        filled with default_friction (matching config.yml's own
        simulation.flooding.default_friction) before scaling, same
        friction_scale_factor as production (see module docstring).
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


def sfincs_domain_coastline_mask(mask: np.ndarray, ocean_code: int = OCEAN_CODE, river_code: int | None = RIVER_CODE) -> np.ndarray:
    """Ocean cells within 1px of land (or river) - deliberately WITHOUT
    flood_model.coastline_mask's own edge-connectivity requirement.

    That requirement exists to reject isolated inland ponds DeltaDTM
    sometimes miscodes as ocean, and relies on GFM's own tile-generation
    process (src/tile_chunking.py) deliberately building each eikonal tile
    with a real "wet edge" - a property SFINCS's own grid never has (it's
    just a rectangular UTM bounding box around the tile polygon plus a
    buffer margin, unrelated to tile_chunking's own wet-edge convention).

    Real, confirmed bug (2026-09, tile 37): reusing coastline_mask's
    edge-connectivity check on the SFINCS subgrid's own domain rejected
    84% of the tile's real coastline (12,829 native coastline cells ->
    only 2,011 survived reprojection, clustered in one small corner,
    because the SFINCS bounding box's own edges mostly cut through land
    far from the coast). This simpler formula matches what
    build_sfincs_tile.py's own _ocean_polygon_wgs84 already does to place
    SFINCS's own real boundary forcing - already validated safe for this
    exact domain shape (used successfully across every SFINCS tile this
    session), since the domain is already tightly built around one real
    coastal tile, not the whole-globe scope the edge-connectivity check
    was originally guarding against.
    """
    ocean = mask == ocean_code
    landlike = mask == LAND_CODE
    if river_code is not None:
        landlike = landlike | (mask == river_code)
    dilated_landlike = ndimage.binary_dilation(landlike, structure=_STRUCTURE_8)
    return dilated_landlike & ocean


def compute_planar_idw_seeds(
    mask: np.ndarray, transform: rasterio.Affine, crs: str,
    boundaries_path: Path, variable: str, k: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Coastline-cell seed_rows/seed_cols/seed_values via PLANAR (Euclidean,
    UTM-metre) IDW from real boundary stations - the projected-grid
    counterpart of flood_model._idw_seed_values (which hard-codes
    haversine/lon-lat and can't be reused directly here). Returns None if
    the boundaries file is empty or no coastline cells exist.

    k=15 matches production's own simulation.flooding.knn - genuinely
    caps each cell's IDW to its k NEAREST stations, not "however many
    stations this tile happens to have". Real, confirmed bug (2026-09):
    an earlier k=20 default, combined with a tile that happened to have
    exactly 20 stations, meant min(k, len(station_values))==20 used EVERY
    station for EVERY coastline cell regardless of distance - collapsing a
    real 1.47-3.12m station spread down to a ~1.73-1.78m tile-wide average
    before it ever reached the solver, silently under-flooding the tile.
    """
    boundaries = retry_transient_io(gpd.read_file, boundaries_path)
    if boundaries.empty:
        return None
    station_values_m = boundaries[variable].to_numpy(dtype=np.float64) / 100.0  # cm -> m
    boundaries_utm = boundaries.to_crs(crs)
    station_x = boundaries_utm.geometry.x.to_numpy()
    station_y = boundaries_utm.geometry.y.to_numpy()

    coast = sfincs_domain_coastline_mask(mask, ocean_code=OCEAN_CODE, river_code=RIVER_CODE)
    seed_rows, seed_cols = np.nonzero(coast)
    if len(seed_rows) == 0:
        return None

    xs, ys = rasterio.transform.xy(transform, seed_rows, seed_cols)
    xs, ys = np.asarray(xs), np.asarray(ys)
    seed_values = idw_interpolate_to_grid(
        station_x, station_y, station_values_m, xs, ys, k=min(k, len(station_values_m)),
    )
    return seed_rows.astype(np.int64), seed_cols.astype(np.int64), seed_values.astype(np.float32)


def compute_bathtub_depth(dem: np.ndarray, mask: np.ndarray, max_waterlevel_m: float) -> np.ndarray:
    """Naive/unconstrained "bathtub" baseline: every LAND cell below
    `max_waterlevel_m` is flooded to exactly that level - NO connectivity,
    NO friction, NO propagation at all, the simplest possible flood model.
    Restricted to land (mask==LAND_CODE) to match the eikonal/SFINCS
    outputs' own land-only depth convention, so all three are directly
    comparable. `max_waterlevel_m` is a single scalar (the tile's own
    highest real boundary station value) - deliberately NOT spatially
    interpolated the way the real solves are, since the whole point of
    this baseline is to be the simplest possible reference, not a third
    variant of the same IDW-seeded approach.
    """
    depth = np.maximum(np.float32(max_waterlevel_m) - dem, np.float32(0.0))
    return np.where(mask == LAND_CODE, depth, np.float32(0.0)).astype(np.float32)


def run_eikonal_on_sfincs_subgrid(
    tile_id: str, root: Path, friction_scale_factor: float = FRICTION_SCALE_FACTOR_DEFAULT,
    max_rounds: int = MAX_ROUNDS_DEFAULT, waterlevel_epsilon_m: float = WATERLEVEL_EPSILON_M_DEFAULT,
    base_dir_name: str = "validation_sfincs_v2",
) -> tuple[np.ndarray, dict, dict] | None:
    """Returns (waterdepth, diagnostics, grid_info) on the SFINCS subgrid's
    own UTM grid, or None if this tile has no usable boundary forcing."""
    sfincs_dir = root / base_dir_name / tile_id / "sfincs_model"
    native_mask_path = root / base_dir_name / tile_id / "inputs" / "mask.tif"
    boundaries_path = root / base_dir_name / tile_id / "inputs" / f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg"

    dem, mask, friction, transform, crs = build_inputs_from_sfincs_subgrid(
        sfincs_dir, native_mask_path, friction_scale_factor,
    )

    seeds = compute_planar_idw_seeds(mask, transform, crs, boundaries_path, WATERLEVEL_NAME)
    if seeds is None:
        return None
    seed_rows, seed_cols, seed_values = seeds

    waterdepth, diagnostics = flood_depth_dense(
        dem, mask, friction, transform,
        seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
        max_rounds=max_rounds, waterlevel_epsilon_m=waterlevel_epsilon_m,
    )
    return waterdepth, diagnostics, {"transform": transform, "crs": crs}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=None, help="config.yml or a resolved_config.yml; defaults to the main repo config.yml")
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2", help="output root directory name under paths.root (default: validation_sfincs_v2)")
    parser.add_argument(
        "--models", nargs="+", choices=["bathtub", "eikonal"], default=["bathtub", "eikonal"],
        help="which of this script's two models to (re-)run (2026-09-24, user direction: "
             "e.g. --models bathtub to skip eikonal entirely while its own sweep-count "
             "calibration is still in progress). Default: both.",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=None,
        help=f"eikonal round cap forwarded to run_eikonal_on_sfincs_subgrid() (default: "
             f"{MAX_ROUNDS_DEFAULT} - 2026-09-24, user direction: intended to eventually take "
             f"the value the sweep-budget calibration study settles on, see tests/ "
             f"test_sweep_budget_calibration.py)",
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        root = Path(config["paths"]["root"])
    else:
        _repo_root = Path(__file__).resolve().parent.parent
        root = read_root(_repo_root / "snakemake_workflow" / "config" / "config.yml")

    out_dir = root / args.base_dir_name / args.tile_id / "outputs"
    bathtub_output_path = out_dir / f"bathtub_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}.tif"
    eikonal_output_path = out_dir / f"eikonal_on_subgrid_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}.tif"

    sfincs_dir = root / args.base_dir_name / args.tile_id / "sfincs_model"
    native_mask_path = root / args.base_dir_name / args.tile_id / "inputs" / "mask.tif"
    boundaries_path = root / args.base_dir_name / args.tile_id / "inputs" / f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg"

    run_bathtub = "bathtub" in args.models
    run_eikonal = "eikonal" in args.models

    already_done = (not run_bathtub or bathtub_output_path.exists()) and (not run_eikonal or eikonal_output_path.exists())
    if already_done:
        print(f"tile {args.tile_id}: already done ({'+'.join(args.models)}), skipping")
        return
    if not boundaries_path.exists():
        print(f"tile {args.tile_id}: SKIP - no {boundaries_path}")
        return
    boundaries = retry_transient_io(gpd.read_file, boundaries_path)
    if boundaries.empty:
        print(f"tile {args.tile_id}: SKIP - boundaries file is empty (no station for this tile)")
        return

    # -- bathtub: cheap, computed and written FIRST, ahead of the (much slower) eikonal solve --
    if not run_bathtub:
        print(f"tile {args.tile_id}: bathtub not requested (--models={args.models}), skipping")
    elif bathtub_output_path.exists():
        print(f"tile {args.tile_id}: bathtub already done, skipping")
    else:
        dem, mask, _friction, transform, crs = build_inputs_from_sfincs_subgrid(sfincs_dir, native_mask_path)
        max_waterlevel_m = float(boundaries[WATERLEVEL_NAME].to_numpy(dtype=np.float64).max() / 100.0)
        bathtub_depth = compute_bathtub_depth(dem, mask, max_waterlevel_m)
        _write_waterdepth(bathtub_depth, transform, crs, bathtub_output_path)
        n_flooded_bt = int((bathtub_depth > 0).sum())
        print(f"tile {args.tile_id}: bathtub - max_waterlevel={max_waterlevel_m:.3f}m, "
              f"{n_flooded_bt} flooded cell(s) of {bathtub_depth.size}, max depth {bathtub_depth.max():.3f} m")
        print(f"Wrote {bathtub_output_path}")

    # -- eikonal (the real, friction/propagation-aware solve) --
    if not run_eikonal:
        print(f"tile {args.tile_id}: eikonal not requested (--models={args.models}), skipping")
        return
    if eikonal_output_path.exists():
        print(f"tile {args.tile_id}: eikonal already done, skipping")
        return
    eikonal_kwargs = {"base_dir_name": args.base_dir_name}
    if args.max_rounds is not None:
        eikonal_kwargs["max_rounds"] = args.max_rounds
    result = run_eikonal_on_sfincs_subgrid(args.tile_id, root, **eikonal_kwargs)
    if result is None:
        print(f"tile {args.tile_id}: SKIP eikonal - no usable coastline in this domain")
        return
    waterdepth, diagnostics, grid_info = result
    _write_waterdepth(waterdepth, grid_info["transform"], grid_info["crs"], eikonal_output_path)

    n_flooded = int((waterdepth > 0).sum())
    print(f"tile {args.tile_id}: eikonal - {n_flooded} flooded cell(s) of {waterdepth.size}, "
          f"max depth {waterdepth.max():.3f} m, diagnostics={diagnostics}")
    print(f"Wrote {eikonal_output_path}")


if __name__ == "__main__":
    main()
