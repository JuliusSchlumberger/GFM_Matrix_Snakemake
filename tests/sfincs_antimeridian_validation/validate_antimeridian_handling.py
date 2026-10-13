"""Regression test for the water-level forcing step near +-180 deg longitude.

hydromt_sfincs's water_level.create() selects the forcing locations with the
polygonized waterlevel-boundary cells, reprojected to the locations' CRS and
merged (union_all). With EPSG:4326 locations, the rotated-grid cell polygons
of a tile at the antimeridian reproject to invalid geometry and GEOS raises
"TopologyException: side location conflict" (5 tiles of the final
calibration run, 2026-10-13). build_sfincs_tile.py therefore passes the
locations in the model's own UTM CRS, where nothing is reprojected.

This test rebuilds tile 1359's grid + boundary-line mask (no subgrid - a few
seconds) from its calibration_roughness_SFINCS files and checks that
water_level.create() succeeds with UTM locations and keeps every forcing
point. Skips (not fails) without the P: share or the tile's files.

Usage:
    python validate_antimeridian_handling.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_sfincs_tile import MAIN_RES_M_DEFAULT, _create_mask  # noqa: E402
from gfm_config import read_root  # noqa: E402

TILE_ID = "1359"  # tile reaches 180.18 deg E
BASE_DIR_NAME = "calibration_roughness_SFINCS"


def test_water_level_create_near_antimeridian() -> None:
    print(f"=== water_level.create() with UTM locations on tile {TILE_ID} (crosses 180 deg) ===")
    try:
        root = read_root(REPO_ROOT / "snakemake_workflow" / "config" / "config.yml")
    except Exception:
        root = None
    tile = root / BASE_DIR_NAME / TILE_ID if root else None
    needed = ["inputs/tile_geometry.gpkg", f"boundary_line_{TILE_ID}.gpkg",
              "sfincs_model/matched_boundary_points.gpkg", "sfincs_model/corrected_hydrographs.csv"]
    if tile is None or not all((tile / f).exists() for f in needed):
        print(f"SKIP: tile {TILE_ID}'s files under {BASE_DIR_NAME} are not available here.\n")
        return

    from hydromt_sfincs import SfincsModel

    tile_gdf = gpd.read_file(tile / "inputs" / "tile_geometry.gpkg")
    points = gpd.read_file(tile / "sfincs_model" / "matched_boundary_points.gpkg")
    hydro = pd.read_csv(tile / "sfincs_model" / "corrected_hydrographs.csv")
    tmp = Path(tempfile.mkdtemp(prefix="antimeridian_wl_"))
    try:
        sf = SfincsModel(root=str(tmp), mode="w+")
        sf.grid.create_from_region(region={"geom": tile_gdf}, res=MAIN_RES_M_DEFAULT, crs="utm", rotated=True)
        _create_mask(sf, tile_gdf, tile / f"boundary_line_{TILE_ID}.gpkg", MAIN_RES_M_DEFAULT)
        cols = [c for c in hydro.columns if c != "elapsed_hr"]
        times = pd.date_range("2026-01-01", periods=len(hydro), freq="10min")
        wl = hydro[cols].copy()
        wl.index = times
        wl.columns = range(len(cols))
        for key, t in (("tref", times[0]), ("tstart", times[0]), ("tstop", times[-1])):
            sf.config.set(key, t.to_pydatetime())
        locations = points.to_crs(sf.crs)
        locations["index"] = range(len(cols))
        sf.water_level.create(timeseries=wl, locations=locations, buffer=25_000.0)
        n_kept = len(sf.water_level.gdf)
        assert n_kept == len(points), f"{n_kept} of {len(points)} forcing points kept"
        print(f"PASS: water_level.create() succeeded, {n_kept}/{len(points)} forcing points kept\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    test_water_level_create_near_antimeridian()


if __name__ == "__main__":
    main()
