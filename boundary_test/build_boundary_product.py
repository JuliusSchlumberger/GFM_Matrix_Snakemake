"""Plan §5.C (H1): tide-replaced boundary levels L_RP at each test tile's boundary cells.

Per station: S_RP = COAST-RP level - MHWS_GTSM (GTSM tidal_elevation). S_RP and the station
MDT are interpolated to the boundary cells by IDW (k = KNN) over water-path distance on the
coarse ocean-connectivity raster (`connectivity_resolution_m`), with haversine distance where a
cell has no water path to any station. Then L_RP = S_RP + MHWS_FES(cell) + MDT (+ SLR, 0 here).
`seed_prod_rp100_m` is the production seed level; `l_prod_rp*_m` a haversine IDW of COAST-RP + MDT
(close to production, which also masks by ocean component) for the other RPs.
Station MDT is read back from the tile's production boundaries (level - COAST-RP RP100).

Writes <out-dir>/boundary_product/<tile>.parquet, one row per boundary cell, in seed order.

Usage:
    python boundary_test/build_boundary_product.py --out-dir .cleanup_baseline/boundary_test \
        --tiles-csv .cleanup_baseline/boundary_test/test_tiles_provisional.csv
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.transform
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from along_water import along_water_distances, along_water_idw  # noqa: E402
from tiles import _scan_mask_dir, mosaic_water_fraction_downsampled  # noqa: E402

DATA_ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
BASE = DATA_ROOT / "calibration_roughness_SFINCS"
MASK_DIR = DATA_ROOT / "inputs" / "DeltaDTM_masks"
RPS = (10, 100, 1000)
KNN = 15  # simulation.flooding.knn
WATER_RES_M = 500.0  # boundary_conditions.connectivity_resolution_m
WATER_FRACTION = 0.5  # boundary_conditions.connectivity_water_fraction_threshold
PAD_DEG = 0.05
EARTH_RADIUS_M = 6_371_000.0
MATCH_M = 100.0


def _nearest(src_lonlat: np.ndarray, query_lonlat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d, i = BallTree(np.radians(src_lonlat[:, ::-1]), metric="haversine").query(np.radians(query_lonlat[:, ::-1]), k=1)
    return i[:, 0], d[:, 0] * EARTH_RADIUS_M


def haversine_matrix(a_lonlat: np.ndarray, b_lonlat: np.ndarray) -> np.ndarray:
    a, b = np.radians(a_lonlat), np.radians(b_lonlat)
    dlat = b[None, :, 1] - a[:, None, 1]
    dlon = b[None, :, 0] - a[:, None, 0]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[:, None, 1]) * np.cos(b[None, :, 1]) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(h))


def station_table(tile_id: str, out: Path) -> pd.DataFrame:
    tiles = pd.read_csv(out / "test_station_tiles.csv", dtype={"tile_id": str})
    st = tiles[tiles.tile_id == tile_id].drop_duplicates("gtsm_index").reset_index(drop=True)
    inv = pd.read_parquet(out / "stations.parquet")
    idx, _ = _nearest(inv[["lon", "lat"]].to_numpy(), st[["lon", "lat"]].to_numpy())
    st = pd.concat([st, inv.iloc[idx][[f"rp{rp}_m" for rp in RPS]].reset_index(drop=True)], axis=1)
    tide = pd.read_parquet(out / "tide_stats_gtsm.parquet").set_index("gtsm_index")
    st["mhws_gtsm_m"] = st.gtsm_index.map(tide.gtsm_mhws_m)
    prod = gpd.read_file(BASE / tile_id / "inputs" / "boundaries_RP100_SLR_0.gpkg")
    p_ll = np.column_stack([prod.geometry.x, prod.geometry.y])
    j, d = _nearest(p_ll, st[["lon", "lat"]].to_numpy())
    level = prod["SLR_0"].to_numpy(dtype=float) / 100.0
    st["mdt_m"] = np.where(d <= MATCH_M, level[j] - st.rp100_m, np.nan)
    return st.dropna(subset=["mhws_gtsm_m", "mdt_m"]).reset_index(drop=True)


def water_raster(lonlat: np.ndarray) -> tuple[np.ndarray, object]:
    bbox = (lonlat[:, 0].min() - PAD_DEG, lonlat[:, 1].min() - PAD_DEG,
            lonlat[:, 0].max() + PAD_DEG, lonlat[:, 1].max() + PAD_DEG)
    frac, transform = mosaic_water_fraction_downsampled(bbox, _scan_mask_dir(MASK_DIR), WATER_RES_M, ocean_code=1)
    return frac >= WATER_FRACTION, transform


def build_tile(tile_id: str, out: Path) -> pd.DataFrame:
    st = station_table(tile_id, out)
    cells = pd.read_parquet(out / "fes_cell_node_map" / f"{tile_id}.parquet")
    fes = pd.read_parquet(out / "tide_stats_fes2022.parquet")
    fes = fes[(fes.tile_id == tile_id) & fes.valid]
    key = lambda lon, lat: (np.round(lon, 6).astype(str) + "_" + np.round(lat, 6).astype(str))  # noqa: E731
    cells["mhws_fes_m"] = key(cells.node_lon, cells.node_lat).map(
        pd.Series(fes.mhws_m.to_numpy(), index=key(fes.node_lon, fes.node_lat)))
    s_ll, c_ll = st[["lon", "lat"]].to_numpy(), cells[["lon", "lat"]].to_numpy()

    water, transform = water_raster(np.vstack([s_ll, c_ll]))
    cell_m = abs(transform.a) * 111_320.0 * np.cos(np.radians(c_ll[:, 1].mean()))
    rc = lambda ll: np.column_stack(rasterio.transform.rowcol(transform, ll[:, 0], ll[:, 1]))  # noqa: E731
    d_water = along_water_distances(water, cell_m, rc(s_ll), rc(c_ll))
    d_hav = haversine_matrix(s_ll, c_ll)

    cells["mdt_m"], fallback = along_water_idw(d_water, st.mdt_m.to_numpy(), KNN, d_hav)
    cells["used_haversine_fallback"] = fallback
    for rp in RPS:
        s_rp = (st[f"rp{rp}_m"] - st.mhws_gtsm_m).to_numpy()
        cells[f"s_rp{rp}_m"], _ = along_water_idw(d_water, s_rp, KNN, d_hav)
        cells[f"l_rp{rp}_m"] = cells[f"s_rp{rp}_m"] + cells.mhws_fes_m + cells.mdt_m
        cells[f"l_prod_rp{rp}_m"], _ = along_water_idw(np.full_like(d_hav, np.inf),
                                                       (st[f"rp{rp}_m"] + st.mdt_m).to_numpy(), KNN, d_hav)
    cells.attrs["n_stations"] = len(st)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tiles-csv", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    (out / "boundary_product").mkdir(exist_ok=True)
    for tile_id in pd.read_csv(args.tiles_csv, dtype={"tile_id": str}).tile_id:
        cells = build_tile(tile_id, out)
        cells.to_parquet(out / "boundary_product" / f"{tile_id}.parquet")
        d = cells.l_rp100_m - cells.seed_prod_rp100_m
        print(f"tile {tile_id}: {len(cells)} cells, {cells.attrs['n_stations']} stations, "
              f"fallback {cells.used_haversine_fallback.mean():.0%}, L_RP100 - production: "
              f"median {d.median():+.2f} m, range {d.min():+.2f} / {d.max():+.2f}", flush=True)


if __name__ == "__main__":
    main()
