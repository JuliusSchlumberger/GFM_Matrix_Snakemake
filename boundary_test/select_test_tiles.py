"""Stratify the SFINCS calibration tiles by tidal regime and pick the boundary-test tiles (plan §2).

Pool: tiles of the 1000-tile friction sweep with a calibration result, a suitable boundary line
and a passed pre-flight. Per tile, its boundary stations (inputs/boundaries_RP100_SLR_0.gpkg)
give the spring high water (FES MHWS once available, else the COAST-HG proxy from
station_inventory), the TC share and the along-tile tide amplification (max / min spring high
water). Estuary = river fraction >= ESTUARY_RIVER_FRAC or amplification >= ESTUARY_AMPLIFICATION.
Picks per stratum prefer tiles with a GESLA site and keep tiles >= MIN_SPACING_DEG apart.

Usage:
    python boundary_test/select_test_tiles.py --out-dir .cleanup_baseline/boundary_test [--mhws-col spring_hw_proxy_m]
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

BASE = Path("P:/11212688-004-global-floodmaps/modelling/calibration_roughness_SFINCS")
STRATA = {"microtidal": 4, "mesotidal": 4, "macrotidal_open": 3, "macrotidal_estuary": 4, "tc_dominated": 3}
MICRO_M, MACRO_M = 0.5, 1.5
TC_SHARE = 0.5
ESTUARY_RIVER_FRAC = 0.01
ESTUARY_AMPLIFICATION = 1.2
OPEN_RIVER_FRAC = 0.003
OPEN_AMPLIFICATION = 1.1
MIN_SPACING_DEG = 2.0
MATCH_KM = 0.1
EARTH_RADIUS_KM = 6371.0
# Held-out validation regions (plan §2, §6): region bboxes of data_catalog_validation.yml, plus
# California for SF Bay; a tile whose centre lies within HELD_OUT_PAD_DEG of one is excluded.
HELD_OUT_BBOXES = {"wales": (-5.45, 51.30, -2.55, 53.50), "scotland": (-8.35, 54.54, -0.70, 60.87),
                   "england": (-6.37, 49.88, 1.77, 55.82), "california": (-124.5, 32.5, -114.1, 42.0)}
HELD_OUT_PAD_DEG = 0.5


def _boundary_lonlat(tile_id: str) -> np.ndarray:
    path = BASE / tile_id / "inputs" / "boundaries_RP100_SLR_0.gpkg"
    if not path.exists():
        return np.zeros((0, 2))
    g = gpd.read_file(path)
    return np.column_stack([g.geometry.x.to_numpy(), g.geometry.y.to_numpy()])


def tile_table(stations: pd.DataFrame, gesla: pd.DataFrame, mhws_col: str) -> pd.DataFrame:
    meta = pd.read_csv(BASE / "tile_selection_metadata.csv", dtype={"tile_id": str})
    meta = meta[meta.line_suitable.astype(bool) & meta.preflight_ok.astype(bool)]
    meta = meta[[(BASE / t / "outputs" / "calibration_result.json").exists() for t in meta.tile_id]]
    with ThreadPoolExecutor(16) as pool:
        lonlats = list(pool.map(_boundary_lonlat, meta.tile_id))
    tree = BallTree(np.radians(stations[["lat", "lon"]].to_numpy()), metric="haversine")
    gesla_tree = BallTree(np.radians(gesla[["lat", "lon"]].to_numpy()), metric="haversine")
    rows = []
    for (_, m), ll in zip(meta.iterrows(), lonlats):
        rec = {"tile_id": m.tile_id, "lon": m.lon, "lat": m.lat, "river_frac": m.river_frac,
               "n_boundary_stations": len(ll)}
        if len(ll):
            d, i = tree.query(np.radians(ll[:, ::-1]), k=1)
            s = stations.iloc[i[d[:, 0] * EARTH_RADIUS_KM <= MATCH_KM, 0]]
            hw = s[mhws_col].to_numpy()
            rec.update(mhws_median_m=float(np.median(hw)), mhws_min_m=float(hw.min()),
                       mhws_max_m=float(hw.max()), tc_share=float(s.tc_dominated.mean()),
                       amplification=float(hw.max() / max(hw.min(), 0.2)),
                       rp100_median_m=float(s.rp100_m.median()))
        near = gesla_tree.query_radius(np.radians([[m.lat, m.lon]]), r=50.0 / EARTH_RADIUS_KM)[0]
        rec["gesla_sites_50km"] = json.dumps(sorted(gesla.file_name.iloc[near].tolist()))
        rec["n_gesla_50km"] = len(near)
        rows.append(rec)
    df = pd.DataFrame(rows).dropna(subset=["mhws_median_m"])
    df["held_out_region"] = df.apply(held_out_region, axis=1)
    df["stratum"] = df.apply(assign_stratum, axis=1).where(df.held_out_region.isna())
    per_tile = pd.read_csv(BASE / "friction_sweep_per_tile_metrics.csv", dtype={"tile_id": str})
    csi = per_tile[per_tile.friction_scale_factor == 12.0].set_index("tile_id").CSI
    df["csi_fsf12"] = df.tile_id.map(csi)
    return df


def held_out_region(r) -> str | None:
    p = HELD_OUT_PAD_DEG
    for name, (x0, y0, x1, y1) in HELD_OUT_BBOXES.items():
        if x0 - p <= r.lon <= x1 + p and y0 - p <= r.lat <= y1 + p:
            return name
    return None


def assign_stratum(r) -> str | None:
    if r.tc_share > TC_SHARE:
        return "tc_dominated"
    if r.mhws_median_m < MICRO_M:
        return "microtidal"
    if r.mhws_median_m <= MACRO_M:
        return "mesotidal"
    if r.river_frac >= ESTUARY_RIVER_FRAC or r.amplification >= ESTUARY_AMPLIFICATION:
        return "macrotidal_estuary"
    if r.river_frac < OPEN_RIVER_FRAC and r.amplification < OPEN_AMPLIFICATION:
        return "macrotidal_open"
    return None


def pick(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    picks = []
    for stratum, n in STRATA.items():
        pool = df[df.stratum == stratum].assign(r=lambda d: rng.random(len(d)))
        pool = pool.sort_values(["n_gesla_50km", "r"], ascending=[False, True])
        chosen = []
        for _, row in pool.iterrows():
            if all(np.hypot(row.lon - c.lon, row.lat - c.lat) >= MIN_SPACING_DEG for c in chosen):
                chosen.append(row)
            if len(chosen) == n:
                break
        picks += chosen
    return pd.DataFrame(picks).drop(columns="r")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mhws-col", default="spring_hw_proxy_m",
                        help="stations.parquet column used as MHWS (FES column once computed)")
    args = parser.parse_args()
    out = Path(args.out_dir)
    stations = pd.read_parquet(out / "stations.parquet")
    gesla = pd.read_parquet(out / "gesla_sites.parquet")
    df = tile_table(stations, gesla, args.mhws_col)
    df.to_csv(out / "tile_strata_all.csv", index=False)
    picks = pick(df)
    picks.to_csv(out / "test_tiles_provisional.csv", index=False)
    print(df.stratum.value_counts(dropna=False).to_string())
    print(picks[["tile_id", "stratum", "lon", "lat", "mhws_median_m", "amplification", "river_frac",
                 "tc_share", "n_gesla_50km", "csi_fsf12"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
