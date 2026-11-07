"""Write a tile's SFINCS boundary forcing points and hydrographs from its station line.

Writes sfincs_model/matched_boundary_points.gpkg (EPSG:4326) and
corrected_hydrographs.csv (elapsed_hr + one column per point). The hydrograph shapes come from
the config's boundary_conditions.storm_tide_source: COAST-HG (RP100) or the storm-tide product's
RP100 hydrographs, the latter put on COAST-HG's time axis (peak at COAST_HG_PEAK_HR).

Usage:
    python build_line_forcing.py --tile-id 1122 --base-dir-name calibration_roughness_SFINCS \\
        --lines-gpkg <root>/calibration_roughness_SFINCS/boundary_lines/1122.gpkg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root, storm_tide_hydrograph_path  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

HYDROGRAPH_VARIABLE = "hydrograph_average_tide_signal"
PRODUCT_VARIABLE = "storm_tide_hydrograph"
FORCING_RP = 100
MAX_MATCH_DIST_DEG = 0.5  # the hydrograph stations are dense along real coastline
# COAST-HG: 894 10-min steps from 2000-01-01, every peak at step 447 (74.5 h). The product's
# hydrographs (peak at time 0) are cut to the same axis, so build_sfincs_tile's truncation
# window keeps the same span around the peak for both sources.
COAST_HG_START = np.datetime64("2000-01-01T00:00")
COAST_HG_PEAK_HR = 74.5
COAST_HG_STEPS = 894


def read_hydrographs(nc_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return (lon, lat, series (n, t) m above local MSL, times) of a COAST-HG or product file.

    Product: RP FORCING_RP, cut to COAST-HG's axis; samples outside a station's window (NaN)
    take the nearest finite value, the window's end level.
    """
    with retry_transient_io(xr.open_dataset, nc_path) as ds:
        lon, lat = ds.station_x_coordinate.values, ds.station_y_coordinate.values
        if PRODUCT_VARIABLE not in ds:
            return lon, lat, ds[HYDROGRAPH_VARIABLE].values, pd.DatetimeIndex(ds.time.values)
        offsets = ds.time.values.astype("timedelta64[m]")
        first = int(np.flatnonzero(offsets == -np.timedelta64(int(COAST_HG_PEAK_HR * 60), "m"))[0])
        sel = slice(first, first + COAST_HG_STEPS)
        series = ds[PRODUCT_VARIABLE].sel(return_period=FORCING_RP).isel(time=sel).values.astype(np.float64)
        offsets = offsets[sel]
    series = pd.DataFrame(series.T).ffill().bfill().to_numpy().T
    times = pd.DatetimeIndex(COAST_HG_START + (offsets - offsets[0]))
    return lon, lat, series, times


def build_line_forcing(lines_gpkg: Path, hydrograph_nc: Path, tile_dir: Path | None = None,
                       root: Path | None = None) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Build forcing points (EPSG:4326) and matching hydrographs for a boundary line.

    One point per used station at its node (node_lon/node_lat: the station
    moved ~1 km out to sea onto the line); SFINCS interpolates between points.
    Hydrograph: the nearest station's RP100 hydrograph in `hydrograph_nc` (`read_hydrographs`),
    shifted so its peak equals the station's MDT-corrected RP100 level:
    offset = boundary_value - hydrograph_max. With `tile_dir`/`root`, also adds
    "interior" points for non-node stations by water that stays active in SFINCS.
    """
    st = retry_transient_io(gpd.read_file, lines_gpkg, layer="stations")
    st = st[st["status"] == "used"].reset_index(drop=True)
    if st.empty:
        raise ValueError(f"{lines_gpkg}: no used station on the boundary line")
    hg_lon, hg_lat, hg, times = read_hydrographs(hydrograph_nc)

    cols, rows = {}, []
    for k, r in st.iterrows():
        # Match on the real station position, not the node.
        d2 = (hg_lon - r.geometry.x) ** 2 + (hg_lat - r.geometry.y) ** 2
        j = int(np.argmin(d2))
        dist_deg = float(np.sqrt(d2[j]))
        if dist_deg > MAX_MATCH_DIST_DEG:
            print(f"  station {r['station']}: nearest hydrograph station {dist_deg:.3f} deg away - dropped")
            continue
        series = hg[j].astype(np.float64)
        offset = float(r["rp100_level_m"]) - float(np.nanmax(series))
        i = len(cols)
        cols[i] = series + offset
        rows.append({"station": int(r["station"]), "station_index": int(r["station_index"]),
                     "rp100_level_m": float(r["rp100_level_m"]), "hydrograph_station": j,
                     "hydrograph_dist_deg": round(dist_deg, 4), "mdt_offset_m": round(offset, 4),
                     "node_shift_km": float(r["node_shift_km"]), "kind": "line",
                     "geometry": gpd.points_from_xy([r["node_lon"]], [r["node_lat"]])[0]})
        print(f"  station {r['station']:>3} -> hydrograph {j} ({dist_deg:.3f} deg): level {r['rp100_level_m']:.3f} m, "
              f"offset {offset:+.3f} m, node shifted {r['node_shift_km']:.2f} km")
    if tile_dir is not None and root is not None:
        from connected_forcing import interior_points
        inner = interior_points(lines_gpkg, tile_dir, root, np.column_stack([hg_lon, hg_lat]))
        for _, r in inner.iterrows():
            series = hg[int(r["hydrograph_station"])].astype(np.float64)
            offset = float(r["rp100_level_m"]) - float(np.nanmax(series))
            cols[len(cols)] = series + offset
            rows.append({"station": int(r["station"]), "station_index": int(r["station_index"]),
                         "rp100_level_m": float(r["rp100_level_m"]), "hydrograph_station": int(r["hydrograph_station"]),
                         "hydrograph_dist_deg": float(r["hydrograph_dist_deg"]), "mdt_offset_m": round(offset, 4),
                         "node_shift_km": 0.0, "kind": "interior", "geometry": r.geometry})
            print(f"  station {r['station']:>3} (interior) -> hydrograph {int(r['hydrograph_station'])}: "
                  f"level {r['rp100_level_m']:.3f} m")
    points = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    elapsed_hr = (times - times[0]).total_seconds() / 3600.0
    hydro = pd.DataFrame(cols, index=times)
    hydro.insert(0, "elapsed_hr", elapsed_hr)
    return points, hydro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--lines-gpkg", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1]
                                                 / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()

    root = read_root(Path(args.config))
    hydrograph_nc = storm_tide_hydrograph_path(Path(args.config))
    print(f"hydrographs: {hydrograph_nc}")
    out_dir = root / args.base_dir_name / args.tile_id / "sfincs_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    points, hydro = build_line_forcing(Path(args.lines_gpkg), hydrograph_nc,
                                       tile_dir=root / args.base_dir_name / args.tile_id, root=root)
    points.to_file(out_dir / "matched_boundary_points.gpkg", driver="GPKG")
    hydro.to_csv(out_dir / "corrected_hydrographs.csv", index=False)
    print(f"{len(points)} forcing point(s) ({int((points['kind'] == 'interior').sum())} interior); "
          f"peaks {points.rp100_level_m.min():.2f}-{points.rp100_level_m.max():.2f} m")
    print(f"Wrote {out_dir / 'matched_boundary_points.gpkg'}")
    print(f"Wrote {out_dir / 'corrected_hydrographs.csv'}")


if __name__ == "__main__":
    main()
