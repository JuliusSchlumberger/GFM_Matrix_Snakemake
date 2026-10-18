"""Boundary forcing for a SFINCS tile built on a station boundary line
(build_station_boundary_lines.py): one forcing point per COAST-RP station
node on the line, each with its own COAST-HG hydrograph - no resampling
along the line, SFINCS interpolates between the points.

Per used station (status "used" in the line's stations layer):
  - forcing point: the station's node (node_lon/node_lat - the station moved
    ~1 km out to sea, where the line passes);
  - hydrograph: the nearest COAST-HG station's average-tide hydrograph,
    shifted so its peak equals the station's COAST-RP RP100 level
    (rp100_level_m, MDT-corrected): offset = boundary_value - hydrograph_max.

Writes the two files build_sfincs_tile.py reads for its forcing step:
  sfincs_model/matched_boundary_points.gpkg  (points at the nodes, EPSG:4326)
  sfincs_model/corrected_hydrographs.csv     (elapsed_hr + one column per point)

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
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

COAST_HG_NC = "inputs/COAST_HG/COAST-HG_RP100.nc"
HYDROGRAPH_VARIABLE = "hydrograph_average_tide_signal"
MAX_MATCH_DIST_DEG = 0.5  # nearest COAST-HG station must be this close (COAST-HG is dense along real coastline)


def build_line_forcing(lines_gpkg: Path, coast_hg_nc: Path, tile_dir: Path | None = None,
                       root: Path | None = None) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Build forcing points and matching hydrographs for a tile's boundary line.

    One point per used station node on the line ("line" kind), plus - when
    `tile_dir`/`root` are given - one per station inside the tile that is
    not a line node but lies by water that stays active in SFINCS
    ("interior" kind, via connected_forcing.interior_points).

    Parameters
    ----------
    lines_gpkg : Path
        build_station_boundary_lines.py output (boundary_line/stations layers).
    coast_hg_nc : Path
        COAST-HG hydrograph NetCDF (COAST-HG_RP100.nc).
    tile_dir : Path, optional
        Tile's base directory (containing inputs/); enables interior points.
    root : Path, optional
        Config root directory; required together with `tile_dir`.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, pandas.DataFrame]
        Forcing points (EPSG:4326) and the matching hydrographs (elapsed_hr
        plus one column per point, in the same order).

    Raises
    ------
    ValueError
        If `lines_gpkg` has no used station.
    """
    st = retry_transient_io(gpd.read_file, lines_gpkg, layer="stations")
    st = st[st["status"] == "used"].reset_index(drop=True)
    if st.empty:
        raise ValueError(f"{lines_gpkg}: no used station on the boundary line")
    with retry_transient_io(xr.open_dataset, coast_hg_nc) as ds:
        hg_lon = ds.station_x_coordinate.values
        hg_lat = ds.station_y_coordinate.values
        hg = ds[HYDROGRAPH_VARIABLE].values  # (station, time)
        times = pd.DatetimeIndex(ds.time.values)

    cols, rows = {}, []
    for k, r in st.iterrows():
        # nearest COAST-HG station to the real COAST-RP station position
        d2 = (hg_lon - r.geometry.x) ** 2 + (hg_lat - r.geometry.y) ** 2
        j = int(np.argmin(d2))
        dist_deg = float(np.sqrt(d2[j]))
        if dist_deg > MAX_MATCH_DIST_DEG:
            print(f"  station {r['station']}: nearest COAST-HG station {dist_deg:.3f} deg away - dropped")
            continue
        series = hg[j].astype(np.float64)
        offset = float(r["rp100_level_m"]) - float(np.nanmax(series))
        i = len(cols)
        cols[i] = series + offset
        rows.append({"station": int(r["station"]), "station_index": int(r["station_index"]),
                     "rp100_level_m": float(r["rp100_level_m"]), "coast_hg_station": j,
                     "coast_hg_dist_deg": round(dist_deg, 4), "mdt_offset_m": round(offset, 4),
                     "node_shift_km": float(r["node_shift_km"]), "kind": "line",
                     "geometry": gpd.points_from_xy([r["node_lon"]], [r["node_lat"]])[0]})
        print(f"  station {r['station']:>3} -> COAST-HG {j} ({dist_deg:.3f} deg): level {r['rp100_level_m']:.3f} m, "
              f"offset {offset:+.3f} m, node shifted {r['node_shift_km']:.2f} km")
    if tile_dir is not None and root is not None:
        from connected_forcing import interior_points
        inner = interior_points(lines_gpkg, tile_dir, root, np.column_stack([hg_lon, hg_lat]))
        for _, r in inner.iterrows():
            series = hg[int(r["coast_hg_station"])].astype(np.float64)
            offset = float(r["rp100_level_m"]) - float(np.nanmax(series))
            cols[len(cols)] = series + offset
            rows.append({"station": int(r["station"]), "station_index": int(r["station_index"]),
                         "rp100_level_m": float(r["rp100_level_m"]), "coast_hg_station": int(r["coast_hg_station"]),
                         "coast_hg_dist_deg": float(r["coast_hg_dist_deg"]), "mdt_offset_m": round(offset, 4),
                         "node_shift_km": 0.0, "kind": "interior", "geometry": r.geometry})
            print(f"  station {r['station']:>3} (interior) -> COAST-HG {int(r['coast_hg_station'])}: "
                  f"level {r['rp100_level_m']:.3f} m")
    points = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    elapsed_hr = (times - times[0]).total_seconds() / 3600.0
    hydro = pd.DataFrame(cols, index=times)
    hydro.insert(0, "elapsed_hr", elapsed_hr)
    return points, hydro


def main() -> None:
    """Parse CLI arguments and write the tile's matched boundary points and
    corrected hydrographs (matched_boundary_points.gpkg, corrected_hydrographs.csv)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument("--lines-gpkg", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1]
                                                 / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()

    root = read_root(Path(args.config))
    out_dir = root / args.base_dir_name / args.tile_id / "sfincs_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    points, hydro = build_line_forcing(Path(args.lines_gpkg), root / COAST_HG_NC,
                                       tile_dir=root / args.base_dir_name / args.tile_id, root=root)
    points.to_file(out_dir / "matched_boundary_points.gpkg", driver="GPKG")
    hydro.to_csv(out_dir / "corrected_hydrographs.csv", index=False)
    print(f"{len(points)} forcing point(s) ({int((points['kind'] == 'interior').sum())} interior); "
          f"peaks {points.rp100_level_m.min():.2f}-{points.rp100_level_m.max():.2f} m")
    print(f"Wrote {out_dir / 'matched_boundary_points.gpkg'}")
    print(f"Wrote {out_dir / 'corrected_hydrographs.csv'}")


if __name__ == "__main__":
    main()
