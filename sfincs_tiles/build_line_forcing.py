"""Write a tile's SFINCS boundary forcing points and hydrographs along its boundary line.

Points every FORCING_SPACING_M along the line's segments near the tile, plus the interior points
(connected_forcing.py). Each point's peak is production's eikonal seed level at its nearest
coastline cell (`production_seed_levels`), so SFINCS and the eikonal models get the same forcing;
the station line only decides where SFINCS's boundary lies. Writes
sfincs_model/matched_boundary_points.gpkg (EPSG:4326) and corrected_hydrographs.csv (elapsed_hr +
one column per point). The hydrograph shapes come from the config's
boundary_conditions.storm_tide_source: COAST-HG (RP100) or the storm-tide product's RP100
hydrographs, the latter put on COAST-HG's time axis (peak at COAST_HG_PEAK_HR).

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
import rasterio
import xarray as xr
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import RETURN_PERIOD, WATERLEVEL_NAME  # noqa: E402
from gfm_config import read_root, storm_tide_hydrograph_path  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flood_model import coastline_mask, idw_seed_values_at  # noqa: E402

HYDROGRAPH_VARIABLE = "hydrograph_average_tide_signal"
PRODUCT_VARIABLE = "storm_tide_hydrograph"
FORCING_RP = 100
MAX_MATCH_KM = 55.0  # ~0.5 deg of latitude; the hydrograph stations are dense along real coastline
KM_PER_DEG = 111.2
# COAST-HG: 894 10-min steps from 2000-01-01, every peak at step 447 (74.5 h). The product's
# hydrographs (peak at time 0) are cut to the same axis, so build_sfincs_tile's truncation
# window keeps the same span around the peak for both sources.
COAST_HG_START = np.datetime64("2000-01-01T00:00")
COAST_HG_PEAK_HR = 74.5
COAST_HG_STEPS = 894
# Finer than the product's coastline nodes (1/30 deg, ~3.7 km), so SFINCS's interpolation between
# boundary points does not smooth over the seed levels' own spatial variation.
FORCING_SPACING_M = 2000.0
# Line samples this far outside the tile polygon still sit next to its boundary cells.
TILE_MARGIN_M = 5000.0
OCEAN_CODE = 1


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


def production_seed_levels(tile_dir: Path, seed_rule: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ((n, 2) lon/lat, (n,) level_m) of production's eikonal seeds on the tile's native mask.

    As flood_model.flood_depth_dense seeds a production tile: coastline_mask cells, each the IDW
    (k = seed_rule["knn"], weights distance^-seed_rule["idw_power"]) of every row of the tile's
    boundaries gpkg (stations and coastline nodes) in its own ocean component.
    """
    with retry_transient_io(rasterio.open, tile_dir / "inputs" / "mask.tif") as src:
        mask, transform = src.read(1), src.transform
    rows, cols = np.nonzero(coastline_mask(mask, ocean_code=OCEAN_CODE, river_code=int(seed_rule["river_code"])))
    if not len(rows):
        raise ValueError(f"{tile_dir}: no coastline cell in inputs/mask.tif - nothing to take the forcing from")
    stations_gpkg = tile_dir / "inputs" / f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg"
    st = retry_transient_io(gpd.read_file, stations_gpkg)
    if st.empty:
        raise ValueError(f"{stations_gpkg}: no station")
    # cell centres, as rasterio.transform.xy
    c, r = cols + 0.5, rows + 0.5
    cells = np.column_stack([transform.c + transform.a * c + transform.b * r,
                             transform.f + transform.d * c + transform.e * r])
    # int16 cm -> float32 m as rasters.decode_waterlevel_cm (rasters needs the gfm env's hydromt)
    station_levels = st[WATERLEVEL_NAME].to_numpy().astype(np.float32) / np.float32(100.0)
    levels = idw_seed_values_at(cells, rows, cols, np.column_stack([st.geometry.x, st.geometry.y]),
                                station_levels, int(seed_rule["knn"]), mask, OCEAN_CODE,
                                float(seed_rule["idw_power"]))
    return cells, levels


def level_at_nearest_seed(points_lonlat: np.ndarray, seed_lonlat: np.ndarray,
                          seed_levels: np.ndarray) -> np.ndarray:
    """Return the level of each point's nearest seed (longitude scaled by cos(latitude))."""
    scale = np.array([np.cos(np.radians(np.mean(seed_lonlat[:, 1]))), 1.0])
    _, j = cKDTree(seed_lonlat * scale).query(points_lonlat * scale)
    return seed_levels[j]


def line_samples(lines_gpkg: Path, tile_geometry_gpkg: Path) -> gpd.GeoDataFrame:
    """Return points (EPSG:4326) at most FORCING_SPACING_M apart along the line near the tile.

    Each segment part within TILE_MARGIN_M of the tile polygon is sampled at both ends and evenly
    in between; `segment_kind` keeps the segment's kind (coastline, tile_edge, ...).
    """
    seg = retry_transient_io(gpd.read_file, lines_gpkg, layer="segments")
    tile = retry_transient_io(gpd.read_file, tile_geometry_gpkg).to_crs("EPSG:4326")
    utm = tile.estimate_utm_crs()
    near = tile.to_crs(utm).geometry.iloc[0].buffer(TILE_MARGIN_M)
    pts, kinds = [], []
    for kind, geom in zip(seg["kind"], seg.to_crs(utm).geometry):
        clipped = geom.intersection(near)
        for part in getattr(clipped, "geoms", [clipped]):
            if part.geom_type != "LineString" or part.length == 0:
                continue
            n = int(np.ceil(part.length / FORCING_SPACING_M))
            pts += [part.interpolate(d) for d in np.linspace(0.0, part.length, n + 1)]
            kinds += [kind] * (n + 1)
    out = gpd.GeoDataFrame({"segment_kind": kinds}, geometry=pts, crs=utm)
    # Segments share their end nodes; a doubled bnd point would weigh twice in SFINCS's interpolation.
    key = pd.Series(list(zip(out.geometry.x.round(), out.geometry.y.round())))
    return out[~key.duplicated().to_numpy()].to_crs("EPSG:4326").reset_index(drop=True)


def attach_hydrographs(points: gpd.GeoDataFrame, hydrograph_nc: Path, knn: int = 1,
                       idw_power: float = 3.0) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Give each point (EPSG:4326, `rp100_level_m`) a hydrograph peaking at its level.

    Shape: the point's `hydrograph_station` where set; else the IDW blend (weights
    distance^-idw_power, longitude scaled by cos(latitude)) of its `knn` nearest hydrograph
    stations within MAX_MATCH_KM, points without one dropped. Blending shapes taken relative to
    their own peak keeps the blend's peak at 0, as every hydrograph peaks at the same time
    (COAST_HG_PEAK_HR), and makes the shape vary smoothly between neighbouring points instead of
    jumping where the nearest station changes. The shape is then shifted, not scaled, to the level.
    """
    hg_lon, hg_lat, hg, times = read_hydrographs(hydrograph_nc)
    shapes = hg.astype(np.float64) - np.nanmax(hg, axis=1, keepdims=True)
    points = points.reset_index(drop=True).copy()
    if "hydrograph_station" not in points.columns:
        points["hydrograph_station"] = np.nan
    blended = np.empty((len(points), shapes.shape[1]), dtype=np.float64)
    fixed = points["hydrograph_station"].notna().to_numpy()
    blended[fixed] = shapes[points.loc[fixed, "hydrograph_station"].to_numpy(dtype=int)]
    keep = np.ones(len(points), dtype=bool)
    if (~fixed).any():
        xy = np.column_stack([points.geometry.x, points.geometry.y])[~fixed]
        scale = np.array([np.cos(np.radians(xy[:, 1].mean())), 1.0])
        k = min(knn, len(hg_lon))
        d, j = cKDTree(np.column_stack([hg_lon, hg_lat]) * scale).query(xy * scale, k=k)
        d, j = d.reshape(len(xy), k) * KM_PER_DEG, j.reshape(len(xy), k)
        near = d <= MAX_MATCH_KM
        w = np.where(near, np.maximum(d, 1e-6) ** -idw_power, 0.0)
        has = near.any(axis=1)
        w[has] /= w[has].sum(axis=1, keepdims=True)
        blended[~fixed] = np.einsum("pk,pkt->pt", w, shapes[j])
        points.loc[~fixed, "hydrograph_station"] = j[:, 0]
        points.loc[~fixed, "hydrograph_dist_km"] = np.round(d[:, 0], 1)
        points.loc[~fixed, "n_hydrographs"] = near.sum(axis=1)
        keep[~fixed] = has
    if not keep.all():
        print(f"  {int((~keep).sum())} point(s) without a hydrograph station within {MAX_MATCH_KM:.0f} km - dropped")
    points, blended = points[keep].reset_index(drop=True), blended[keep]
    if points.empty:
        raise ValueError("no forcing point has a hydrograph station nearby")
    points["hydrograph_station"] = points["hydrograph_station"].astype(int)
    level = points["rp100_level_m"].to_numpy(dtype=np.float64)
    nearest = hg[points["hydrograph_station"].to_numpy()].astype(np.float64)
    points["mdt_offset_m"] = np.round(level - np.nanmax(nearest, axis=1), 4)  # vs the nearest station's own peak
    hydro = pd.DataFrame(blended.T + level, index=times)
    hydro.insert(0, "elapsed_hr", (times - times[0]).total_seconds() / 3600.0)
    return points, hydro


def build_line_forcing(lines_gpkg: Path, hydrograph_nc: Path, tile_dir: Path, root: Path,
                       seed_rule: dict) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Build forcing points (EPSG:4326) and matching hydrographs for a tile's boundary line.

    `seed_rule`: production's seeding parameters, knn and idw_power (config simulation.flooding)
    and river_code (tile_generation).
    """
    from connected_forcing import interior_points
    line = line_samples(lines_gpkg, tile_dir / "inputs" / "tile_geometry.gpkg")
    if line.empty:
        raise ValueError(f"{lines_gpkg}: no boundary-line segment near the tile")
    line = line.assign(kind="line", station=-1, station_index=-1)
    hg_lon, hg_lat, _, _ = read_hydrographs(hydrograph_nc)
    inner = interior_points(lines_gpkg, tile_dir, root, np.column_stack([hg_lon, hg_lat]))
    points = gpd.GeoDataFrame(pd.concat([line, inner.assign(segment_kind="")], ignore_index=True),
                              geometry="geometry", crs="EPSG:4326")
    seed_lonlat, seed_levels = production_seed_levels(tile_dir, seed_rule)
    points["rp100_level_m"] = level_at_nearest_seed(
        np.column_stack([points.geometry.x, points.geometry.y]), seed_lonlat, seed_levels)
    return attach_hydrographs(points, hydrograph_nc, knn=int(seed_rule["knn"]),
                              idw_power=float(seed_rule["idw_power"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--lines-gpkg", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1]
                                                 / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()

    root = read_root(Path(args.config))
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    seed_rule = {"knn": config["simulation"]["flooding"]["knn"],
                 "idw_power": config["simulation"]["flooding"]["idw_power"],
                 "river_code": config["tile_generation"]["river_code"]}
    hydrograph_nc = storm_tide_hydrograph_path(Path(args.config))
    print(f"hydrographs: {hydrograph_nc}")
    out_dir = root / args.base_dir_name / args.tile_id / "sfincs_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    points, hydro = build_line_forcing(Path(args.lines_gpkg), hydrograph_nc,
                                       root / args.base_dir_name / args.tile_id, root, seed_rule)
    points.to_file(out_dir / "matched_boundary_points.gpkg", driver="GPKG")
    hydro.to_csv(out_dir / "corrected_hydrographs.csv", index=False)
    print(f"{len(points)} forcing point(s) ({int((points['kind'] == 'interior').sum())} interior); "
          f"peaks {points.rp100_level_m.min():.2f}-{points.rp100_level_m.max():.2f} m")
    print(f"Wrote {out_dir / 'matched_boundary_points.gpkg'}")
    print(f"Wrote {out_dir / 'corrected_hydrographs.csv'}")


if __name__ == "__main__":
    main()
