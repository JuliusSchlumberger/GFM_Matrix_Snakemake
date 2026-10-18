"""The SFINCS waterlevel-boundary line of a calibration tile, built from
COAST-RP stations as nodes: a coast-following boundary ~1 km offshore
instead of the open-sea tile edge (whose distance from the coast lets the
tide amplify across a -10 m-floored sea, see build_elevation.py's
MIN_BATHYMETRY_M). build_sfincs_tile.py masks the line's ocean side inactive
and forces the model along it (build_line_forcing.py).

Rules:
  1. Nodes: every station of the tile's own production selection
     (inputs/boundaries_RP100_SLR_0.gpkg: production's search box - the tile
     bbox + boundary_conditions.station_search_buffer_deg, at least
     station_search_min_size_deg wide - minus stations in separate water
     bodies) that lies at (or within STATION_OCEAN_TOL_M of) open
     ocean - ocean water bodies touching the search area's edge or larger
     than OPEN_OCEAN_MIN_KM2; stations on land or in closed lagoons are
     dropped before anything is built, and so are stations in a bay whose
     mouth closes when the ocean is shrunk by BAY_CLOSE_M (cut off from the
     open ocean: dropped_closed_bay).
  2. Link geometry, per candidate pair (each station with its
     CANDIDATE_NEIGHBOURS nearest stations): for seaward offset d = 0, then
     SEAWARD_OFFSETS_M, two shapes are built -
       coastline  the coastline between the two stations (both projected onto
                  the coastline piece they are both near, within COAST_SNAP_M;
                  for d > 0 the ocean polygon's edge shrunk by d, which also
                  closes narrow bays), shifted to start/end exactly at the
                  stations, the shift fading out over RAMP_FACTOR x each
                  station's own offset;
       straight   the straight line, for d > 0 shifted d to the sea side with
                  the same fade towards both stations (up to MAX_STRAIGHT_M).
     At the smallest d where one clears the land: the coastline if it is not
     rugged (<= RUGGED_RATIO x the straight distance), else the straight
     line ("coastline-first"). Bay mouths, rugged coasts, stations on
     separate coastline pieces and offshore stations therefore get straight
     links. Nothing clears the land: the plain coastline path is kept and
     flagged (crosses_land / coastline_long), or there is no link at all.
  3. Linking: candidate links cheapest (shortest actual geometry, flagged
     ones last) first; a link is accepted if both stations have fewer than
     two links so far and it does not close a loop. Result: one
     MultiLineString per tile, one part per separate line.
  4. Domain check: the lines split the tile into faces; a face that is mostly
     ocean becomes inactive (the ocean side of the boundary), the rest stays
     active. Conflicts: land inside inactive faces (> CONFLICT_LAND_FRAC of
     the tile's land) or open ocean left active (> CONFLICT_OCEAN_FRAC of
     the tile's ocean - a line that doesn't close off the sea). A tile with
     conflicts is rebuilt "straight-first" (at each offset the straight line
     before the coastline - bay mouths instead of bays); if that has fewer
     conflicts it is kept. Still conflicting: suitable=False (not usable
     for SFINCS without hand edits).
  5. Every line must cross the tile edge at both ends or be a closed ring: a
     line end inside the tile is extended along the (seaward-offset)
     coastline, away from its neighbouring station, until it leaves the
     tile - only if that is not rugged (<= RUGGED_RATIO x straight, i.e. not
     into a bay) - else straight on in the line's own direction
     (extension). A line with both ends stuck inside the tile is dropped
     with its stations (dropped_dangling) and the lines are rebuilt; a single
     stuck end on an otherwise valid line is an open-end conflict.
  0. Offshore: every station's node (where its line passes) is moved to the
     nearest point MIN_OFFSHORE_M out at open sea (at most MAX_NODE_SHIFT_M),
     coastline links follow the coastline shifted >= MIN_OFFSHORE_M seaward,
     and a link only clears the land if it keeps LAND_CLEARANCE_M from it.
  7. Final scan per line: untangle (2-opt over up to UNTANGLE_WINDOW
     consecutive stations, e.g. 90-92-91-93 -> 90-91-92-93) and shortcuts
     (two stations whose boundary in between is > SHORTCUT_RATIO x a straight
     link that clears the land, <= MAX_SHORTCUT_M: the stretch is replaced
     by that link, the stations skipped are dropped_shortcut) - kept only if
     the tile's conflicts do not get worse.
  8. Tile-edge fallback (last): tile-edge stretches still running through
     open sea on the active side - sea no station line closes off - become
     boundary lines themselves (tile_edge), unless shorter than EDGE_MIN_M.
  6. Relevance: after the split, a station inside the tile further than
     RELEVANCE_TOL_M from the active/inactive interface plays no part in the
     boundary (a chain inside a cut-off bay, or out at sea): it is dropped
     and the lines are rebuilt without it (up to MAX_RELEVANCE_ROUNDS).

The coastline is traced from the global DeltaDTM mask VRT (ocean code 1) over
the same buffered bbox, decimated to at most MAX_MASK_PX pixels per side, one
ocean ring at a time, with the bbox's own outer edge removed (an ocean
polygon's boundary along the bbox edge is not coastline).

Outputs under {base-dir-name}/boundary_lines/:
  {tile_id}.gpkg   layers: boundary_line (one MultiLineString - the one to
                   edit by hand), segments (per link, with its kind and
                   offset), stations, inactive (the tile faces that would be
                   inactive SFINCS cells)
  figures/{tile_id}.png   (a) the lines on the land/sea mask with the tile
                   outline (and SFINCS boundary cells if the tile has a model), (b) the
                   tile's active/inactive split and its conflicts
  summary.csv      per tile counts/flags, to find the tiles needing hand edits

Existing {tile_id}.gpkg files are never overwritten without --overwrite (so
hand edits survive a rerun); the figure is always redrawn from the gpkg.

Tracking: every finished tile writes boundary_lines/status/{tile_id}.json
(safe with many jobs at once; --skip-done resumes); --summarize collects
them into summary.csv, suitable_tiles.txt and unsuitable_tiles.txt. The
inspection figure goes to figures/ (suitable) or figures_unsuitable/ (not).

Usage:
    python build_station_boundary_lines.py [--tile-ids 841 2905 | --sample 30] [--workers 8]
    python build_station_boundary_lines.py --tile-ids-file tiles_000.txt --overwrite --skip-done   # one HPC job
    python build_station_boundary_lines.py --summarize
    (HPC jobs: generate_calibration_jobs.py stage1)
"""

from __future__ import annotations

import argparse

import json

import math

import os

import random

import sys

import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed

from concurrent.futures.process import BrokenProcessPool

from pathlib import Path

import geopandas as gpd

import matplotlib

matplotlib.use("Agg")


import numpy as np

import pandas as pd



import shapely


import yaml








from shapely.geometry import MultiLineString, Point, box


sys.path.insert(0, str(Path(__file__).resolve().parent))

from gfm_config import read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# Geometry core and plotting live in boundary_lines_geometry.py / boundary_lines_plot.py; the
# names below are imported for build_tile and the CLI, and re-exported for
# connected_forcing.py (bsl.MASK_VRT, bsl.OCEAN_CODE, bsl._read_mask_window, bsl._search_box)
# and analysis/compute_delta_flood_extent.py.
from boundary_lines_geometry import (  # noqa: E402,F401
    BAY_CLOSE_M,
    Coast,
    FLAGGED_KINDS,
    MASK_VRT,
    MAX_NODE_SHIFT_M,
    MIN_FORCING_STATIONS,
    MIN_OFFSHORE_M,
    OCEAN_CODE,
    SEGMENT_KINDS,
    STATION_OCEAN_TOL_M,
    TILE_STATIONS_GPKG,
    _build_lines,
    _clean,
    _domain_split,
    _edge_fallback,
    _finalize,
    _merge_lines,
    _n_with_coast_hg,
    _read_mask_window,
    _search_box,
    _tile_stations,
)
from boundary_lines_plot import _plot  # noqa: E402


def build_tile(root: Path, tile_dir: Path, out_dir: Path, buffer_deg: float, overwrite: bool,
               plot: str = "suitable", min_size_deg: float = 0.0) -> dict:
    """Build one tile's SFINCS waterlevel-boundary line and write its outputs.

    Writes {tile_id}.gpkg under `out_dir` (layers: stations, boundary_line,
    segments, inactive/land_inactive/ocean_active) unless it already exists
    and `overwrite` is False, and redraws the inspection figure according to
    `plot`. Also looks up each used station's COAST-HG hydrograph to flag
    whether the tile has enough forcing stations to be usable.

    Parameters
    ----------
    root : Path
        Project root (paths.root), used to open the DeltaDTM mask VRT and
        the COAST-HG hydrograph file.
    tile_dir : Path
        Tile directory holding inputs/tile_geometry.gpkg and the tile's own
        station selection (TILE_STATIONS_GPKG).
    out_dir : Path
        boundary_lines output directory to write {tile_id}.gpkg and figures
        under.
    buffer_deg : float
        Degrees added to the tile bbox for the station search box.
    overwrite : bool
        Rebuild an existing {tile_id}.gpkg instead of keeping it (loses any
        hand edits).
    plot : str, default "suitable"
        "all", "suitable" or "none": which tiles get the inspection figure.
    min_size_deg : float, default 0.0
        Minimum width/height (degrees) of the station search box.

    Returns
    -------
    dict
        One row of per-tile counts/flags, also written to
        boundary_lines/status/{tile_id}.json and collected by `summarize`.
    """
    tile_id = tile_dir.name
    gpkg = out_dir / f"{tile_id}.gpkg"
    row = {"tile_id": int(tile_id)}
    tile = gpd.read_file(tile_dir / "inputs" / "tile_geometry.gpkg").to_crs("EPSG:4326")
    bounds = _search_box(tile, buffer_deg, min_size_deg)
    utm_crs = tile.estimate_utm_crs()
    area_utm = gpd.GeoSeries([box(*bounds)], crs="EPSG:4326").to_crs(utm_crs).iloc[0]

    stations = _tile_stations(tile_dir)
    row.update(n_stations=len(stations),
               n_stations_in_tile=int(stations.within(tile.geometry.iloc[0]).sum()))

    mask, transform = _read_mask_window(root / MASK_VRT, bounds)
    pixel_m = abs(transform.a) * 111_320 * math.cos(math.radians(0.5 * (bounds[1] + bounds[3])))
    coast = Coast(mask, transform, area_utm, utm_crs, pixel_m)

    if gpkg.exists() and not overwrite:
        row["status"] = "kept_existing"
    elif len(stations) < 2:
        row["status"] = "lt2_stations"
    else:
        st_utm = stations.to_crs(utm_crs)
        xy = np.column_stack([st_utm.geometry.x, st_utm.geometry.y])
        tile_utm = tile.to_crs(utm_crs).geometry.iloc[0]
        status = np.array(["used"] * len(xy), dtype=object)
        open_dist = (np.array([coast.open_ocean.distance(Point(p)) for p in xy]) if not coast.open_ocean.is_empty
                     else np.full(len(xy), np.inf))
        status[open_dist > STATION_OCEAN_TOL_M] = "dropped_not_open_ocean"
        open_bay = coast.open_eroded(BAY_CLOSE_M)
        bay_dist = (np.array([open_bay.distance(Point(p)) for p in xy]) if not open_bay.is_empty
                    else np.full(len(xy), np.inf))
        status[(status == "used") & (bay_dist > BAY_CLOSE_M + STATION_OCEAN_TOL_M)] = "dropped_closed_bay"
        ids = [i for i in range(len(xy)) if status[i] == "used"]
        # nodes: each station moved MIN_OFFSHORE_M out to open sea (lines pass there)
        node_sea = coast.open_eroded(MIN_OFFSHORE_M)
        node_xy = xy.copy()
        if not node_sea.is_empty:
            for i in ids:
                pt = Point(xy[i])
                if not node_sea.contains(pt):
                    q = shapely.ops.nearest_points(node_sea, pt)[0]
                    if q.distance(pt) <= MAX_NODE_SHIFT_M:
                        node_xy[i] = q.coords[0]
        station_xy, xy = xy, node_xy
        cache: dict = {}
        best = _build_lines(xy, ids, coast, tile_utm, False, cache)
        mode = "coastline_first"
        if best["n_conflicts"]:
            alt = _build_lines(xy, ids, coast, tile_utm, True, cache)
            if alt["score"] < best["score"]:
                best, mode = alt, "straight_first"
        final = _finalize(best, xy, coast, tile_utm, mode == "straight_first", cache)
        if final["score"] <= best["score"]:
            best = final
        edge_links = _edge_fallback(best["links"], best["split"], tile_utm, coast)
        if edge_links:
            split_e = _domain_split(_merge_lines(best["links"] + edge_links), tile_utm, coast)
            n_conf_e = split_e["n_conflicts"] + int(best["n_open_ends"] > 0)
            best = {**best, "links": best["links"] + edge_links, "split": split_e, "n_conflicts": n_conf_e}
        links, split = best["links"], best["split"]
        status[best["dropped"]] = "dropped_irrelevant"
        status[best.get("shortcut", [])] = "dropped_shortcut"
        status[best["dangling"]] = "dropped_dangling"
        linked = {l["station_a"] for l in links} | {l["station_b"] for l in links}
        status[[i for i in best["ids"] if i not in linked]] = "unlinked"
        coast_lines = MultiLineString(coast.pieces(0.0))
        coast_dist = (np.array([coast_lines.distance(Point(p)) for p in station_xy]) if not coast_lines.is_empty
                      else np.full(len(xy), np.nan))
        nodes = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=utm_crs).to_crs("EPSG:4326")
        stations = stations.assign(status=status, open_ocean_dist_km=np.round(open_dist / 1000, 2),
                                   node_lon=nodes.x.to_numpy(), node_lat=nodes.y.to_numpy(),
                                   node_shift_km=np.round(np.hypot(*(xy - station_xy).T) / 1000, 2))
        st_out = stations.assign(station=range(len(stations)), coast_dist_km=coast_dist / 1000)
        out_dir.mkdir(parents=True, exist_ok=True)
        if gpkg.exists():
            gpkg.unlink()
        st_out.to_file(gpkg, layer="stations", driver="GPKG")
        layers = {"inactive": split["inactive"], "land_inactive": split["land_inactive"],
                  "ocean_active": split["ocean_active"]}
        for name, geom in layers.items():
            if not geom.is_empty:
                out = gpd.GeoSeries([geom], crs=utm_crs).to_crs("EPSG:4326").iloc[0]
                gpd.GeoDataFrame({"tile_id": [int(tile_id)]}, geometry=[_clean(out)], crs="EPSG:4326").to_file(
                    gpkg, layer=name, driver="GPKG")
        if links:
            segs = gpd.GeoDataFrame(links, crs=utm_crs)
            merged = _merge_lines(links)
            gpd.GeoDataFrame({"tile_id": [int(tile_id)], "n_parts": [len(merged.geoms)]},
                             geometry=[merged], crs=utm_crs).to_crs("EPSG:4326").to_file(
                gpkg, layer="boundary_line", driver="GPKG")
            segs.to_crs("EPSG:4326").to_file(gpkg, layer="segments", driver="GPKG")
            row.update(n_links=len(segs), n_parts=len(merged.geoms),
                       **{f"n_{k}": int((segs["kind"] == k).sum()) for k in SEGMENT_KINDS},
                       n_flagged=int(segs["kind"].isin(FLAGGED_KINDS).sum()),
                       max_link_km=round(float(segs["straight_km"].max()), 1))
        else:
            row.update(n_links=0, n_parts=0, n_flagged=0)
        degree = np.bincount([s for l in links for s in (l["station_a"], l["station_b"]) if s >= 0],
                             minlength=len(xy))
        row.update(n_unlinked_stations=int((status == "unlinked").sum()),
                   n_dropped_not_open_ocean=int((status == "dropped_not_open_ocean").sum()),
                   n_dropped_closed_bay=int((status == "dropped_closed_bay").sum()),
                   n_dropped_irrelevant=int((status == "dropped_irrelevant").sum()),
                   n_dropped_dangling=int((status == "dropped_dangling").sum()),
                   n_dropped_shortcut=int((status == "dropped_shortcut").sum()),
                   tile_edge_km=round(sum(l["length_km"] for l in links if l["kind"] == "tile_edge"), 1),
                   n_open_ends=best["n_open_ends"], max_station_degree=int(degree.max()),
                   max_station_coast_dist_km=round(float(np.nanmax(coast_dist)) / 1000, 2),
                   mode=mode, land_inactive_pct=round(100 * split["land_inactive_frac"], 1),
                   ocean_active_pct=round(100 * split["ocean_active_frac"], 1),
                   ocean_enclosed_pct=round(100 * split["ocean_enclosed_frac"], 1))
        used = stations[status == "used"]
        n_forcing = _n_with_coast_hg(root, np.column_stack([used.geometry.x, used.geometry.y]))
        row.update(n_used_stations=len(used), n_forcing_stations=n_forcing,
                   suitable=best["n_conflicts"] == 0 and n_forcing >= MIN_FORCING_STATIONS)
        row["status"] = "written"

    # suitable tiles -> figures/, unsuitable ones -> figures_unsuitable/
    suitable = bool(row.get("suitable"))
    fig_path = out_dir / ("figures" if suitable else "figures_unsuitable") / f"{tile_id}.png"
    stale = out_dir / ("figures_unsuitable" if suitable else "figures") / f"{tile_id}.png"
    if plot == "all" or (plot == "suitable" and suitable):
        _plot(tile_id, tile_dir, gpkg, mask, bounds, tile, fig_path, row)
    for path in ([stale] if plot != "none" else []) + ([fig_path] if plot == "suitable" and not suitable else []):
        if path.exists():
            try:
                path.unlink()  # figure from an earlier run in which the tile was the other way round
            except OSError as exc:  # e.g. open in a viewer - never fail the tile for it
                print(f"  tile {tile_id}: could not remove stale figure {path} ({exc})", flush=True)
    return row


def _status_path(out_dir: Path, tile_id) -> Path:
    return out_dir / "status" / f"{tile_id}.json"


def _write_status(out_dir: Path, row: dict) -> None:
    """One small JSON per tile, written atomically as soon as the tile is done -
    many jobs can run at once, and a restart can skip what is finished."""
    path = _status_path(out_dir, row["tile_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(row, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def _running_marker(out_dir: Path, tile_id) -> Path:
    return out_dir / "status" / f"{tile_id}.running"


def _safe_build(args):
    root, tile_dir, out_dir, buffer_deg, overwrite, plot, min_size_deg = args
    marker = _running_marker(out_dir, tile_dir.name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(os.getpid()))  # tells the parent which tiles were running if this worker dies
    try:
        row = build_tile(root, tile_dir, out_dir, buffer_deg, overwrite, plot, min_size_deg)
    except Exception as exc:  # one broken tile must not stop the batch
        row = {"tile_id": int(tile_dir.name), "status": f"error: {exc!r}",
               "traceback": traceback.format_exc(limit=3)}
    _write_status(out_dir, row)
    marker.unlink(missing_ok=True)
    return row


def summarize(out_dir: Path) -> pd.DataFrame:
    """Collect every tile's status JSON into summary.csv, suitable_tiles.txt
    and unsuitable_tiles.txt.

    Parameters
    ----------
    out_dir : Path
        boundary_lines output directory holding status/*.json.

    Returns
    -------
    pandas.DataFrame
        The combined per-tile summary, also written to out_dir/summary.csv.
    """
    rows = [json.loads(p.read_text()) for p in sorted((out_dir / "status").glob("*.json"))]
    summary = pd.DataFrame(rows).sort_values("tile_id")
    summary.drop(columns=["traceback"], errors="ignore").to_csv(out_dir / "summary.csv", index=False)
    written = summary[summary["status"] == "written"]
    suitable = written[written["suitable"].astype(bool)]["tile_id"].astype(int)
    unsuitable = summary[~summary["tile_id"].isin(suitable)]["tile_id"].astype(int)
    (out_dir / "suitable_tiles.txt").write_text("".join(f"{t}\n" for t in suitable), newline="\n")
    (out_dir / "unsuitable_tiles.txt").write_text("".join(f"{t}\n" for t in unsuitable), newline="\n")
    _print_summary(summary)
    print(f"Wrote {out_dir / 'summary.csv'}, suitable_tiles.txt ({len(suitable)}), "
          f"unsuitable_tiles.txt ({len(unsuitable)})")
    return summary


def _print_summary(summary: pd.DataFrame) -> None:
    print(summary["status"].astype(str).str.split(":").str[0].value_counts().to_string())
    w = summary[summary["status"] == "written"]
    if len(w):
        print(f"suitable for SFINCS: {int(w['suitable'].astype(bool).sum())}/{len(w)} "
              f"({int((w['mode'] == 'straight_first').sum())} after the straight-first rebuild)")


def _report(row: dict, k: int, n: int) -> None:
    if row["status"].startswith("error") or k % 10 == 0 or n <= 50:
        print(f"  [{k}/{n}] tile {row['tile_id']}: {row['status']}"
              + (f", suitable={row.get('suitable')}" if row["status"] == "written" else ""), flush=True)


def _run_jobs(jobs: list, workers: int, out_dir: Path) -> list[dict]:
    """Runs the tiles over `workers` processes. A worker killed outright (out
    of memory, native crash) breaks the whole pool: the tiles that were still
    in flight are then retried one at a time in a fresh single-worker pool -
    a tile that kills even that one is recorded as crashed - and the rest
    continues in a new pool."""
    rows, pending, n = [], list(jobs), len(jobs)
    while pending:
        in_flight: dict = {}
        try:
            with ProcessPoolExecutor(workers) as ex:
                in_flight = {ex.submit(_safe_build, job): job for job in pending}
                for fut in as_completed(in_flight):
                    row = fut.result()
                    rows.append(row)
                    pending.remove(in_flight[fut])
                    _report(row, len(rows), n)
        except BrokenProcessPool:
            print(f"  worker process died - retrying the {len(pending)} unfinished tile(s), the in-flight ones alone",
                  flush=True)
            # the tiles a worker was busy with when it died still have their .running marker
            suspects = [job for job in pending if _running_marker(out_dir, job[1].name).exists()]
            for job in suspects:
                _running_marker(out_dir, job[1].name).unlink(missing_ok=True)
                try:
                    with ProcessPoolExecutor(1) as solo:
                        row = solo.submit(_safe_build, job).result()
                except BrokenProcessPool:
                    row = {"tile_id": int(job[1].name), "status": "error: worker crashed (likely out of memory)"}
                    _write_status(out_dir, row)
                    _running_marker(out_dir, job[1].name).unlink(missing_ok=True)
                rows.append(row)
                pending.remove(job)
                _report(row, len(rows), n)
    return rows


def main() -> None:
    """Parse CLI arguments and build boundary lines for the selected tiles,
    or (with --summarize) just collect existing results into summary.csv.

    Runs tiles over a process pool (--workers), writing each tile's
    {tile_id}.gpkg, status JSON and inspection figure under
    {base-dir-name}/boundary_lines/ beneath paths.root.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1]
                                                 / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS",
                        help="where boundary_lines/ is written, under paths.root")
    parser.add_argument("--tile-ids", nargs="*", help="default: every tile in --tiles-dir-name with a station file")
    parser.add_argument("--tile-ids-file", default=None, help="one tile id per line (e.g. one HPC job's share)")
    parser.add_argument("--sample", type=int, default=0, help="random sample of N tiles (seed 0) instead of all")
    parser.add_argument("--station-buffer-deg", type=float, default=None,
                        help="search-box buffer (default: the config's boundary_conditions.station_search_buffer_deg)")
    parser.add_argument("--tiles-dir-name", default="model_outputs",
                        help="where the tile dirs (with inputs/) live, under paths.root (default: model_outputs)")
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing {tile_id}.gpkg (loses hand edits)")
    parser.add_argument("--skip-done", action="store_true",
                        help="skip tiles that already have a status file (not an error one) - resume a run")
    parser.add_argument("--plot", choices=["suitable", "all", "none"], default="all",
                        help="which tiles get the inspection figure - suitable ones go to figures/, "
                             "unsuitable ones to figures_unsuitable/ (default: all)")
    parser.add_argument("--summarize", action="store_true",
                        help="only collect boundary_lines/status/*.json into summary.csv + suitable/unsuitable lists")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base = root / args.base_dir_name
    out_dir = base / "boundary_lines"
    tiles_root = root / args.tiles_dir_name
    with open(args.config, encoding="utf-8") as f:
        bc = (yaml.safe_load(f) or {}).get("boundary_conditions", {})
    buffer_deg = args.station_buffer_deg if args.station_buffer_deg is not None else float(bc.get("station_search_buffer_deg", 1.0))
    min_size_deg = float(bc.get("station_search_min_size_deg", 2.0))
    if args.summarize:
        summarize(out_dir)
        return
    if args.tile_ids or args.tile_ids_file:
        ids = list(args.tile_ids or []) + (
            [t.strip() for t in Path(args.tile_ids_file).read_text().split() if t.strip()] if args.tile_ids_file else [])
        tile_dirs = [tiles_root / t for t in ids]
    else:
        tile_dirs = sorted((d for d in tiles_root.iterdir() if d.name.isdigit() and (d / TILE_STATIONS_GPKG).exists()),
                           key=lambda d: int(d.name))
        if args.sample:
            tile_dirs = sorted(random.Random(0).sample(tile_dirs, min(args.sample, len(tile_dirs))),
                               key=lambda d: int(d.name))
    if args.skip_done:
        def done(d):
            p = _status_path(out_dir, d.name)
            return p.exists() and not json.loads(p.read_text())["status"].startswith("error")
        n0 = len(tile_dirs)
        tile_dirs = [d for d in tile_dirs if not done(d)]
        print(f"--skip-done: {n0 - len(tile_dirs)} tile(s) already done")
    print(f"{len(tile_dirs)} tile(s) -> {out_dir}", flush=True)

    jobs = [(root, d, out_dir, buffer_deg, args.overwrite, args.plot, min_size_deg) for d in tile_dirs]
    rows = _run_jobs(jobs, args.workers, out_dir)
    if rows:
        print()
        _print_summary(pd.DataFrame(rows))
    print("(collect every job's results with: --summarize)")


if __name__ == "__main__":
    main()
