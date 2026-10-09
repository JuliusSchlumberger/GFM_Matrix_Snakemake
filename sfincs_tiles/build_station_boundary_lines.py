"""Test: a candidate SFINCS waterlevel-boundary line per calibration tile,
built from COAST-RP stations as nodes - first step towards a coast-following
boundary instead of the current open-sea domain edge (whose distance from the
coast lets the tide amplify across a -10 m-floored sea, see build_elevation.py's
MIN_BATHYMETRY_M).

Rules (fourth iteration):
  1. Nodes: every COAST-RP station (global station set, the RP100 SLR_0
     scenario file) inside the tile's bbox buffered by --station-buffer-deg
     (default 0.5 deg) that lies at (or within STATION_OCEAN_TOL_M of) open
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
                   outline and the current SFINCS boundary cells, (b) the
                   tile's active/inactive split and its conflicts
  summary.csv      per tile counts/flags, to find the tiles needing hand edits

Existing {tile_id}.gpkg files are never overwritten without --overwrite (so
hand edits survive a rerun); the figure is always redrawn from the gpkg.

Tracking: every finished tile writes boundary_lines/status/{tile_id}.json
(safe with many jobs at once; --skip-done resumes); --summarize collects
them into summary.csv, suitable_tiles.txt and unsuitable_tiles.txt. The
inspection figure goes to figures/ (suitable) or figures_unsuitable/ (not).

Usage:
    python build_station_boundary_lines.py --base-dir-name sfincs_calibration [--tile-ids 841 2905 | --sample 30] [--workers 8]
    python build_station_boundary_lines.py --tile-ids-file tiles_000.txt --overwrite --skip-done   # one HPC job
    python build_station_boundary_lines.py --summarize
    (HPC jobs: generate_boundary_lines_jobs.py)
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import shapely
import xarray as xr
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point, box, shape
from shapely.ops import linemerge, substring, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import WATER_COLOR  # noqa: E402

OCEAN_CODE = 1
STATIONS_NC = "processed_inputs/WL_scenarios/COAST-RP_EWL_RP100_SLR_0.nc"
MASK_VRT = "inputs/DeltaDTM_masks/deltadtm_mask.vrt"
MAX_MASK_PX = 3000            # coastline traced on the mask decimated to at most this many px per side
MIN_OCEAN_AREA_KM2 = 0.5      # drop ocean specks (misclassified pixels) from the coastline
CANDIDATE_NEIGHBOURS = 5      # each station is only ever linked to one of its 5 nearest stations
COAST_SNAP_M = 10_000.0       # both stations must be this close to the coastline piece that is copied
RUGGED_RATIO = 1.5            # coastline link longer than this x the straight distance: rugged, go straight
COAST_LONG_RATIO = 3.0        # flagged fallback: coastline path this many x the straight distance
MAX_STRAIGHT_M = 60_000.0     # never a straight link between stations further apart
LAND_CROSSING_TOL_M = 100.0   # a link "crosses land" above this much length inside the eroded land
SEAWARD_OFFSETS_M = (500.0, 1000.0, 2000.0, 4000.0)  # extra seaward shift, tried in order after d = 0
MIN_OFFSHORE_M = 1000.0       # boundary lines stay ~this far out at sea: nodes moved out, coastline shifted
MAX_NODE_SHIFT_M = 5000.0     # a station's node is moved out to sea by at most this
LAND_CLEARANCE_M = 700.0      # a link "crosses land" once it comes closer to land than this
SHORTCUT_RATIO = 1.5          # final scan: boundary between two stations > this x a clear straight link
MAX_SHORTCUT_M = 40_000.0     # ... of at most this length: replaced by that link
UNTANGLE_WINDOW = 4           # final scan: 2-opt over up to this many consecutive stations
EDGE_MIN_M = 2000.0           # tile-edge fallback: stretches shorter than this ignored
RAMP_FACTOR = 3.0             # a station's offset from the coastline fades out over 3x that offset
STATION_OCEAN_TOL_M = 2000.0  # station further than this from open ocean: dropped
BAY_CLOSE_M = 2000.0          # ocean shrunk by this closes bays with a mouth < ~2x this; stations
                              # then cut off from the open ocean are in a closed bay: dropped
OPEN_OCEAN_MIN_KM2 = 1000.0   # an ocean water body this large (or touching the search edge) is open ocean
RELEVANCE_TOL_M = 1000.0      # in-tile station further than this from the active/inactive interface: dropped
MAX_RELEVANCE_ROUNDS = 3
EXTENSION_OVERSHOOT_M = 2000.0  # an extended line end continues this far past the tile edge
CONFLICT_LAND_FRAC = 0.02     # > 2 % of the tile's land in inactive faces: conflict
CONFLICT_OCEAN_FRAC = 0.5     # > 50 % of the tile's ocean in active faces: conflict (sea not closed off)
SEGMENT_KINDS = {"coastline": "#d62728", "coastline_offset": "#ff7f0e", "straight": "#2ca02c",
                 "straight_offset": "#17becf", "shortcut": "#e377c2", "tile_edge": "#000000",
                 "extension": "#9467bd",
                 "extension_offset": "#bcbd22",
                 "coastline_long": "#8c564b", "crosses_land": "#c51b8a"}
STATION_STYLE = {"used": ("o", "white"), "unlinked": ("o", "#fdd0a2"), "dropped_dangling": ("X", "#d62728"),
                 "dropped_shortcut": ("X", "#e377c2"),
                 "dropped_not_open_ocean": ("X", "#7f7f7f"), "dropped_closed_bay": ("X", "#17becf"),
                 "dropped_irrelevant": ("X", "#9467bd")}
FLAGGED_KINDS = ("coastline_long", "crosses_land")


def _read_mask_window(vrt: Path, bounds: tuple[float, float, float, float]):
    with rasterio.open(vrt) as src:
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        f = max(1, math.ceil(max(window.height, window.width) / MAX_MASK_PX))
        out_shape = (math.ceil(window.height / f), math.ceil(window.width / f))
        nodata = src.nodata if src.nodata is not None else 255
        arr = src.read(1, window=window, out_shape=out_shape, resampling=Resampling.nearest,
                       boundless=True, fill_value=nodata)
        wt = src.window_transform(window)
        transform = wt * wt.scale(window.width / out_shape[1], window.height / out_shape[0])
    # The VRT has no tiles over open ocean or far inland: a no-data block whose
    # valid neighbouring pixels are mostly ocean is ocean, otherwise land -
    # never a fake coastline along a block edge.
    missing = arr == nodata
    if missing.any():
        labels, n = ndimage.label(missing)
        rim = ndimage.binary_dilation(missing) & ~missing
        n_rim = ndimage.sum(rim, ndimage.grey_dilation(labels, size=3) * rim, index=np.arange(1, n + 1))
        n_ocean = ndimage.sum(rim & (arr == OCEAN_CODE), ndimage.grey_dilation(labels, size=3) * rim,
                              index=np.arange(1, n + 1))
        ocean_blocks = np.arange(1, n + 1)[(n_rim > 0) & (n_ocean > 0.5 * n_rim)]
        arr = arr.copy()
        arr[missing] = 0
        arr[np.isin(labels, ocean_blocks)] = OCEAN_CODE
    return arr, transform


def _ring_pieces(ocean_geom, area_utm, simplify_m: float, edge_m: float) -> list[LineString]:
    """Coastline pieces of `ocean_geom`: its rings, simplified, minus the
    area's own outer edge. One ring at a time: merging all rings' lines
    together would split them wherever two rings touch at a pixel corner (a
    junction linemerge stops at)."""
    edge = area_utm.exterior.buffer(edge_m)
    pieces = []
    for poly in shapely.get_parts(ocean_geom):
        if poly.geom_type != "Polygon":
            continue
        for ring in [poly.exterior, *poly.interiors]:
            coords = ring.simplify(simplify_m).coords
            if len(coords) < 4:  # ring simplified away (sub-pixel speck)
                continue
            lines = [g for g in shapely.get_parts(LineString(coords).difference(edge))
                     if g.geom_type == "LineString" and not g.is_empty]
            if not lines:
                continue
            merged = linemerge(lines) if len(lines) > 1 else lines[0]  # rejoins a ring cut at its start point
            pieces.extend(g for g in shapely.get_parts(merged) if g.length > 0)
    return pieces


class Coast:
    """Ocean/land polygons (UTM) of the search area, coastline pieces at the
    real coast (offset 0) and at seaward offsets (the edge of the ocean
    polygon shrunk by d - narrow bays close by themselves), cached per d."""

    def __init__(self, mask: np.ndarray, transform, area_utm, utm_crs, pixel_m: float):
        self.area_utm, self.pixel_m = area_utm, pixel_m
        self._pieces: dict[float, list[LineString]] = {}
        ocean = (mask == OCEAN_CODE).astype(np.uint8)
        polys = [shape(g) for g, v in shapes(ocean, mask=ocean.astype(bool), transform=transform) if v == 1]
        ocean_utm = gpd.GeoSeries(polys, crs="EPSG:4326").to_crs(utm_crs) if polys else gpd.GeoSeries([], crs=utm_crs)
        ocean_utm = ocean_utm[ocean_utm.area >= MIN_OCEAN_AREA_KM2 * 1e6]
        self.ocean = (unary_union(list(ocean_utm)).buffer(0).intersection(area_utm) if len(ocean_utm)
                      else shapely.Polygon())
        # stations sit on the coast and the coastline is simplified to ~1 px, so
        # a link only "crosses land" once it is clearly inside it
        # "land" for the crossing test: the land grown by LAND_CLEARANCE_M, so a
        # link that clears it stays that far out at sea (name kept from when it was eroded)
        self.land_eroded = area_utm.difference(self.ocean).buffer(LAND_CLEARANCE_M)
        shapely.prepare(self.land_eroded)
        shapely.prepare(self.ocean)
        edge = area_utm.exterior.buffer(2 * pixel_m)
        open_parts = [g for g in shapely.get_parts(self.ocean)
                      if g.area >= OPEN_OCEAN_MIN_KM2 * 1e6 or g.intersects(edge)]
        self.open_ocean = unary_union(open_parts) if open_parts else shapely.Polygon()
        shapely.prepare(self.open_ocean)
        self.land_parts = list(shapely.get_parts(area_utm.difference(self.ocean)))
        self.land_tree = shapely.STRtree(self.land_parts)

    def eroded(self, offset_m: float):
        if not hasattr(self, "_eroded"):
            self._eroded = {}
        if offset_m not in self._eroded:
            self._eroded[offset_m] = self.ocean if offset_m == 0 else self.ocean.buffer(-offset_m)
        return self._eroded[offset_m]

    def open_eroded(self, offset_m: float):
        """Parts of the ocean shrunk by offset_m that are still open ocean
        (touch the search area's edge or are large) - bays whose mouth closed
        at this offset drop out."""
        edge = self.area_utm.exterior.buffer(offset_m + 2 * self.pixel_m)
        parts = [g for g in shapely.get_parts(self.eroded(offset_m))
                 if g.area >= OPEN_OCEAN_MIN_KM2 * 1e6 or g.intersects(edge)]
        return unary_union(parts) if parts else shapely.Polygon()

    def pieces(self, offset_m: float = 0.0) -> list[LineString]:
        if offset_m not in self._pieces:
            geom = self.eroded(offset_m)
            self._pieces[offset_m] = _ring_pieces(geom, self.area_utm, max(self.pixel_m, offset_m / 4),
                                                  1.5 * self.pixel_m + offset_m)
        return self._pieces[offset_m]

    def crosses_land(self, line: LineString, stations: tuple[Point, ...] = (), min_ramp_m: float = 1000.0) -> bool:
        """More than LAND_CROSSING_TOL_M of `line` inside the (eroded) land,
        ignoring the approach to each station that sits inland itself - the
        stretch over which _coast_path fades that station's offset in
        (RAMP_FACTOR x its distance from the coastline, at least min_ramp_m,
        + 300 m) has to cross land anyway."""
        if self.land_eroded.is_empty or not self.land_eroded.intersects(line):
            return False
        if not hasattr(self, "_coast_lines"):
            self._coast_lines = MultiLineString(self.pieces(0.0)) if self.pieces(0.0) else None
        for p in stations:
            if self._coast_lines is not None and self.land_eroded.contains(p):
                ramp = max(RAMP_FACTOR * self._coast_lines.distance(p), min_ramp_m)
                line = line.difference(p.buffer(ramp + 300.0))
        return line.intersection(self.land_eroded).length > LAND_CROSSING_TOL_M


def _coast_path(pa: Point, pb: Point, pieces: list[LineString], min_ramp_m: float = 1000.0):
    """Coastline between pa and pb, shifted to start/end exactly at them, or
    None if they are not both near one coastline piece. Each station's offset
    from the coastline fades out over RAMP_FACTOR x that offset (at least
    min_ramp_m) from its end - in between the path stays on the coastline,
    instead of the whole path being dragged towards a station that sits a
    bit inland or offshore."""
    if not pieces:
        return None
    da = np.array([g.distance(pa) for g in pieces])
    db = np.array([g.distance(pb) for g in pieces])
    score = np.where((da <= COAST_SNAP_M) & (db <= COAST_SNAP_M), da + db, np.inf)
    if not np.isfinite(score).any():
        return None
    line = pieces[int(np.argmin(score))]
    sa, sb = line.project(pa), line.project(pb)
    lo, hi = sorted((sa, sb))
    path = substring(line, lo, hi)
    if line.is_ring and line.length - (hi - lo) < hi - lo:  # the other way round the ring is shorter
        parts = [g for g in (substring(line, hi, line.length), substring(line, 0, lo))
                 if g.geom_type == "LineString" and g.length > 0]  # a part ending on the ring's start is a point
        path = linemerge(parts) if len(parts) > 1 else (parts[0] if parts else LineString())
    qa, qb = line.interpolate(sa), line.interpolate(sb)
    if path.is_empty or path.geom_type != "LineString" or path.length == 0:
        return LineString([pa, pb])  # both project onto the same coastline point
    xy = np.asarray(path.coords)
    if np.hypot(*(xy[0] - qa.coords[0])) > np.hypot(*(xy[0] - qb.coords[0])):
        xy = xy[::-1]
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    off_a = np.subtract(pa.coords[0], qa.coords[0])
    off_b = np.subtract(pb.coords[0], qb.coords[0])
    ramp_a = min(max(RAMP_FACTOR * np.hypot(*off_a), min_ramp_m), dist[-1] / 2)
    ramp_b = min(max(RAMP_FACTOR * np.hypot(*off_b), min_ramp_m), dist[-1] / 2)
    w_a = np.clip(1 - dist / ramp_a, 0, 1)
    w_b = np.clip(1 - (dist[-1] - dist) / ramp_b, 0, 1)
    xy = xy + w_a[:, None] * off_a + w_b[:, None] * off_b
    xy[0], xy[-1] = pa.coords[0], pb.coords[0]
    return LineString(xy)


def _offset_straight(pa: Point, pb: Point, offset_m: float, coast: Coast, ramp_m: float) -> LineString:
    """Straight pa-pb, shifted offset_m to the sea side in between, fading
    back to the stations over ramp_m at both ends."""
    a, b = np.asarray(pa.coords[0]), np.asarray(pb.coords[0])
    length = float(np.hypot(*(b - a)))
    if offset_m == 0 or length == 0:
        return LineString([a, b])
    u = (b - a) / length
    n = np.array([-u[1], u[0]])
    mid = 0.5 * (a + b)
    # sea side: the side whose shifted midpoint is at sea (else the one crossing less land)
    sides = [(coast.ocean.contains(Point(mid + sgn * offset_m * n)),
              -LineString([a + sgn * offset_m * n, b + sgn * offset_m * n]).intersection(coast.land_eroded).length,
              sgn) for sgn in (1.0, -1.0)]
    sgn = max(sides)[2]
    ramp = min(ramp_m, length / 3)
    if ramp <= 0:
        return LineString([a, mid + sgn * offset_m * n, b])
    return LineString([a, a + ramp * u + sgn * offset_m * n, b - ramp * u + sgn * offset_m * n, b])


def _link_geometry(pa: Point, pb: Point, dist: float, coast: Coast, straight_first: bool = False):
    """(geometry, kind, offset_m) for one link, or None if there is none.
    For d = 0, then SEAWARD_OFFSETS_M: a coastline path (not rugged) and a
    straight line, each shifted d seaward; the first that clears the land
    wins - at each d the coastline first, or the straight line first with
    straight_first (bay mouths instead of following into bays)."""
    ends = (pa, pb)
    coast0 = None
    for offset in (0.0, *SEAWARD_OFFSETS_M):
        ramp = max(2 * offset, 1000.0)
        options = []
        path = _coast_path(pa, pb, coast.pieces(MIN_OFFSHORE_M + offset), min_ramp_m=ramp)
        if offset == 0:
            coast0 = path
        if path is not None and path.length <= RUGGED_RATIO * max(dist, 1.0):
            options.append((path, "coastline" if offset == 0 else "coastline_offset"))
        if dist <= MAX_STRAIGHT_M:
            options.append((_offset_straight(pa, pb, offset, coast, ramp), "straight" if offset == 0 else "straight_offset"))
        if straight_first:
            options.sort(key=lambda o: not o[1].startswith("straight"))
        for geom, kind in options:
            if not coast.crosses_land(geom, ends, min_ramp_m=ramp):
                return geom, kind, offset
    # no straight line clears the land at any offset (e.g. round an island's
    # tip): accept a clear coastline path up to COAST_LONG_RATIO after all
    for offset in (0.0, *SEAWARD_OFFSETS_M):
        ramp = max(2 * offset, 1000.0)
        path = _coast_path(pa, pb, coast.pieces(MIN_OFFSHORE_M + offset), min_ramp_m=ramp)
        if (path is not None and path.length <= COAST_LONG_RATIO * max(dist, 1.0)
                and not coast.crosses_land(path, ends, min_ramp_m=ramp)):
            return path, "coastline" if offset == 0 else "coastline_offset", offset
    if coast0 is None:
        return None
    return coast0, ("coastline_long" if coast0.length > COAST_LONG_RATIO * max(dist, 1.0) else "crosses_land"), 0.0


def _link_stations(xy_all: np.ndarray, ids: list[int], coast: Coast, straight_first: bool = False,
                   cache: dict | None = None) -> list[dict]:
    """Links between the stations `ids` (global indices into xy_all): every
    candidate pair's link first, then cheapest (shortest geometry, flagged
    last) first; each station gets at most two links, no loops (except a
    ring around an island). Link geometries are cached across calls."""
    cache = {} if cache is None else cache
    n = len(ids)
    if n < 2:
        return []
    xy = xy_all[ids]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    k = min(CANDIDATE_NEIGHBOURS + 1, n)
    d, nb = cKDTree(xy).query(xy, k=k)
    pairs = {(min(a, int(b)), max(a, int(b))): float(dd)
             for a in range(n) for b, dd in zip(nb[a, 1:], d[a, 1:])}
    candidates = []
    for (a, b), dist in pairs.items():
        key = (ids[a], ids[b], straight_first)
        if key not in cache:
            cache[key] = _link_geometry(Point(xy[a]), Point(xy[b]), dist, coast, straight_first)
        if cache[key] is not None:
            geom, kind, offset = cache[key]
            candidates.append((geom.length * (10 if kind in FLAGGED_KINDS else 1), a, b, dist, geom, kind, offset))
    degree = np.zeros(n, dtype=int)
    links = []
    for _, a, b, dist, geom, kind, offset in sorted(candidates, key=lambda c: c[0]):
        if degree[a] >= 2 or degree[b] >= 2:
            continue
        if find(a) == find(b) and not _closes_island(links, find, a, geom, coast):
            continue  # no loops - except a ring around an island
        links.append({"station_a": a, "station_b": b, "kind": kind, "seaward_offset_m": offset,
                      "straight_km": dist / 1000, "length_km": geom.length / 1000, "geometry": geom})
        degree[a] += 1
        degree[b] += 1
        parent[find(a)] = find(b)
    for l in links:  # local -> global station numbers
        l["station_a"], l["station_b"] = ids[l["station_a"]], ids[l["station_b"]]
    return links


def _extend_open_ends(links: list[dict], xy_all: np.ndarray, coast: Coast, tile_utm) -> tuple[list[dict], list, int, list]:
    """Extensions for line ends inside the tile: along the (seaward-offset)
    coastline, away from the neighbouring station, until EXTENSION_OVERSHOOT_M
    past the tile edge. Returns (extension links, the parts with BOTH ends
    stuck inside the tile - no boundary at all, dropped by the caller -, the
    number of other ends that could not be extended - an open-end conflict)."""
    merged = _merge_lines(links)
    if merged is None:
        return [], [], 0, []
    tree = cKDTree(xy_all)
    extensions, stuck_parts, n_open = [], [], 0
    open_parts = []
    for part in merged.geoms:
        if part.is_closed:
            continue
        failed = 0
        for at_start in (True, False):
            end = Point(part.coords[0] if at_start else part.coords[-1])
            if not tile_utm.contains(end):
                continue
            step = min(2000.0, part.length / 2)
            nb = part.interpolate(step if at_start else part.length - step)
            ext = _extension(end, nb, coast, tile_utm)
            if ext is None:
                failed += 1
                continue
            geom, offset = ext
            extensions.append({"station_a": int(tree.query(end.coords[0])[1]), "station_b": -1,
                               "kind": "extension" if offset == 0 else "extension_offset",
                               "seaward_offset_m": offset, "straight_km": 0.0,
                               "length_km": geom.length / 1000, "geometry": geom})
        if failed == 2:
            stuck_parts.append(part)
        elif failed:
            n_open += failed
            open_parts.append(part)
    return extensions, stuck_parts, n_open, open_parts


def _extension(end: Point, nb: Point, coast: Coast, tile_utm):
    for offset in (0.0, *SEAWARD_OFFSETS_M):
        pieces = coast.pieces(MIN_OFFSHORE_M + offset)
        if not pieces:
            continue
        dists = [g.distance(end) for g in pieces]
        line = pieces[int(np.argmin(dists))]
        if min(dists) > COAST_SNAP_M or line.is_closed:  # an island ring cannot lead out of the tile
            continue
        s0, sn = line.project(end), line.project(nb)
        if sn < s0:
            sub = substring(line, s0, line.length)
        else:
            sub = substring(line, 0, s0)
            sub = LineString(sub.coords[::-1]) if sub.geom_type == "LineString" else sub
        if sub.geom_type != "LineString" or sub.length == 0:
            continue
        exits = [sub.project(q) for q in shapely.get_parts(sub.intersection(tile_utm.exterior))]
        exits = [e for e in exits if e > 1.0]
        if not exits:
            continue  # this coastline never leaves the tile
        sub = substring(sub, 0, min(min(exits) + EXTENSION_OVERSHOOT_M, sub.length))
        if sub.length > RUGGED_RATIO * max(end.distance(Point(sub.coords[-1])), 1.0):
            continue  # follows the coast into a bay
        xy = np.asarray(sub.coords)
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        off = np.subtract(end.coords[0], xy[0])
        ramp = min(max(RAMP_FACTOR * np.hypot(*off), 2 * offset, 1000.0), dist[-1] / 2)
        xy = xy + np.clip(1 - dist / ramp, 0, 1)[:, None] * off
        xy[0] = end.coords[0]
        geom = LineString(xy)
        if not coast.crosses_land(geom, (end,), min_ramp_m=max(2 * offset, 1000.0)):
            return geom, offset
    # straight on, in the direction the line arrives at this end
    u = np.subtract(end.coords[0], nb.coords[0])
    if np.hypot(*u) == 0:
        return None
    u = u / np.hypot(*u)
    far = LineString([end.coords[0], np.add(end.coords[0], u * 1e7)])
    exits = [far.project(q) for q in shapely.get_parts(far.intersection(tile_utm.exterior))]
    exits = [e for e in exits if e > 1.0]
    if not exits:
        return None
    geom = LineString([end.coords[0], np.add(end.coords[0], u * (min(exits) + EXTENSION_OVERSHOOT_M))])
    if not coast.crosses_land(geom, (end,)):
        return geom, 0.0
    return None


def _irrelevant_stations(ids: list[int], xy_all: np.ndarray, split: dict, tile_utm) -> list[int]:
    """In-tile stations further than RELEVANCE_TOL_M from the active/inactive
    interface (empty if the tile has no interface at all)."""
    if split["inactive"].is_empty or split["active"].is_empty:
        return []
    if split["land_inactive_frac"] > CONFLICT_LAND_FRAC:
        return []  # a split that already puts land on the ocean side is no basis to judge stations by
    interface = split["inactive"].boundary.difference(tile_utm.exterior.buffer(5.0))
    if interface.is_empty:
        return []
    out = []
    for i in ids:
        pt = Point(xy_all[i])
        if not tile_utm.contains(pt) or interface.distance(pt) <= RELEVANCE_TOL_M:
            continue
        # out at sea on the inactive side: irrelevant; on the active side only
        # if it is not next to open sea still leaking in (then the lines are
        # what is wrong, not the station)
        if split["inactive"].contains(pt) or split["ocean_active"].distance(pt) > RELEVANCE_TOL_M:
            out.append(i)
    return out


def _build_lines(xy_all: np.ndarray, ids: list[int], coast: Coast, tile_utm, straight_first: bool,
                 cache: dict, retry_open: bool = True) -> dict:
    """Link, extend open ends, split the tile, drop irrelevant stations and
    repeat (up to MAX_RELEVANCE_ROUNDS)."""
    ids, dropped, dangling = list(ids), [], []
    for round_ in range(MAX_RELEVANCE_ROUNDS + 1):
        links = _link_stations(xy_all, ids, coast, straight_first, cache)
        extensions, stuck_parts, n_open, open_parts = _extend_open_ends(links, xy_all, coast, tile_utm)
        if stuck_parts and round_ < MAX_RELEVANCE_ROUNDS:
            # a line with both ends stuck inside the tile is no boundary: drop it, rebuild
            gone = [i for i in ids if any(p.distance(Point(xy_all[i])) < 1.0 for p in stuck_parts)]
            dangling += gone
            ids = [i for i in ids if i not in gone]
            continue
        split = _domain_split(_merge_lines(links + extensions), tile_utm, coast)
        irrelevant = _irrelevant_stations(ids, xy_all, split, tile_utm)
        if not irrelevant or round_ >= MAX_RELEVANCE_ROUNDS:
            break
        dropped += irrelevant
        ids = [i for i in ids if i not in irrelevant]
    n_open += 2 * len(stuck_parts)
    n_conflicts = split["n_conflicts"] + int(n_open > 0)
    result = {"links": links + extensions, "split": split, "n_open_ends": n_open, "ids": ids,
              "dropped": dropped, "dangling": dangling, "n_conflicts": n_conflicts,
              "score": (n_conflicts, split["land_inactive_frac"] + split["ocean_active_frac"])}
    if n_open and retry_open:
        # a line with one end stuck: either a real boundary missing its last
        # bit, or a chain inside a wide bay - try without it, keep the better
        gone = [i for i in ids if any(p.distance(Point(xy_all[i])) < 1.0 for p in open_parts)]
        alt = _build_lines(xy_all, [i for i in ids if i not in gone], coast, tile_utm, straight_first, cache,
                           retry_open=False)
        if alt["score"] < result["score"]:
            alt["dropped"] = dropped + alt["dropped"]
            alt["dangling"] = dangling + gone + alt["dangling"]
            return alt
    return result


def _chains(station_links: list[dict]) -> list[tuple[list[int], bool]]:
    """(station sequence, is_closed) per connected chain of station links."""
    adj: dict[int, list[int]] = {}
    for l in station_links:
        adj.setdefault(l["station_a"], []).append(l["station_b"])
        adj.setdefault(l["station_b"], []).append(l["station_a"])
    seen, chains = set(), []
    for start in sorted(adj, key=lambda k: len(adj[k])):  # path ends (degree 1) first
        if start in seen:
            continue
        seq, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seq.append(cur)
            seen.add(cur)
        chains.append((seq, len(adj[start]) == 2 and seq[0] in adj[seq[-1]] and len(seq) > 2))
    return chains


def _cached_link(a: int, b: int, xy_all: np.ndarray, coast: Coast, straight_first: bool, cache: dict):
    lo, hi = min(a, b), max(a, b)
    key = (lo, hi, straight_first)
    if key not in cache:
        cache[key] = _link_geometry(Point(xy_all[lo]), Point(xy_all[hi]),
                                    float(np.hypot(*(xy_all[lo] - xy_all[hi]))), coast, straight_first)
    return cache[key]


def _link_record(a: int, b: int, xy_all: np.ndarray, geom, kind: str, offset: float) -> dict:
    return {"station_a": a, "station_b": b, "kind": kind, "seaward_offset_m": offset,
            "straight_km": float(np.hypot(*(xy_all[a] - xy_all[b]))) / 1000,
            "length_km": geom.length / 1000, "geometry": geom}


def _untangle(seq: list[int], cost) -> list[int]:
    """2-opt over windows of up to UNTANGLE_WINDOW consecutive stations."""
    improved = True
    while improved:
        improved = False
        for i in range(1, len(seq) - 1):
            for j in range(i + 1, min(i + UNTANGLE_WINDOW, len(seq) - 1)):
                before = cost(seq[i - 1], seq[i]) + cost(seq[j], seq[j + 1])
                after = cost(seq[i - 1], seq[j]) + cost(seq[i], seq[j + 1])
                if after < before - 1.0:
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    improved = True
    return seq


def _shortcuts(seq: list[int], seg_len, xy_all: np.ndarray, coast: Coast) -> tuple[list[int], dict, list[int]]:
    """Repeatedly replace the stretch between two stations whose boundary is
    > SHORTCUT_RATIO x a clear straight link (<= MAX_SHORTCUT_M) by that link.
    Returns (new sequence, {(a, b): (geom, offset)} shortcut links, skipped stations)."""
    special: dict = {}
    skipped: list[int] = []
    for _ in range(50):
        along = np.concatenate([[0.0], np.cumsum([special[(a, b)][0].length if (a, b) in special else seg_len(a, b)
                                                  for a, b in zip(seq[:-1], seq[1:])])])
        best = None
        for i in range(len(seq) - 2):
            for j in range(i + 2, len(seq)):
                d = float(np.hypot(*(xy_all[seq[i]] - xy_all[seq[j]])))
                if d > MAX_SHORTCUT_M or along[j] - along[i] <= SHORTCUT_RATIO * d:
                    continue
                gain = along[j] - along[i] - d
                if best is not None and gain <= best[0]:
                    continue
                pa, pb = Point(xy_all[seq[i]]), Point(xy_all[seq[j]])
                for offset in (0.0, *SEAWARD_OFFSETS_M):
                    geom = _offset_straight(pa, pb, offset, coast, max(2 * offset, 1000.0))
                    if not coast.crosses_land(geom, (pa, pb), min_ramp_m=max(2 * offset, 1000.0)):
                        best = (gain, i, j, geom, offset)
                        break
        if best is None:
            break
        _, i, j, geom, offset = best
        skipped += seq[i + 1:j]
        special[(seq[i], seq[j])] = (geom, offset)
        seq = seq[:i + 1] + seq[j:]
    return seq, special, skipped


def _finalize(result: dict, xy_all: np.ndarray, coast: Coast, tile_utm, straight_first: bool, cache: dict) -> dict:
    """Final scan of the built lines: untangle, then shortcuts; extensions and
    the domain split are redone on the result."""
    station_links = [l for l in result["links"] if l["station_b"] >= 0]
    lookup = {(min(l["station_a"], l["station_b"]), max(l["station_a"], l["station_b"])): l for l in station_links}

    def seg_len(a, b):
        l = lookup.get((min(a, b), max(a, b)))
        if l is not None:
            return l["geometry"].length
        link = _cached_link(a, b, xy_all, coast, straight_first, cache)
        return np.inf if link is None or link[1] in FLAGGED_KINDS else link[0].length

    new_links, skipped = [], []
    for seq, closed in _chains(station_links):
        if closed or len(seq) < 3:
            new_links += [l for l in station_links if l["station_a"] in seq and l["station_b"] in seq]
            continue
        seq = _untangle(list(seq), seg_len)
        seq, special, gone = _shortcuts(seq, seg_len, xy_all, coast)
        skipped += gone
        for a, b in zip(seq[:-1], seq[1:]):
            if (a, b) in special:
                geom, offset = special[(a, b)]
                new_links.append(_link_record(a, b, xy_all, geom, "shortcut", offset))
            elif (min(a, b), max(a, b)) in lookup:
                new_links.append(lookup[(min(a, b), max(a, b))])
            else:
                geom, kind, offset = _cached_link(a, b, xy_all, coast, straight_first, cache)
                new_links.append(_link_record(a, b, xy_all, geom, kind, offset))
    extensions, stuck_parts, n_open, _ = _extend_open_ends(new_links, xy_all, coast, tile_utm)
    n_open += 2 * len(stuck_parts)
    split = _domain_split(_merge_lines(new_links + extensions), tile_utm, coast)
    n_conflicts = split["n_conflicts"] + int(n_open > 0)
    return {**result, "links": new_links + extensions, "split": split, "n_open_ends": n_open,
            "ids": [i for i in result["ids"] if i not in skipped], "shortcut": skipped,
            "n_conflicts": n_conflicts,
            "score": (n_conflicts, split["land_inactive_frac"] + split["ocean_active_frac"])}


def _edge_fallback(links: list[dict], split: dict, tile_utm, coast: Coast) -> list[dict]:
    """Tile-edge stretches still running through open sea on the active side
    after all other rules (no station lines close that sea off): those
    stretches become boundary lines themselves, unless shorter than
    EDGE_MIN_M (e.g. the ~1 km strip where a line crosses the tile edge)."""
    active_sea = split["active"].intersection(coast.open_ocean)
    if active_sea.is_empty:
        return []
    edge = tile_utm.exterior.intersection(active_sea.buffer(1.0))
    pieces = [g for g in shapely.get_parts(edge) if g.geom_type == "LineString"]
    merged = linemerge(pieces) if len(pieces) > 1 else (pieces[0] if pieces else LineString())
    return [{"station_a": -1, "station_b": -1, "kind": "tile_edge", "seaward_offset_m": 0.0,
             "straight_km": 0.0, "length_km": g.length / 1000, "geometry": g}
            for g in shapely.get_parts(merged) if g.geom_type == "LineString" and g.length >= EDGE_MIN_M]


def _closes_island(links: list[dict], find, a: int, geom: LineString, coast: Coast) -> bool:
    """Does closing a's chain with `geom` make a ring around mostly land?"""
    root = find(a)
    chain = [l["geometry"] for l in links if find(l["station_a"]) == root] + [geom]
    ring = linemerge(chain)
    if ring.geom_type != "LineString" or not ring.is_closed:  # is_ring would also demand a simple ring
        return False
    poly = shapely.Polygon(ring.coords).buffer(0)
    if poly.area == 0 or poly.difference(coast.ocean).area <= 0.5 * poly.area:
        return False
    # an island: the land pieces the ring touches lie (almost) entirely inside
    # it - a loop cutting off part of the mainland/a peninsula is no island
    touched = [coast.land_parts[i] for i in coast.land_tree.query(poly, predicate="intersects")]
    touched_area = sum(p.area for p in touched)
    return touched_area > 0 and sum(p.intersection(poly).area for p in touched) >= 0.8 * touched_area


def _domain_split(lines, tile_utm, coast: Coast) -> dict:
    """Faces of the tile cut by the boundary lines; mostly-ocean faces are
    inactive. Returns the inactive/active polygons and the conflict areas."""
    # Split the whole search area, not just the tile: whether sea left on the
    # active side inside the tile is a closed-off bay or open sea leaking in
    # past a line end is decided outside the tile. Line ends inside the area
    # are carried straight on to its edge (for this split only), so a line
    # that just stops does not let the sea flow round its end.
    area = coast.area_utm
    tile_ocean = tile_utm.intersection(coast.ocean)
    tile_land = tile_utm.difference(coast.ocean)
    if lines is None or lines.is_empty or not lines.intersects(tile_utm):
        faces = [area]
    else:
        linework = unary_union([area.exterior, lines.intersection(area), *_rays_to_edge(lines, area)])
        faces = [f for f in shapely.get_parts(shapely.polygonize(shapely.get_parts(linework))) if f.area > 0] or [area]
    inactive, active = [], []
    for f in faces:
        (inactive if f.intersection(coast.ocean).area >= 0.5 * f.area else active).append(f)
    inactive_area = unary_union(inactive) if inactive else shapely.Polygon()
    active_area = unary_union(active) if active else shapely.Polygon()
    # sea left active is only a problem where it is open sea (reaches the
    # search area's edge); a bay/lagoon closed off behind a line is meant to be active
    area_edge = area.exterior.buffer(10.0)
    leaking = [g for g in shapely.get_parts(active_area.intersection(coast.ocean)) if g.intersects(area_edge)]
    inactive_u = inactive_area.intersection(tile_utm)
    active_u = active_area.intersection(tile_utm)
    land_inactive = inactive_u.intersection(tile_land)
    ocean_active_all = active_u.intersection(tile_ocean)
    ocean_active = (unary_union(leaking) if leaking else shapely.Polygon()).intersection(tile_utm)
    land_frac = land_inactive.area / tile_land.area if tile_land.area else 0.0
    ocean_frac = ocean_active.area / tile_ocean.area if tile_ocean.area > 1e6 else 0.0
    enclosed_frac = (ocean_active_all.area - ocean_active.area) / tile_ocean.area if tile_ocean.area > 1e6 else 0.0
    return {"inactive": inactive_u, "active": active_u, "land_inactive": land_inactive,
            "ocean_active": ocean_active, "land_inactive_frac": land_frac, "ocean_active_frac": ocean_frac,
            "ocean_enclosed_frac": enclosed_frac,
            "n_conflicts": int(land_frac > CONFLICT_LAND_FRAC) + int(ocean_frac > CONFLICT_OCEAN_FRAC)}


def _rays_to_edge(lines, area) -> list[LineString]:
    """Straight continuations of every open line end inside `area` to its edge."""
    rays = []
    for part in lines.geoms:
        if part.is_closed or part.length == 0:
            continue
        for at_start in (True, False):
            end = np.asarray(part.coords[0] if at_start else part.coords[-1])
            if not area.contains(Point(end)):
                continue
            step = min(2000.0, part.length / 2)
            nb = np.asarray(part.interpolate(step if at_start else part.length - step).coords[0])
            u = end - nb
            if np.hypot(*u) == 0:
                continue
            u = u / np.hypot(*u)
            far = LineString([end, end + u * 1e7])
            hits = [far.project(q) for q in shapely.get_parts(far.intersection(area.exterior))]
            hits = [h for h in hits if h > 1.0]
            if hits:
                rays.append(LineString([end, end + u * (min(hits) + 10.0)]))
    return rays


def _merge_lines(links: list[dict]):
    if not links:
        return None
    merged = linemerge([l["geometry"] for l in links])
    return MultiLineString([merged]) if isinstance(merged, LineString) else merged


def _stations_in(root: Path, bounds) -> gpd.GeoDataFrame:
    with xr.open_dataset(root / STATIONS_NC) as ds:
        x = ds["station_x_coordinate"].values.astype(float)
        y = ds["station_y_coordinate"].values.astype(float)
        level = ds["COAST-RP_EWL_RP100_SLR_0"].values.astype(float)
    sel = (x >= bounds[0]) & (x <= bounds[2]) & (y >= bounds[1]) & (y <= bounds[3])
    return gpd.GeoDataFrame({"station_index": np.nonzero(sel)[0], "rp100_level_m": level[sel]},
                            geometry=gpd.points_from_xy(x[sel], y[sel]), crs="EPSG:4326")


def build_tile(root: Path, tile_dir: Path, out_dir: Path, buffer_deg: float, overwrite: bool,
               plot: str = "suitable") -> dict:
    tile_id = tile_dir.name
    gpkg = out_dir / f"{tile_id}.gpkg"
    row = {"tile_id": int(tile_id)}
    tile = gpd.read_file(tile_dir / "inputs" / "tile_geometry.gpkg").to_crs("EPSG:4326")
    minx, miny, maxx, maxy = tile.total_bounds
    bounds = (minx - buffer_deg, max(miny - buffer_deg, -89.0), maxx + buffer_deg, min(maxy + buffer_deg, 89.0))
    utm_crs = tile.estimate_utm_crs()
    area_utm = gpd.GeoSeries([box(*bounds)], crs="EPSG:4326").to_crs(utm_crs).iloc[0]

    stations = _stations_in(root, bounds)
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
                gpd.GeoDataFrame({"tile_id": [int(tile_id)]}, geometry=[geom], crs=utm_crs).to_crs(
                    "EPSG:4326").to_file(gpkg, layer=name, driver="GPKG")
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
                   ocean_enclosed_pct=round(100 * split["ocean_enclosed_frac"], 1),
                   suitable=best["n_conflicts"] == 0)
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


def _plot(tile_id, tile_dir, gpkg, mask, bounds, tile, out_path: Path, row: dict) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 9.5))
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    land_sea = np.where(mask == OCEAN_CODE, 0, 1)
    layers = {name for name, _ in pyogrio.list_layers(gpkg)} if gpkg.exists() else set()
    segs = gpd.read_file(gpkg, layer="segments") if "segments" in layers else None
    st = gpd.read_file(gpkg, layer="stations") if "stations" in layers else None

    # (a) lines
    ax.imshow(land_sea, extent=extent, cmap=ListedColormap([WATER_COLOR, "#e4e4e0"]), vmin=0, vmax=1,
              interpolation="nearest")
    tile.boundary.plot(ax=ax, color="black", linewidth=1.0)
    ax.plot([], [], color="black", linewidth=1.0, label="tile")
    msk_path = tile_dir / "sfincs_model" / "sfincs.msk"
    if msk_path.exists():
        from plot_sfincs_tile_diagnostics import BOUNDARY, _read_sfincs_mask
        m2d, tr, crs = _read_sfincs_mask(tile_dir / "sfincs_model")
        r, c = np.nonzero(m2d == BOUNDARY)
        lon, lat = warp_transform(crs, "EPSG:4326", *(tr * (c + 0.5, r + 0.5)))
        ax.scatter(lon, lat, s=1.5, color="#1f4e8c", alpha=0.6, label="current SFINCS boundary cells", zorder=3)
    if segs is not None:
        for kind, color in SEGMENT_KINDS.items():
            sel = segs[segs["kind"] == kind]
            if len(sel):
                sel.plot(ax=ax, color=color, linewidth=2.2, zorder=4)
                ax.plot([], [], color=color, linewidth=2.2, label=f"{kind} link (n={len(sel)})")
    if st is not None and "node_lon" in st:
        for x0, y0, x1, y1 in zip(st.geometry.x, st.geometry.y, st["node_lon"], st["node_lat"]):
            ax.plot([x0, x1], [y0, y1], color="#555555", linewidth=0.6, zorder=4)
    if st is not None:
        for status, (marker, color) in STATION_STYLE.items():
            sel = st[st["status"] == status] if "status" in st else (st if status == "used" else st.iloc[:0])
            if len(sel):
                ax.scatter(sel.geometry.x, sel.geometry.y, s=40 if marker == "o" else 55, marker=marker,
                           color=color, edgecolors="black", linewidths=0.6, zorder=5,
                           label=f"station: {status} (n={len(sel)})")
        for k, p in zip(st["station"], st.geometry):
            ax.annotate(str(k), (p.x, p.y), xytext=(4, 4), textcoords="offset points", fontsize=7, zorder=6)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    flags = [f"{k}={row[k]}" for k in ("n_parts", "n_open_ends", "n_flagged", "mode") if k in row]
    ax.set_title(f"(a) tile {tile_id}  boundary lines   " + "  ".join(flags), fontsize=10)
    ax.legend(loc="lower left", fontsize=8)

    # (b) active / inactive split of the tile
    tb = tile.total_bounds
    ax2.imshow(land_sea, extent=extent, cmap=ListedColormap([WATER_COLOR, "#e4e4e0"]), vmin=0, vmax=1,
               interpolation="nearest")
    handles = []
    for name, style, label in [
        ("inactive", dict(facecolor="#4d4d4d", alpha=0.55, edgecolor="none"), "inactive cells (ocean side)"),
        ("land_inactive", dict(facecolor="#d62728", alpha=0.85, edgecolor="none"), "CONFLICT: land inactive"),
        ("ocean_active", dict(facecolor="none", hatch="////", edgecolor="#ff7f0e", linewidth=0.0),
         "open sea left active (reaches tile edge)"),
    ]:
        if name in layers:
            gpd.read_file(gpkg, layer=name).plot(ax=ax2, **style, zorder=3)
            handles.append(plt.Rectangle((0, 0), 1, 1, **style, label=label))
    if segs is not None:
        segs.plot(ax=ax2, color="black", linewidth=1.6, zorder=4)
        handles.append(plt.Line2D([0], [0], color="black", linewidth=1.6, label="boundary line"))
    tile.boundary.plot(ax=ax2, color="black", linewidth=1.0, linestyle="--", zorder=4)
    ax2.set_xlim(tb[0], tb[2])
    ax2.set_ylim(tb[1], tb[3])
    verdict = "suitable" if row.get("suitable") else "NOT suitable (conflicts)"
    ax2.set_title(f"(b) tile domain: {verdict}   land inactive {row.get('land_inactive_pct', '-')} % of land, "
                  f"open sea active {row.get('ocean_active_pct', '-')} % of ocean", fontsize=10)
    if handles:
        ax2.legend(handles=handles, loc="lower left", fontsize=8)

    for a_ in (ax, ax2):
        a_.set_aspect(1 / math.cos(math.radians(0.5 * (bounds[1] + bounds[3]))))
        a_.set_xlabel("longitude (°E)")
        a_.set_ylabel("latitude (°N)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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
    root, tile_dir, out_dir, buffer_deg, overwrite, plot = args
    marker = _running_marker(out_dir, tile_dir.name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(os.getpid()))  # tells the parent which tiles were running if this worker dies
    try:
        row = build_tile(root, tile_dir, out_dir, buffer_deg, overwrite, plot)
    except Exception as exc:  # one broken tile must not stop the batch
        row = {"tile_id": int(tile_dir.name), "status": f"error: {exc!r}",
               "traceback": traceback.format_exc(limit=3)}
    _write_status(out_dir, row)
    marker.unlink(missing_ok=True)
    return row


def summarize(out_dir: Path) -> pd.DataFrame:
    """All per-tile status files -> summary.csv, suitable_tiles.txt, unsuitable_tiles.txt."""
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1]
                                                 / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="sfincs_calibration")
    parser.add_argument("--tile-ids", nargs="*", help="default: every tile with a built SFINCS model")
    parser.add_argument("--tile-ids-file", default=None, help="one tile id per line (e.g. one HPC job's share)")
    parser.add_argument("--sample", type=int, default=0, help="random sample of N tiles (seed 0) instead of all")
    parser.add_argument("--station-buffer-deg", type=float, default=0.5)
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
    if args.summarize:
        summarize(out_dir)
        return
    if args.tile_ids or args.tile_ids_file:
        ids = list(args.tile_ids or []) + (
            [t.strip() for t in Path(args.tile_ids_file).read_text().split() if t.strip()] if args.tile_ids_file else [])
        tile_dirs = [base / t for t in ids]
    else:
        tile_dirs = sorted((d for d in base.iterdir() if d.name.isdigit() and (d / "sfincs_model" / "sfincs.msk").exists()),
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

    jobs = [(root, d, out_dir, args.station_buffer_deg, args.overwrite, args.plot) for d in tile_dirs]
    rows = _run_jobs(jobs, args.workers, out_dir)
    if rows:
        print()
        _print_summary(pd.DataFrame(rows))
    print("(collect every job's results with: --summarize)")


if __name__ == "__main__":
    main()
