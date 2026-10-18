"""Geometry core of the SFINCS station boundary lines (see build_station_boundary_lines.py).

Coastline tracing, station linking, extension, conflict resolution and the domain
check, plus the tuning constants they share. Split out of
build_station_boundary_lines.py (cleanup, 2026-10-18) as a pure move: every function
and constant is unchanged; that module re-exports the names other modules use.
"""

from __future__ import annotations
import math
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
import shapely
import xarray as xr
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point, shape
from shapely.ops import linemerge, substring, unary_union

OCEAN_CODE = 1
TILE_STATIONS_GPKG = "inputs/boundaries_RP100_SLR_0.gpkg"  # the tile's own production station selection
COAST_HG_NC = "inputs/COAST_HG/COAST-HG_RP100.nc"
MAX_HG_MATCH_DEG = 0.5  # same backstop build_line_forcing.py uses
MIN_FORCING_STATIONS = 2  # a tile needs this many used stations with a COAST-HG hydrograph
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
    polygon shrunk by d - narrow bays close by themselves), cached per d.

    Parameters
    ----------
    mask : np.ndarray
        Land/ocean mask read from the DeltaDTM VRT (OCEAN_CODE marks ocean).
    transform : affine.Affine
        Pixel-to-CRS (lon/lat) transform of `mask`.
    area_utm : shapely.Polygon
        Search-area bounding box, in `utm_crs`.
    utm_crs : pyproj.CRS
        UTM CRS the polygons and coastline pieces are built in.
    pixel_m : float
        Approximate mask pixel size in metres, used to size buffers.
    """

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


def _search_box(tile: gpd.GeoDataFrame, buffer_deg: float, min_size_deg: float) -> tuple:
    """Production's station search box (boundaries.select_stations_for_tile):
    the envelope of the tile bbox + buffer_deg and a min_size_deg box centred
    on the tile."""
    minx, miny, maxx, maxy = tile.total_bounds
    cx, cy, half = (minx + maxx) / 2, (miny + maxy) / 2, min_size_deg / 2
    return (min(minx - buffer_deg, cx - half), max(min(miny - buffer_deg, cy - half), -89.0),
            max(maxx + buffer_deg, cx + half), min(max(maxy + buffer_deg, cy + half), 89.0))


def _tile_stations(tile_dir: Path) -> gpd.GeoDataFrame:
    """The tile's own pre-selected COAST-RP stations (production's
    boundaries_RP100_SLR_0.gpkg: search box + ocean-connectivity filter,
    MDT-corrected, int16 cm)."""
    st = gpd.read_file(tile_dir / TILE_STATIONS_GPKG).to_crs("EPSG:4326").reset_index(drop=True)
    level = st["SLR_0"].to_numpy(dtype=np.float64) / 100.0
    return gpd.GeoDataFrame({"station_index": np.arange(len(st)), "rp100_level_m": level},
                            geometry=st.geometry.values, crs="EPSG:4326")


_HG_XY = {}


def _n_with_coast_hg(root: Path, lonlat: np.ndarray) -> int:
    """How many of these stations have a COAST-HG hydrograph within MAX_HG_MATCH_DEG."""
    if "xy" not in _HG_XY:
        with xr.open_dataset(root / COAST_HG_NC) as ds:
            _HG_XY["xy"] = np.column_stack([ds.station_x_coordinate.values, ds.station_y_coordinate.values])
            _HG_XY["tree"] = cKDTree(_HG_XY["xy"])
    if len(lonlat) == 0:
        return 0
    d, _ = _HG_XY["tree"].query(lonlat)
    return int((d <= MAX_HG_MATCH_DEG).sum())


def _clean(geom):
    """Valid, single-type geometry for writing (make_valid can return a
    GeometryCollection with stray points/lines next to the polygons)."""
    geom = shapely.make_valid(geom)
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in shapely.get_parts(geom) if g.geom_type in ("Polygon", "MultiPolygon")]
        geom = unary_union(parts) if parts else shapely.Polygon()
    return geom
