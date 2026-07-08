"""Functions for loading tiles from the overlapping tile grid and selecting
tiles for which the DEM has data coverage.
"""

from pathlib import Path

import geopandas as gpd
import hydromt
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box
from shapely.ops import unary_union


def load_tile_grid(tile_grid_path: str | Path) -> gpd.GeoDataFrame:
    """Load the overlapping tile grid produced by `tile_mask_creation.py`."""
    return gpd.read_file(tile_grid_path)


def get_tile_geometry(tile_grid: gpd.GeoDataFrame, tile_id: int) -> gpd.GeoDataFrame:
    """Select a single tile from the tile grid by its tile_id.

    Args:
        tile_grid: The full tile grid, as returned by `load_tile_grid`.
        tile_id: Unique identifier of the tile to select.

    Returns:
        A single-row GeoDataFrame containing the selected tile.

    Raises:
        ValueError: If `tile_id` is not present in `tile_grid`.
    """
    tile = tile_grid[tile_grid["tile_id"] == tile_id]
    if tile.empty:
        raise ValueError(f"tile_id {tile_id} not found in tile grid")
    return tile.reset_index(drop=True)


def save_tile_geometry(tile: gpd.GeoDataFrame, output_path: str | Path) -> None:
    """Save a single tile geometry to a GeoPackage file."""
    tile.to_file(output_path, driver="GPKG")


def _coord_str(lat: int, lon: int) -> str:
    """Format a 1°×1° SW-corner (lat, lon) as the {NS}{lat:02d}{EW}{lon:03d} token."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"


def _scan_mask_dir(mask_dir: Path) -> dict[str, Path]:
    """Scan *mask_dir* once and return a {coord_str: path} dict for every .tif found.

    Files are indexed by their embedded {NS}{lat:02d}{EW}{lon:03d} coordinate
    token so any naming prefix or suffix is tolerated.
    """
    import re
    coord_re = re.compile(r"([NS])(\d{2})([EW])(\d{3})")
    index: dict[str, Path] = {}
    for p in mask_dir.glob("*.tif"):
        m = coord_re.search(p.name)
        if m:
            index[m.group(0)] = p
    return index


def _degree_tiles_for_bbox(bbox: tuple[float, float, float, float]):
    """Yield (lat_sw, lon_sw) integer pairs for all 1°×1° cells that overlap *bbox*."""
    import math
    minx, miny, maxx, maxy = bbox
    for lat in range(math.floor(miny), math.ceil(maxy)):
        for lon in range(math.floor(minx), math.ceil(maxx)):
            yield lat, lon


def _mask_tile_has_land(path: Path) -> bool:
    """Return True if the mask raster contains any non-ocean, non-nodata value.

    Mask values: 0=land, 1=ocean, 2=lake, 3=river, 255=nodata.
    A tile is considered to have DEM coverage when at least one pixel has a
    value other than 1 (ocean) or 255 (nodata).
    """
    with rasterio.open(path) as src:
        data = src.read(1)
    return bool(np.any((data != 1) & (data != 255)))


def filter_tiles_by_dem_mask(
    tile_grid: gpd.GeoDataFrame,
    mask_dir: str | Path,
    discarded_tiles_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Filter a tile grid to tiles covered by the DeltaDTM validity mask.

    For each tile the function checks all 1°×1° DeltaDTM mask files whose
    extent overlaps the tile's bounding box.  A tile is **kept** if at least
    one mask pixel in any of those files has a value of 0 (land), 2 (lake),
    or 3 (river).  A tile is **discarded** if every overlapping mask file
    contains only ocean (1) and nodata (255) pixels — meaning DeltaDTM has
    no terrain data there.

    The mask directory is scanned once upfront so each per-tile lookup is an
    O(1) dict look-up rather than a filesystem search.

    Args:
        tile_grid: Tile grid, as returned by `load_tile_grid`.
        mask_dir: Directory containing the 1°×1° DeltaDTM mask GeoTIFF tiles.
            Filenames must embed the standard {NS}{lat:02d}{EW}{lon:03d}
            coordinate token (any prefix/suffix is accepted).
        discarded_tiles_path: If given, the discarded tiles (full rows, with
            geometry) are written to this GeoPackage file for visual
            confirmation (e.g. in QGIS).

    Returns:
        The subset of `tile_grid` with DeltaDTM mask coverage.
    """
    mask_dir = Path(mask_dir)
    mask_index = _scan_mask_dir(mask_dir)

    keep: list[bool] = []

    for _, row in tile_grid.iterrows():
        tile_id = int(row["tile_id"])
        bbox = row.geometry.bounds   # (minx, miny, maxx, maxy)
        has_coverage = False

        for lat, lon in _degree_tiles_for_bbox(bbox):
            coord = _coord_str(lat, lon)
            mask_path = mask_index.get(coord)
            if mask_path is not None and _mask_tile_has_land(mask_path):
                has_coverage = True
                break  # one covering tile with land is enough

        keep.append(has_coverage)
        if not has_coverage:
            print(f"Tile {tile_id}: no DeltaDTM mask coverage, discarding")

    discarded_gdf = tile_grid[[not k for k in keep]]
    if len(discarded_gdf) and discarded_tiles_path is not None:
        discarded_gdf.to_file(discarded_tiles_path, driver="GPKG")
        print(f"Wrote {len(discarded_gdf)} discarded tiles to {discarded_tiles_path}")

    return tile_grid[keep].reset_index(drop=True)


def filter_tiles_by_exposure(
    tile_grid: gpd.GeoDataFrame,
    data_catalog: hydromt.DataCatalog,
    population_source: str,
    discarded_tiles_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Filter a tile grid to tiles with at least some population exposure.

    For each tile, clips `population_source` (WorldPop population count,
    ~1km resolution - see config.yml's exposure.population_source) to the
    tile's bounding box. A tile is **kept** if the clipped area has any
    positive population; **discarded** if every valid (non-nodata) pixel is
    zero, or the tile's bbox has no coverage in the population raster at
    all (e.g. far outside WorldPop's own latitude range) - either way,
    there is nobody there for a flood to expose.

    Args:
        tile_grid: Tile grid, as returned by `filter_tiles_by_dem_mask`.
        data_catalog: HydroMT data catalog containing `population_source`.
        population_source: Catalog key for the population RasterDataset.
        discarded_tiles_path: If given, the discarded tiles (full rows, with
            geometry) are written to this GeoPackage file for visual
            confirmation (e.g. in QGIS).

    Returns:
        The subset of `tile_grid` with at least some population exposure.
    """
    keep: list[bool] = []

    for _, row in tile_grid.iterrows():
        tile_id = int(row["tile_id"])
        bbox = list(row.geometry.bounds)
        try:
            da = data_catalog.get_rasterdataset(population_source, bbox=bbox)
        except Exception:
            has_exposure = False
        else:
            vals = da.values
            nodata = da.raster.nodata
            valid = vals != nodata if nodata is not None else np.isfinite(vals)
            has_exposure = bool(np.nansum(vals[valid]) > 0) if valid.any() else False

        keep.append(has_exposure)
        if not has_exposure:
            print(f"Tile {tile_id}: no population exposure, discarding")

    discarded_gdf = tile_grid[[not k for k in keep]]
    if len(discarded_gdf) and discarded_tiles_path is not None:
        discarded_gdf.to_file(discarded_tiles_path, driver="GPKG")
        print(f"Wrote {len(discarded_gdf)} discarded tiles to {discarded_tiles_path}")

    return tile_grid[keep].reset_index(drop=True)


def _compute_fractions_from_tiles(
    bbox: tuple[float, float, float, float],
    mask_index: dict[str, Path],
) -> tuple[float, float, float]:
    """Compute ocean, land, and mask-coverage fractions from individual mask files.

    Reads only the pixels that fall within ``bbox`` by clipping each 1°×1°
    mask tile to its intersection with the bounding box.  This avoids the VRT
    gap ambiguity: pixels in uncovered areas are simply not counted rather than
    being filled with an arbitrary value that could be misinterpreted.

    Valid mask values are 0 (land), 1 (ocean), 2 (lake), 3 (river).  The
    mask files have no defined nodata value; any pixel inside a file is real.

    Returns:
        ``(ocean_fraction, land_fraction, mask_fraction)`` where:
        - ``ocean_fraction`` = ocean pixels / total covered pixels
        - ``land_fraction``  = land/lake/river pixels / total covered pixels
        - ``mask_fraction``  = covered pixels / estimated total tile pixels
    """
    minx, miny, maxx, maxy = bbox
    # Approximate total pixel count at 1 arcsec resolution
    total_pixels = max(1, round((maxx - minx) * 3600) * round((maxy - miny) * 3600))

    n_ocean = 0
    n_land = 0
    n_covered = 0

    for lat, lon in _degree_tiles_for_bbox(bbox):
        coord = _coord_str(lat, lon)
        path = mask_index.get(coord)
        if path is None:
            continue

        # Clip to the intersection of this 1°×1° mask tile and the grid bbox
        ix0 = max(minx, lon)
        iy0 = max(miny, lat)
        ix1 = min(maxx, lon + 1)
        iy1 = min(maxy, lat + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue

        with rasterio.open(path) as src:
            # fill_value=255 guards against the rare sub-pixel floating-point
            # overshoot at tile boundaries; 255 is not a valid mask value.
            window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
            data = src.read(1, window=window, boundless=True, fill_value=255)

        valid = data <= 3  # values 0-3 are real; 255 (fill) is excluded
        n_ocean   += int(np.sum((data == 1) & valid))
        n_land    += int(np.sum(np.isin(data, [0, 2, 3]) & valid))
        n_covered += int(valid.sum())

    mask_fraction = n_covered / total_pixels
    if n_covered == 0:
        return 0.0, 0.0, mask_fraction
    return n_ocean / n_covered, n_land / n_covered, mask_fraction


def _mosaic_mask_for_trim(
    bbox: tuple[float, float, float, float],
    mask_index: dict[str, Path],
) -> tuple[np.ndarray, Affine] | None:
    """Mosaic all 1°×1° DeltaDTM mask tiles overlapping ``bbox`` into one array.

    Used only by `compute_trimmed_bbox` — kept separate from
    `_compute_fractions_from_tiles` (which stays on its original per-file-loop
    implementation, unchanged) because pre-rounding each source window to
    place it into a shared array shifts pixel content by fractions of a
    pixel relative to the unrounded fractional-window reads
    `_compute_fractions_from_tiles` relies on (confirmed empirically: the two
    approaches differ by ~1e-4 in the resulting fractions - small, but a real,
    unexplained bias not worth introducing into an already-relied-upon
    computation that drives real merge/drop decisions). The shave algorithm
    below only needs "is this row/column pure ocean" booleans, so the ~1
    pixel-scale imprecision from rounding is inconsequential, especially
    against the generous buffer_arcsec margin kept beyond the land edge.

    Safe at single-tile scale (a handful of source files, ~180M pixels worst
    case for the largest nominal tile) — unlike the *global*-mosaic approach
    abandoned elsewhere in this codebase (see
    analysis/compute_exposure_analysis.py's docstring: merging 100 scenarios
    x 750M pixels each needed hundreds of GB and was replaced by chunk
    streaming).

    Returns ``(band, transform)`` for the single mask band, or ``None`` if no
    mask file overlaps ``bbox`` at all. Areas inside ``bbox`` not covered by
    any mask file are filled with 255 (not a valid mask value).

    DeltaDTM mask tiles do NOT all share one native resolution: y-resolution
    is a constant 1 arcsec, but x-resolution coarsens at higher latitudes to
    compensate for longitude convergence near the poles (confirmed
    empirically, e.g. 3 arcsec at 76-77N vs 5 arcsec at 80-81N). A bbox
    spanning such a boundary would place mismatched array shapes if source
    windows were read at each file's own resolution. To handle this, the
    combined array uses the FINEST resolution found among the overlapping
    files, and every source window is read with `out_shape`/nearest-neighbour
    resampling to exactly match its destination slot - safe for this
    categorical mask (values 0/1/2/3/255), unlike averaging resamplers.
    """
    minx, miny, maxx, maxy = bbox
    windows = []  # (path, ix0, iy0, ix1, iy1)
    resolutions = []  # (px_size_x, px_size_y) per overlapping file
    for lat, lon in _degree_tiles_for_bbox(bbox):
        coord = _coord_str(lat, lon)
        path = mask_index.get(coord)
        if path is None:
            continue
        ix0 = max(minx, lon)
        iy0 = max(miny, lat)
        ix1 = min(maxx, lon + 1)
        iy1 = min(maxy, lat + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        windows.append((path, ix0, iy0, ix1, iy1))
        with rasterio.open(path) as src:
            resolutions.append((abs(src.transform.a), abs(src.transform.e)))

    if not windows:
        return None

    # Finest (smallest) resolution among all overlapping files - avoids
    # downsampling/losing precision from any single source.
    px_size_x = min(r[0] for r in resolutions)
    px_size_y = min(r[1] for r in resolutions)

    transform = Affine.translation(minx, maxy) * Affine.scale(px_size_x, -px_size_y)
    out_width = max(1, round((maxx - minx) / px_size_x))
    out_height = max(1, round((maxy - miny) / px_size_y))
    out = np.full((out_height, out_width), 255, dtype=np.uint8)

    for path, ix0, iy0, ix1, iy1 in windows:
        dst_window = from_bounds(ix0, iy0, ix1, iy1, transform).round_lengths().round_offsets()
        r0, c0 = int(dst_window.row_off), int(dst_window.col_off)
        h, w = int(dst_window.height), int(dst_window.width)
        if h <= 0 or w <= 0:
            continue
        with rasterio.open(path) as src:
            src_window = from_bounds(ix0, iy0, ix1, iy1, src.transform)
            data = src.read(
                1, window=src_window, boundless=True, fill_value=255,
                out_shape=(h, w), resampling=Resampling.nearest,
            )
        out[r0:r0 + h, c0:c0 + w] = data

    return out, transform


def compute_trimmed_bbox(
    bbox: tuple[float, float, float, float],
    mask_index: dict[str, Path],
    buffer_arcsec: float,
) -> tuple[float, float, float, float]:
    """Shave pure-ocean/nodata edge rows and columns from ``bbox``.

    Mosaics all overlapping 1°×1° mask tiles (via `_mosaic_mask_for_trim`),
    then peels rows/columns off each of the four edges inward while every
    pixel in that row/column is ocean (1) or uncovered/nodata (255) —
    stopping at the first row/column containing any land/lake/river pixel
    (0, 2 or 3). ``buffer_arcsec`` worth of additional pure-ocean/nodata
    rows/columns are then kept beyond that edge as a coastal margin, clamped
    to the mosaic's own array bounds. The result can never exceed the input
    ``bbox``, since the mosaic itself is built from it — no separate
    clip-back step is needed.

    If no land/lake/river pixel is found anywhere in ``bbox`` (pure ocean, or
    no DeltaDTM coverage at all), ``bbox`` is returned unchanged. This
    matters because pre-merge trim runs *before* `merge_undersized_tiles`'s
    ``land_fraction == 0`` drop, so it must degrade gracefully here.
    """
    mosaic = _mosaic_mask_for_trim(bbox, mask_index)
    if mosaic is None:
        return bbox
    band, transform = mosaic

    is_land = np.isin(band, [0, 2, 3])
    if not is_land.any():
        return bbox

    row_has_land = is_land.any(axis=1)
    col_has_land = is_land.any(axis=0)
    r0 = int(np.argmax(row_has_land))
    r1 = len(row_has_land) - 1 - int(np.argmax(row_has_land[::-1]))
    c0 = int(np.argmax(col_has_land))
    c1 = len(col_has_land) - 1 - int(np.argmax(col_has_land[::-1]))

    buffer_px = round((buffer_arcsec / 3600.0) / abs(transform.a))
    r0 = max(0, r0 - buffer_px)
    r1 = min(band.shape[0] - 1, r1 + buffer_px)
    c0 = max(0, c0 - buffer_px)
    c1 = min(band.shape[1] - 1, c1 + buffer_px)

    window = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
    return window_bounds(window, transform)


def compute_trimmed_geometries(
    tile_grid: gpd.GeoDataFrame,
    mask_dir: str | Path,
    buffer_arcsec: float,
) -> gpd.GeoSeries:
    """Compute a trimmed bbox (as a shapely box) for every row of ``tile_grid``.

    Reads each row's ``geometry`` bounds and calls `compute_trimmed_bbox`.
    Returns a GeoSeries aligned to ``tile_grid``'s index and CRS. Called
    before merging (trimming becomes every tile's real working geometry from
    then on) and again after (to re-tighten any slack left by bbox unions).
    """
    mask_index = _scan_mask_dir(Path(mask_dir))
    trimmed = []
    n = len(tile_grid)
    print(f"Computing trimmed bboxes for {n} tiles...", flush=True)
    for i, (_, row) in enumerate(tile_grid.iterrows()):
        trimmed_bbox = compute_trimmed_bbox(row.geometry.bounds, mask_index, buffer_arcsec)
        trimmed.append(box(*trimmed_bbox))
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} tiles", flush=True)
    return gpd.GeoSeries(trimmed, index=tile_grid.index, crs=tile_grid.crs)


def compute_tile_fractions(
    tile_grid: gpd.GeoDataFrame,
    mask_dir: str | Path,
) -> gpd.GeoDataFrame:
    """Compute ocean, land, and DeltaDTM coverage fractions for each tile.

    Reads individual 1°×1° DeltaDTM mask tile files (not the VRT mosaic) to
    avoid VRT gap ambiguity: uncovered areas are simply absent rather than
    carrying a potentially misleading fill value.

    Three fractions are computed per tile:

    - ``ocean_fraction``: share of covered pixels that are ocean (mask = 1).
    - ``land_fraction``:  share of covered pixels that are land, lake or river
      (mask = 0, 2 or 3).  ``ocean_fraction + land_fraction = 1``.
    - ``mask_fraction``:  share of the tile's total pixel area (at 1 arcsec
      resolution) that is covered by at least one mask tile.  A low value
      means the tile mostly lies outside the DeltaDTM coastal coverage zone.
    - ``nodata_fraction``: ``1 - mask_fraction`` - the complementary share of
      the tile with no real DeltaDTM data at all. Diagnostic only (not used
      by `merge_undersized_tiles`'s merge decisions - a tile's ocean/land
      split among its *covered* pixels is what drives those).

    Args:
        tile_grid: Tile grid, as returned by `load_tile_grid`.
        mask_dir: Directory containing the 1°×1° DeltaDTM mask GeoTIFF tiles.

    Returns:
        ``tile_grid`` with added ``ocean_fraction``, ``land_fraction``,
        ``mask_fraction`` and ``nodata_fraction`` columns.
    """
    mask_index = _scan_mask_dir(Path(mask_dir))
    ocean_fractions: list[float] = []
    land_fractions:  list[float] = []
    mask_fractions:  list[float] = []

    n = len(tile_grid)
    print(f"Computing fractions for {n} tiles...", flush=True)
    for i, (_, row) in enumerate(tile_grid.iterrows()):
        ocean_f, land_f, mask_f = _compute_fractions_from_tiles(
            row.geometry.bounds, mask_index
        )
        ocean_fractions.append(ocean_f)
        land_fractions.append(land_f)
        mask_fractions.append(mask_f)
        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  {i + 1}/{n} tiles", flush=True)

    tile_grid = tile_grid.copy()
    tile_grid["ocean_fraction"]  = ocean_fractions
    tile_grid["land_fraction"]   = land_fractions
    tile_grid["mask_fraction"]   = mask_fractions
    tile_grid["nodata_fraction"] = [1.0 - m for m in mask_fractions]
    return tile_grid


def _is_cardinal_neighbor(geom_a, geom_b, threshold: float = 0.5) -> bool:
    """Check whether two axis-aligned tile geometries are immediate top/bottom/left/right neighbors.

    Two tiles are cardinal neighbors if their bounding boxes overlap and
    that overlap spans at least `threshold` of the smaller tile's width (a
    north/south relationship) or height (an east/west relationship). Tiles
    that only overlap in a small corner square (diagonal neighbors) fail
    both checks and are excluded.

    Args:
        geom_a: First tile geometry.
        geom_b: Second tile geometry.
        threshold: Minimum fraction of the smaller tile's width or height
            that the overlap must span to count as a cardinal neighbor.

    Returns:
        True if `geom_a` and `geom_b` are cardinal (N/S/E/W) neighbors.
    """
    ax0, ay0, ax1, ay1 = geom_a.bounds
    bx0, by0, bx1, by1 = geom_b.bounds

    overlap_x = min(ax1, bx1) - max(ax0, bx0)
    overlap_y = min(ay1, by1) - max(ay0, by0)
    if overlap_x <= 0 or overlap_y <= 0:
        return False

    width = min(ax1 - ax0, bx1 - bx0)
    height = min(ay1 - ay0, by1 - by0)
    return overlap_x >= threshold * width or overlap_y >= threshold * height


def merge_undersized_tiles(
    tile_grid: gpd.GeoDataFrame,
    min_coast_fraction: float,
    max_merge_count: int = 4,
) -> gpd.GeoDataFrame:
    """Merge tiles with too little ocean or land into a neighbour, in two phases.

    Expects ``tile_grid``'s ``geometry`` column to ALREADY be each tile's
    trimmed working shape (call `compute_trimmed_geometries` first) — this
    function is purely geometric/tabular (no mask/raster access) and treats
    ``geometry`` as final for both adjacency and the merged output.

    ``ocean_fraction``/``land_fraction`` (from `compute_tile_fractions`, on
    each tile's ORIGINAL nominal footprint) drive classification throughout
    and are never recomputed here — trimming a tile first would trivially
    inflate its own fraction and defeat the thresholds' purpose. Since both
    fractions are complementary (sum to 1, computed over the same covered
    pixels) and ``min_coast_fraction`` is well under 0.5, a tile can only
    ever be water-deficient or land-deficient, never both at once.

    0. Tiles with ``land_fraction == 0`` (pure ocean, nothing to model) are
       dropped outright.
    1. **Water-deficient tiles** (``ocean_fraction < min_coast_fraction``)
       each unionize with whichever cardinal neighbour (any status — good,
       water-deficient or land-deficient) has the HIGHEST ``ocean_fraction``
       — directly targets the tile's actual deficiency, rather than
       re-testing the merged union's own fractions (which a diluted-by-
       nominal-footprint recheck could wrongly reject, e.g. two
       overlapping-grid tiles whose only shared content is one small
       island). A single one-shot merge per tile; if the result is still
       water-deficient, it isn't reprocessed within this phase.
    2. **Land-deficient tiles** (``land_fraction < min_coast_fraction``,
       excluding the ``== 0`` tiles already dropped in step 0) each unionize
       with whichever cardinal neighbour has the highest ``land_fraction``,
       mirroring phase 1.

    Both phases respect ``max_merge_count`` (tracked across both phases, not
    reset in between) and fall back to keeping a tile as-is when no cardinal
    neighbour exists at all.

    Args:
        tile_grid: Tile grid with ``ocean_fraction`` and ``land_fraction``
            columns (as returned by `compute_tile_fractions`) and an
            already-trimmed ``geometry`` column.
        min_coast_fraction: Minimum required ocean fraction and land fraction.
        max_merge_count: Maximum number of original tiles that may be combined
            into a single output tile (default: 4), enforced across both
            phases together.

    Returns:
        ``tile_grid`` with deficient tiles merged or dropped and the
        fraction columns removed (they no longer apply to tiles whose
        geometry changed).
    """
    grid = tile_grid.set_index("tile_id", drop=False)

    # 0. Drop tiles with no land at all (100% ocean) outright.
    no_land_ids = grid.index[grid["land_fraction"] == 0].tolist()
    for tile_id in no_land_ids:
        print(f"  tile {tile_id}: drop (no land)", flush=True)
    grid = grid.drop(index=no_land_ids)

    tile_count: dict = {tid: 1 for tid in grid.index}

    def _run_phase(bad_ids: list, rank_column: str, phase_name: str) -> None:
        nonlocal grid
        n = len(bad_ids)
        print(f"Phase: resolving {n} {phase_name} tiles...", flush=True)
        for i, tile_id in enumerate(bad_ids):
            if tile_id not in grid.index:
                continue  # already absorbed by an earlier merge this phase

            own_geom = grid.loc[tile_id, "geometry"]
            candidates = []
            for other_id in grid.index:
                if other_id == tile_id:
                    continue
                if tile_count[tile_id] + tile_count[other_id] > max_merge_count:
                    continue
                other_geom = grid.loc[other_id, "geometry"]
                if not _is_cardinal_neighbor(own_geom, other_geom):
                    continue
                candidates.append((other_id, grid.loc[other_id, rank_column]))

            if not candidates:
                print(f"  [{i+1}/{n}] tile {tile_id}: keep (no suitable neighbour)", flush=True)
                continue

            chosen_id = max(candidates, key=lambda c: c[1])[0]
            print(f"  [{i+1}/{n}] tile {tile_id}: unionize with {chosen_id} (highest {rank_column})", flush=True)
            merged_geom = box(*unary_union([own_geom, grid.loc[chosen_id, "geometry"]]).bounds)
            grid.loc[chosen_id, "geometry"] = merged_geom
            tile_count[chosen_id] += tile_count[tile_id]
            grid = grid.drop(index=tile_id)

    # 1. Water-deficient tiles unionize with their highest-ocean_fraction neighbour.
    water_bad_ids = grid.index[grid["ocean_fraction"] < min_coast_fraction].tolist()
    _run_phase(water_bad_ids, "ocean_fraction", "water-deficient")

    # 2. Land-deficient tiles unionize with their highest-land_fraction neighbour.
    land_bad_ids = grid.index[grid["land_fraction"] < min_coast_fraction].tolist()
    _run_phase(land_bad_ids, "land_fraction", "land-deficient")

    drop_cols = [c for c in ("ocean_fraction", "land_fraction", "mask_fraction", "nodata_fraction") if c in grid.columns]
    return grid.drop(columns=drop_cols).reset_index(drop=True)


def deduplicate_overlapping_tiles(
    tile_grid: gpd.GeoDataFrame,
    iou_threshold: float = 0.8,
    max_merge_count: int = 4,
) -> gpd.GeoDataFrame:
    """Consolidate tiles whose geometries are near-total duplicates of each other.

    `merge_undersized_tiles` only ever compares a BAD tile against its
    neighbours, so two tiles that both individually pass the fraction
    thresholds - but happen to trim down to (nearly) the same physical
    feature, e.g. two overlapping-grid tiles whose only shared land content
    is one small island - are never compared to each other there and can
    both survive as near-duplicate final tiles. This runs as a separate pass
    over ALL tiles (regardless of original good/bad status), intended to be
    called on the already-trimmed final geometry (post `merge_undersized_tiles`
    + post-merge trim).

    Uses intersection-over-union (IoU), not a one-sided "overlap / smaller
    tile's area" ratio: the overlapping tile grid design deliberately gives
    many legitimately-distinct neighbouring tiles high one-sided overlap
    (e.g. a small trimmed tile fully contained in a much larger neighbour's
    trimmed box without covering the same feature), which a one-sided ratio
    would falsely flag. IoU only approaches 1.0 when both tiles are close to
    identical in extent, which correctly isolates true duplicates - confirmed
    empirically on a regional test grid: the one genuine duplicate pair had
    IoU=1.000, while the next-highest (legitimate, distinct) neighbouring
    pair topped out at IoU=0.645, a wide margin below the 0.8 default.

    Args:
        tile_grid: Tile grid to deduplicate (operates on whatever
            ``geometry`` column it's given - call after trimming).
        iou_threshold: Minimum intersection-over-union to consider two tiles
            duplicates worth consolidating (default 0.8).
        max_merge_count: Safety cap on how many original tiles may be
            combined into one, tracked independently of any prior merge step.

    Returns:
        ``tile_grid`` with duplicate pairs consolidated (bounding-box union).
    """
    grid = tile_grid.set_index("tile_id", drop=False)
    tile_count: dict = {tid: 1 for tid in grid.index}

    merged_any = True
    while merged_any:
        merged_any = False
        sindex = grid.sindex
        id_array = np.array(grid.index)  # snapshot matching this sindex build
        processed = set()
        for tile_id in list(grid.index):
            if tile_id in processed or tile_id not in grid.index:
                continue
            geom = grid.loc[tile_id, "geometry"]
            for pos in sindex.intersection(geom.bounds):
                other_id = id_array[pos]
                if other_id == tile_id or other_id in processed or other_id not in grid.index:
                    continue
                if tile_count[tile_id] + tile_count[other_id] > max_merge_count:
                    continue
                other_geom = grid.loc[other_id, "geometry"]
                inter = geom.intersection(other_geom).area
                if inter <= 0:
                    continue
                union_area = geom.union(other_geom).area
                if union_area <= 0 or inter / union_area < iou_threshold:
                    continue
                print(f"  dedup: merge {tile_id} and {other_id} (IoU={inter / union_area:.2f})", flush=True)
                geom = box(*unary_union([geom, other_geom]).bounds)
                grid.loc[tile_id, "geometry"] = geom
                tile_count[tile_id] += tile_count[other_id]
                grid = grid.drop(index=other_id)
                processed.add(other_id)
                merged_any = True
            processed.add(tile_id)

    return grid.reset_index(drop=True)
