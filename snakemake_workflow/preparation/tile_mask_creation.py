"""Prepare an overlapping 2.5-degree tile grid for use as processing masks.

Step 1: build a 5x5 degree grid from the `deltadtm_mask` VRT's constituent
1x1 degree tiles (build_five_deg_grid_from_deltadtm). A 5-degree cell is
kept if at least one DeltaDTM tile falls inside it. DeltaDTM's mask tiles
cover roughly -69 to 84 degrees latitude, far beyond the +-60 degree extent
of the DiluviumDEM-derived `five_deg_grid` catalog source.

Step 2: filter that grid to cells that also have COAST-RP storm-tide
station coverage nearby (filter_grid_by_coastrp), so tiles are never
created where there is DeltaDTM terrain but no possible boundary forcing.
COAST-RP's raw station coverage is itself near-global (-84.7..83.65 deg
latitude), but prepare_boundary_conditions.py drops every station south of
boundary_conditions.coastrp_min_lat (config.yml; Antarctic stations,
deemed unreliable) before any station reaches extract_boundaries.py — so a
cell south of that latitude would never get boundary forcing regardless of
DeltaDTM coverage. This step applies that same cutoff plus a
coastline-proximity check, so it is filtered out here too rather than only
discovered later via an empty-boundaries skip in run_aqueduct.py.

Step 3: each 5x5 degree tile from that grid is split into four 2.5x2.5
degree quadrants. Each quadrant is then scaled by a factor of 1.5 around
its centre point, producing 3.75x3.75 degree tiles that overlap
neighbouring tiles by 0.625 degrees on each side. Each output tile receives
a unique tile_id.

Not a standalone entry point - exposes `run(config)`, called from
run_preparation.py (`python run_preparation.py tile_mask_creation`).
"""

import pathlib
import re
import sys

import geopandas as gpd
import rasterio
import xarray as xr
from shapely.affinity import scale
from shapely.geometry import box

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from config_utils import get_data_catalog  # noqa: E402

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent

# Matches the SW-corner tile naming convention shared by DeltaDTM and
# DiluviumDEM, e.g. "DeltaDTM_v1_1_N51W176.tif" -> lat_south=51, lon_west=-176.
_TILE_NAME_RE = re.compile(r"([NS])(\d{2})([EW])(\d{3})")


def _tile_sw_corner(filename: str) -> tuple[int, int] | None:
    """Parse a tile filename's SW corner as (lat_south, lon_west) in degrees.

    Returns None if the filename doesn't match the expected naming convention.
    """
    m = _TILE_NAME_RE.search(filename)
    if m is None:
        return None
    ns, lat, ew, lon = m.groups()
    lat_south = int(lat) * (1 if ns == "N" else -1)
    lon_west = int(lon) * (1 if ew == "E" else -1)
    return lat_south, lon_west


def build_five_deg_grid_from_deltadtm(mask_vrt_path: pathlib.Path) -> gpd.GeoDataFrame:
    """Build a 5x5 degree grid from the DeltaDTM mask VRT's constituent tiles.

    Every 1x1 degree DeltaDTM tile referenced by the VRT is binned into its
    enclosing 5-degree cell (cell_left = 5*floor(lon/5), cell_bottom =
    5*floor(lat/5)); a cell is kept if at least one DeltaDTM tile falls
    inside it. This mirrors how the DiluviumDEM-derived five_deg_grid was
    built (grouping that dataset's own 1-degree tile index into 5-degree
    blocks — see inputs/mask/readme.txt), just against DeltaDTM's tile
    index instead, which reaches much higher/lower latitudes.
    """
    with rasterio.open(mask_vrt_path) as src:
        tile_paths = [f for f in src.files if f.lower().endswith((".tif", ".tiff"))]

    cells: dict[tuple[int, int], list[str]] = {}
    for tile_path in tile_paths:
        corner = _tile_sw_corner(pathlib.Path(tile_path).name)
        if corner is None:
            continue
        lat_south, lon_west = corner
        cell_left = 5 * (lon_west // 5)
        cell_bottom = 5 * (lat_south // 5)
        cells.setdefault((cell_left, cell_bottom), []).append(pathlib.Path(tile_path).stem)

    rows = []
    for tile_id, ((left, bottom), tile_names) in enumerate(
        sorted(cells.items(), key=lambda kv: (-kv[0][1], kv[0][0]))
    ):
        top, right = bottom + 5, left + 5
        ns, ew = ("N" if bottom >= 0 else "S"), ("E" if left >= 0 else "W")
        grid_name = f"{ns}{abs(bottom):02d}{ew}{abs(left):03d}"
        rows.append(
            {
                "id": tile_id,
                "left": left,
                "right": right,
                "bottom": bottom,
                "top": top,
                "tiles_1deg": sorted(tile_names),
                "grid": grid_name,
                "geometry": box(left, bottom, right, top),
            }
        )

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def load_coastrp_stations(
    data_catalog, min_lat: float, x_var: str, y_var: str
) -> gpd.GeoDataFrame:
    """Load COAST-RP station points, dropping stations south of `min_lat`.

    `min_lat` should be boundary_conditions.coastrp_min_lat in config.yml —
    the same threshold applied by prepare_boundary_conditions.py's
    preprocess_coastrp — so this reflects what actually reaches
    extract_boundaries.py rather than COAST-RP's raw (near-global) station
    coverage. `x_var`/`y_var` should be boundary_conditions.station_x_var/
    station_y_var, matching the coordinate variable names
    prepare_boundary_conditions.py and extract_boundaries.py use.
    """
    coastrp_path = data_catalog.get_source("coast_rp").path
    ds = xr.open_dataset(coastrp_path)
    lon = ds[x_var].values
    lat = ds[y_var].values
    ds.close()
    keep = lat > min_lat
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon[keep], lat[keep]), crs="EPSG:4326")


def filter_grid_by_coastrp(
    grid: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, buffer_deg: float
) -> gpd.GeoDataFrame:
    """Keep only cells with at least one (non-Antarctic) COAST-RP station nearby.

    `buffer_deg` should match the quadrant-scaling overflow
    ((scale_factor-1) * quadrant_size / 2): a station just outside a
    5-degree cell's own bounds can still end up inside one of that cell's
    scaled quadrant tiles, so the coverage test must allow for that.
    Cells with no boundary forcing available would only ever produce
    empty-boundaries placeholders in run_aqueduct.py — filtering them out
    here avoids preprocessing/simulating tiles that can never flood.

    Re-numbers `id` sequentially (0..n-1) over the surviving rows.
    """
    buffered = grid.copy()
    buffered["geometry"] = buffered.geometry.buffer(buffer_deg)
    joined = gpd.sjoin(buffered, stations, predicate="intersects", how="inner")
    kept = grid[grid["id"].isin(joined["id"].unique())].reset_index(drop=True)
    kept["id"] = kept.index
    return kept


def split_into_quadrants(row):
    """Split a 5x5 degree tile into four 2.5x2.5 degree quadrant bounding boxes."""
    left, right, bottom, top = row["left"], row["right"], row["bottom"], row["top"]
    mid_x = (left + right) / 2
    mid_y = (bottom + top) / 2
    return {
        0: (left, bottom, mid_x, mid_y),  # SW
        1: (mid_x, bottom, right, mid_y),  # SE
        2: (left, mid_y, mid_x, top),  # NW
        3: (mid_x, mid_y, right, top),  # NE
    }


def run(config: dict) -> None:
    data_catalog = get_data_catalog(
        repo_root / config["paths"]["hydromt_data_catalog"], logger_name="Prepare tile masks"
    )

    mask_vrt_path = pathlib.Path(data_catalog.get_source("deltadtm_mask").path)
    grid = build_five_deg_grid_from_deltadtm(mask_vrt_path)
    n_deltadtm_only = len(grid)

    bc_cfg = config["boundary_conditions"]
    stations = load_coastrp_stations(
        data_catalog, bc_cfg["coastrp_min_lat"], bc_cfg["station_x_var"], bc_cfg["station_y_var"]
    )
    quadrant_size_deg = 5.0 / 2  # each 5-degree cell splits into four 2.5x2.5 quadrants
    buffer_deg = (config["one_off_edits"]["scale_factor"] - 1) * quadrant_size_deg / 2
    grid = filter_grid_by_coastrp(grid, stations, buffer_deg)

    five_deg_grid_path = pathlib.Path(config["one_off_edits"]["five_deg_grid_deltadtm"])
    five_deg_grid_path.parent.mkdir(parents=True, exist_ok=True)
    n_tiles_1deg = sum(len(t) for t in grid["tiles_1deg"])
    # GPKG/fiona has no list field type - serialize for the write only,
    # keeping the in-memory `grid` (used above/below) as real lists.
    grid_to_write = grid.assign(tiles_1deg=grid["tiles_1deg"].apply(",".join))
    grid_to_write.to_file(five_deg_grid_path, driver="GPKG")
    print(
        f"DeltaDTM coverage: {n_deltadtm_only} five-degree cells. "
        f"After requiring COAST-RP coverage too ({len(stations)} non-Antarctic "
        f"stations): {len(grid)} cells (from {n_tiles_1deg} DeltaDTM tiles, "
        f"latitude range {grid['bottom'].min()}..{grid['top'].max()}). "
        f"Saved to {five_deg_grid_path}"
    )

    tiles = []
    for _, row in grid.iterrows():
        for quadrant_id, bounds in split_into_quadrants(row).items():
            geom = scale(box(*bounds), xfact=config["one_off_edits"]["scale_factor"], yfact=config["one_off_edits"]["scale_factor"], origin="center")
            tiles.append(
                {
                    "tile_id": int(row["id"]) * 10 + quadrant_id,
                    "parent_grid": row["grid"],
                    "geometry": geom,
                }
            )

    tiles_gdf = gpd.GeoDataFrame(tiles, crs=grid.crs)
    smaller_tiles_path = config["one_off_edits"]["smaller_tiles"]
    tiles_gdf.to_file(smaller_tiles_path, driver="GPKG")
    print(f"Saved {len(tiles_gdf)} tiles to {smaller_tiles_path}")


if __name__ == "__main__":
    sys.exit(
        "tile_mask_creation.py is no longer a standalone entry point.\n"
        "Run it via: python run_preparation.py tile_mask_creation\n"
        "See run_preparation.py --help for the full list of steps."
    )
