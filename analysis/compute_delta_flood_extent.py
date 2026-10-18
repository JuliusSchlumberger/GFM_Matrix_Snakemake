"""Per-delta flooded-extent CSV across both delta scenarios (ssp126/ssp245)
and all 5 SLR magnitudes, RP100 - flooded area in km2 AND as a percentage of
each delta's own REAL outline area (not its bounding box).

Deltas come from `inputs/DeltasGlobal/all_deltas.shp` (EPSG:4326, 49
polygons, fields `Id`/`Area` only - same file
`preparation/build_delta_tile_subset.py` already uses to select tiles, same
hardcoded path convention). Delta `Id=44` (Volga) is dropped - confirmed
zero tile coverage in both scenarios.

Tile<->delta attribution is NOT persisted anywhere upstream (the tile
selection step's own sjoin, against each delta's buffered BBOX for
simulation-coverage purposes, is never saved) - this script redoes that
join itself, against each delta's REAL polygon, via the shared
`deltas_global_tiles.gpkg` tile grid both scenarios' materialized configs
point at.

Production tiles genuinely OVERLAP each other (confirmed directly: tile
grid core polygons for tiles covering the Amazon delta sum to 1.65x their
own union area) - a first version of this script naively summed each
covering tile's own contribution independently and got 120-380% "coverage"
for every delta as a result. Fixed by applying the SAME per-cell MAXIMUM
combine rule `src.merge.merge_tile_rasters_chunk` uses in production (see
that function's own docstring for why max, not sum/mean) - each delta's
covering tiles are merged onto one grid (chunk_bounds = the delta's own
bbox) BEFORE any area/flood accounting happens, so every real cell is
counted exactly once regardless of how many tiles' footprints reach it.

Merging happens IN MEMORY, not via that production function directly - a
single delta's own bbox comfortably fits in memory (confirmed: even the
largest, Amazon, is ~287M pixels at 30m resolution, ~1.1GB as float64),
unlike the full 5deg production postprocessing chunks that function is
built to stream to disk in bounded-memory blocks. Skipping the
write-compressed-GeoTIFF-then-reread round trip that function would
otherwise impose per (delta, SLR) was the single biggest speedup found
live on this run (see `merge_in_memory`/`_tile_geometry` below) - this is
NOT a safe change to make in `src/merge.py` itself, since a dense,
continent-spanning production chunk genuinely can exceed available memory
if loaded whole; it's only safe here because a delta's bbox is reliably
small. Each tile's own window-intersection geometry (which output pixels
it covers) is also computed ONCE per delta and reused across all 5 SLR
magnitudes, since every SLR raster for the same tile shares an identical
pixel grid (production's own guarantee - see src/merge.py's module
docstring: "all tiles share the same origin and pixel size").

Flooded extent uses depth > 0 (any real flood depth, matching production's
own binary flood classification in `flood_model.py` and
`validation.primary_threshold_m`'s own "any real depth" reasoning) - NOT the
0.1m exposure threshold, which is a separate, not-yet-implemented
extension (see this script's own module docstring in the project plan: the
exposure pipeline needs a full merge/chunk prerequisite run first).

`delta_name_inferred`: best-effort names matched by hand against each
delta's real centroid during planning (cross-checked against the one
already-known case - Id=44 is confirmed Volga - as a sanity check on the
method) - NOT from any authoritative source in this repo (none exists).
Treat as a convenience label, never a join key; `delta_id` is authoritative.

`pct_flooded_of_modeled_area` is computed against `delta_area_km2_modeled`
(the portion of the delta's real outline actually covered by tiles THIS
scenario modeled), not the shapefile's full `Area` - a delta whose outline
extends beyond the simulated tile footprint would otherwise read an
artificially low percentage. `pct_delta_covered_by_model` makes that gap
visible per row.

Flood protection (FLOPROS, 2026-10): each cell of a delta carries the FLOPROS design standard of its
WRI geogunit_107 unit - `Coastal`, else `Riverine`, else `protection.default_rp` (the same convention
as the exposure pipeline, `compute_exposure_analysis.py`). Flooding counts only where it is allowed:
never in cells protected to >= PROTECTION_THRESHOLD_YEARS (100), and in cells below that only if they
are connected to the sea through other cells below it - an enclosed low-standard area behind a
high-standard one does not flood either. Decided per cell, not per delta (a delta that is mostly
protected still floods in its connected low-standard part). Connectivity: 8-connected cells below the
threshold (land and water alike) that reach open ocean of the DeltaDTM land/sea mask, on the same
~30 m grid the flood depths are merged on. A delta in which no land cell may flood (at that SLR) gets
`protection_status = "protected"` and 0 flooded area; it stays in the CSV and the figures.
`protection_main_rp_years` (the standard covering most of the delta's land) is for reporting only.
Protection failure (as the main pipeline's baseline, compute_exposure_analysis.py): each protection
class (FLOPROS standard >= 100 yr floored to a simulated RP, max 1000) holds up to the flood extent of
its design event at SLR_0. With the delta-wide flooded area as proxy, a class fails at a given SLR once
the RP100 flood there covers more of the class's area than its design event did at SLR_0; its cells then
count as unprotected (and connect the low-standard cells behind them). The design floods above RP100 come
from `--design-outputs` (prepare_delta_design_runs.py + run_delta_floodmaps.py --scenarios design).
Deltas in NEVER_FAILING_DELTA_IDS (Rhine-Meuse) skip the failure rule - protection holds at every SLR.
`protection_status` per row: unprotected / partly_protected (holding) / protection_failed / protected
(no cell may flood - flooded 0); `failed_design_rps` lists the failed classes.
`flooded_area_km2` / `pct_flooded_of_modeled_area` are the protection-adjusted values;
`flooded_area_km2_unprotected` keeps the old, unadjusted number for comparison.

Usage:
    python compute_delta_flood_extent.py
    python compute_delta_flood_extent.py --scenarios ssp126
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from merge import AQUEDUCT_NODATA, _make_chunk_transform, _open_overlapping_tiles, _tile_id_from_path  # noqa: E402
from plotting import pixel_area_km2_grid  # noqa: E402
from validation import tri_domain_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sfincs_tiles"))
from build_station_boundary_lines import MASK_VRT, OCEAN_CODE  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
RETURN_PERIOD = "RP100"
SLR_SCENARIOS = ["SLR_0", "SLR_500", "SLR_1000", "SLR_1500", "SLR_2000"]
SCENARIOS_DEFAULT = ["ssp126", "ssp245"]
VOLGA_ID = 44  # confirmed zero tile coverage in both scenarios - excluded
# Deltas whose protection is assumed to hold against the RP100 event at every SLR level. Rhine-Meuse:
# Dutch standards (1/1,000-1/30,000 yr) are capped at COAST-RP's RP1000, whose tail is far too flat on
# this coast (RP1000-RP100 ~0.2 m vs ~1.1-1.3 m between the official 1/100 and 1/10,000 levels at Hoek
# van Holland), so the failure rule would fail the dikes at the first SLR step.
NEVER_FAILING_DELTA_IDS = {35}
PROTECTION_THRESHOLD_YEARS = 100.0  # areas protected to at least this design RP do not flood
CONNECT_PAD_DEG = (0.25, 0.5, 1.0)  # sea-connectivity grid = merge grid + this on every side (first with ocean)
SIMULATED_RPS = (2, 5, 10, 25, 50, 100, 250, 500, 1000)  # COAST-RP return periods a design standard snaps to


def snap_rp(rp):
    """Design standard -> the nearest simulated return period at or BELOW it (floor; above the largest
    -> the largest). Floored, not rounded up as in compute_exposure_analysis.py: the FLOPROS standards in
    the deltas sit just above a COAST-RP step (101, 112, 134, 150, 156, 500, 527 yr). Scalar or array."""
    steps = np.asarray(SIMULATED_RPS)
    out = steps[np.clip(np.searchsorted(steps, rp, side="right") - 1, 0, len(steps) - 1)]
    return int(out) if np.ndim(out) == 0 else out.astype("int32")
GEOGUNIT_SOURCE = "geogunit_protection_units"  # catalog key (data_catalog_gfm.yml)

# Best-effort Id -> name, matched by centroid location during planning - see
# module docstring. NOT authoritative; cross-checked against Id=44 (Volga,
# independently confirmed via deltas_ssp126.yml's own comment) as the one
# available sanity check on the method.
DELTA_NAME_INFERRED = {
    1: "Amazon", 2: "Amur", 3: "Fitzroy (Australia)", 4: "Colorado (Mexico/US)",
    5: "Congo", 6: "Chao Phraya", 7: "Danube", 8: "Dnieper", 9: "Ebro",
    10: "Fly", 11: "Ganges-Brahmaputra", 12: "Godavari", 13: "Grijalva-Usumacinta",
    14: "Han", 15: "Indus", 16: "Irrawaddy", 17: "Krishna", 18: "Lena",
    19: "Limpopo", 20: "Mackenzie", 21: "Magdalena", 22: "Mahakam", 23: "Mahanadi",
    24: "Mekong", 25: "Mississippi", 26: "Moulouya", 27: "Niger", 28: "Nile",
    29: "Orinoco", 30: "Parana-Rio de la Plata", 31: "Pearl", 32: "Po",
    33: "Red River", 34: "Rio Grande", 35: "Rhine-Meuse", 36: "Rhone",
    37: "Sebou", 38: "Senegal", 39: "Sao Francisco", 40: "Tana", 41: "Shatt al-Arab",
    42: "Tone", 43: "Vistula", 44: "Volga", 45: "Volta", 46: "Yangtze",
    47: "Yellow River", 48: "Yukon-Kuskokwim", 49: "Zambezi",
}


def build_tile_delta_join(tile_grid: gpd.GeoDataFrame, deltas: gpd.GeoDataFrame) -> dict[int, list[int]]:
    """{delta Id: [tile_id, ...]} - tiles whose geometry actually intersects
    the delta's REAL polygon (not the wider simulation-selection buffer)."""
    joined = gpd.sjoin(tile_grid[["tile_id", "geometry"]], deltas[["Id", "geometry"]],
                        how="inner", predicate="intersects")
    return joined.groupby("Id")["tile_id"].apply(list).to_dict()


def _tile_geometry(tile_rasters: list[Path], bbox: tuple[float, float, float, float]):
    """Computes the shared output grid (transform/width/height) and each
    surviving tile's own window-intersection geometry ONCE, from a
    reference set of tile rasters (SLR_0) - reused across every other SLR
    magnitude for this delta (see module docstring for why that's safe).
    Returns (out_transform, out_w, out_h, [(tile_id, row_off, col_off,
    n_rows, n_cols, src_r0, src_c0), ...]) - a tile whose footprint doesn't
    actually reach the delta's bbox is silently dropped here (same
    `_open_overlapping_tiles` behaviour production's own merge relies on).
    """
    with rasterio.open(tile_rasters[0]) as ref:
        ref_transform = ref.transform
    out_transform, out_w, out_h = _make_chunk_transform(bbox, ref_transform)
    metas = _open_overlapping_tiles(tile_rasters, out_transform, out_w, out_h)
    geometry = [
        (_tile_id_from_path(tm.path), tm.row_off, tm.col_off, tm.n_rows, tm.n_cols, tm.src_r0, tm.src_c0)
        for tm in metas
    ]
    for tm in metas:
        tm.src.close()
    return out_transform, out_w, out_h, geometry


def merge_in_memory(model_outputs: Path, slr: str, out_w: int, out_h: int, geometry: list[tuple],
                    rp: str = RETURN_PERIOD) -> np.ndarray | None:
    """In-memory equivalent of merge.merge_tile_rasters_chunk's per-cell
    MAXIMUM combine (same production rule - see module docstring and that
    function's own docstring for why max, not sum/mean) - skips the
    GeoTIFF write+re-read round trip entirely. NaN wherever no tile's
    footprint reaches a cell. Returns None if no tile actually has this
    SLR's output on disk (should only happen for a genuinely incomplete run)."""
    best = np.full((out_h, out_w), -np.inf, dtype="float64")
    covered = np.zeros((out_h, out_w), dtype=bool)
    any_found = False
    for tile_id, row_off, col_off, n_rows, n_cols, src_r0, src_c0 in geometry:
        path = model_outputs / str(tile_id) / "results" / f"waterdepth_{rp}_{slr}.tif"
        if not path.exists():
            print(f"    WARNING: {path} missing (had SLR_0 but not {slr}) - excluded from this merge", flush=True)
            continue
        any_found = True
        with rasterio.open(path) as src:
            patch = src.read(1, window=Window(src_c0, src_r0, n_cols, n_rows)).astype("float64")
        valid = patch < AQUEDUCT_NODATA
        sl = (slice(row_off, row_off + n_rows), slice(col_off, col_off + n_cols))
        sub_best = best[sl]
        wins = valid & (patch > sub_best)
        sub_best[wins] = patch[wins]
        best[sl] = sub_best
        covered[sl] |= valid
    if not any_found:
        return None
    return np.where(covered, best, np.nan).astype("float32")


def _read_mask_on_grid(root: Path, transform, width: int, height: int) -> np.ndarray:
    """DeltaDTM land/sea mask (0 land, 1 ocean, 2 lake, 3 river) on the merge grid, nearest neighbour.
    The global VRT has no tiles over open ocean or far inland: a no-data block whose valid neighbouring
    pixels are mostly ocean is ocean, otherwise land (same rule as build_station_boundary_lines.py)."""
    with rasterio.open(root / MASK_VRT) as src:
        nodata = src.nodata if src.nodata is not None else 255
        arr = np.full((height, width), nodata, dtype="uint8")
        reproject(rasterio.band(src, 1), arr, dst_transform=transform, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest, src_nodata=nodata, dst_nodata=nodata)
    missing = arr == nodata
    if missing.any():
        labels, n = ndimage.label(missing)
        rim = ndimage.binary_dilation(missing) & ~missing
        lab_rim = ndimage.grey_dilation(labels, size=3) * rim
        n_rim = ndimage.sum(rim, lab_rim, index=np.arange(1, n + 1))
        n_ocean = ndimage.sum(rim & (arr == OCEAN_CODE), lab_rim, index=np.arange(1, n + 1))
        ocean_blocks = np.arange(1, n + 1)[(n_rim > 0) & (n_ocean > 0.5 * n_rim)]
        arr[missing] = 0
        arr[np.isin(labels, ocean_blocks)] = OCEAN_CODE
    return arr


def _protection_rp(catalog, rp_by_geogunit: pd.Series, default_rp: float, transform, width: int, height: int):
    """FLOPROS design RP per cell of the given grid (geogunit unit -> coastal, else riverine, else
    default_rp; cells without a geogunit unit get default_rp too)."""
    left, top = transform.c, transform.f
    bbox = [left, top + height * transform.e, left + width * transform.a, top]
    try:
        geo = catalog.get_rasterdataset(GEOGUNIT_SOURCE, bbox=bbox, variables=["Geogunits"]).squeeze(drop=True)
    except Exception as e:  # noqa: BLE001 - no geogunit coverage: production's default standard everywhere
        print(f"    (no geogunit coverage: {e}) - default protection {default_rp:g} yr everywhere", flush=True)
        return np.full((height, width), default_rp, dtype="float32")
    gid = np.full((height, width), -1, dtype="int32")
    reproject(np.nan_to_num(geo.values.astype("float64"), nan=-1).astype("int32"), gid,
              src_transform=geo.raster.transform, src_crs=geo.raster.crs, dst_transform=transform,
              dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=-1, dst_nodata=-1)
    ids, inv = np.unique(gid, return_inverse=True)  # map per unique unit, not per cell
    mapped = pd.Series(ids).map(rp_by_geogunit).to_numpy(dtype="float64")
    return np.where(np.isnan(mapped), default_rp, mapped)[inv].reshape(height, width).astype("float32")


class ProtectionGrid:
    """FLOPROS protection of one delta on the merge grid, extended by CONNECT_PAD_DEG on every side for
    the sea-connectivity check (the sea a delta drains to can lie just outside its bbox - Amur: no ocean
    cell in it at all), widened until it holds ocean; results are cropped back to the merge grid.

    floodable(failed): cells that may flood when the protection classes in `failed` (floored design RPs)
    have failed - below PROTECTION_THRESHOLD_YEARS or in a failed class, and sea-connected through such
    cells (on this ~30 m grid)."""

    def __init__(self, prot_inputs: dict, domain: np.ndarray, transform, width: int, height: int):
        catalog, rp_by_geogunit = prot_inputs["catalog"], prot_inputs["rp_by_geogunit"]
        default_rp, root = prot_inputs["default_rp"], prot_inputs["root"]
        for pad_deg in CONNECT_PAD_DEG:
            pad = int(round(pad_deg / abs(transform.a)))
            tr = transform * Affine.translation(-pad, -pad)
            H, W = height + 2 * pad, width + 2 * pad
            ocean = _read_mask_on_grid(root, tr, W, H) == OCEAN_CODE
            if ocean.any():
                break
        rp = _protection_rp(catalog, rp_by_geogunit, default_rp, tr, W, H)
        self.ocean = ocean
        self.rp_class = np.where(rp >= PROTECTION_THRESHOLD_YEARS,
                                 snap_rp(rp), 0).astype("int32")
        self.crop = (slice(pad, pad + height), slice(pad, pad + width))
        self._cache: dict[frozenset, np.ndarray] = {}
        rp_c = rp[self.crop]
        self.land = domain & ~ocean[self.crop]
        vals, counts = np.unique(rp_c[self.land], return_counts=True)
        self.main_rp = float(vals[np.argmax(counts)]) if len(vals) else float("nan")
        cls_c = self.rp_class[self.crop]
        self.classes = sorted(int(c) for c in np.unique(cls_c[self.land]) if c > 0)  # protection classes present
        self.class_masks = {c: self.land & (cls_c == c) for c in self.classes}
        self.n_land = max(int(self.land.sum()), 1)
        self.pct_protected = 100.0 * float((self.land & (cls_c > 0)).sum()) / self.n_land

    def floodable(self, failed: frozenset) -> np.ndarray:
        if failed not in self._cache:
            low = (self.rp_class == 0) | np.isin(self.rp_class, list(failed))
            labels, _ = ndimage.label(low | self.ocean, structure=np.ones((3, 3)))
            sea_ids = np.unique(labels[self.ocean])
            self._cache[failed] = np.isin(labels[self.crop], sea_ids[sea_ids > 0]) & low[self.crop]
        return self._cache[failed]


def score_delta(delta_row, candidate_tiles: list[int], model_outputs: Path, prot_inputs: dict | None = None,
                design_outputs: Path | None = None) -> dict:
    """Per-(scenario, delta): merges every covering tile's own waterdepth
    raster onto one common grid (chunk_bounds = the delta's own bbox),
    in memory, once per SLR magnitude - each real cell is counted exactly
    once regardless of how many tiles' footprints reach it.
    `n_tiles`/`delta_area_km2_modeled` come from the SLR_0 merge (the same
    set of tiles, hence the same footprint, covers every SLR magnitude)."""
    delta_id = int(delta_row["Id"])
    delta_gdf = gpd.GeoDataFrame({"Id": [delta_id]}, geometry=[delta_row.geometry], crs="EPSG:4326")
    bbox = tuple(delta_row.geometry.bounds)

    slr0_rasters = [
        model_outputs / str(tid) / "results" / f"waterdepth_{RETURN_PERIOD}_SLR_0.tif"
        for tid in candidate_tiles
    ]
    slr0_rasters = [p for p in slr0_rasters if p.exists()]
    if not slr0_rasters:
        zero = {slr: 0.0 for slr in SLR_SCENARIOS}
        return {"n_tiles": 0, "delta_area_km2_modeled": 0.0,
                "delta_area_km2_shapefile": float(delta_row["Area"]),
                "pct_delta_covered_by_model": 0.0, "flooded_km2_by_slr": zero, "raw_km2_by_slr": dict(zero),
                "protection": None, "per_slr_protection": {}}

    out_transform, out_w, out_h, geometry = _tile_geometry(slr0_rasters, bbox)

    per_slr_flooded_km2: dict[str, float] = {}
    per_slr_raw_km2: dict[str, float] = {}
    n_tiles = len(geometry)
    prot = None
    design_km2: dict[int, float] = {}  # protection class -> its area flooded by the design event at SLR_0
    per_slr_prot: dict[str, dict] = {}
    modeled_km2 = 0.0
    domain = None
    area_grid = None
    for slr in SLR_SCENARIOS:
        depth = merge_in_memory(model_outputs, slr, out_w, out_h, geometry)
        if depth is None:
            per_slr_flooded_km2[slr] = per_slr_raw_km2[slr] = 0.0
            continue

        if domain is None:  # SLR-independent (same footprint/grid every SLR) - build once
            # NOT model_domain_mask (depth != nodata) - merge_in_memory's own nodata
            # sentinel is NaN, and NaN != NaN is always True in numpy, so an equality
            # check would silently treat every uncovered cell as "covered".
            covered = ~np.isnan(depth)
            domain = tri_domain_mask(delta_gdf, out_transform, "EPSG:4326", (out_h, out_w)) & covered
            area_grid = pixel_area_km2_grid(out_transform, out_w, out_h)
            modeled_km2 = float((domain * area_grid).sum())
            if prot_inputs is not None:
                prot = ProtectionGrid(prot_inputs, domain, out_transform, out_w, out_h)
                # design-event flood at SLR_0 per protection class: RP100 from this run (its SLR_0 merge,
                # SLR_SCENARIOS[0]), higher design RPs from the design runs (prepare_delta_design_runs.py)
                never_fails = int(delta_row["Id"]) in NEVER_FAILING_DELTA_IDS
                for c in prot.classes:
                    if never_fails:
                        design_km2[c] = float("inf")
                        continue
                    if c == 100:
                        d_wet = (depth > 0) & domain
                    else:
                        d_depth = (merge_in_memory(design_outputs, "SLR_0", out_w, out_h, geometry, f"RP{c}")
                                   if design_outputs is not None else None)
                        if d_depth is None:
                            print(f"    WARNING: no RP{c} SLR_0 design flood for this delta - its RP{c} "
                                  f"protection is treated as never failing", flush=True)
                            design_km2[c] = float("inf")
                            continue
                        d_wet = (d_depth > 0) & domain
                    design_km2[c] = float(((d_wet & prot.class_masks[c]) * area_grid).sum())

        wet = (depth > 0) & domain  # depth>0 is already False wherever depth is NaN - no separate NaN guard needed
        per_slr_raw_km2[slr] = float((wet * area_grid).sum())
        if prot is None:
            per_slr_flooded_km2[slr] = per_slr_raw_km2[slr]
            continue
        # a protection class fails once this flood covers more of its area than its design event did at SLR_0
        failed = frozenset(c for c in prot.classes
                           if float(((wet & prot.class_masks[c]) * area_grid).sum()) > design_km2[c])
        floodable = prot.floodable(failed)
        per_slr_flooded_km2[slr] = float(((wet & floodable) * area_grid).sum())
        per_slr_prot[slr] = {"failed": sorted(failed),
                             "pct_floodable": 100.0 * float((prot.land & floodable).sum()) / prot.n_land}

    shapefile_km2 = float(delta_row["Area"])
    pct_covered = 100.0 * modeled_km2 / shapefile_km2 if shapefile_km2 > 0 else float("nan")

    return {
        "n_tiles": n_tiles,
        "delta_area_km2_modeled": modeled_km2,
        "delta_area_km2_shapefile": shapefile_km2,
        "pct_delta_covered_by_model": pct_covered,
        "flooded_km2_by_slr": per_slr_flooded_km2,
        "raw_km2_by_slr": per_slr_raw_km2,
        "protection": None if prot is None else {"main_rp": prot.main_rp, "pct_protected": prot.pct_protected,
                                                 "classes": prot.classes, "design_km2": design_km2},
        "per_slr_protection": per_slr_prot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", default=str(_REPO_ROOT / "snakemake_workflow" / "config"))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS_DEFAULT)
    parser.add_argument("--out", default=None, help="default: {root}/deltas_floodmaps/delta_flood_extent.csv")
    parser.add_argument("--design-outputs", default=None,
                        help="model_outputs of the design-event runs (prepare_delta_design_runs.py); default: "
                             "{root}/deltas_floodmaps/design/model_outputs")
    parser.add_argument("--delta-ids", nargs="+", type=int, default=None, help="only these delta Ids (testing)")
    args = parser.parse_args()

    # FLOPROS protection standards - scenario-independent, computed once per delta. Loaded up front
    # (not just at point of first use below) since the deltas shapefile path also comes from here.
    prod_cfg = load_config(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml")

    deltas = retry_transient_io(gpd.read_file, Path(prod_cfg["paths"]["deltas_shapefile"]))
    deltas = deltas[deltas["Id"] != VOLGA_ID].reset_index(drop=True)
    if args.delta_ids:
        deltas = deltas[deltas["Id"].isin(args.delta_ids)].reset_index(drop=True)
    centroids = deltas.geometry.centroid
    deltas["centroid_lon"] = centroids.x
    deltas["centroid_lat"] = centroids.y
    print(f"{len(deltas)} delta(s) (Id={VOLGA_ID} Volga excluded - zero tile coverage)")

    first_cfg = load_config(Path(args.config_dir) / f"deltas_{args.scenarios[0]}_materialized.yml")
    tile_grid = retry_transient_io(gpd.read_file, Path(first_cfg["tile_grid"]["path"]))
    if tile_grid.crs != deltas.crs:
        tile_grid = tile_grid.to_crs(deltas.crs)
    tiles_by_delta = build_tile_delta_join(tile_grid, deltas)
    n_unmatched = sum(1 for _, row in deltas.iterrows() if int(row["Id"]) not in tiles_by_delta)
    if n_unmatched:
        print(f"WARNING: {n_unmatched} delta(s) matched zero tiles in the shared tile grid")

    root = Path(first_cfg["paths"]["root"])
    out_path = Path(args.out) if args.out else root / "deltas_floodmaps" / "delta_flood_extent.csv"

    default_rp = float(prod_cfg.get("protection", {}).get("default_rp", 5.0))
    catalog = get_data_catalog(_REPO_ROOT / prod_cfg["paths"]["hydromt_data_catalog"], root=prod_cfg["paths"]["root"])
    flopros = catalog.get_dataframe("flopros_protection_standards")
    rp_by_geogunit = flopros["Coastal"].fillna(flopros["Riverine"]).fillna(default_rp)
    rp_by_geogunit.index = rp_by_geogunit.index.astype("int64")
    prot_inputs = {"catalog": catalog, "rp_by_geogunit": rp_by_geogunit, "default_rp": default_rp, "root": root}
    design_outputs = Path(args.design_outputs) if args.design_outputs else root / "deltas_floodmaps" / "design" / "model_outputs"
    if not design_outputs.exists():
        print(f"WARNING: {design_outputs} missing - protections above 100 yr are treated as never failing")
        design_outputs = None
    print(f"\nFLOPROS protection: coastal, else riverine, else {default_rp:g} yr; cells >= "
          f"{PROTECTION_THRESHOLD_YEARS:g} yr and enclosed low-standard cells do not flood", flush=True)

    rows = []
    t_start = time.time()
    n_deltas = len(deltas)
    for scenario in args.scenarios:
        cfg = load_config(Path(args.config_dir) / f"deltas_{scenario}_materialized.yml")
        model_outputs = Path(cfg["simulation"]["model_outputs"])
        print(f"\n=== {scenario} (model_outputs={model_outputs}) ===", flush=True)

        for i, (_, delta_row) in enumerate(deltas.iterrows(), start=1):
            delta_id = int(delta_row["Id"])
            delta_name = DELTA_NAME_INFERRED.get(delta_id, f"delta_{delta_id}")
            candidate_tiles = tiles_by_delta.get(delta_id, [])
            print(f"  [{i}/{n_deltas}] delta {delta_id:2d} ({delta_name}): "
                  f"{len(candidate_tiles)} candidate tile(s) - merging...", flush=True)
            t0 = time.time()
            result = score_delta(delta_row, candidate_tiles, model_outputs, prot_inputs, design_outputs)
            prot = result["protection"] or {"main_rp": float("nan"), "pct_protected": float("nan"), "classes": [],
                                             "design_km2": {}}
            elapsed = time.time() - t0
            modeled_km2 = result["delta_area_km2_modeled"]

            print(f"    -> {result['n_tiles']} tile(s), "
                  f"modeled {modeled_km2:.1f}/{result['delta_area_km2_shapefile']:.1f} km2 "
                  f"({result['pct_delta_covered_by_model']:.1f}% covered) - {elapsed:.1f}s "
                  f"[{(time.time() - t_start) / 60:.1f}min elapsed]", flush=True)
            if prot["classes"]:
                fails = {slr: v["failed"] for slr, v in result["per_slr_protection"].items() if v["failed"]}
                print(f"       protection: {prot['pct_protected']:.1f}% >= {PROTECTION_THRESHOLD_YEARS:g} yr, "
                      f"design classes {prot['classes']} (design-event flooded km2 "
                      f"{ {c: round(v, 1) for c, v in prot['design_km2'].items()} }); failed: {fails or 'never'}",
                      flush=True)

            for slr in SLR_SCENARIOS:
                sp = result["per_slr_protection"].get(slr, {"failed": [], "pct_floodable": 100.0})
                status = ("unprotected" if not prot["classes"] else
                          "protection_failed" if sp["failed"] else
                          "protected" if sp["pct_floodable"] == 0.0 else "partly_protected")
                flooded_km2 = result["flooded_km2_by_slr"][slr]
                pct_flooded = 100.0 * flooded_km2 / modeled_km2 if modeled_km2 > 0 else float("nan")
                rows.append({
                    "delta_id": delta_id,
                    "delta_name_inferred": delta_name,
                    "centroid_lon": round(float(delta_row["centroid_lon"]), 4),
                    "centroid_lat": round(float(delta_row["centroid_lat"]), 4),
                    "ssp_scenario": scenario,
                    "slr_scenario": slr,
                    "return_period": RETURN_PERIOD,
                    "n_tiles": result["n_tiles"],
                    "flooded_area_km2": round(flooded_km2, 4),
                    "delta_area_km2_modeled": round(modeled_km2, 4),
                    "delta_area_km2_shapefile": round(result["delta_area_km2_shapefile"], 4),
                    "pct_delta_covered_by_model": round(result["pct_delta_covered_by_model"], 2),
                    "pct_flooded_of_modeled_area": round(pct_flooded, 4),
                    "flooded_area_km2_unprotected": round(result["raw_km2_by_slr"][slr], 4),
                    "protection_main_rp_years": prot["main_rp"],
                    "pct_area_protected_ge100": round(prot["pct_protected"], 2),
                    "pct_area_floodable": round(sp["pct_floodable"], 2),
                    "failed_design_rps": " ".join(str(c) for c in sp["failed"]),
                    "protection_status": status,
                })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retry_transient_io(df.to_csv, out_path, index=False)
    print(f"\nWrote {len(df)} row(s) to {out_path}")

    low_coverage = df[["delta_id", "delta_name_inferred", "ssp_scenario", "pct_delta_covered_by_model"]].drop_duplicates()
    low_coverage = low_coverage[low_coverage["pct_delta_covered_by_model"] < 90]
    if not low_coverage.empty:
        print(f"\nWARNING: {len(low_coverage)} (delta, scenario) pair(s) below 90% modeled coverage:")
        print(low_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
