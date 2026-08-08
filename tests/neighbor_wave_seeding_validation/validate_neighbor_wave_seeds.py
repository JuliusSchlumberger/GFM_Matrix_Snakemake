"""Validate `collect_neighbor_wave_seeds` (src/boundaries.py, 2026-08) - the
hop>=1 hinterland eikonal-seed collection mechanism: gathers real,
already-computed wet-cell water levels from lower-hop_distance neighbour
tile(s), snapped onto the target tile's own grid, taking the max where
sources disagree.

Writes small real GeoTIFFs (matching this pipeline's own int16-centimetre
DEM/waterdepth encoding - `rasters.encode_dem_cm`/`encode_waterdepth_cm`) to
a temp directory and exercises the actual function end-to-end, not a mock.

All source tiles share the target's own grid exactly (same origin/
resolution/shape) except the deliberately-disjoint one - this keeps the
row/col snapping arithmetic trivially exact (a source cell center maps back
onto the identical target cell, no floating-point boundary ambiguity),
since `rasterio.transform.xy`/`rowcol` round-tripping itself is not what
this test is checking; the windowing/decode/wet-filter/max-conflict
aggregation logic is. The "points outside target dropped" case from the
implementation plan is covered here via a fully geographically-disjoint
source tile (real neighbour tiles are only ever proposed as candidates
because they DO overlap, so this is the realistic manifestation of that
guard - `from_bounds(...).round_offsets().round_lengths()` was confirmed
by direct experimentation to never produce a window that pads past the
target's own bounds for aligned, equal-resolution, non-rotated grids like
this pipeline's, so the post-transform in-bounds numpy filter is a
defensive guard for degenerate/misaligned geometry rather than something
reachable through this specific construction).

Usage:
    python validate_neighbor_wave_seeds.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(r"C:/Users/Schlu005/GFM")
sys.path.insert(0, str(REPO_ROOT / "snakemake_workflow" / "src"))

import boundaries as bnd  # noqa: E402
from merge import WATERDEPTH_NODATA_INT16  # noqa: E402
from rasters import encode_dem_cm, encode_waterdepth_cm  # noqa: E402

RES = 0.1
ORIGIN = (0.0, 1.0)  # west, north
SHAPE = (4, 4)  # height, width


def _write_dem(path: Path, dem_m: np.ndarray, origin: tuple[float, float] = ORIGIN) -> None:
    transform = from_origin(origin[0], origin[1], RES, RES)
    encoded = encode_dem_cm(dem_m)
    profile = dict(
        driver="GTiff", dtype="int16", count=1, nodata=None,
        crs="EPSG:4326", transform=transform,
        width=dem_m.shape[1], height=dem_m.shape[0],
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(encoded, indexes=1)


def _write_mask(path: Path, mask_vals: np.ndarray, origin: tuple[float, float] = ORIGIN) -> None:
    transform = from_origin(origin[0], origin[1], RES, RES)
    profile = dict(
        driver="GTiff", dtype="uint8", count=1, nodata=255,
        crs="EPSG:4326", transform=transform,
        width=mask_vals.shape[1], height=mask_vals.shape[0],
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask_vals.astype("uint8"), indexes=1)


def _write_waterdepth(
    path: Path, depth_m: np.ndarray,
    nodata_cells: list[tuple[int, int]] | None = None,
    origin: tuple[float, float] = ORIGIN,
) -> None:
    transform = from_origin(origin[0], origin[1], RES, RES)
    encoded = encode_waterdepth_cm(depth_m)
    for r, c in nodata_cells or []:
        encoded[r, c] = WATERDEPTH_NODATA_INT16
    profile = dict(
        driver="GTiff", dtype="int16", count=1, nodata=WATERDEPTH_NODATA_INT16,
        crs="EPSG:4326", transform=transform,
        width=depth_m.shape[1], height=depth_m.shape[0],
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(encoded, indexes=1)


def test_max_conflict_dry_nodata_and_disjoint_source(tmp_dir: Path) -> None:
    print("=== multi-source aggregation: max-wins both directions, dry/nodata/no-wet-cells/disjoint sources excluded ===")
    target_dem = tmp_dir / "target_dem.tif"
    _write_dem(target_dem, np.full(SHAPE, 5.0))  # target's own DEM values are never read by this function

    # All sources are all-land (mask=0) here, so effective_dem(dem, mask) ==
    # dem - this test is about the windowing/decode/wet-filter/max-conflict
    # logic, not the effective_dem substitution (see
    # test_non_land_source_cell_uses_effective_dem below for that).
    all_land = np.zeros(SHAPE, dtype="uint8")

    # source1: higher elevation, wins the (0,0) conflict
    source1_dem, source1_mask, source1_wd = tmp_dir / "s1_dem.tif", tmp_dir / "s1_mask.tif", tmp_dir / "s1_wd.tif"
    _write_dem(source1_dem, np.full(SHAPE, 2.0))
    _write_mask(source1_mask, all_land)
    s1_depth = np.zeros(SHAPE)
    s1_depth[0, 0] = 0.5  # level 2.5
    s1_depth[3, 3] = 0.5  # level 2.5 - will lose the (3,3) conflict to source2
    _write_waterdepth(source1_wd, s1_depth, nodata_cells=[(1, 1)])  # (1,1): forced not-computed

    # source2: wins the (3,3) conflict despite lower elevation (higher depth more than compensates)
    source2_dem, source2_mask, source2_wd = tmp_dir / "s2_dem.tif", tmp_dir / "s2_mask.tif", tmp_dir / "s2_wd.tif"
    _write_dem(source2_dem, np.full(SHAPE, 1.0))
    _write_mask(source2_mask, all_land)
    s2_depth = np.zeros(SHAPE)
    s2_depth[0, 0] = 0.3  # level 1.3 - loses the (0,0) conflict to source1
    s2_depth[3, 3] = 1.8  # level 2.8 - wins the (3,3) conflict over source1
    _write_waterdepth(source2_wd, s2_depth)

    # source3: real overlap, but computed nothing but dry cells - must contribute nothing
    source3_dem, source3_mask, source3_wd = tmp_dir / "s3_dem.tif", tmp_dir / "s3_mask.tif", tmp_dir / "s3_wd.tif"
    _write_dem(source3_dem, np.full(SHAPE, 9.0))
    _write_mask(source3_mask, all_land)
    _write_waterdepth(source3_wd, np.zeros(SHAPE))

    # source4: geographically disjoint from the target entirely - must be silently skipped,
    # even though its (fabricated) values would dominate every conflict above if wrongly included
    source4_dem, source4_mask, source4_wd = tmp_dir / "s4_dem.tif", tmp_dir / "s4_mask.tif", tmp_dir / "s4_wd.tif"
    far_origin = (100.0, 1.0)
    _write_dem(source4_dem, np.full(SHAPE, 3.0), origin=far_origin)
    _write_mask(source4_mask, all_land, origin=far_origin)
    _write_waterdepth(source4_wd, np.full(SHAPE, 9.0), origin=far_origin)

    seed_rows, seed_cols, seed_values = bnd.collect_neighbor_wave_seeds(
        target_dem,
        [
            (source1_dem, source1_mask, source1_wd),
            (source2_dem, source2_mask, source2_wd),
            (source3_dem, source3_mask, source3_wd),
            (source4_dem, source4_mask, source4_wd),
        ],
    )
    seeds = {(int(r), int(c)): float(v) for r, c, v in zip(seed_rows, seed_cols, seed_values)}
    expected = {(0, 0): 2.5, (3, 3): 2.8}
    assert seeds.keys() == expected.keys(), seeds
    for key, exp_val in expected.items():
        assert abs(seeds[key] - exp_val) < 0.01, (key, seeds[key], exp_val)
    print("PASS: exactly the two expected seed cells, both conflict directions resolved by max, "
          "dry/nodata/no-wet-cells/disjoint sources correctly excluded")
    print()


def test_empty_source_paths() -> None:
    print("=== empty source_paths -> empty result ===")
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        target_dem = tmp_dir / "target_dem.tif"
        _write_dem(target_dem, np.full(SHAPE, 5.0))
        seed_rows, seed_cols, seed_values = bnd.collect_neighbor_wave_seeds(target_dem, [])
    assert len(seed_rows) == 0 and len(seed_cols) == 0 and len(seed_values) == 0
    print("PASS: empty source_paths produces empty seed arrays")
    print()


def test_non_land_source_cell_uses_effective_dem(tmp_dir: Path) -> None:
    print("=== non-land source cell: seed value uses effective_dem (0), not raw dem ===")
    # Regression test for a real bug found 2026-08: flood_depth_dense saves
    # `depth = waterlevel - effective_dem`, which is 0 for ocean/lake/river
    # cells - so `depth` there already IS the full water level. Using the
    # RAW (un-zeroed) dem when reconstructing the seed value double-counts
    # that cell's real terrain elevation. Caught on a real river cell
    # (dem=29.89m, true water level ~5.74m) whose seed value came out as
    # 35.63m instead - see boundaries.collect_neighbor_wave_seeds' docstring.
    target_dem = tmp_dir / "target_dem.tif"
    _write_dem(target_dem, np.full(SHAPE, 5.0))

    source_dem, source_mask, source_wd = tmp_dir / "sr_dem.tif", tmp_dir / "sr_mask.tif", tmp_dir / "sr_wd.tif"
    # (0,0): land, raw dem is real and used as-is.
    # (1,1): river (mask=3) with a large raw dem reading - must NOT be added
    #        on top of depth (which is already the full water level there).
    dem_vals = np.full(SHAPE, 2.0)
    dem_vals[1, 1] = 29.89
    mask_vals = np.zeros(SHAPE, dtype="uint8")
    mask_vals[1, 1] = 3  # river
    _write_dem(source_dem, dem_vals)
    _write_mask(source_mask, mask_vals)
    depth_vals = np.zeros(SHAPE)
    depth_vals[0, 0] = 1.0  # level = 2.0 (land) + 1.0 = 3.0
    depth_vals[1, 1] = 5.74  # level = effective_dem(0, river) + 5.74 = 5.74, NOT 29.89 + 5.74
    _write_waterdepth(source_wd, depth_vals)

    seed_rows, seed_cols, seed_values = bnd.collect_neighbor_wave_seeds(
        target_dem, [(source_dem, source_mask, source_wd)],
    )
    seeds = {(int(r), int(c)): float(v) for r, c, v in zip(seed_rows, seed_cols, seed_values)}
    assert abs(seeds[(0, 0)] - 3.0) < 0.01, seeds
    assert abs(seeds[(1, 1)] - 5.74) < 0.01, (
        f"expected river cell's seed value to use effective_dem=0, got {seeds[(1, 1)]} "
        f"(would be 35.63 if wrongly using raw dem=29.89)"
    )
    print("PASS: river-cell seed value correctly uses effective_dem (0), not raw dem")
    print()


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        test_max_conflict_dry_nodata_and_disjoint_source(Path(td))
    test_empty_source_paths()
    with tempfile.TemporaryDirectory() as td:
        test_non_land_source_cell_uses_effective_dem(Path(td))
    print("All neighbour-wave-seeding validation checks passed.")


if __name__ == "__main__":
    main()
