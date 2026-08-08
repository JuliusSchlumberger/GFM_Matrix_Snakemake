"""Build the DeltaDTM-tile-based chunk manifest -> tile_grid.path (2026-08).

Replaces the adaptive parent/child pipeline (build_parent_grid/
build_child_grid/refine_tile_grid, src/tile_generation.py - retired) with a
fixed-DeltaDTM-tile chunking approach - see src/tile_chunking.py for the
actual algorithms and memory.md's FIXED-TILE CHUNKING REDESIGN entry (which
supersedes the earlier TILE GENERATION REDESIGN / REAL-DATA VALIDATION /
MERGE-SPLIT REDESIGN entries) for the full design/reasoning. Originated as a
one-off prototype in tests/deltadtm_coverage/ before being promoted here.

Frozen geometry DAG: depends only on the DeltaDTM mask, elevation, and
tile_generation.elev_threshold_m - never on scenario (Coast-RP station
VALUES, SLR, return period). No parent/child distinction and no separate
forcing-context grid this time - each chunk gets its own boundary-condition
stations directly (extract_boundaries.py), same as every other tile in the
Snakemake DAG.

Stages, in order (each delegates to src/tile_chunking.py) - also the
CHECKPOINTS list below, one debug GeoPackage per stage:
  1. build_tile_index        - one 1x1deg polygon per real DeltaDTM mask tile.
  2. filter_floodable_tiles  - keep tiles with >=1 floodable cell; mark
                                river-mouth (ocean-river mixing) tiles.
  3. build_chunks (+ cleanup)- greedy maximal-rectangle covering, seeded at
                                river mouths first, then reduce_overlap /
                                add_minimum_overlap / add_connector_chunks.
  4. filter_and_shave_chunks - Stage A: exposure filter + coarse shave,
                                possibly splitting off a disconnected
                                interior feature (2026-08) in the process.
  5. [RETIRED, 2026-08] merge_dry_chunks - used to merge any chunk with no
                                wet edge into a connected wet neighbour (or
                                drop it if unreachable), the actual source
                                of this pipeline's >1e9-cell oversized
                                chunks (its dry-to-wet merge deliberately
                                ignores max_extent). Superseded by step 6's
                                compute_run_order: a dry chunk no longer
                                needs a wet edge of its own, since it can
                                get boundary forcing from an
                                already-simulated neighbour instead - see
                                src/tile_chunking.py's module docstring.
  6. drop_redundant_chunks (1st pass) / split_oversized_chunks /
     drop_redundant_chunks (2nd pass, 2026-08) / cap_overlap_density /
     compute_run_order (2026-08) - post-cleanup: dedup (a chunk covered by
     the UNION of its overlapping neighbours by >= dedup_min_coverage_
     fraction adds nothing), split anything still oversized, dedup AGAIN
     (splitting can create new redundancy - tile_chunking.py's own module
     docstring already called for this "before AND after step 10", the
     orchestration just never did the "after" part until now), cap
     redundant overlap density, then compute the hop-distance-from-ocean
     run order (drops any chunk with no path to ocean via any chain of
     neighbours - the sole remaining case retired step 5 used to catch)
     and reorder the manifest by it - `tile_id` IS this order (see
     tile_chunking.py's compute_run_order). NOT YET IMPLEMENTED: sourcing
     a hop>=1 chunk's real boundary points from an already-simulated
     earlier neighbour - see the NOT IMPLEMENTED note where Stage 13 ends
     below, and tile_chunking.py's own equivalent note.

Output: tile_grid.path (tile_id + geometry per row) - the manifest everything
downstream (Snakefile, extract_tile_geometry, extract_boundaries) reads from.
`tile_id`'s numeric value now encodes hop-distance-aware simulation run
order (2026-08), not just an arbitrary index - see compute_run_order.
Also writes a 50-bin histogram of each final chunk's native (1 arcsecond)
pixel count - Aqueduct's OOM risk is dominated by tile pixel count (see
src/tile_split.py's own docstring), so this is a cheap way to anticipate
trouble before running the expensive simulation.

Debug output (2026-08, tile_generation.write_debug_gpkg): one GeoPackage PER
PIPELINE STAGE, numbered to match the list above, written to
tile_generation.debug_gpkg_dir - both intermediate chunk-shape snapshots
(e.g. `03_chunks_built.gpkg`) and, wherever a stage cleanly drops items
rather than transforming their geometry, what got dropped and why (e.g.
`07_dropped_no_exposure_or_shave_empty.gpkg`). Lets every drop and every
intermediate shape change be inspected in QGIS, not just the final
tile_grid.path output.

Resumable (2026-08): `run(config, start_from=CHECKPOINT_NAME)` skips every
stage up to and including whichever produced `CHECKPOINT_NAME`, loading that
checkpoint's own debug GeoPackage as the input to continue from - added
after two real production runs each crashed 50-70 minutes in (an I/O
hiccup, an OOM) with every earlier stage's valid output sitting unused on
disk. See CHECKPOINTS below for valid names; `python run_preparation.py
tile_generation --start-from <name>` is the CLI entry point.

Not a standalone entry point - exposes `run(config, start_from=None)`,
called from run_preparation.py (`python run_preparation.py tile_generation`).
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config_utils import get_data_catalog, retry_transient_io  # noqa: E402
from tile_chunking import (  # noqa: E402
    add_connector_chunks,
    add_minimum_overlap,
    bbox_to_grid_rect,
    bboxes_to_geodataframe,
    build_chunks,
    build_presence_grid,
    build_river_mouth_grids,
    build_tile_index,
    cap_overlap_density,
    compute_run_order,
    drop_redundant_chunks,
    filter_and_shave_chunks,
    filter_floodable_tiles,
    grid_rect_to_bbox,
    reduce_overlap,
    split_oversized_chunks,
)
from tiles import _scan_mask_dir  # noqa: E402

# Ordered checkpoint names, one per pipeline stage - index in this list is
# what `start_from` resolves to. "Resuming from CHECKPOINT" means that
# checkpoint's own file already holds valid output; every stage up to and
# including it is skipped, and the pipeline continues with the next one.
# Stage 8 (merge_dry_chunks) was retired 2026-08 (see module docstring) -
# the numbering gap between 07 and 09 is intentional, kept for continuity
# with older debug-GeoPackage archives rather than renumbering everything.
CHECKPOINTS = [
    "01_tile_index",
    "02_floodable_tiles",
    "03_chunks_built",
    "04_chunks_reduce_overlap",
    "05_chunks_add_minimum_overlap",
    "06_chunks_add_connector_chunks",
    "07_chunks_after_shave",
    "09_chunks_dedup",
    "10_chunks_split_oversized",
    "11_chunks_dedup_after_split",
    "12_chunks_final",
    "13_chunks_ordered",
]


def _write_gpkg(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    retry_transient_io(gdf.to_file, path, driver="GPKG")


def _chunk_snapshot_gdf(bboxes: list[tuple[float, float, float, float]]) -> gpd.GeoDataFrame:
    if not bboxes:
        return gpd.GeoDataFrame(columns=["tile_id", "geometry"], geometry="geometry", crs="EPSG:4326")
    return bboxes_to_geodataframe(bboxes)


def _grid_rect_snapshot_gdf(
    rects: list[tuple[int, int, int, int]], lat_min: int, lon_min: int,
) -> gpd.GeoDataFrame:
    return _chunk_snapshot_gdf([grid_rect_to_bbox(rect, lat_min, lon_min) for rect in rects])


def _dropped_gdf(dropped: list[tuple[tuple[float, float, float, float], str]]) -> gpd.GeoDataFrame:
    """`dropped`: list of (bbox, reason) pairs, as returned by
    filter_and_shave_chunks/merge_dry_chunks, or assembled by the caller
    for stages that don't return this natively (see _diff_dropped)."""
    if not dropped:
        return gpd.GeoDataFrame(columns=["tile_id", "reason", "geometry"], geometry="geometry", crs="EPSG:4326")
    rows = [{"tile_id": i, "reason": reason, "geometry": box(*bbox)} for i, (bbox, reason) in enumerate(dropped)]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _diff_dropped(
    before: list[tuple[float, float, float, float]], after: list[tuple[float, float, float, float]], reason: str,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Multiset (Counter) difference of `before` minus `after`, tagged with
    `reason` - safe for stages that are PURE filters (every surviving bbox
    is byte-identical to its input, no shave/split/merge in between), e.g.
    drop_redundant_chunks and cap_overlap_density. NOT valid for a
    stage that also transforms geometry (e.g. filter_and_shave_chunks'
    shave step) - those report their own drops directly instead.
    """
    from collections import Counter
    remaining = Counter(before) - Counter(after)
    return [(bbox, reason) for bbox in remaining.elements()]


def _load_bboxes(path: Path) -> list[tuple[float, float, float, float]]:
    gdf = retry_transient_io(gpd.read_file, path)
    return [tuple(geom.bounds) for geom in gdf.geometry]


def _write_size_histogram(final: gpd.GeoDataFrame, path: Path) -> None:
    """50-bin histogram of each final chunk's approximate native (1
    arcsecond) pixel count - a direct proxy for Aqueduct's OOM risk, which
    is dominated by tile pixel count (see src/tile_split.py's own
    docstring). Log-scale y-axis since chunk sizes span orders of
    magnitude (a small river-mouth connector chunk vs. a full 4x4deg
    chunk).
    """
    n_cells = []
    for geom in final.geometry:
        minx, miny, maxx, maxy = geom.bounds
        width_arcsec = round((maxx - minx) * 3600)
        height_arcsec = round((maxy - miny) * 3600)
        n_cells.append(width_arcsec * height_arcsec)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(n_cells, bins=50, color="#3b6ea5", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Native (1 arcsecond) pixel count per chunk")
    ax.set_ylabel("Number of chunks (log scale)")
    ax.set_title(f"Final chunk size distribution - n={len(final)} chunks, "
                 f"median={int(np.median(n_cells)):,}, max={max(n_cells):,} px")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(config: dict, start_from: str | None = None) -> None:
    tg_cfg = config["tile_generation"]
    output_path = Path(config["tile_grid"]["path"])
    write_debug = tg_cfg.get("write_debug_gpkg", False)
    debug_dir = Path(tg_cfg["debug_gpkg_dir"]) if write_debug else None

    def _debug(gdf: gpd.GeoDataFrame, filename: str) -> None:
        if write_debug:
            _write_gpkg(gdf, debug_dir / filename)

    if start_from is not None:
        if start_from not in CHECKPOINTS:
            raise ValueError(
                f"Unrecognized start_from={start_from!r}. Valid checkpoints: {CHECKPOINTS}"
            )
        if debug_dir is None:
            raise ValueError(
                "start_from requires tile_generation.write_debug_gpkg=true (the checkpoint files "
                "it resumes from ARE the debug GeoPackages)."
            )
        start_index = CHECKPOINTS.index(start_from)
        print(f"Resuming from checkpoint {start_from!r} (skipping stages 1-{start_index + 1})", flush=True)
    else:
        start_index = -1

    repo_root = Path(__file__).resolve().parent.parent.parent
    catalog_path = repo_root / config["paths"]["hydromt_data_catalog"]
    catalog_root = config["paths"]["root"]
    catalog = get_data_catalog(catalog_path, root=catalog_root)
    mask_dir = Path(catalog["deltadtm_mask"].path).parent
    dem_dir = Path(catalog["deltadtm"].path).parent

    elev_threshold_m = tg_cfg["elev_threshold_m"]
    ocean_code = tg_cfg["ocean_code"]
    river_code = tg_cfg["river_code"]
    coarse_resolution_m = tg_cfg["coarse_resolution_m"]
    max_extent = tg_cfg["max_extent_tiles"]
    mask_index = _scan_mask_dir(mask_dir)
    dem_index = _scan_mask_dir(dem_dir)

    # ---- Stage 1: build_tile_index ----------------------------------------
    if start_index < 0:
        print("Stage 1: build_tile_index", flush=True)
        tiles = build_tile_index(mask_dir)
        print(f"  {len(tiles)} DeltaDTM tiles found", flush=True)
        _debug(tiles, "01_tile_index.gpkg")
    elif start_index == 0:
        # Resuming exactly from checkpoint "01_tile_index": Stage 2 (below)
        # still needs `tiles` loaded (for its own dropped_not_floodable diff).
        print(f"Stage 1: loading {debug_dir / '01_tile_index.gpkg'}", flush=True)
        tiles = retry_transient_io(gpd.read_file, debug_dir / "01_tile_index.gpkg")
        print(f"  {len(tiles)} DeltaDTM tiles loaded", flush=True)
    else:
        tiles = None  # Stage 2 also being skipped (start_index >= 1) - never referenced below

    # ---- Stage 2: filter_floodable_tiles (+ river-mouth marking) ----------
    if start_index < 1:
        print("\nStage 2: filter_floodable_tiles (+ river-mouth marking)", flush=True)
        floodable = filter_floodable_tiles(
            tiles, mask_index, dem_index,
            elev_threshold_m, coarse_resolution_m, ocean_code, river_code,
            tg_cfg["river_mouth_ocean_frac_min"], tg_cfg["river_mouth_river_frac_min"],
            tg_cfg["river_mouth_speckle_buffer_deg"], tg_cfg["river_mouth_min_coastal_component_cells"],
        )
        n_mouths = int(floodable["is_river_mouth"].sum()) if len(floodable) else 0
        print(f"  {len(floodable)}/{len(tiles)} tiles kept ({n_mouths} river-mouth)", flush=True)
        _debug(floodable, "02_floodable_tiles.gpkg")
        if write_debug:
            dropped_tiles = tiles[~tiles["coord"].isin(floodable["coord"])].reset_index(drop=True)
            _debug(dropped_tiles, "02_dropped_not_floodable.gpkg")
    else:
        print(f"\nStage 2: loading {debug_dir / '02_floodable_tiles.gpkg'}", flush=True)
        floodable = retry_transient_io(gpd.read_file, debug_dir / "02_floodable_tiles.gpkg")
        print(f"  {len(floodable)} floodable tiles loaded", flush=True)

    # river_mouth_seeds.gpkg is always rewritten from whatever `floodable`
    # ends up being (loaded or computed) - cheap, and keeps it from ever
    # going stale relative to a resumed run.
    river_mouth_seeds_path = Path(tg_cfg["river_mouth_seeds_path"])
    river_mouth_seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seed_cols = ["coord", "lat", "lon", "ocean_frac", "river_frac", "geometry"]
    seeds = floodable.loc[floodable["is_river_mouth"], seed_cols].reset_index(drop=True)
    retry_transient_io(seeds.to_file, river_mouth_seeds_path, driver="GPKG")
    print(f"  {len(seeds)} river-mouth seed tiles -> {river_mouth_seeds_path}", flush=True)

    grid, lat_min, lon_min = build_presence_grid(floodable)

    # ---- Stage 3: build_chunks (river-mouth seeded) ------------------------
    if start_index < 2:
        print("\nStage 3: build_chunks (river-mouth seeded)", flush=True)
        river_mouth, ocean_present = build_river_mouth_grids(floodable, lat_min, lon_min, grid.shape)
        chunks = build_chunks(grid, max_extent, river_mouth, ocean_present)
        print(f"  {len(chunks)} chunks before overlap cleanup", flush=True)
        _debug(_grid_rect_snapshot_gdf(chunks, lat_min, lon_min), "03_chunks_built.gpkg")
    else:
        print(f"\nStage 3: loading {debug_dir / '03_chunks_built.gpkg'}", flush=True)
        chunks = [bbox_to_grid_rect(b, lat_min, lon_min) for b in _load_bboxes(debug_dir / "03_chunks_built.gpkg")]
        print(f"  {len(chunks)} chunks loaded", flush=True)

    # ---- Stage 4: reduce_overlap -------------------------------------------
    if start_index < 3:
        print("\nStage 4: reduce_overlap", flush=True)
        cleaned = reduce_overlap(chunks, grid.shape)
        print(f"  {len(cleaned)} chunks after reduce_overlap", flush=True)
        _debug(_grid_rect_snapshot_gdf(cleaned, lat_min, lon_min), "04_chunks_reduce_overlap.gpkg")
    else:
        print(f"\nStage 4: loading {debug_dir / '04_chunks_reduce_overlap.gpkg'}", flush=True)
        cleaned = [bbox_to_grid_rect(b, lat_min, lon_min) for b in _load_bboxes(debug_dir / "04_chunks_reduce_overlap.gpkg")]
        print(f"  {len(cleaned)} chunks loaded", flush=True)

    # ---- Stage 5: add_minimum_overlap --------------------------------------
    if start_index < 4:
        print("\nStage 5: add_minimum_overlap", flush=True)
        repaired, unresolved = add_minimum_overlap(cleaned, grid, max_extent)
        print(f"  {len(repaired)} chunks after add_minimum_overlap ({unresolved} pairs left unresolved)", flush=True)
        _debug(_grid_rect_snapshot_gdf(repaired, lat_min, lon_min), "05_chunks_add_minimum_overlap.gpkg")
    else:
        print(f"\nStage 5: loading {debug_dir / '05_chunks_add_minimum_overlap.gpkg'}", flush=True)
        repaired = [bbox_to_grid_rect(b, lat_min, lon_min) for b in _load_bboxes(debug_dir / "05_chunks_add_minimum_overlap.gpkg")]
        print(f"  {len(repaired)} chunks loaded", flush=True)

    # ---- Stage 6: add_connector_chunks -------------------------------------
    if start_index < 5:
        print("\nStage 6: add_connector_chunks", flush=True)
        connected, pairs_bridged = add_connector_chunks(repaired)
        print(f"  {len(connected)} chunks after add_connector_chunks "
              f"({len(connected) - len(repaired)} connectors, bridging {pairs_bridged} pairs)", flush=True)
        _debug(_grid_rect_snapshot_gdf(connected, lat_min, lon_min), "06_chunks_add_connector_chunks.gpkg")

        covered = np.zeros_like(grid)
        for r0, r1, c0, c1 in connected:
            covered[r0:r1 + 1, c0:c1 + 1] = True
        assert (covered == grid).all(), "overlap cleanup broke full coverage - some tile is no longer in any chunk"

        bboxes = [grid_rect_to_bbox(rect, lat_min, lon_min) for rect in connected]
    else:
        print(f"\nStage 6: loading {debug_dir / '06_chunks_add_connector_chunks.gpkg'}", flush=True)
        bboxes = _load_bboxes(debug_dir / "06_chunks_add_connector_chunks.gpkg")
        print(f"  {len(bboxes)} chunks loaded", flush=True)

    # ---- Stage 7: filter_and_shave_chunks (+ internal split, 2026-08) -----
    if start_index < 6:
        print(f"\nStage 7: filter_and_shave_chunks ({len(bboxes)} chunks)", flush=True)
        shaved, dropped_exposure = filter_and_shave_chunks(
            bboxes, mask_index, dem_index, elev_threshold_m, ocean_code,
            catalog_path, catalog_root, tg_cfg["population_source"],
            coarse_resolution_m, tg_cfg["shave_sample_resolution_m"], tg_cfg["shave_unfloodable_fraction"],
            tg_cfg["min_split_gap_coarse_cells"],
        )
        print(f"  {len(shaved)} pieces kept from {len(bboxes)} input chunks", flush=True)
        _debug(_chunk_snapshot_gdf(shaved), "07_chunks_after_shave.gpkg")
        _debug(_dropped_gdf(dropped_exposure), "07_dropped_no_exposure_or_shave_empty.gpkg")
    else:
        print(f"\nStage 7: loading {debug_dir / '07_chunks_after_shave.gpkg'}", flush=True)
        shaved = _load_bboxes(debug_dir / "07_chunks_after_shave.gpkg")
        print(f"  {len(shaved)} chunks loaded", flush=True)

    # ---- Stage 8 [RETIRED, 2026-08]: merge_dry_chunks ----------------------
    # No longer called - a dry chunk no longer needs a wet edge of its own
    # (see module docstring / tile_chunking.py's compute_run_order); Stage 9
    # now takes Stage 7's `shaved` output directly.

    # ---- Stage 9: drop_redundant_chunks (1st pass) -------------------------
    if start_index < 7:
        print(f"\nStage 9: drop_redundant_chunks (1st pass, {len(shaved)} chunks)", flush=True)
        deduped = drop_redundant_chunks(shaved, tg_cfg["dedup_min_coverage_fraction"])
        print(f"  {len(deduped)} chunks after subset dedup", flush=True)
        _debug(_chunk_snapshot_gdf(deduped), "09_chunks_dedup.gpkg")
        _debug(_dropped_gdf(_diff_dropped(shaved, deduped, "redundant_covered_by_other_chunks")),
               "09_dropped_fully_contained.gpkg")
    else:
        print(f"\nStage 9: loading {debug_dir / '09_chunks_dedup.gpkg'}", flush=True)
        deduped = _load_bboxes(debug_dir / "09_chunks_dedup.gpkg")
        print(f"  {len(deduped)} chunks loaded", flush=True)

    # ---- Stage 10: split_oversized_chunks -----------------------------------
    if start_index < 8:
        print("\nStage 10: split_oversized_chunks", flush=True)
        split = split_oversized_chunks(
            deduped, mask_index, ocean_code, river_code,
            tg_cfg["river_mouth_ocean_frac_min"], tg_cfg["river_mouth_river_frac_min"],
            tg_cfg["river_mouth_check_band_cells"], coarse_resolution_m, max_extent,
        )
        n_still_oversized = sum(
            1 for b in split if round(b[2] - b[0]) > max_extent or round(b[3] - b[1]) > max_extent
        )
        if n_still_oversized:
            print(f"  note: {n_still_oversized} chunk(s) remain oversized - every split of them would "
                  f"have either stranded a dry piece or cut through a river mouth", flush=True)
        print(f"  {len(split)} chunks after split_oversized_chunks", flush=True)
        _debug(_chunk_snapshot_gdf(split), "10_chunks_split_oversized.gpkg")
    else:
        print(f"\nStage 10: loading {debug_dir / '10_chunks_split_oversized.gpkg'}", flush=True)
        split = _load_bboxes(debug_dir / "10_chunks_split_oversized.gpkg")
        print(f"  {len(split)} chunks loaded", flush=True)

    # ---- Stage 11: drop_redundant_chunks (2nd pass, 2026-08) ---------------
    # split_oversized_chunks can create NEW redundancy (a split piece
    # covered by some other, unrelated chunk(s)) that the 1st dedup pass
    # (Stage 9, which ran BEFORE splitting) never saw - tile_chunking.py's
    # own module docstring already called for running this "before AND
    # after step 10"; the orchestration just never did the "after" part
    # until now.
    if start_index < 9:
        print("\nStage 11: drop_redundant_chunks (2nd pass, after split)", flush=True)
        deduped_after_split = drop_redundant_chunks(split, tg_cfg["dedup_min_coverage_fraction"])
        print(f"  {len(deduped_after_split)} chunks after 2nd subset dedup", flush=True)
        _debug(_chunk_snapshot_gdf(deduped_after_split), "11_chunks_dedup_after_split.gpkg")
        _debug(_dropped_gdf(_diff_dropped(split, deduped_after_split, "redundant_covered_by_other_chunks")),
               "11_dropped_fully_contained_after_split.gpkg")
    else:
        print(f"\nStage 11: loading {debug_dir / '11_chunks_dedup_after_split.gpkg'}", flush=True)
        deduped_after_split = _load_bboxes(debug_dir / "11_chunks_dedup_after_split.gpkg")
        print(f"  {len(deduped_after_split)} chunks loaded", flush=True)

    # ---- Stage 12: cap_overlap_density --------------------------------------
    if start_index < 10:
        print("\nStage 12: cap_overlap_density", flush=True)
        capped = cap_overlap_density(deduped_after_split, tg_cfg["max_overlap_per_cell"])
        print(f"  {len(capped)} chunks after cap_overlap_density", flush=True)
        _debug(_dropped_gdf(_diff_dropped(deduped_after_split, capped, "cap_overlap_density")),
               "12_dropped_overlap_cap.gpkg")
        _debug(_chunk_snapshot_gdf(capped), "12_chunks_final.gpkg")
    else:
        print(f"\nStage 12: loading {debug_dir / '12_chunks_final.gpkg'}", flush=True)
        capped = _load_bboxes(debug_dir / "12_chunks_final.gpkg")
        print(f"  {len(capped)} chunks loaded", flush=True)

    # ---- Stage 13: compute_run_order (hop-distance/groups/ordering, 2026-08) ---
    # See src/tile_chunking.py's module docstring / compute_run_order - this
    # is what actually writes tile_grid.path now (`tile_id` = list position
    # after reordering by run order), and is the sole remaining place a
    # chunk gets dropped for lacking a path to the ocean (superseding the
    # retired Stage 8/merge_dry_chunks).
    if start_index < 11:
        print(f"\nStage 13: compute_run_order ({len(capped)} chunks)", flush=True)
        order, unreachable, diagnostics = compute_run_order(
            capped, mask_index, ocean_code, seeds, coarse_resolution_m,
        )
        if unreachable:
            _debug(
                _dropped_gdf([(capped[i], "no_path_to_ocean_via_any_chunk_chain") for i in unreachable]),
                "13_dropped_unreachable_via_any_chain.gpkg",
            )

        ordered_bboxes = [capped[i] for i in order]
        hop_distance_ordered = [diagnostics["hop_distance"][i] for i in order]

        final = bboxes_to_geodataframe(ordered_bboxes)
        # hop_distance rides along on the PRODUCTION manifest itself (2026-08,
        # not just the debug checkpoint below) - it's the one piece every
        # downstream consumer (Snakefile/HPC dispatch, whenever the deferred
        # DAG/HPC-orchestration work lands) needs to tell wave-0 chunks apart
        # from hinterland ones without cross-referencing a debug-only file
        # that may not even exist (tile_generation.write_debug_gpkg=false).
        # Purely additive - get_tile_geometry/TILE_IDS-building both select
        # columns by name, so this is safe for every existing reader.
        final["hop_distance"] = hop_distance_ordered
        retry_transient_io(final.to_file, output_path, driver="GPKG")
        print(f"\nWrote {len(final)} final chunks to {output_path}")

        final_with_diagnostics = final.copy()
        for col in ("group_id", "is_river_mouth_seed", "degree", "ocean_fraction"):
            final_with_diagnostics[col] = [diagnostics[col][i] for i in order]
        _debug(final_with_diagnostics, "13_chunks_ordered.gpkg")

        # NOT IMPLEMENTED (2026-08): a hop>=1 chunk's real neighbour-water-
        # level boundary points still need to come from somewhere. A
        # tile-generation-time precompute (candidate point LOCATIONS from
        # frozen geometry alone, gridding each hop>=1 chunk's overlap with
        # its earlier neighbour(s)) was tried and rejected here - real
        # overlaps in this pipeline span up to hundreds of km, not thin
        # border strips, so any fixed grid resolution produces a huge,
        # mostly-dry-land point count (a real test: 8.65M points across
        # only 119 chunks). The right place to pick points is AFTER a
        # wave-0 chunk has actually been simulated - read its real
        # waterdepth output and keep only the (very few, genuinely wet)
        # non-zero/non-NaN cells within the overlap - which has to run
        # BETWEEN wave-0 and wave>=1 simulations, a step that doesn't
        # exist yet (see the inter-chunk boundary propagation plan's
        # "Deferred consideration" section, and tile_chunking.py's own
        # NOT IMPLEMENTED note just above compute_run_order's output
        # section). boundaries.sample_waterlevel_at_points (the raster
        # point-lookup primitive) already exists and is ready for that
        # step once it's built.
    else:
        print(f"\nStage 13: loading {debug_dir / '13_chunks_ordered.gpkg'} (regenerating histogram only)", flush=True)
        final = retry_transient_io(gpd.read_file, debug_dir / "13_chunks_ordered.gpkg")
        print(f"  {len(final)} final chunks loaded", flush=True)

    if write_debug:
        histogram_path = debug_dir / "13_chunks_final_size_histogram.png"
        _write_size_histogram(final, histogram_path)
        print(f"Wrote chunk-size histogram -> {histogram_path}")
        print(f"Debug GeoPackages (one per stage) written to {debug_dir}")


if __name__ == "__main__":
    sys.exit(
        "build_tile_manifest.py is no longer a standalone entry point.\n"
        "Run it via: python run_preparation.py tile_generation\n"
        "See run_preparation.py --help for the full list of steps."
    )
