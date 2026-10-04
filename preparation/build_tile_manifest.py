"""Build the connectivity-first domain manifest -> tile_grid.path (2026-10).

Replaces the old 13-stage greedy-covering pipeline (build_chunks/reduce_overlap/
filter_and_shave_chunks/.../compute_run_order - retired, see src/tile_chunking.py's
own RETIRED note) with src/connectivity_tiling.py's connectivity-first method - see
that module's docstring for the phases and docs/methods_01_tile_processing_and_
waterlevels.md section 3 for the full conceptual design, the "why this is correct"
argument, and real global validation numbers.

Frozen geometry DAG: depends only on the DeltaDTM mask, elevation, and
tile_generation.elev_threshold_m - never on scenario (Coast-RP station VALUES,
SLR, return period). Each tile gets its own boundary-condition stations directly
(extract_boundaries.py), same as before.

Stages, in order (each delegates to src/connectivity_tiling.py):
  0. load_raw_tile_index       - one 1x1deg polygon per real DeltaDTM mask tile.
  1. build_connectivity_graph  - floodable-LAND-only tile adjacency.
  2. connected_components      - plain union-find.
  3. split_component_to_budget - per component, budget-aware recursive split
                                  (checkpointed - see below).
  4. merge_small_domains_tiered - post-hoc merge of undersized domains.
  5. compute_hop_distances     - BFS from real ocean-touching domains; drops
                                  any domain with no path to ocean in its own
                                  component.
  6. pad_hop0_domains_bbox     - explicit coastal buffer, hop=0 domains only.
  Final: assign_tile_id        - writes tile_grid.path.

Output: tile_grid.path (tile_id/hop_distance/geometry, plus split_reason/
component_id/approx_cells_M as harmless extra diagnostic columns - nothing
downstream is column-position-sensitive). `tile_id` is assigned by hop_distance-
ascending run order, same spirit as the old pipeline's `compute_run_order`.

Debug output (tile_generation.write_debug_gpkg): one GeoPackage per stage,
written to tile_generation.debug_gpkg_dir, lets every intermediate shape be
inspected in QGIS, not just the final tile_grid.path output. Also writes a
50-bin histogram of each final domain's native (1 arcsecond) pixel count -
Aqueduct's OOM risk is dominated by tile pixel count (see src/tile_split.py's
own docstring).

Checkpointed (independent of write_debug_gpkg - this is the step's resilience
mechanism, not just a QA convenience): Phase 3 is by far the most expensive stage
(a real global run: ~1.2h out of ~5.3h total, dominated by per-component raster
reads), so every component's result is appended to tile_generation_phase3_
checkpoint.jsonl (under debug_gpkg_dir) immediately after that component
finishes, including an explicit marker line even for a component that produces
zero domains (a pure-ocean/ice singleton tile) - without that marker, a
zero-domain component would look indistinguishable from "not yet processed" on
resume and get silently, harmlessly redone. Re-running `run()` after any
interruption automatically resumes from whatever's already checkpointed; delete
the checkpoint file to force a clean re-run.

Not a standalone entry point - exposes `run(config)`, called from
run_preparation.py (`python run_preparation.py tile_generation`).
"""

import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config_utils import get_data_catalog, retry_transient_io  # noqa: E402
from connectivity_tiling import (  # noqa: E402
    ConnectivityConfig,
    Domain,
    assign_tile_id,
    bbox_to_polygon,
    build_connectivity_graph,
    compute_hop_distances,
    connected_components,
    load_raw_tile_index,
    merge_small_domains_tiered,
    pad_hop0_domains_bbox,
    split_component_to_budget,
)
from tiles import _scan_mask_dir  # noqa: E402


def _write_gpkg(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    retry_transient_io(gdf.to_file, path, driver="GPKG")


def _domains_to_gdf(domains: list[Domain]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "split_reason": [d.split_reason for d in domains],
            "component_id": [d.component_id for d in domains],
            "approx_cells_M": [round(d.approx_native_cells / 1e6, 2) for d in domains],
        },
        geometry=[bbox_to_polygon(d.bbox) for d in domains],
        crs="EPSG:4326",
    )


def _load_checkpoint(checkpoint_path: Path) -> tuple[list[Domain], set[int]]:
    """Returns (domains already on disk, component_ids already done). See
    this module's own docstring for why done_ids is built from BOTH domain
    records and the unconditional per-component marker record."""
    if not checkpoint_path.exists():
        return [], set()
    domains: list[Domain] = []
    done_ids: set[int] = set()
    for line in checkpoint_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        done_ids.add(rec["component_id"])
        if rec.get("marker") == "component_done":
            continue
        domains.append(Domain(
            bbox=tuple(rec["bbox"]), split_reason=rec["split_reason"],
            approx_native_cells=rec["approx_native_cells"], component_id=rec["component_id"],
        ))
    return domains, done_ids


def _append_checkpoint(checkpoint_path: Path, domains: list[Domain], component_id: int) -> None:
    with open(checkpoint_path, "a") as f:
        for d in domains:
            f.write(json.dumps({
                "bbox": list(d.bbox), "split_reason": d.split_reason,
                "approx_native_cells": d.approx_native_cells, "component_id": component_id,
            }) + "\n")
        f.write(json.dumps({"marker": "component_done", "component_id": component_id}) + "\n")


def _write_size_histogram(final: gpd.GeoDataFrame, path: Path) -> None:
    """50-bin histogram of each final domain's approximate native (1
    arcsecond) pixel count - a direct proxy for Aqueduct's OOM risk, which
    is dominated by tile pixel count (see src/tile_split.py's own
    docstring). Log-scale y-axis since domain sizes span orders of
    magnitude.
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
    ax.set_xlabel("Native (1 arcsecond) pixel count per domain")
    ax.set_ylabel("Number of domains (log scale)")
    ax.set_title(
        f"Final domain size distribution - n={len(final)} domains\n"
        f"min={min(n_cells):,}, median={int(np.median(n_cells)):,}, "
        f"mean={int(np.mean(n_cells)):,}, max={max(n_cells):,} px"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(config: dict) -> None:
    tg_cfg = config["tile_generation"]
    output_path = Path(config["tile_grid"]["path"])
    write_debug = tg_cfg.get("write_debug_gpkg", False)
    debug_dir = Path(tg_cfg["debug_gpkg_dir"])

    def _debug(gdf: gpd.GeoDataFrame, filename: str) -> None:
        if write_debug:
            _write_gpkg(gdf, debug_dir / filename)

    repo_root = Path(__file__).resolve().parent.parent
    catalog_path = repo_root / config["paths"]["hydromt_data_catalog"]
    catalog_root = config["paths"]["root"]
    catalog = get_data_catalog(catalog_path, root=catalog_root)
    mask_dir = Path(catalog["deltadtm_mask"].path).parent
    dem_dir = Path(catalog["deltadtm"].path).parent
    mask_index = _scan_mask_dir(mask_dir)
    dem_index = _scan_mask_dir(dem_dir)

    cfg = ConnectivityConfig(
        elev_threshold_m=tg_cfg["elev_threshold_m"],
        ocean_code=tg_cfg["ocean_code"],
        coarse_resolution_m=tg_cfg["coarse_resolution_m"],
        strip_resolution_m=tg_cfg["strip_resolution_m"],
        strip_width_deg=tg_cfg["strip_width_deg"],
        budget_cells=tg_cfg["budget_cells"],
        min_split_gap_coarse_cells=tg_cfg["min_split_gap_coarse_cells"],
        overlap_target_km=tg_cfg["overlap_target_km"],
        merge_trigger_cells=tg_cfg["merge_trigger_cells"],
        preferred_ceiling_cells=tg_cfg["preferred_ceiling_cells"],
        hard_ceiling_cells=tg_cfg["hard_ceiling_cells"],
        coastal_buffer_km=tg_cfg["coastal_buffer_km"],
    )

    # ---- Phase 0: raw tile index --------------------------------------------
    print("Phase 0: load_raw_tile_index", flush=True)
    sub = load_raw_tile_index(mask_dir)
    print(f"  {len(sub)} raw DeltaDTM tiles", flush=True)
    _debug(sub, "00_raw_tile_index.gpkg")

    # ---- Phase 1: connectivity graph ----------------------------------------
    print("\nPhase 1: build_connectivity_graph", flush=True)
    t0 = time.time()
    edges = build_connectivity_graph(sub, mask_index, dem_index, cfg)
    print(f"  {len(edges)} floodable-land connection(s), t={time.time() - t0:.0f}s", flush=True)

    # ---- Phase 2: connected components --------------------------------------
    clusters = connected_components(len(sub), edges)
    sizes = sorted(((len(members), root_idx) for root_idx, members in clusters.items()), reverse=True)
    print(f"Phase 2: {len(sizes)} connected component(s)", flush=True)

    # ---- Phase 3: budget-aware splitting, per component, checkpointed ------
    debug_dir.mkdir(parents=True, exist_ok=True)  # checkpoint needs this dir regardless of write_debug_gpkg
    checkpoint_path = debug_dir / "tile_generation_phase3_checkpoint.jsonl"
    all_domains, done_ids = _load_checkpoint(checkpoint_path)
    if done_ids:
        print(f"Resuming Phase 3 - {len(done_ids)} component(s) already checkpointed in {checkpoint_path}", flush=True)

    n_total = len(sizes)
    t_phase3 = time.time()
    for component_id, (n_members, root_idx) in enumerate(sizes):
        if component_id in done_ids:
            continue
        members = clusters[root_idx]
        bounds = sub.loc[members, "geometry"].total_bounds
        cluster_bbox = tuple(bounds)
        member_coords = set(sub.loc[members, "coord"])
        pieces = split_component_to_budget(cluster_bbox, mask_index, dem_index, cfg, member_coords=member_coords)
        for d in pieces:
            d.component_id = component_id
        _append_checkpoint(checkpoint_path, pieces, component_id)
        all_domains.extend(pieces)
        print(
            f"  [{component_id + 1}/{n_total}] {n_members} tile(s) -> {len(pieces)} domain(s) "
            f"(elapsed {(time.time() - t_phase3) / 60:.1f} min)",
            flush=True,
        )

    print(f"\nPhase 3 complete: {len(all_domains)} domain(s) across {n_total} component(s), "
          f"t={time.time() - t_phase3:.0f}s", flush=True)
    _debug(_domains_to_gdf(all_domains), "01_domains_presplit.gpkg")

    # ---- Phase 4: post-hoc merge of undersized domains ----------------------
    merged = merge_small_domains_tiered(
        all_domains, cfg.merge_trigger_cells, cfg.preferred_ceiling_cells, cfg.hard_ceiling_cells,
    )
    print(f"Phase 4 merge: {len(all_domains)} -> {len(merged)} domain(s)", flush=True)
    _debug(_domains_to_gdf(merged), "02_domains_merged.gpkg")

    # ---- Phase 5: hop-distance BFS, drop unreachable ------------------------
    hop, unreachable = compute_hop_distances(merged, mask_index, dem_index, cfg)
    n_hop0 = sum(1 for h in hop if h == 0)
    print(f"Phase 5 hop-distance: hop=0 {n_hop0}/{len(merged)}, unreachable (dropped) {len(unreachable)}", flush=True)
    unreachable_set = set(unreachable)
    if write_debug and unreachable:
        _debug(_domains_to_gdf([merged[i] for i in unreachable]), "03_dropped_unreachable.gpkg")
    final_domains = [d for i, d in enumerate(merged) if i not in unreachable_set]
    final_hop = [h for i, h in enumerate(hop) if i not in unreachable_set]

    # ---- Phase 6: coastal buffer pad, hop=0 domains only --------------------
    padded_domains, n_capped = pad_hop0_domains_bbox(
        final_domains, final_hop, cfg.coastal_buffer_km, cfg.hard_ceiling_cells,
    )
    if n_capped:
        print(f"Phase 6 coastal pad: {n_capped} hop=0 domain(s) left unpadded "
              f"(padding would exceed the hard ceiling)", flush=True)

    # ---- Final: assign tile_id, write tile_grid.path ------------------------
    final_gdf = assign_tile_id(padded_domains, final_hop)
    retry_transient_io(final_gdf.to_file, output_path, driver="GPKG")
    print(f"\nWrote {len(final_gdf)} domains to {output_path}", flush=True)
    _debug(final_gdf, "04_tile_grid_final.gpkg")

    if write_debug:
        histogram_path = debug_dir / "tile_generation_final_size_histogram.png"
        _write_size_histogram(final_gdf, histogram_path)
        print(f"Wrote domain-size histogram -> {histogram_path}")
        print(f"Debug GeoPackages written to {debug_dir}")


if __name__ == "__main__":
    sys.exit(
        "build_tile_manifest.py is no longer a standalone entry point.\n"
        "Run it via: python run_preparation.py tile_generation\n"
        "See run_preparation.py --help for the full list of steps."
    )
