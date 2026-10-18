"""Build the connectivity-first domain manifest (tile_grid.path).

Entry point: run(config), via `python run_preparation.py tile_generation`.
Stages are in src/connectivity_tiling.py; design in docs/methods_01 section 3.
Scenario-independent: depends only on the DeltaDTM mask, elevation and
tile_generation.elev_threshold_m. tile_id is assigned in hop_distance-ascending
run order. Phase 3 (~1.2 h of ~5.3 h globally) checkpoints each component,
with a marker line even for zero-domain components so resume skips them;
delete the checkpoint to force a clean run.
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

from plot_domain_maps import plot_dropped_unreachable_map, plot_final_domains_map  # noqa: E402


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
    # done_ids includes zero-domain components via their "component_done" marker.
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
    """Plot native (1 arcsecond) pixel counts per domain, the solver's OOM-risk proxy."""
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
    """Build the connectivity-first domain manifest and write tile_grid.path.

    Also writes per-phase GeoPackages, a size histogram and methods maps next to
    it. Phase 3 resumes from tile_generation_phase3_checkpoint.jsonl.
    """
    tg_cfg = config["tile_generation"]
    output_path = Path(config["tile_grid"]["path"])
    intermediate_dir = output_path.parent

    def _write_intermediate(gdf: gpd.GeoDataFrame, filename: str) -> None:
        _write_gpkg(gdf, intermediate_dir / filename)

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

    print("Phase 0: load_raw_tile_index", flush=True)
    sub = load_raw_tile_index(mask_dir)
    print(f"  {len(sub)} raw DeltaDTM tiles", flush=True)
    _write_intermediate(sub, "00_raw_tile_index.gpkg")

    print("\nPhase 1: build_connectivity_graph", flush=True)
    t0 = time.time()
    edges = build_connectivity_graph(sub, mask_index, dem_index, cfg)
    print(f"  {len(edges)} floodable-land connection(s), t={time.time() - t0:.0f}s", flush=True)

    clusters = connected_components(len(sub), edges)
    sizes = sorted(((len(members), root_idx) for root_idx, members in clusters.items()), reverse=True)
    print(f"Phase 2: {len(sizes)} connected component(s)", flush=True)

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = intermediate_dir / "tile_generation_phase3_checkpoint.jsonl"
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
    _write_intermediate(_domains_to_gdf(all_domains), "01_domains_presplit.gpkg")

    merged = merge_small_domains_tiered(
        all_domains, cfg.merge_trigger_cells, cfg.preferred_ceiling_cells, cfg.hard_ceiling_cells,
    )
    print(f"Phase 4 merge: {len(all_domains)} -> {len(merged)} domain(s)", flush=True)
    _write_intermediate(_domains_to_gdf(merged), "02_domains_merged.gpkg")

    hop, unreachable = compute_hop_distances(merged, mask_index, dem_index, cfg)
    n_hop0 = sum(1 for h in hop if h == 0)
    print(f"Phase 5 hop-distance: hop=0 {n_hop0}/{len(merged)}, unreachable (dropped) {len(unreachable)}", flush=True)
    unreachable_set = set(unreachable)
    if unreachable:
        _write_intermediate(_domains_to_gdf([merged[i] for i in unreachable]), "03_dropped_unreachable.gpkg")
    final_domains = [d for i, d in enumerate(merged) if i not in unreachable_set]
    final_hop = [h for i, h in enumerate(hop) if i not in unreachable_set]

    padded_domains, n_capped = pad_hop0_domains_bbox(
        final_domains, final_hop, cfg.coastal_buffer_km, cfg.hard_ceiling_cells,
    )
    if n_capped:
        print(f"Phase 6 coastal pad: {n_capped} hop=0 domain(s) left unpadded "
              f"(padding would exceed the hard ceiling)", flush=True)

    final_gdf = assign_tile_id(padded_domains, final_hop)
    retry_transient_io(final_gdf.to_file, output_path, driver="GPKG")
    print(f"\nWrote {len(final_gdf)} domains to {output_path}", flush=True)
    _write_intermediate(final_gdf, "04_tile_grid_final.gpkg")

    histogram_path = intermediate_dir / "tile_generation_final_size_histogram.png"
    _write_size_histogram(final_gdf, histogram_path)
    print(f"Wrote domain-size histogram -> {histogram_path}")
    print(f"Intermediate per-phase GeoPackages written to {intermediate_dir}")

    plot_final_domains_map(final_gdf, intermediate_dir)
    if unreachable:
        dropped_gdf = _domains_to_gdf([merged[i] for i in unreachable])
        plot_dropped_unreachable_map(dropped_gdf, intermediate_dir)


if __name__ == "__main__":
    sys.exit(
        "build_tile_manifest.py is no longer a standalone entry point.\n"
        "Run it via: python run_preparation.py tile_generation\n"
        "See run_preparation.py --help for the full list of steps."
    )
