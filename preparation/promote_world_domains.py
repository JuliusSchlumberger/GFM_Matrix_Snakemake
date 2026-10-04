"""One-time promotion of the already-computed global connectivity-tiling result
(tests/connectivity_tiling_prototype/output/world_domains.gpkg, 4,504 domains,
2026-10) to production's `tile_grid.path`.

Two things happen here that the production `tile_generation` step (once rewired to
call `src/connectivity_tiling.py` directly - see docs/methods_01_tile_processing_and_
waterlevels.md) will also do for future regenerations, but applied here as a cheap
one-off fix-up so this promotion doesn't require re-running the ~5.3h world
computation:

1. Coastal buffer pad, hop_distance==0 domains ONLY. Phase 3's trim-to-content step
   only gives an IMPLICIT buffer around a coastal nose/headland (water within the
   land's own row/column bounding extent survives, since domains are rectangles, not
   per-pixel masks) - not the old pipeline's EXPLICIT minimum-ocean-margin guarantee.
   This only matters for hop=0 domains (self-forced directly from the open
   coast/COAST-RP, where a headland cutting too close to the domain edge would
   matter) - a hop>=1 hinterland domain is forced from an already-simulated
   neighbour's wave, not from direct coastline geometry, so it has no analogous
   concern. Applied strictly AFTER hop-distance is already known, so no BFS/adjacency
   re-derivation is needed - padding a hop=0 domain's bbox doesn't change the fact
   that it's hop=0.
2. `tile_id` assignment. The prototype's domains never had one (nothing downstream
   can consume a tile grid without it - Snakemake wildcards, HPC batch generators,
   model_outputs/{tile_id}/... paths all require it). Assigned sequentially, ordered
   by hop_distance ascending (wave-0 first) then component_id then original row
   order - preserves the old schema's "hop_distance correlates with run order" spirit
   without inventing a new tie-break scheme.

Usage:
    python preparation/promote_world_domains.py [--dry-run]
"""

import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
WORLD_DOMAINS_PATH = REPO / "tests" / "connectivity_tiling_prototype" / "output" / "world_domains.gpkg"

COASTAL_BUFFER_KM_DEFAULT = 20.0
KM_PER_DEG = 111.32
HARD_CEILING_CELLS = 100_000_000.0
DEG_TO_NATIVE_PX = 3600.0


def _native_cells(bbox) -> float:
    minx, miny, maxx, maxy = bbox
    return (maxx - minx) * DEG_TO_NATIVE_PX * (maxy - miny) * DEG_TO_NATIVE_PX


def pad_hop0_bbox(bbox, buffer_deg: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = bbox
    padded = (minx - buffer_deg, miny - buffer_deg, maxx + buffer_deg, maxy + buffer_deg)
    if _native_cells(padded) <= HARD_CEILING_CELLS:
        return padded
    return bbox  # padding would exceed the hard ceiling - leave this one unpadded


def promote(buffer_km: float = COASTAL_BUFFER_KM_DEFAULT, dry_run: bool = False) -> gpd.GeoDataFrame:
    gdf = retry_transient_io(gpd.read_file, WORLD_DOMAINS_PATH)
    print(f"Loaded {len(gdf)} domains from {WORLD_DOMAINS_PATH}")

    buffer_deg = buffer_km / KM_PER_DEG
    is_hop0 = gdf["hop_distance"] == 0
    n_hop0 = int(is_hop0.sum())
    print(f"Padding {n_hop0} hop_distance==0 domain(s) by {buffer_km} km ({buffer_deg:.4f} deg) on each side...")

    n_capped = 0
    new_geoms = []
    for idx, row in gdf.iterrows():
        if not is_hop0.loc[idx]:
            new_geoms.append(row.geometry)
            continue
        bbox = row.geometry.bounds
        padded_bbox = pad_hop0_bbox(bbox, buffer_deg)
        if padded_bbox == bbox:
            n_capped += 1
        from shapely.geometry import box
        new_geoms.append(box(*padded_bbox))
    gdf["geometry"] = new_geoms
    if n_capped:
        print(f"  {n_capped} hop=0 domain(s) left unpadded - padding would have exceeded the "
              f"{HARD_CEILING_CELLS / 1e6:.0f}M-cell hard ceiling.")

    # tile_id: sequential, ordered by hop_distance asc, then component_id, then
    # original row order (stable sort preserves the last tiebreak automatically).
    gdf = gdf.sort_values(["hop_distance", "component_id"], kind="stable").reset_index(drop=True)
    gdf["tile_id"] = gdf.index

    cols = ["tile_id", "hop_distance", "split_reason", "component_id", "approx_cells_M", "geometry"]
    gdf = gdf[cols]

    print(f"Final: {len(gdf)} domains, tile_id 0..{len(gdf) - 1}")
    print(gdf["hop_distance"].value_counts().sort_index().to_string())

    if not dry_run:
        cfg = load_config(REPO / "snakemake_workflow" / "config" / "config.yml")
        out_path = Path(cfg["tile_grid"]["path"])
        tmp_path = out_path.with_name(f"{out_path.name}.tmp")
        retry_transient_io(gdf.to_file, tmp_path, driver="GPKG")
        import os
        retry_transient_io(os.replace, tmp_path, out_path)
        print(f"Wrote {out_path}")

    return gdf


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--buffer-km", type=float, default=COASTAL_BUFFER_KM_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    promote(buffer_km=args.buffer_km, dry_run=args.dry_run)
