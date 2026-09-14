"""Build hop_distance-closed tile subsets for the ESP/FRA/NOR HPC calibration
sweep (docs/calibration_sweep_plan.md) - two groups, `esp_fra` (simulated at
RP100) and `nor` (RP250), each written as its own GeoPackage under
{processed_inputs_dir}/mask/calibration/.

For each group: finds every coastal benchmark catalog entry for its member
countries, then unions tile IDs intersecting each NAMED region SEPARATELY -
NEVER a bbox union across regions. Spain's `mainland`/`canary_islands` and
France's `metropole`/`guadeloupe`/`martinique`/`guyane`/`mayotte`/`reunion`
regions are scattered across most of the globe; naively unioning their
bboxes into one rectangle per group would select a wildly wrong "core" tile
set (confirmed 2026-09 - France's regions alone span longitude -62.6 to
45.8, latitude -13.5 to 51.5). This mirrors how validate_country.py already
processes multi-region countries, not a new pattern.

The resulting "core" set is then expanded to its full hop_distance
transitive closure using EXACTLY the same neighbour-candidate query
run_aqueduct.py/run_aqueduct_cli.py use at runtime, repeated to a fixed
point - a hop>=1 (hinterland) tile is seeded at runtime from its own
lower-hop neighbours' already-written output, a dependency invisible to
Snakemake's own DAG (see rules/simulation.smk's docstring), so a subset run
must include every such neighbour itself or those tiles silently underflood.

hop_distance and tile_id are copied VERBATIM from the production tile grid,
never recomputed on the subset - hop_distance is a BFS distance over the
FULL production adjacency graph; recomputing it on an isolated subset would
silently produce a different, non-equivalent wave ordering. Closure
computation itself is RP-independent (hop_distance is pure geometry/BFS, not
tied to any scenario), so one run here covers both groups' eventual RPs.

This script's printed report (core vs. closure size, hop_distance breakdown)
is a hard gate - review it before writing/using any scenario config that
depends on these output paths.

Usage:
    python select_country_tiles.py --config <config.yml> [--out-dir <dir>]
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from tiles import load_tile_grid  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]

GROUPS = {
    "esp_fra": ["ESP", "FRA"],
    "nor": ["NOR"],
}


def _find_benchmark_keys(catalog, country_isos: list[str]) -> list[str]:
    """Every coastal benchmark catalog entry for any of `country_isos`.

    Mirrors validate_country.py's own private `_find_benchmark_keys` filter
    (country_iso + hazard_type == "coastal") - inlined here rather than
    imported across the validation/ vs snakemake_workflow/ package boundary
    for a private, underscore-prefixed function.
    """
    keys = []
    for key in catalog.sources.keys():
        meta = catalog.get_source(key).meta or {}
        if meta.get("country_iso") in country_isos and meta.get("hazard_type") == "coastal":
            keys.append(key)
    return sorted(keys)


def _core_tile_ids(tile_grid: gpd.GeoDataFrame, catalog, country_isos: list[str]) -> set[int]:
    """Union of tile IDs intersecting each named benchmark region SEPARATELY
    - see module docstring for why a bbox union across regions is wrong.
    """
    core: set[int] = set()
    seen = []
    for key in _find_benchmark_keys(catalog, country_isos):
        meta = catalog.get_source(key).meta or {}
        regions = meta.get("regions") or {}
        if not regions:
            print(f"  WARNING: {key} has no meta.regions - skipping (cannot safely bbox it)")
            continue
        for region_name, bbox in regions.items():
            region_geom = shapely_box(*bbox)
            hits = tile_grid[tile_grid.geometry.intersects(region_geom)]
            core |= set(hits["tile_id"].astype(int))
            seen.append(f"{key}:{region_name} ({len(hits)} tiles)")
    print(f"  regions processed ({len(seen)}):")
    for line in seen:
        print(f"    {line}")
    return core


def _hop_distance_closure(tile_grid: gpd.GeoDataFrame, core_ids: set[int]) -> set[int]:
    """Transitive hop_distance closure - see module docstring. Fixed-point
    iteration: a newly-added lower-hop tile may itself be hop>=1 and need
    further upstream neighbours.
    """
    closure = set(core_ids)
    changed = True
    while changed:
        changed = False
        for tid in list(closure):
            this_tile = tile_grid[tile_grid["tile_id"] == tid]
            hop = int(this_tile["hop_distance"].iloc[0])
            if hop == 0:
                continue
            this_geom = this_tile.geometry.iloc[0]
            candidates = tile_grid[
                (tile_grid["hop_distance"] < hop)
                & (tile_grid["tile_id"] != tid)
                & tile_grid.geometry.intersects(this_geom)
            ]
            new_ids = set(candidates["tile_id"].astype(int)) - closure
            if new_ids:
                closure |= new_ids
                changed = True
    return closure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = _REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--out-dir", default=None,
        help="output directory for the subset GeoPackages "
             "(default: {processed_inputs_dir}/mask/calibration/)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    val_cfg = cfg["validation"]

    tile_grid = load_tile_grid(cfg["tile_grid"]["path"])
    tile_grid["tile_id"] = tile_grid["tile_id"].astype(int)
    print(f"Production tile grid: {len(tile_grid)} tiles\n")

    bench_catalog = get_data_catalog(
        _REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"],
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["processed_inputs_dir"]) / "mask" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    for group_name, country_isos in GROUPS.items():
        print(f"=== {group_name} ({'+'.join(country_isos)}) ===")
        core_ids = _core_tile_ids(tile_grid, bench_catalog, country_isos)
        closure_ids = _hop_distance_closure(tile_grid, core_ids)

        pulled_in = closure_ids - core_ids
        hop_counts: dict[int, int] = {}
        for tid in closure_ids:
            hop = int(tile_grid.loc[tile_grid["tile_id"] == tid, "hop_distance"].iloc[0])
            hop_counts[hop] = hop_counts.get(hop, 0) + 1

        print(f"  core tiles: {len(core_ids)}")
        print(f"  closure tiles: {len(closure_ids)} ({len(pulled_in)} pulled in as pure dependency)")
        print(f"  by hop_distance: {dict(sorted(hop_counts.items()))}")
        if core_ids and len(closure_ids) > 2 * len(core_ids):
            print("  ** WARNING: closure is more than 2x core size - review before using **")

        subset = tile_grid.loc[
            tile_grid["tile_id"].isin(closure_ids), ["tile_id", "hop_distance", "geometry"]
        ].copy()
        out_path = out_dir / f"{group_name}_tiles.gpkg"
        subset.to_file(out_path, driver="GPKG")
        print(f"  wrote {out_path} ({len(subset)} tiles)\n")


if __name__ == "__main__":
    main()
