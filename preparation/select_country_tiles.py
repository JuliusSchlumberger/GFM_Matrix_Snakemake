"""Build hop_distance-closed tile subsets for country-scoped validation runs.

Core = tiles intersecting each named benchmark region SEPARATELY (multi-region
countries span the globe, so a bbox union is wrong). Core is closed over lower-
hop neighbours with the same intersects query as run_eikonal at runtime;
missing neighbours would silently underflood hop>=1 tiles. hop_distance and
tile_id are copied from the production grid, never recomputed on the subset.
Review the printed core-vs-closure report before using the output.

Usage:
    python select_country_tiles.py --config <config.yml> [--out-dir <dir>]
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from tiles import load_tile_grid  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]

GROUPS = {
    # GBR coastal benchmark entries are Wales and Scotland only (no England/N.Ireland).
    "gbr_wales_scotland": ["GBR"],
    "new_brunswick": ["CAN"],
}


def _find_benchmark_keys(catalog, country_isos: list[str]) -> list[str]:
    """Return coastal benchmark catalog keys for country_isos (as validate_country.py)."""
    keys = []
    for key in catalog.sources.keys():
        meta = catalog.get_source(key).meta or {}
        if meta.get("country_iso") in country_isos and meta.get("hazard_type") == "coastal":
            keys.append(key)
    return sorted(keys)


def _core_tile_ids(tile_grid: gpd.GeoDataFrame, catalog, country_isos: list[str]) -> set[int]:
    # Per named region, never a bbox union across regions.
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
    # Fixed point: a newly added lower-hop tile may itself be hop>=1.
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
    parser.add_argument(
        "--groups", nargs="+", default=None, choices=list(GROUPS),
        help=f"which GROUPS entries to (re-)build (default: all of them - "
             f"{', '.join(GROUPS)}). Pass this to add/refresh one group without "
             f"touching another, already-complete group's own output file.",
    )
    args = parser.parse_args()
    groups = {k: GROUPS[k] for k in args.groups} if args.groups else GROUPS

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

    for group_name, country_isos in groups.items():
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
