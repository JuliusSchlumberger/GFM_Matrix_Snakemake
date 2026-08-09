"""Report preprocessing progress across the tile grid by counting which
per-tile output files already exist on disk - no Snakemake invocation
needed, safe to run anytime (including while an HPC run is in progress)
since it only reads, never writes.

Preprocessing produces several files per tile, roughly in this order:
    tile_geometry.gpkg  (extract_tile_geometry)
    model_bbox.json      (compute_model_bbox)
    dem.tif, mask.tif, friction.tif   (extract_dem / extract_dem_mask / compute_friction - roughly parallel)
    boundaries_{RP}_{SLR}.gpkg   (extract_boundaries - one per return_period x waterlevel_name)
    aqueduct_{RP}_{SLR}.toml     (write_aqueduct_config - one per return_period x waterlevel_name)

Counting each gives a rough per-stage progress picture, not just a single
"done/not done" per tile - e.g. many tiles with a DEM but no boundaries yet
tells you the run is mid-way through extract_dem-family rules, not stuck.

Usage:
    python snakemake_workflow/check_preprocess_progress.py [--config path/to/config.yml] [--watch SECONDS]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config_utils import load_config, merged_slr_scenarios  # noqa: E402
from tiles import load_tile_grid  # noqa: E402


def _count_progress(model_outputs: Path, tile_ids: list[int], n_scenarios_per_tile: int) -> dict:
    counts = {
        "tile_geometry.gpkg": 0,
        "model_bbox.json": 0,
        "dem.tif": 0,
        "mask.tif": 0,
        "friction.tif": 0,
    }
    total_boundaries = 0
    total_tomls = 0
    for tid in tile_ids:
        inputs_dir = model_outputs / str(tid) / "inputs"
        if not inputs_dir.is_dir():
            continue
        entries = set(p.name for p in inputs_dir.iterdir())
        for key in counts:
            if key in entries:
                counts[key] += 1
        total_boundaries += sum(1 for n in entries if n.startswith("boundaries_") and n.endswith(".gpkg"))
        total_tomls += sum(1 for n in entries if n.startswith("aqueduct_") and n.endswith(".toml"))

    n_tiles = len(tile_ids)
    expected_scenario_files = n_tiles * n_scenarios_per_tile
    return {
        "n_tiles": n_tiles,
        "counts": counts,
        "boundaries": total_boundaries,
        "boundaries_expected": expected_scenario_files,
        "tomls": total_tomls,
        "tomls_expected": expected_scenario_files,
    }


def _print_report(report: dict) -> None:
    n_tiles = report["n_tiles"]
    print(f"Tile grid: {n_tiles} tiles")
    for key, n in report["counts"].items():
        pct = 100 * n / n_tiles if n_tiles else 0
        bar_len = 30
        filled = int(bar_len * n / n_tiles) if n_tiles else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"  {key:<20} [{bar}] {n:>5}/{n_tiles} ({pct:5.1f}%)")

    for label, done, expected in [
        ("boundaries_*.gpkg", report["boundaries"], report["boundaries_expected"]),
        ("aqueduct_*.toml", report["tomls"], report["tomls_expected"]),
    ]:
        pct = 100 * done / expected if expected else 0
        bar_len = 30
        filled = int(bar_len * done / expected) if expected else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        print(f"  {label:<20} [{bar}] {done:>6}/{expected} ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(Path(__file__).resolve().parent / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg, help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                         help="re-check and reprint every SECONDS until interrupted (Ctrl+C), "
                              "instead of a single one-shot report")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    tile_gdf = load_tile_grid(cfg["tile_grid"]["path"])
    tile_ids = sorted(tile_gdf["tile_id"].astype(int).tolist())
    model_outputs = Path(cfg["simulation"]["model_outputs"])

    bc_cfg = cfg["boundary_conditions"]
    waterlevel_names = merged_slr_scenarios(bc_cfg, cfg["adaptation"])
    n_scenarios_per_tile = len(bc_cfg["return_periods"]) * len(waterlevel_names)

    if args.watch:
        try:
            while True:
                print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
                report = _count_progress(model_outputs, tile_ids, n_scenarios_per_tile)
                _print_report(report)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        report = _count_progress(model_outputs, tile_ids, n_scenarios_per_tile)
        _print_report(report)


if __name__ == "__main__":
    main()
