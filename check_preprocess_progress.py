"""Report preprocessing progress by counting per-tile input files (read-only).

Usage:
    python check_preprocess_progress.py [--config path/to/config.yml] [--watch SECONDS]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config_utils import load_config, merged_slr_scenarios  # noqa: E402
from progress_report import progress_bar, run_report  # noqa: E402
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
    for tid in tile_ids:
        inputs_dir = model_outputs / str(tid) / "inputs"
        if not inputs_dir.is_dir():
            continue
        entries = set(p.name for p in inputs_dir.iterdir())
        for key in counts:
            if key in entries:
                counts[key] += 1
        total_boundaries += sum(1 for n in entries if n.startswith("boundaries_") and n.endswith(".gpkg"))

    n_tiles = len(tile_ids)
    expected_scenario_files = n_tiles * n_scenarios_per_tile
    return {
        "n_tiles": n_tiles,
        "counts": counts,
        "boundaries": total_boundaries,
        "boundaries_expected": expected_scenario_files,
    }


def _print_report(report: dict) -> None:
    n_tiles = report["n_tiles"]
    print(f"Tile grid: {n_tiles} tiles")
    for key, n in report["counts"].items():
        pct = 100 * n / n_tiles if n_tiles else 0
        print(f"  {key:<20} [{progress_bar(n, n_tiles)}] {n:>5}/{n_tiles} ({pct:5.1f}%)")

    for label, done, expected in [
        ("boundaries_*.gpkg", report["boundaries"], report["boundaries_expected"]),
    ]:
        pct = 100 * done / expected if expected else 0
        print(f"  {label:<20} [{progress_bar(done, expected)}] {done:>6}/{expected} ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(Path(__file__).resolve().parent / "snakemake_workflow" / "config" / "config.yml")
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

    run_report(lambda: _print_report(_count_progress(model_outputs, tile_ids, n_scenarios_per_tile)), args.watch)


if __name__ == "__main__":
    main()
