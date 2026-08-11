"""Report simulation progress by counting waterdepth_*.tif files per tile,
broken down by wave (hop_distance) - no Snakemake invocation needed, safe to
run anytime (including while an HPC run is in progress) since it only
reads, never writes.

A tile's own results/ dir gets exactly one waterdepth_{rp}_{slr}.tif per
scenario regardless of HOW that scenario finished (real solve, confidently-
zero skip, or OOM/nodata placeholder - see aqueduct_runner.
tile_output_complete's own docstring), so counting these files is a
complete, coarse-grained progress signal.

Broken down per wave (not just a single flat total) because waves run
strictly sequentially - a wave with 0% done really does mean "hasn't
started yet, waiting on the previous wave's SLURM dependency barrier", not
"stuck".

Usage:
    python snakemake_workflow/check_simulation_progress.py [--config path/to/config.yml] [--watch SECONDS]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from aqueduct_runner import oom_marker_path  # noqa: E402
from config_utils import load_config, merged_slr_scenarios  # noqa: E402
from tiles import load_tile_grid  # noqa: E402


def _count_progress(model_outputs: Path, tile_ids_by_wave: dict[int, list[int]], n_scenarios_per_tile: int) -> dict:
    oom_dir = model_outputs / "oom_tiles"
    per_wave = {}
    for wave, tile_ids in sorted(tile_ids_by_wave.items()):
        n_files_done = 0
        n_tiles_complete = 0
        n_oom = 0
        for tid in tile_ids:
            results_dir = model_outputs / str(tid) / "results"
            n = sum(1 for _ in results_dir.glob("waterdepth_*.tif")) if results_dir.is_dir() else 0
            n_files_done += n
            if n >= n_scenarios_per_tile:
                n_tiles_complete += 1
            if oom_marker_path(oom_dir, str(tid)).exists():
                n_oom += 1
        per_wave[wave] = {
            "n_tiles": len(tile_ids),
            "n_tiles_complete": n_tiles_complete,
            "n_files_done": n_files_done,
            "n_files_expected": len(tile_ids) * n_scenarios_per_tile,
            "n_oom": n_oom,
        }
    return per_wave


def _bar(done: int, expected: int, bar_len: int = 30) -> str:
    filled = int(bar_len * done / expected) if expected else 0
    return "#" * filled + "-" * (bar_len - filled)


def _print_report(per_wave: dict) -> None:
    for wave, s in per_wave.items():
        pct = 100 * s["n_files_done"] / s["n_files_expected"] if s["n_files_expected"] else 0
        oom_note = f", {s['n_oom']} OOM" if s["n_oom"] else ""
        print(
            f"  wave {wave:<3} [{_bar(s['n_files_done'], s['n_files_expected'])}] "
            f"{s['n_files_done']:>6}/{s['n_files_expected']:<6} ({pct:5.1f}%)  "
            f"tiles complete: {s['n_tiles_complete']:>4}/{s['n_tiles']:<4}{oom_note}"
        )

    total_tiles = sum(s["n_tiles"] for s in per_wave.values())
    total_tiles_complete = sum(s["n_tiles_complete"] for s in per_wave.values())
    total_files_done = sum(s["n_files_done"] for s in per_wave.values())
    total_files_expected = sum(s["n_files_expected"] for s in per_wave.values())
    total_oom = sum(s["n_oom"] for s in per_wave.values())
    pct = 100 * total_files_done / total_files_expected if total_files_expected else 0
    oom_note = f", {total_oom} OOM" if total_oom else ""
    print(
        f"\n  {'TOTAL':<8} [{_bar(total_files_done, total_files_expected)}] "
        f"{total_files_done:>6}/{total_files_expected:<6} ({pct:5.1f}%)  "
        f"tiles complete: {total_tiles_complete:>4}/{total_tiles:<4}{oom_note}"
    )


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
    tile_ids_by_wave: dict[int, list[int]] = {}
    for tid, hop in zip(tile_gdf["tile_id"].astype(int), tile_gdf["hop_distance"].astype(int)):
        tile_ids_by_wave.setdefault(int(hop), []).append(int(tid))
    model_outputs = Path(cfg["simulation"]["model_outputs"])

    bc_cfg = cfg["boundary_conditions"]
    waterlevel_names = merged_slr_scenarios(bc_cfg, cfg["adaptation"])
    n_scenarios_per_tile = len(bc_cfg["return_periods"]) * len(waterlevel_names)

    if args.watch:
        try:
            while True:
                print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
                per_wave = _count_progress(model_outputs, tile_ids_by_wave, n_scenarios_per_tile)
                _print_report(per_wave)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        per_wave = _count_progress(model_outputs, tile_ids_by_wave, n_scenarios_per_tile)
        _print_report(per_wave)


if __name__ == "__main__":
    main()
