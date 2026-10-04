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

Also breaks down each wave by size class (small/large, same
hpc.large_tile_pixel_threshold + area_deg2*3600**2 pixel-count proxy
hpc_dispatch.smk's own HPC batching uses - not re-reading the DEM, since
this must stay cheap enough to run anytime) and by per-tile status (none
done / partial - at least one scenario but not all / complete) - the large
class is the one usually worth watching since it's the one that regularly
blows past its own sbatch_large time limit (see snakemake_workflow/hpc.md's
"Tile-size routing" section).

Usage:
    python check_simulation_progress.py [--config path/to/config.yml] [--watch SECONDS]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from aqueduct_runner import oom_marker_path  # noqa: E402
from config_utils import load_config, merged_slr_scenarios  # noqa: E402
from tiles import load_tile_grid  # noqa: E402


def _empty_class_bucket() -> dict:
    return {"n_tiles": 0, "n_none": 0, "n_partial": 0, "n_complete": 0, "n_files_done": 0, "n_files_expected": 0}


def _count_progress(
    model_outputs: Path, tile_ids_by_wave: dict[int, list[int]], n_scenarios_per_tile: int,
    size_class_by_tile: dict[str, str] | None = None,
) -> dict:
    oom_dir = model_outputs / "oom_tiles"
    per_wave = {}
    for wave, tile_ids in sorted(tile_ids_by_wave.items()):
        n_files_done = 0
        n_tiles_complete = 0
        n_oom = 0
        by_size = {"small": _empty_class_bucket(), "large": _empty_class_bucket()}
        for tid in tile_ids:
            results_dir = model_outputs / str(tid) / "results"
            n = sum(1 for _ in results_dir.glob("waterdepth_*.tif")) if results_dir.is_dir() else 0
            n_files_done += n
            if n >= n_scenarios_per_tile:
                n_tiles_complete += 1
            if oom_marker_path(oom_dir, str(tid)).exists():
                n_oom += 1

            size_class = (size_class_by_tile or {}).get(str(tid), "small")
            bucket = by_size[size_class]
            bucket["n_tiles"] += 1
            bucket["n_files_done"] += n
            bucket["n_files_expected"] += n_scenarios_per_tile
            if n == 0:
                bucket["n_none"] += 1
            elif n >= n_scenarios_per_tile:
                bucket["n_complete"] += 1
            else:
                bucket["n_partial"] += 1

        per_wave[wave] = {
            "n_tiles": len(tile_ids),
            "n_tiles_complete": n_tiles_complete,
            "n_files_done": n_files_done,
            "n_files_expected": len(tile_ids) * n_scenarios_per_tile,
            "n_oom": n_oom,
            "by_size": by_size,
        }
    return per_wave


def _bar(done: int, expected: int, bar_len: int = 30) -> str:
    filled = int(bar_len * done / expected) if expected else 0
    return "#" * filled + "-" * (bar_len - filled)


def _print_size_class_line(label: str, b: dict) -> None:
    if b["n_tiles"] == 0:
        return
    pct = 100 * b["n_files_done"] / b["n_files_expected"] if b["n_files_expected"] else 0
    print(
        f"           {label:<6} [{_bar(b['n_files_done'], b['n_files_expected'])}] "
        f"{b['n_files_done']:>6}/{b['n_files_expected']:<6} ({pct:5.1f}%)  "
        f"complete: {b['n_complete']:>4}  partial (>=1 scenario): {b['n_partial']:>4}  "
        f"none started: {b['n_none']:>4}  (of {b['n_tiles']} {label} tile(s))"
    )


def _print_report(per_wave: dict, by_size: bool = False) -> None:
    for wave, s in per_wave.items():
        pct = 100 * s["n_files_done"] / s["n_files_expected"] if s["n_files_expected"] else 0
        oom_note = f", {s['n_oom']} OOM" if s["n_oom"] else ""
        print(
            f"  wave {wave:<3} [{_bar(s['n_files_done'], s['n_files_expected'])}] "
            f"{s['n_files_done']:>6}/{s['n_files_expected']:<6} ({pct:5.1f}%)  "
            f"tiles complete: {s['n_tiles_complete']:>4}/{s['n_tiles']:<4}{oom_note}"
        )
        if by_size:
            _print_size_class_line("large", s["by_size"]["large"])
            _print_size_class_line("small", s["by_size"]["small"])

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
    if by_size:
        total_large = _empty_class_bucket()
        total_small = _empty_class_bucket()
        for s in per_wave.values():
            for key in total_large:
                total_large[key] += s["by_size"]["large"][key]
                total_small[key] += s["by_size"]["small"][key]
        _print_size_class_line("large", total_large)
        _print_size_class_line("small", total_small)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(Path(__file__).resolve().parent / "snakemake_workflow" / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg, help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                         help="re-check and reprint every SECONDS until interrupted (Ctrl+C), "
                              "instead of a single one-shot report")
    parser.add_argument("--by-size", action="store_true",
                         help="also break each wave down by small/large tile-size class "
                              "(hpc.large_tile_pixel_threshold) and by per-tile status "
                              "(none/partial/complete) - the large class is the one that "
                              "regularly blows past its own sbatch_large time limit")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    tile_gdf = load_tile_grid(cfg["tile_grid"]["path"])
    tile_ids_by_wave: dict[int, list[int]] = {}
    for tid, hop in zip(tile_gdf["tile_id"].astype(int), tile_gdf["hop_distance"].astype(int)):
        tile_ids_by_wave.setdefault(int(hop), []).append(int(tid))
    model_outputs = Path(cfg["simulation"]["model_outputs"])

    size_class_by_tile = None
    if args.by_size:
        threshold = cfg["hpc"]["large_tile_pixel_threshold"]
        bounds = tile_gdf.geometry.bounds
        size_class_by_tile = {}
        for tid, minx, miny, maxx, maxy in zip(
            tile_gdf["tile_id"].astype(int), bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"],
        ):
            # Same proxy hpc_dispatch.smk's own HPC batching uses: area_deg2*3600**2
            # (DeltaDTM's ~1 arcsec native resolution), not a real DEM read.
            approx_pixels = float(maxx - minx) * float(maxy - miny) * 3600.0 * 3600.0
            size_class_by_tile[str(tid)] = "large" if approx_pixels >= threshold else "small"

    bc_cfg = cfg["boundary_conditions"]
    waterlevel_names = merged_slr_scenarios(bc_cfg, cfg["adaptation"])
    n_scenarios_per_tile = len(bc_cfg["return_periods"]) * len(waterlevel_names)

    if args.watch:
        try:
            while True:
                print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
                per_wave = _count_progress(model_outputs, tile_ids_by_wave, n_scenarios_per_tile, size_class_by_tile)
                _print_report(per_wave, by_size=args.by_size)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        per_wave = _count_progress(model_outputs, tile_ids_by_wave, n_scenarios_per_tile, size_class_by_tile)
        _print_report(per_wave, by_size=args.by_size)


if __name__ == "__main__":
    main()
