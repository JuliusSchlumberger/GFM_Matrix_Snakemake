"""Report sfincs_calibration progress by counting which per-tile output
files already exist on disk - no script invocation needed, safe to run
anytime (including while an HPC run is in progress) since it only reads,
never writes.

Unlike the repo-root check_preprocess_progress.py/check_simulation_progress.py/
check_postprocess_progress.py (which track the main production pipeline
against the FULL tile grid), this tracks one calibration study's own tile
list (`{base_dir_name}/tile_ids.txt`) against its own directory layout
(`{base_dir_name}/{tile_id}/{inputs,sfincs_model,outputs}/`) - see
run_one_tile.sh for the real per-tile execution order this mirrors:

    inputs copied-in (dem/mask/friction/boundaries/tile_geometry/model_bbox)
    -> SFINCS build (elevation_combined.tif, manning_n.tif,
       matched_boundary_points.gpkg, sfincs.inp)
    -> SFINCS run (sfincs_map.nc)
    -> SFINCS postprocess (hmax.tif, flood_extent.tif)
    -> bathtub (bathtub_waterdepth_RP100_SLR_0.tif)
    -> eikonal, one point per (friction_scale_factor, max_outer_iterations)
       pair - the "roughness/friction sweep" - each its own tagged file
       (eikonal_on_subgrid_waterdepth_RP100_SLR_0{_fsf<v>}{_outer<o>}.tif,
       see run_eikonal_on_sfincs_subgrid.py's own _fsf_tag/_outer_tag)
    -> postprocess_tile_summary.py (summary_bathtub.json, summary_sfincs.json,
       summary_eikonal.json)

Eikonal/sweep completion is read from the tagged .tif filenames directly,
NOT from summary_eikonal.json's own eikonal_* fields: postprocess_tile_
summary.py's MODEL_WATERDEPTH_FILENAME only ever looks for the bare,
untagged eikonal_on_subgrid_waterdepth_RP100_SLR_0.tif, so for any study
run with a non-default --max-outer-iterations (sfincs_calibration used 5,
production's own default is 4), that file never exists and
summary_eikonal.json's eikonal fields are silently null regardless of how
much of the sweep is actually done. summary_*.json presence is still
reported, as a "was postprocess re-triggered at all" signal, not an
eikonal-progress signal.

Usage:
    python check_sfincs_calibration_progress.py
    python check_sfincs_calibration_progress.py --base-dir-name sfincs_calibration --watch 60
    python check_sfincs_calibration_progress.py --friction-scale-factors 3 6 9 12 15 18 21 24 27 30 --max-outer-iterations 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402

RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"

# Matches run_eikonal_on_sfincs_subgrid.py's own FRICTION_SCALE_FACTOR_DEFAULT/
# MAX_OUTER_ITERATIONS_DEFAULT - a point at these exact values gets NO tag at
# all (bare filename), any other value gets its own _fsf<v>/_outer<o> tag.
FRICTION_SCALE_FACTOR_DEFAULT = 30.0
MAX_OUTER_ITERATIONS_DEFAULT = 4

# The actual sweep this study was dispatched with (generate_friction_sweep_jobs.py's
# own FRICTION_SCALE_FACTORS_DEFAULT list, run via generate_validation_batch_jobs.py
# --defer-eikonal --defer-eikonal-max-outer-iterations 5 - confirmed live in
# hpc_jobs/*.sbatch). Override via --friction-scale-factors/--max-outer-iterations
# if a different sweep is ever run.
FRICTION_SCALE_FACTORS_DEFAULT = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
MAX_OUTER_ITERATIONS_SWEEP_DEFAULT = 5

INPUT_FILES = ("dem.tif", "mask.tif", "friction.tif", f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg",
               "tile_geometry.gpkg", "model_bbox.json")
BUILD_FILES = ("elevation_combined.tif", "manning_n.tif", "matched_boundary_points.gpkg", "sfincs.inp")


def _eikonal_filename(fsf: float, max_outer_iterations: int) -> str:
    fsf_tag = "" if fsf == FRICTION_SCALE_FACTOR_DEFAULT else f"_fsf{fsf:g}"
    outer_tag = "" if max_outer_iterations == MAX_OUTER_ITERATIONS_DEFAULT else f"_outer{max_outer_iterations}"
    return f"eikonal_on_subgrid_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}{fsf_tag}{outer_tag}.tif"


def _count_progress(base_dir: Path, tile_ids: list[str], friction_scale_factors: list[float],
                     max_outer_iterations: int) -> dict:
    sweep_filenames = {fsf: _eikonal_filename(fsf, max_outer_iterations) for fsf in friction_scale_factors}

    counts = {k: 0 for k in (
        "inputs copied-in", "SFINCS build", "SFINCS run (sfincs_map.nc)",
        "SFINCS postprocess (hmax/flood_extent)", "bathtub",
        "summary_bathtub.json", "summary_sfincs.json", "summary_eikonal.json",
    )}
    sweep_counts = {fsf: 0 for fsf in friction_scale_factors}
    n_tiles_full_sweep = 0

    for tid in tile_ids:
        tile_dir = base_dir / tid
        inputs_entries = _entries(tile_dir / "inputs")
        sfincs_entries = _entries(tile_dir / "sfincs_model")
        output_entries = _entries(tile_dir / "outputs")

        if all(f in inputs_entries for f in INPUT_FILES):
            counts["inputs copied-in"] += 1
        if all(f in sfincs_entries for f in BUILD_FILES):
            counts["SFINCS build"] += 1
        if "sfincs_map.nc" in sfincs_entries:
            counts["SFINCS run (sfincs_map.nc)"] += 1
        if "hmax.tif" in output_entries and "flood_extent.tif" in output_entries:
            counts["SFINCS postprocess (hmax/flood_extent)"] += 1
        if f"bathtub_waterdepth_{RETURN_PERIOD}_{WATERLEVEL_NAME}.tif" in output_entries:
            counts["bathtub"] += 1
        for key in ("summary_bathtub.json", "summary_sfincs.json", "summary_eikonal.json"):
            if key in output_entries:
                counts[key] += 1

        n_present = 0
        for fsf, fname in sweep_filenames.items():
            if fname in output_entries:
                sweep_counts[fsf] += 1
                n_present += 1
        if n_present == len(friction_scale_factors):
            n_tiles_full_sweep += 1

    n_tiles = len(tile_ids)
    return {
        "n_tiles": n_tiles,
        "counts": counts,
        "sweep_counts": sweep_counts,
        "n_tiles_full_sweep": n_tiles_full_sweep,
        "max_outer_iterations": max_outer_iterations,
    }


def _entries(d: Path) -> set[str]:
    if not d.is_dir():
        return set()
    return set(p.name for p in d.iterdir())


def _bar(done: int, expected: int, bar_len: int = 30) -> str:
    filled = int(bar_len * done / expected) if expected else 0
    return "#" * filled + "-" * (bar_len - filled)


def _print_line(label: str, done: int, expected: int, label_width: int = 36) -> None:
    pct = 100 * done / expected if expected else 0
    print(f"  {label:<{label_width}} [{_bar(done, expected)}] {done:>5}/{expected} ({pct:5.1f}%)")


def _print_report(report: dict) -> None:
    n_tiles = report["n_tiles"]
    print(f"sfincs_calibration: {n_tiles} target tile(s) (tile_ids.txt)")
    for key, n in report["counts"].items():
        _print_line(key, n, n_tiles)

    print(f"\n  roughness/friction sweep (max_outer_iterations={report['max_outer_iterations']}):")
    for fsf, n in report["sweep_counts"].items():
        tag = "default" if fsf == FRICTION_SCALE_FACTOR_DEFAULT else f"fsf={fsf:g}"
        _print_line(f"    {tag}", n, n_tiles, label_width=34)
    _print_line("    ALL sweep points present", report["n_tiles_full_sweep"], n_tiles, label_width=34)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo_root = Path(__file__).resolve().parent.parent
    _default_cfg = str(_repo_root / "snakemake_workflow" / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg, help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--base-dir-name", default="sfincs_calibration")
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", default=FRICTION_SCALE_FACTORS_DEFAULT,
                         help=f"default: {FRICTION_SCALE_FACTORS_DEFAULT} (the sweep this study was actually "
                              f"dispatched with)")
    parser.add_argument("--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS_SWEEP_DEFAULT,
                         help=f"default: {MAX_OUTER_ITERATIONS_SWEEP_DEFAULT} (what this study was actually "
                              f"dispatched with; production's own default is {MAX_OUTER_ITERATIONS_DEFAULT})")
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS",
                         help="re-check and reprint every SECONDS until interrupted (Ctrl+C), "
                              "instead of a single one-shot report")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    tile_ids_path = base_dir / "tile_ids.txt"
    tile_ids = [line.strip() for line in tile_ids_path.read_text().splitlines() if line.strip()]

    if args.watch:
        try:
            while True:
                print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
                report = _count_progress(base_dir, tile_ids, args.friction_scale_factors, args.max_outer_iterations)
                _print_report(report)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        report = _count_progress(base_dir, tile_ids, args.friction_scale_factors, args.max_outer_iterations)
        _print_report(report)


if __name__ == "__main__":
    main()
