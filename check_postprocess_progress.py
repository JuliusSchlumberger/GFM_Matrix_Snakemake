"""Report postprocessing + exposure-analysis progress per phase (read-only).

Phases are SLURM-dependency gated: 0% means "waiting", not "stuck".

Usage:
    python check_postprocess_progress.py [--config path/to/config.yml] [--watch SECONDS]
"""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config_utils import load_config, merged_slr_scenarios  # noqa: E402
from progress_report import progress_bar, run_report  # noqa: E402
from tiles import build_chunk_grid, load_tile_grid  # noqa: E402



def _print_line(label: str, done: int, expected: int) -> None:
    pct = 100 * done / expected if expected else 0
    print(f"  {label:<32} [{progress_bar(done, expected)}] {done:>6}/{expected:<6} ({pct:5.1f}%)")


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
    chunk_ids = build_chunk_grid(tile_gdf, cfg["postprocessing"]["chunk_size_deg"])["chunk_id"].tolist()
    n_chunks = len(chunk_ids)

    bc_cfg = cfg["boundary_conditions"]
    waterlevel_names = merged_slr_scenarios(bc_cfg, cfg["adaptation"])
    n_scenarios_per_chunk = len(bc_cfg["return_periods"]) * len(waterlevel_names)

    merged_dir = Path(cfg["postprocessing"]["merged_outputs"])
    chunks_dir = merged_dir / "chunks"
    flood_frac_dir = chunks_dir / "flood_fraction"
    exposure_out_dir = merged_dir / "exposure"
    exposure_jobs_dir = Path(cfg["hpc"]["jobs_dir"]) / "exposure"

    n_nodes = cfg["hpc"]["n_nodes"]
    n_exposure_batches = min(n_nodes, n_chunks) if n_chunks else 0

    def _report() -> None:
        n_merge = sum(1 for _ in chunks_dir.glob("waterdepth_*.tif")) if chunks_dir.is_dir() else 0
        n_grid = sum(1 for _ in chunks_dir.glob("exposure_population_grid_*.tif")) if chunks_dir.is_dir() else 0
        n_ff = sum(1 for _ in flood_frac_dir.glob("flood_fraction_*.tif")) if flood_frac_dir.is_dir() else 0

        print("Postprocessing:")
        _print_line("phase 1  merge_chunk", n_merge, n_chunks * n_scenarios_per_chunk)
        _print_line("phase 2  prepare_exposure_grid_chunk", n_grid, n_chunks)
        _print_line("phase 3  compute_flood_fraction_chunk", n_ff, n_chunks * n_scenarios_per_chunk)

        n_pass1 = sum(1 for _ in exposure_jobs_dir.glob("pass1_batch_*.json")) if exposure_jobs_dir.is_dir() else 0
        shares_done = (exposure_jobs_dir / "shares_by_intensity.json").exists()
        n_pass2 = sum(1 for _ in exposure_jobs_dir.glob("pass2_batch_*.pkl")) if exposure_jobs_dir.is_dir() else 0
        n_csvs = sum(1 for _ in exposure_out_dir.glob("*.csv")) if exposure_out_dir.is_dir() else 0

        print("\nExposure analysis:")
        _print_line("pass 1   shares", n_pass1, n_exposure_batches)
        print(f"  {'reduce_shares':<32} [{progress_bar(1 if shares_done else 0, 1)}] "
              f"{'done' if shares_done else 'not yet':>6}/{'done':<6}")
        _print_line("pass 2   EAI", n_pass2, n_exposure_batches)
        print(f"  {'reduce_write (output CSVs)':<32} {n_csvs} CSV file(s) written"
              + (f" to {exposure_out_dir}" if n_csvs else ""))

    run_report(_report, args.watch)


if __name__ == "__main__":
    main()
