"""Report postprocessing + exposure-analysis progress - no Snakemake
invocation needed, safe to run anytime (including while an HPC run is in
progress) since it only reads, never writes.

Covers the full submit_postprocess_and_exposure.sh chain (see
generate_hpc_postprocess_job.py):
  postprocessing phase 1  merge_chunk
  postprocessing phase 2  prepare_exposure_grid_chunk
  postprocessing phase 3  compute_flood_fraction_chunk
  exposure analysis       pass 1 (shares) -> reduce_shares -> pass 2 (EAI) -> reduce_write

Each phase is reported separately (not just one flat total) since the whole
chain is phase-gated: a phase sitting at 0% really does mean "hasn't started
yet, waiting on the previous phase's SLURM dependency barrier", not "stuck".

Uses one directory glob per phase, not one stat() call per expected file -
postprocessing's chunk outputs all live in a handful of FLAT directories
(one file per chunk/scenario, not one subdirectory per chunk the way
simulation's per-tile results/ dirs work), so a single glob() count is cheap
regardless of total file count (tens of thousands at real scale).

Usage:
    python snakemake_workflow/check_postprocess_progress.py [--config path/to/config.yml] [--watch SECONDS]
"""

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config_utils import load_config, merged_slr_scenarios  # noqa: E402
from tiles import load_tile_grid  # noqa: E402


def _build_chunk_grid(tile_gdf: gpd.GeoDataFrame, chunk_size_deg: float) -> list[str]:
    """Mirrors the root Snakefile's own _build_chunk_grid (also duplicated
    in scripts/generate_hpc_postprocess_job.py) - returns just the
    chunk_id list, all this script needs.
    """
    minx, miny, maxx, maxy = tile_gdf.total_bounds
    sz = chunk_size_deg
    xs = np.arange(np.floor(minx / sz) * sz, np.ceil(maxx / sz) * sz, sz)
    ys = np.arange(np.floor(miny / sz) * sz, np.ceil(maxy / sz) * sz, sz)
    chunk_ids = []
    for x in xs:
        for y in ys:
            cell = shapely_box(x, y, x + sz, y + sz)
            if not tile_gdf.geometry.intersects(cell).any():
                continue
            xi, yi = int(round(x)), int(round(y))
            lat = f"N{yi:02d}" if yi >= 0 else f"S{-yi:02d}"
            lon = f"E{xi:03d}" if xi >= 0 else f"W{-xi:03d}"
            chunk_ids.append(f"{lat}{lon}")
    return chunk_ids


def _bar(done: int, expected: int, bar_len: int = 30) -> str:
    filled = int(bar_len * done / expected) if expected else 0
    filled = min(filled, bar_len)
    return "#" * filled + "-" * (bar_len - filled)


def _print_line(label: str, done: int, expected: int) -> None:
    pct = 100 * done / expected if expected else 0
    print(f"  {label:<32} [{_bar(done, expected)}] {done:>6}/{expected:<6} ({pct:5.1f}%)")


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
    chunk_ids = _build_chunk_grid(tile_gdf, cfg["postprocessing"]["chunk_size_deg"])
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
        print(f"  {'reduce_shares':<32} [{_bar(1 if shares_done else 0, 1)}] "
              f"{'done' if shares_done else 'not yet':>6}/{'done':<6}")
        _print_line("pass 2   EAI", n_pass2, n_exposure_batches)
        print(f"  {'reduce_write (output CSVs)':<32} {n_csvs} CSV file(s) written"
              + (f" to {exposure_out_dir}" if n_csvs else ""))

    if args.watch:
        try:
            while True:
                print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
                _report()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        _report()


if __name__ == "__main__":
    main()
