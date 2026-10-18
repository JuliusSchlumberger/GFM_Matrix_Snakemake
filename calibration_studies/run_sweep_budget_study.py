"""Trace the non-coupling eikonal solve sweep by sweep on calibration tiles.

Tiles come from select_study_candidate_tiles.py's candidate_tiles.txt. Seeds and
friction (decode -> scale -> floor) match production; sweeps follow
_ORTHANT_ORDER[i % 4] as solve_eikonal_dense's sweep_budget (one continuous pass,
avoiding O(N^2) re-solves). A round is 4 sweeps; the tile stops once a complete
round's max raw change <= --epsilon (0.03 m) or after MAX_ROUNDS_CEILING rounds.
A tile dry at sweep 1 is logged DRY and stopped (the candidate pool is
over-provisioned). Writes one CSV per tile with per-sweep depth-change metrics.

Usage:
    python run_sweep_budget_study.py --out-dir <output_dir> [--config <config.yml>] [--tile-ids id [id ...]]

A positional <output_dir> is still accepted in place of --out-dir.
"""
import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cli_args import add_option_with_positional, require  # noqa: E402
from config_utils import load_config  # noqa: E402
from eikonal import _ORTHANT_ORDER, _dense_sweep  # noqa: E402
from flood_model import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask, prune_to_coast_connected  # noqa: E402
from rasters import decode_dem_cm, decode_friction_int16, decode_waterlevel_cm  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"

# tile_generation.ocean_code / river_code in config.yml.
OCEAN_CODE = 1
RIVER_CODE = 3

# Smoke-test fallback when no --tile-ids are given.
TILES = [2083, 2514, 1421, 2569, 2858, 2314, 753]

SWEEPS_PER_ROUND = 4  # one full cycle of _ORTHANT_ORDER
MAX_ROUNDS_CEILING = 100
MAX_SWEEPS = MAX_ROUNDS_CEILING * SWEEPS_PER_ROUND
N_TILES_WANTED_DEFAULT = 500  # only reachable in a single sequential run, not per HPC node
DEFAULT_EPSILON_M = 0.03  # = simulation.flooding.waterlevel_epsilon_m


def load_tile(tile_id: int, model_outputs: Path, knn: int, friction_scale_factor: float, on_no_stations: str = "dry"):
    """Load a tile's decoded solver inputs and IDW-seeded coastline values.

    Shared by the calibration studies (run_obstacle_coupling_study.py,
    diagnose_large_residual.py), so all of them seed exactly as production does.

    Parameters
    ----------
    on_no_stations : {"dry", "raise"}
        A tile with zero boundary stations: "dry" seeds nothing (sweep 1 then
        reports DRY; production's NO_STATIONS_REASON), "raise" raises ValueError
        (studies whose tile list should already exclude such tiles; BallTree would
        crash on them).

    Returns
    -------
    tuple
        (dem, mask, friction, coastline, coastline_rows, coastline_cols, initial,
        max_waterlevel); max_waterlevel is the highest station level in m, None
        for a "dry" tile without stations.
    """
    if on_no_stations not in ("dry", "raise"):
        raise ValueError(f"on_no_stations must be 'dry' or 'raise', not {on_no_stations!r}")
    tile_dir = model_outputs / str(tile_id)
    scenario = f"{RETURN_PERIOD}_{WATERLEVEL_NAME}"
    inputs = tile_dir / "inputs"

    with rasterio.open(inputs / "dem.tif") as src:
        dem = decode_dem_cm(src.read(1))
        transform = src.transform
    with rasterio.open(inputs / "mask.tif") as src:
        # int8: int64 would OOM on the largest tiles.
        mask = src.read(1).astype(np.int8)
    with rasterio.open(inputs / "friction.tif") as src:
        friction = decode_friction_int16(src.read(1))
    boundaries = gpd.read_file(inputs / f"boundaries_{scenario}.gpkg")

    dem = effective_dem(dem, mask)
    # decode -> scale -> floor, as in eikonal_runner: a zero-friction cell floors
    # to 0.001 regardless of friction_scale_factor.
    friction = friction * friction_scale_factor
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))
    coastline = coastline_mask(mask, ocean_code=OCEAN_CODE, river_code=RIVER_CODE)
    coastline_rows, coastline_cols = np.nonzero(coastline)

    # Int16 cm-encoded levels. Zero stations are expected for small/isolated tiles.
    station_values = decode_waterlevel_cm(boundaries[WATERLEVEL_NAME].to_numpy())
    if len(station_values) == 0:
        if on_no_stations == "raise":
            raise ValueError(
                f"tile {tile_id} has zero boundary stations - should have been excluded "
                "from wet_tiles_selected.txt by run_sweep_budget_study.py"
            )
        coastline_rows = np.array([], dtype=coastline_rows.dtype)
        coastline_cols = np.array([], dtype=coastline_cols.dtype)
        initial = np.array([], dtype=np.float64)
        return dem, mask, friction, coastline, coastline_rows, coastline_cols, initial, None

    stations_lonlat = np.column_stack(
        [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
    )
    initial = _idw_seed_values(
        coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
        min(knn, len(station_values)), mask, OCEAN_CODE,
    )
    max_waterlevel = float(station_values.max())
    return dem, mask, friction, coastline, coastline_rows, coastline_cols, initial, max_waterlevel


def _full_depth_array(t, dem, mask, coastline) -> np.ndarray:
    """Return tile-shaped depth (0 where dry), same math as flood_depth_dense without coupling."""
    waterlevel = -t[1:, 1:]
    flood = (waterlevel > dem) & (mask != OCEAN_CODE)
    flood = prune_to_coast_connected(flood, coastline)
    depth = np.zeros_like(dem)
    depth[flood] = waterlevel[flood] - dem[flood]
    return depth


def _depth_change_metrics(depth_prev: np.ndarray, depth_curr: np.ndarray, n_cells: int) -> dict:
    """Median/mean use changed cells only (flooding is a small minority of a tile).

    No un-flooded count: without blocking, sweeps only lower t, so flooding only grows.
    """
    abs_diff = np.abs(depth_curr - depth_prev)
    changed = abs_diff > 0
    n_changed = int(changed.sum())
    changed_vals = abs_diff[changed]
    n_newly_flooded = int(((depth_prev == 0) & (depth_curr > 0)).sum())
    return {
        "max_depth_change_abs": round(float(abs_diff.max()), 4) if abs_diff.size else 0.0,
        "median_depth_change_abs": round(float(np.median(changed_vals)), 4) if changed_vals.size else 0.0,
        "mean_depth_change_abs": round(float(changed_vals.mean()), 4) if changed_vals.size else 0.0,
        "n_newly_flooded": n_newly_flooded,
        "n_cells_changed": n_changed,
        "pct_cells_changed": round(100.0 * n_changed / n_cells, 6),
    }


def run_tile_sweep_trace(
    tile_id: int, max_sweeps: int, model_outputs: Path, knn: int, friction_scale_factor: float,
    epsilon: float,
) -> tuple[list[dict], bool, bool]:
    """Trace one tile sweep by sweep; return (rows, is_wet, round_converged).

    is_wet is False iff sweep 1 has zero flooding. round_converged is True iff a
    complete round's max raw change reached <= epsilon before max_sweeps.
    """
    dem, mask, friction, coastline, seed_rows, seed_cols, initial, _ = load_tile(
        tile_id, model_outputs, knn, friction_scale_factor,
    )
    seed_values = -initial
    dtype = friction.dtype
    n_cells = dem.size

    m, n = friction.shape
    # Unseeded t=+99 (waterlevel -99 m) = never flooded, as in solve_eikonal_dense.
    t = np.full((m + 1, n + 1), 99.0, dtype=dtype)
    t[seed_rows, seed_cols] = seed_values
    neg_two = dtype.type(-2.0)
    eight = dtype.type(8.0)
    four = dtype.type(4.0)

    rows = []
    prev_depth = np.zeros_like(dem)  # implicit all-dry "sweep 0"
    t0 = time.perf_counter()

    for sweep in range(1, max_sweeps + 1):
        # Raw whole-array t update (not depth-pruned): what solve_eikonal_dense maxes
        # over 4 sweeps and compares to epsilon, so round convergence can be rebuilt.
        sweep_max_change_raw = _dense_sweep(t, friction, _ORTHANT_ORDER[(sweep - 1) % 4], neg_two, eight, four)
        depth = _full_depth_array(t, dem, mask, coastline)
        n_inundated = int((depth > 0).sum())
        change = _depth_change_metrics(prev_depth, depth, n_cells)

        row = {
            "tile": tile_id,
            "n_cells": n_cells,
            "sweep_count": sweep,
            "cum_time_s": round(time.perf_counter() - t0, 3),
            "sweep_max_change_raw": round(float(sweep_max_change_raw), 8),
            "n_inundated": n_inundated,
            "depth_mean": round(float(depth[depth > 0].mean()), 4) if n_inundated else 0.0,
            "depth_sum": round(float(depth.sum()), 2),
            "depth_max": round(float(depth.max()), 4),
            "status": "ok",
        }
        row.update(change)
        rows.append(row)

        if sweep == 1 and n_inundated == 0:
            return rows, False, False  # dry: skip remaining sweeps

        prev_depth = depth

        # Round-level early exit only at round boundaries, as in solve_eikonal_dense.
        if sweep % SWEEPS_PER_ROUND == 0:
            round_max_change = max(r["sweep_max_change_raw"] for r in rows[-SWEEPS_PER_ROUND:])
            if round_max_change <= epsilon:
                return rows, True, True

    return rows, True, False


FIELDNAMES = [
    "tile", "n_cells", "sweep_count", "cum_time_s", "sweep_max_change_raw", "n_inundated", "depth_mean",
    "depth_sum", "depth_max", "max_depth_change_abs", "median_depth_change_abs",
    "mean_depth_change_abs", "n_newly_flooded", "n_cells_changed", "pct_cells_changed",
    "status",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_option_with_positional(parser, "--out-dir", "out_dir")
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--tile-ids", type=int, nargs="*", default=None)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON_M, help=f"round-level convergence threshold (default: {DEFAULT_EPSILON_M}, production's own simulation.flooding.waterlevel_epsilon_m)")
    parser.add_argument("--n-tiles-wanted", type=int, default=N_TILES_WANTED_DEFAULT, help=f"stop early once this many wet tiles are found, in this process's own single-process run only (default: {N_TILES_WANTED_DEFAULT})")
    parser.add_argument(
        "--write-wet-tiles-summary", action="store_true",
        help="write wet_tiles_selected.txt from THIS process's own tiles only - only safe for a "
             "single sequential run that sees every candidate tile. Off by default: under any "
             "parallel HPC split (array job OR N-node batch job), every concurrent "
             "process would otherwise race to overwrite the same file with just its own tiny "
             "slice. Use aggregate_wet_tiles.py after a parallel run instead - it scans every "
             "tile's own CSV post-hoc and is safe regardless of how the work was split.",
    )
    args = parser.parse_args()
    require(parser, args, "out_dir", "--out-dir")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = args.tile_ids if args.tile_ids else TILES
    n_tiles_wanted = args.n_tiles_wanted

    cfg = load_config(args.config)
    # Not root/model_outputs: calibration studies keep their own model_outputs tree.
    model_outputs = Path(cfg["simulation"]["model_outputs"])
    knn = int(cfg["simulation"]["flooding"]["knn"])
    friction_scale_factor = float(cfg["simulation"]["flooding"]["friction_scale_factor"])
    print(f"model_outputs={model_outputs}  knn={knn}  friction_scale_factor={friction_scale_factor}  "
          f"max_sweeps={MAX_SWEEPS} ({MAX_ROUNDS_CEILING} rounds)  epsilon={args.epsilon}  "
          f"n_tiles_wanted={n_tiles_wanted}", flush=True)

    wet_tiles: list[int] = []
    errored_tiles: list[int] = []

    for tile_id in tiles:
        print(f"=== tile {tile_id} ===", flush=True)
        tile_csv = out_dir / f"{tile_id}.csv"
        with open(tile_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            try:
                tile_rows, is_wet, round_converged = run_tile_sweep_trace(
                    tile_id, MAX_SWEEPS, model_outputs, knn, friction_scale_factor, args.epsilon,
                )
                for row in tile_rows:
                    print(f"  sweeps={row['sweep_count']:3d}  cum_time={row['cum_time_s']}s  "
                          f"n_inundated={row['n_inundated']:,}  depth_sum={row['depth_sum']}  "
                          f"max_change={row['max_depth_change_abs']}  "
                          f"median_change={row['median_depth_change_abs']}  "
                          f"mean_change={row['mean_depth_change_abs']}  "
                          f"n_newly_flooded={row['n_newly_flooded']:,}  "
                          f"n_changed={row['n_cells_changed']:,} ({row['pct_cells_changed']}%)", flush=True)
                    writer.writerow(row)
                    f.flush()
                if is_wet:
                    wet_tiles.append(tile_id)
                    n_rounds = len(tile_rows) // SWEEPS_PER_ROUND
                    conv_label = f"round-converged at round {n_rounds}" if round_converged else \
                        f"did NOT converge within {MAX_ROUNDS_CEILING} rounds"
                    print(f"  SUMMARY: wet, {conv_label} ({len(wet_tiles)}/{n_tiles_wanted} selected so far)", flush=True)
                else:
                    print("  SUMMARY: DRY at sweep 1 - excluded", flush=True)
            except Exception as exc:
                print(f"  FAILED: {exc}", flush=True)
                traceback.print_exc()
                err_row = {k: "" for k in FIELDNAMES}
                err_row["tile"] = tile_id
                err_row["status"] = f"error: {exc}"
                writer.writerow(err_row)
                errored_tiles.append(tile_id)

        if len(wet_tiles) >= n_tiles_wanted:
            print(f"\n{n_tiles_wanted} wet tiles found - stopping early, "
                  f"{len(tiles) - tiles.index(tile_id) - 1} remaining candidate(s) not needed", flush=True)
            break

    print(f"\nDone. Per-tile CSVs written to {out_dir}")
    if errored_tiles:
        print(f"WARNING: {len(errored_tiles)} tile(s) errored (status 'error' row, neither wet nor dry): "
              f"{errored_tiles}", flush=True)
    # Off by default: parallel processes would race on this file (use aggregate_wet_tiles.py).
    if args.write_wet_tiles_summary:
        selected = wet_tiles[:n_tiles_wanted]
        wet_file = out_dir / "wet_tiles_selected.txt"
        with open(wet_file, "w") as f:
            for tile_id in selected:
                f.write(f"{tile_id}\n")
        print(f"{len(selected)}/{n_tiles_wanted} wet tiles selected, written to {wet_file}")
        if len(selected) < n_tiles_wanted:
            print(f"WARNING: only found {len(selected)} wet tiles out of {len(tiles)} candidates - "
                  f"need more candidates from select_study_candidate_tiles.py", flush=True)


if __name__ == "__main__":
    main()
