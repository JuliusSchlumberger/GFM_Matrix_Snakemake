"""Calibration run of the production, non-obstacle-coupling sweep count
(`flood_model.flood_depth_dense`'s `sweep_budget`/round-based `max_rounds`),
across a representative pool of wave-0 tiles. Tile IDs come from
`select_calibration_tiles.py`'s `candidate_tiles.txt`.

For each tile, seeds exactly like production (`coastline_mask` +
`_idw_seed_values`), then runs ONE continuous sequence of individual
`_dense_sweep` calls in `_ORTHANT_ORDER[i % 4]` order - exactly
`solve_eikonal_dense`'s own `sweep_budget` semantics, so results are
directly comparable to the real production code path rather than a
reimplementation that could drift - reporting EVERY individual SWEEP, not
round (a sweep is one single directional Gauss-Seidel pass, a round is 4
sweeps, one full cycle through `_ORTHANT_ORDER`; production's own
`max_rounds`/convergence check only ever operates at round granularity,
never mid-round), from 1 up to MAX_SWEEPS. Doing this as a single
continuous pass (not N independent from-scratch solves) avoids O(N^2)
redundant work on the largest tiles.

Round-level early exit: after each COMPLETE round (every 4th sweep), checks
that round's own max_change (the max of `_dense_sweep`'s own raw per-cell
update magnitude over its 4 sweeps - exactly what `solve_eikonal_dense`'s
real round loop computes and compares against `epsilon`) and stops the
tile right there if it's already <= `--epsilon` - matching production's
real round-based early-exit semantics exactly, instead of always burning
the full MAX_SWEEPS ceiling on tiles that settle in a handful of rounds.
`--epsilon` defaults to config's real `simulation.flooding.
waterlevel_epsilon_m` (0.03m) - this script characterizes production's
actual non-coupling behavior, so it uses production's actual threshold. A
tile that never satisfies this within all MAX_ROUNDS_CEILING available
rounds is naturally recognizable downstream by its CSV having the full
MAX_SWEEPS rows with the last round's max_change still > epsilon -
`plot_sweep_budget_convergence.py`'s post-hoc reconstruction logic handles
a variable-length trace correctly (driven by how many complete rounds a
tile's own CSV actually contains, never a hardcoded row count).

load_tile() reads `knn` from `config.yml`'s `simulation.flooding.knn` and
applies `simulation.flooding.friction_scale_factor` in the same
decode -> scale -> floor order as `aqueduct_runner.py`
(`friction = friction * friction_scale_factor` BEFORE the `friction > 0`
floor), so results are directly comparable to what production actually
runs.

Dry-tile handling: if sweep 1 already shows zero flooding, the tile is
logged DRY and the remaining sweeps are skipped for it (no point spending
more sweeps on a tile already known uninformative) - this is how "replace
dry tiles" is implemented: the candidate pool is deliberately
over-provisioned beyond the number of wet tiles actually wanted, and this
script writes `wet_tiles_selected.txt` (the first N_TILES_WANTED
confirmed-wet tile_ids, in the order given) for
`test_obstacle_coupling_calibration.py` to consume directly.

At each sweep, compares the full tile-shaped depth array (zeros where dry)
to the previous sweep's (sweep 1 compared against an implicit all-dry
"sweep 0"): max absolute depth change (whole array - well-defined
regardless), median/mean absolute depth change (restricted to cells that
actually changed - a whole-array median/mean would trivially read ~0 since
flooding covers a small minority of any tile), count of cells newly
flooded (depth 0 -> >0 - no "un-flooded" count needed: within one
un-blocked sweep sequence, Fast Sweeping's relaxation only ever lowers each
cell's eikonal travel-time cost, so flooding can only grow, never shrink),
and count/percent of all cells whose depth changed at all (broader than
"newly flooded" - includes already-flooded cells still settling deeper/
shallower).

Output: ONE CSV per tile, written incrementally, in CSV_DIR, plus a shared
progress log on stdout.

Usage:
    python test_sweep_budget_calibration.py <output_dir> [--config <config.yml>] [--tile-ids id [id ...]]
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
from config_utils import load_config  # noqa: E402
from eikonal import _ORTHANT_ORDER, _dense_sweep  # noqa: E402
from flood_extent import effective_dem  # noqa: E402
from flood_model import _idw_seed_values, coastline_mask, prune_to_coast_connected  # noqa: E402
from rasters import decode_dem_cm, decode_friction_int16, decode_waterlevel_cm  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"

# tile_generation.river_code / ocean_code in config.yml - hardcoded here,
# matching test_obstacle_coupling_calibration.py's own convention.
OCEAN_CODE = 1
RIVER_CODE = 3

# Fallback for a quick standalone smoke test if no tile_ids are given on the
# command line - the real study always passes an explicit list (see module
# docstring). These are real wet tile_ids under the current tile grid, but
# always pass --tile-ids explicitly for an actual calibration run.
TILES = [2083, 2514, 1421, 2569, 2858, 2314, 753]

SWEEPS_PER_ROUND = 4  # one full cycle of _ORTHANT_ORDER - see module docstring
MAX_ROUNDS_CEILING = 100  # ceiling for this study's own round-level early exit on convergence
MAX_SWEEPS = MAX_ROUNDS_CEILING * SWEEPS_PER_ROUND
N_TILES_WANTED_DEFAULT = 500  # a CLI default (--n-tiles-wanted), not a hardcoded target, so a
# differently-sized study doesn't need a code edit. Only matters for this script's own
# single-process early-exit/summary path (see --write-wet-tiles-summary's docstring below) - the
# real batched-HPC flow (generate_calibration_batch_jobs.py) never has enough tiles in one node's
# slice to reach this, so changing it has no effect there either way.
DEFAULT_EPSILON_M = 0.03  # production's own simulation.flooding.waterlevel_epsilon_m default -
# this script characterizes production's real non-coupling behaviour, so it uses production's
# real threshold


def load_tile(tile_id: int, model_outputs: Path, knn: int, friction_scale_factor: float):
    tile_dir = model_outputs / str(tile_id)
    scenario = f"{RETURN_PERIOD}_{WATERLEVEL_NAME}"
    inputs = tile_dir / "inputs"

    with rasterio.open(inputs / "dem.tif") as src:
        dem = decode_dem_cm(src.read(1))
        transform = src.transform
    with rasterio.open(inputs / "mask.tif") as src:
        # int8, not int64: mask only ever holds a handful of small codes,
        # every downstream use is a plain equality/inequality comparison -
        # int64 would be 8x more memory than this array ever needs, enough
        # to OOM on the largest real tiles.
        mask = src.read(1).astype(np.int8)
    with rasterio.open(inputs / "friction.tif") as src:
        friction = decode_friction_int16(src.read(1))
    boundaries = gpd.read_file(inputs / f"boundaries_{scenario}.gpkg")

    dem = effective_dem(dem, mask)
    # decode -> scale -> floor, matching aqueduct_runner.py's own production order exactly
    # (friction_scale_factor applied BEFORE the zero/negative floor, so a genuinely
    # zero-friction cell always floors to a flat 0.001 regardless of the scale factor,
    # while every real friction value gets scaled first) - see module docstring.
    friction = friction * friction_scale_factor
    friction = np.where(friction > 0, friction, friction.dtype.type(0.001))
    coastline = coastline_mask(mask, ocean_code=OCEAN_CODE, river_code=RIVER_CODE)
    coastline_rows, coastline_cols = np.nonzero(coastline)

    # boundaries.gpkg stores int16-centimetre-encoded water levels
    # (rasters.encode_waterlevel_cm) - decode before use. An empty
    # boundaries file (no COAST-RP station found for this tile/scenario -
    # see extract_boundaries.py) is a real, expected outcome for small/
    # isolated tiles, matching production's own NO_STATIONS_REASON skip -
    # _idw_seed_values can't build a BallTree from zero stations, so skip
    # it entirely and seed nothing, same as a genuinely dry tile (empty
    # seed arrays -> t stays all-zero -> n_inundated correctly comes out 0
    # at sweep 1, triggering the same DRY handling as zero-flooding tiles).
    station_values = decode_waterlevel_cm(boundaries[WATERLEVEL_NAME].to_numpy())
    if len(station_values) == 0:
        coastline_rows = np.array([], dtype=coastline_rows.dtype)
        coastline_cols = np.array([], dtype=coastline_cols.dtype)
        initial = np.array([], dtype=np.float64)
        return dem, mask, friction, coastline, coastline_rows, coastline_cols, initial

    stations_lonlat = np.column_stack(
        [boundaries.geometry.x.to_numpy(), boundaries.geometry.y.to_numpy()]
    )
    initial = _idw_seed_values(
        coastline_rows, coastline_cols, transform, stations_lonlat, station_values,
        min(knn, len(station_values)), mask, OCEAN_CODE,
    )
    return dem, mask, friction, coastline, coastline_rows, coastline_cols, initial


def _full_depth_array(t, dem, mask, coastline) -> np.ndarray:
    """Identical math to flood_depth_dense's own non-coupling path, but
    returns the FULL tile-shaped depth array (zeros where dry) rather than
    just flooded-cell aggregate stats - needed to diff consecutive sweeps
    cell-by-cell.
    """
    waterlevel = -t[1:, 1:]
    flood = (waterlevel > dem) & (mask != OCEAN_CODE)
    flood = prune_to_coast_connected(flood, coastline)
    depth = np.zeros_like(dem)
    depth[flood] = waterlevel[flood] - dem[flood]
    return depth


def _depth_change_metrics(depth_prev: np.ndarray, depth_curr: np.ndarray, n_cells: int) -> dict:
    """Per-sweep change metrics comparing depth_curr to depth_prev - see
    module docstring for why median/mean are restricted to changed cells
    while max and the newly-flooded/changed counts use the whole array.
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
    """Returns (rows, is_wet, round_converged). is_wet is False iff sweep 1
    already shows zero flooding - callers should skip such a tile for
    wet_tiles_selected.txt. round_converged is True iff the trace stopped
    early because a complete round's own max_change already dropped to/below
    epsilon (production's real round-based early-exit criterion) - False iff
    it ran all the way to max_sweeps without ever satisfying that (a
    genuinely slow/complex tile, or a dry tile, where round_converged is
    trivially False since is_wet is already False and no round check ever ran).
    """
    dem, mask, friction, coastline, seed_rows, seed_cols, initial = load_tile(
        tile_id, model_outputs, knn, friction_scale_factor,
    )
    seed_values = -initial
    dtype = friction.dtype
    n_cells = dem.size

    m, n = friction.shape
    # t defaults to +99 (waterlevel=-t=-99m), not 0 - matches src/eikonal.py's
    # solve_eikonal_dense: a cell no seed's influence ever reaches should
    # read as "never flooded", not "flooded at exactly sea level". This
    # script reimplements the sweep loop directly (bypassing
    # solve_eikonal_dense) for per-sweep instrumentation, so needs the same
    # sentinel applied here explicitly to stay consistent with production.
    t = np.full((m + 1, n + 1), 99.0, dtype=dtype)
    t[seed_rows, seed_cols] = seed_values
    neg_two = dtype.type(-2.0)
    eight = dtype.type(8.0)
    four = dtype.type(4.0)

    rows = []
    prev_depth = np.zeros_like(dem)  # implicit all-dry "sweep 0" reference for sweep 1's diff
    t0 = time.perf_counter()

    for sweep in range(1, max_sweeps + 1):
        # sweep_max_change_raw: _dense_sweep's OWN return value - the raw
        # per-cell t-array update magnitude for this one directional sweep,
        # over the WHOLE array (not depth-restricted/connectivity-pruned
        # like max_depth_change_abs below). This is exactly what
        # solve_eikonal_dense's round loop maxes over 4 sweeps and compares
        # against waterlevel_epsilon_m for its real convergence check - kept
        # here so "how many rounds would production's own epsilon-based
        # early-exit have taken" can be reconstructed exactly from this one
        # continuous trace after the fact, without a second, separate
        # round-based solve pass.
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
            return rows, False, False  # dry - caller stops here, doesn't burn the remaining sweeps

        prev_depth = depth

        # Round-level early exit, checked only at a complete round boundary
        # (every 4th sweep), matching solve_eikonal_dense's own round loop,
        # which never checks mid-round either.
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
    parser.add_argument("out_dir")
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = args.tile_ids if args.tile_ids else TILES
    n_tiles_wanted = args.n_tiles_wanted

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    # cfg["simulation"]["model_outputs"], NOT a hardcoded root/"model_outputs" - a calibration
    # study's solver outputs are deliberately isolated under its own study directory (e.g.
    # {root}/calibration_500_tiles/model_outputs), not the shared production tree, and
    # load_config() already expands simulation.model_outputs to the right absolute path for
    # whichever --config is passed, so reading it directly keeps this script isolation-correct
    # for both plain production config.yml and any scenario config that overrides the path.
    model_outputs = Path(cfg["simulation"]["model_outputs"])
    knn = int(cfg["simulation"]["flooding"]["knn"])
    friction_scale_factor = float(cfg["simulation"]["flooding"]["friction_scale_factor"])
    print(f"model_outputs={model_outputs}  knn={knn}  friction_scale_factor={friction_scale_factor}  "
          f"max_sweeps={MAX_SWEEPS} ({MAX_ROUNDS_CEILING} rounds)  epsilon={args.epsilon}  "
          f"n_tiles_wanted={n_tiles_wanted}", flush=True)

    wet_tiles: list[int] = []

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

        if len(wet_tiles) >= n_tiles_wanted:
            print(f"\n{n_tiles_wanted} wet tiles found - stopping early, "
                  f"{len(tiles) - tiles.index(tile_id) - 1} remaining candidate(s) not needed", flush=True)
            break

    print(f"\nDone. Per-tile CSVs written to {out_dir}")
    # wet_tiles_selected.txt: see --write-wet-tiles-summary's own help text
    # above for why this defaults off - under ANY parallel HPC split (array
    # job, one tile per task, OR N-node batch job, several tiles per node),
    # every concurrent process only ever sees ITS OWN slice, so every one of
    # them writing "the" wet_tiles_selected.txt would race and overwrite
    # each other down to whichever process wrote last. Use
    # aggregate_wet_tiles.py after a parallel run instead.
    if args.write_wet_tiles_summary:
        selected = wet_tiles[:n_tiles_wanted]
        wet_file = out_dir / "wet_tiles_selected.txt"
        with open(wet_file, "w") as f:
            for tile_id in selected:
                f.write(f"{tile_id}\n")
        print(f"{len(selected)}/{n_tiles_wanted} wet tiles selected, written to {wet_file}")
        if len(selected) < n_tiles_wanted:
            print(f"WARNING: only found {len(selected)} wet tiles out of {len(tiles)} candidates - "
                  f"need more candidates from select_calibration_tiles.py", flush=True)


if __name__ == "__main__":
    main()
