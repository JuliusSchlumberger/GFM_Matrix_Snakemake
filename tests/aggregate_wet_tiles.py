"""Builds wet_tiles_selected.txt from a sweep_budget output directory's own
per-tile CSVs, post-hoc (2026-09-24) - the counterpart to
test_sweep_budget_calibration.py's own sequential-run wet_tiles_selected.txt
write, which is deliberately skipped when that script runs as one tile per
call (the HPC array job - see that script's own main() comment for why:
~300 concurrent single-tile tasks all writing "the" same file would race).

A tile is "wet" iff its CSV has more than 1 row - the trace loop writes
exactly 1 row and stops immediately when sweep 1 shows zero flooding (see
run_tile_sweep_trace's own dry-tile handling), so row count alone
distinguishes dry (1 row) from wet (>1 row) without re-deriving n_inundated
from the CSV content itself. A 0-byte/unreadable CSV means that task never
started or is still running - treated as "not yet known", not dry.

Usage:
    python aggregate_wet_tiles.py <sweep_budget_dir> [--n-tiles-wanted 260]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_budget_dir")
    parser.add_argument("--n-tiles-wanted", type=int, default=260)
    args = parser.parse_args()

    sweep_budget_dir = Path(args.sweep_budget_dir)
    csvs = sorted(sweep_budget_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)

    wet_tiles: list[int] = []
    n_dry = 0
    n_unreadable = 0
    for p in csvs:
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            n_unreadable += 1
            continue
        if df.empty:
            n_unreadable += 1
            continue
        if len(df) > 1:
            wet_tiles.append(int(df["tile"].iloc[0]))
        else:
            n_dry += 1

    print(f"{len(csvs)} CSV(s) found: {len(wet_tiles)} wet, {n_dry} dry, {n_unreadable} unreadable/still-running")

    selected = wet_tiles[:args.n_tiles_wanted]
    wet_file = sweep_budget_dir / "wet_tiles_selected.txt"
    with open(wet_file, "w") as f:
        for tile_id in selected:
            f.write(f"{tile_id}\n")
    print(f"{len(selected)}/{args.n_tiles_wanted} wet tiles selected, written to {wet_file}")
    if len(selected) < args.n_tiles_wanted:
        print(f"WARNING: only found {len(selected)} wet tiles - need more candidates from "
              f"select_calibration_tiles.py, or the array job hasn't finished yet "
              f"({n_unreadable} tile(s) still unreadable/running)")


if __name__ == "__main__":
    main()
