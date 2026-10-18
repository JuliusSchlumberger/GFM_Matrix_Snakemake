"""Build wet_tiles_selected.txt from a sweep_budget dir's per-tile CSVs.

Wet = CSV has >1 row (the trace writes 1 row and stops on a dry tile); an
empty/unreadable CSV is a task not yet finished, not dry.

Usage:
    python aggregate_wet_tiles.py <sweep_budget_dir> [--n-tiles-wanted 500]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate_wet_tiles(sweep_budget_dir: Path, n_tiles_wanted: int) -> list[int]:
    """Write and return the first n_tiles_wanted wet tile ids (CSV mtime order)."""
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

    selected = wet_tiles[:n_tiles_wanted]
    wet_file = sweep_budget_dir / "wet_tiles_selected.txt"
    with open(wet_file, "w") as f:
        for tile_id in selected:
            f.write(f"{tile_id}\n")
    print(f"{len(selected)}/{n_tiles_wanted} wet tiles selected, written to {wet_file}")
    if len(selected) < n_tiles_wanted:
        print(f"WARNING: only found {len(selected)} wet tiles - need more candidates from "
              f"select_study_candidate_tiles.py, or the array job hasn't finished yet "
              f"({n_unreadable} tile(s) still unreadable/running)")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_budget_dir")
    parser.add_argument("--n-tiles-wanted", type=int, default=500)
    args = parser.parse_args()
    aggregate_wet_tiles(Path(args.sweep_budget_dir), args.n_tiles_wanted)


if __name__ == "__main__":
    main()
