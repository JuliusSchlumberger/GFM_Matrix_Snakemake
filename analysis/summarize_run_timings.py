"""Summarise the per-run timing records of the main simulation (EA-bathtub runs).

Every run (tile, return period, SLR scenario) writes
<model_outputs>/run_timings/{tile_id}_{return_period}_{waterlevel_name}.json with its wall-clock
seconds (`elapsed_s`, src/eikonal_runner.log_run_timing) and, for the HPC path, an `outcome`
("solved", "skip_*", "oom"). This script reads them all and prints:
  - number of runs and tiles, by outcome
  - total computational hours (sum of elapsed_s over all runs)
  - seconds per run (mean / median / p95), all runs and solved runs only
  - seconds per domain (= tile: sum over its runs; mean / median / p95)
and writes the per-run table to --csv if given.

Run it where the files are (Hydrax) - ~200k small files are slow over a network share:
    python analysis/summarize_run_timings.py --dir <model_outputs>/run_timings [--csv timings.csv]
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd


def _read(path: str) -> dict | None:
    try:
        with open(path) as f:
            r = json.load(f)
        return {"tile_id": r.get("tile_id"), "return_period": r.get("return_period"),
                "waterlevel_name": r.get("waterlevel_name"), "elapsed_s": float(r.get("elapsed_s", float("nan"))),
                "outcome": r.get("outcome", "solved")}
    except (OSError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="<model_outputs>/run_timings")
    parser.add_argument("--csv", default=None, help="optional: write the per-run table here")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    paths = [e.path for e in os.scandir(args.dir) if e.name.endswith(".json")]
    with ThreadPoolExecutor(args.threads) as ex:
        rows = [r for r in ex.map(_read, paths, chunksize=500) if r is not None]
    d = pd.DataFrame(rows)
    print(f"{len(paths)} timing files, {len(d)} readable, {d.tile_id.nunique()} tiles")
    print("\nRuns by outcome:")
    print(d.outcome.value_counts().to_string())

    def stats(s: pd.Series) -> str:
        return f"mean {s.mean():8.1f} s | median {s.median():8.1f} s | p95 {s.quantile(0.95):8.1f} s | max {s.max():8.0f} s"

    hours = d.elapsed_s.sum() / 3600
    print(f"\nTotal computational time (sum over all runs): {hours:,.0f} h ({hours / 24:,.1f} days)")
    print(f"  of which solved runs: {d.loc[d.outcome == 'solved', 'elapsed_s'].sum() / 3600:,.0f} h")
    print(f"\nPer run, all runs:     {stats(d.elapsed_s)}")
    print(f"Per run, solved only:  {stats(d.loc[d.outcome == 'solved', 'elapsed_s'])}")
    per_tile = d.groupby("tile_id").elapsed_s.agg(["sum", "count"])
    print(f"Per domain (tile, sum over its {int(per_tile['count'].median())} runs): {stats(per_tile['sum'])}")
    if args.csv:
        d.to_csv(args.csv, index=False)
        print(f"\nPer-run table written to {args.csv}")


if __name__ == "__main__":
    main()
