"""Summarize Aqueduct run-time timing logs written by aqueduct_runner.log_run_timing.

Each successful `aqueduct.exe` invocation (run_aqueduct.py) writes one
`{tile_id}_{return_period}_{waterlevel_name}.json` file to
`{model_outputs}/run_timings/`, recording wall-clock seconds and the
DEM pixel count actually passed to Aqueduct. This script aggregates those
into a summary, and - given a second, baseline run's `model_outputs` - a
matched before/after speedup comparison for `simulation.flood_extent_crop`.

To measure the speedup:
  1. Run the pipeline once with `simulation.flood_extent_crop.enabled: false`
     (the default) against one `model_outputs` directory.
  2. Run it again with `enabled: true` against a *different* `model_outputs`
     directory (or the same tiles re-run after moving `run_timings/` aside),
     so the two runs' timing logs don't overwrite each other.
  3. python summarize_run_timings.py <cropped_model_outputs> --baseline <uncropped_model_outputs>

Usage:
    python summarize_run_timings.py <model_outputs_dir> [--baseline <model_outputs_dir>]
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_timings(model_outputs: str | Path) -> pd.DataFrame:
    """Load every run_timings/*.json record for one model_outputs directory into a DataFrame."""
    timings_dir = Path(model_outputs) / "run_timings"
    records = [json.loads(p.read_text()) for p in timings_dir.glob("*.json")]
    if not records:
        raise SystemExit(f"No timing records found in {timings_dir} - has run_aqueduct.py run yet?")
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame, label: str) -> None:
    """Print summary statistics for one run's timing log."""
    print(f"\n=== {label} ({len(df)} runs) ===")
    print(f"total wall time:  {df['elapsed_s'].sum():10.1f} s")
    print(f"mean per run:     {df['elapsed_s'].mean():10.2f} s")
    print(f"median per run:   {df['elapsed_s'].median():10.2f} s")
    slowest = df.loc[df["elapsed_s"].idxmax()]
    print(f"slowest run:      {slowest['elapsed_s']:10.2f} s (tile {slowest['tile_id']}, "
          f"{slowest['return_period']}_{slowest['waterlevel_name']}, {int(slowest['cropped_pixels']):,} px)")
    if (df["original_pixels"] > 0).all():
        reduction = 1 - df["cropped_pixels"] / df["original_pixels"]
        print(f"mean pixel reduction from crop: {reduction.mean() * 100:5.1f}%")


def compare(df: pd.DataFrame, df_baseline: pd.DataFrame) -> None:
    """Print a matched (tile_id, return_period, waterlevel_name) speedup comparison."""
    key = ["tile_id", "return_period", "waterlevel_name"]
    merged = df.merge(df_baseline, on=key, suffixes=("_this", "_baseline"))
    if merged.empty:
        print("\nNo matching (tile_id, return_period, waterlevel_name) runs between the two directories.")
        return
    merged["speedup"] = merged["elapsed_s_baseline"] / merged["elapsed_s_this"]
    print(f"\n=== matched speedup ({len(merged)} runs in common) ===")
    print(f"mean speedup:     {merged['speedup'].mean():6.2f}x")
    print(f"median speedup:   {merged['speedup'].median():6.2f}x")
    biggest = merged.loc[merged["speedup"].idxmax()]
    print(f"largest speedup:  {biggest['speedup']:6.2f}x (tile {biggest['tile_id']}, "
          f"{int(biggest['original_pixels_this']):,} -> {int(biggest['cropped_pixels_this']):,} px)")
    print(f"total time (this):     {merged['elapsed_s_this'].sum():10.1f} s")
    print(f"total time (baseline): {merged['elapsed_s_baseline'].sum():10.1f} s")
    print(f"overall speedup:       {merged['elapsed_s_baseline'].sum() / merged['elapsed_s_this'].sum():6.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_outputs", help="model_outputs directory to summarize (contains run_timings/)")
    parser.add_argument("--baseline", help="A second model_outputs directory to compare against (e.g. the uncropped run)")
    args = parser.parse_args()

    df = load_timings(args.model_outputs)
    summarize(df, args.model_outputs)

    if args.baseline:
        df_baseline = load_timings(args.baseline)
        summarize(df_baseline, args.baseline)
        compare(df, df_baseline)


if __name__ == "__main__":
    main()
