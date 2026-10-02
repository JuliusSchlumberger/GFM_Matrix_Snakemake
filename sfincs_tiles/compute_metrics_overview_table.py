"""Pooled key-metrics overview table: HC-bathtub (hydraulically-connected
bathtub - elevation threshold + ocean-connectivity pruning, see
flood_agreement.prune_to_ocean_connected) vs EA-bathtub (Eikonal-attenuated
bathtub - the friction/propagation-aware eikonal solve), pooled across a
validation batch - one row per model:
  - HT/FAR/CSI/bias, from matched/model_only/sfincs_only km2 summed across
    tiles first (see flood_agreement.metrics_from_counts).
  - r/bias_m/rmse_m/median_error_m/pct_within_0.2m, cell-level depth
    agreement at every mutually-wet cell, pooled the same way (see
    flood_agreement.depth_error_metrics_from_pooled).

Pure JSON aggregation - reads postprocess_tile_summary.py's
summary_bathtub.json/summary_eikonal.json, no raster I/O. Also run
automatically at the end of plot_validation_results.py's main() - call
build_and_write_table(base_dir) directly to reuse without CLI args.

Usage:
    python compute_metrics_overview_table.py --base-dir-name validation_sfincs_v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flood_agreement import depth_error_metrics_from_pooled, metrics_from_counts, pool_depth_joint  # noqa: E402
from gfm_config import read_root  # noqa: E402

ROWS = ["bathtub", "eikonal"]
ROW_LABELS = {"bathtub": "HC-bathtub", "eikonal": "EA-bathtub"}


def collect_rows(base_dir: Path, tile_ids: list[str]) -> pd.DataFrame:
    records = []
    for tile_id in tile_ids:
        tile_dir = base_dir / tile_id
        row = {"tile_id": tile_id}
        for prefix in ROWS:
            summary_path = tile_dir / "outputs" / f"summary_{prefix}.json"
            data = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            for k in (f"{prefix}_km2", f"{prefix}_matched_km2", f"{prefix}_only_km2", f"{prefix}_sfincs_only_km2"):
                row[k] = data.get(k)
        records.append(row)
    return pd.DataFrame.from_records(records)


def pooled_row(df: pd.DataFrame, base_dir: Path, prefix: str) -> dict:
    sub = df.dropna(subset=[f"{prefix}_matched_km2"])
    matched = sub[f"{prefix}_matched_km2"].sum()
    model_only = sub[f"{prefix}_only_km2"].sum()
    sfincs_only = sub[f"{prefix}_sfincs_only_km2"].sum()
    metrics = metrics_from_counts(matched, model_only, sfincs_only)
    depth_metrics = depth_error_metrics_from_pooled(pool_depth_joint(base_dir, prefix))
    return {
        "model": ROW_LABELS[prefix],
        "n_tiles_pooled": int(len(sub)),
        "total_km2": float(df[f"{prefix}_km2"].sum(skipna=True)),
        "matched_km2": float(matched),
        "model_only_km2": float(model_only),
        "sfincs_only_km2": float(sfincs_only),
        **metrics,
        **depth_metrics,
    }


def build_and_write_table(base_dir: Path) -> pd.DataFrame:
    """Rebuilds and writes metrics_overview_per_tile.csv/metrics_overview_table.csv
    for `base_dir`. Called by both this script's own main() and
    plot_validation_results.py's main()."""
    tile_ids = [line.strip() for line in (base_dir / "tile_ids.txt").read_text().splitlines() if line.strip()]
    print(f"{len(tile_ids)} tile(s) from tile_ids.txt")

    df = collect_rows(base_dir, tile_ids)
    per_tile_path = base_dir / "metrics_overview_per_tile.csv"
    df.to_csv(per_tile_path, index=False)
    print(f"Wrote {per_tile_path} ({len(df)} tile(s))")

    table = pd.DataFrame([pooled_row(df, base_dir, prefix) for prefix in ROWS])
    table_path = base_dir / "metrics_overview_table.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {table_path}\n")

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(table.to_string(index=False))
    return table


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    build_and_write_table(base_dir)


if __name__ == "__main__":
    main()
