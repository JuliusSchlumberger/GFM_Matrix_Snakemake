"""Aggregate the per-tile flood-extent agreement counts that
postprocess_tile_summary.py already writes into each tile's own
outputs/summary.json (bathtub_{matched,only,sfincs_only}_km2 and
eikonal_{matched,only,sfincs_only}_km2) into pooled HT/FAR/CSI/bias for
SFINCS-vs-bathtub and SFINCS-vs-eikonal.

Pure JSON aggregation - no raster is opened here. Moved out of this script
(2026-09-24, user direction) because the per-tile counts require reading
and reprojecting each tile's rasters, which postprocess_tile_summary.py
already does, per-tile, in parallel, on the HPC node that just produced
those rasters - redoing that work here, sequentially, from one machine over
the network mount, was pure waste. This script now only sums what's already
in summary.json and computes ratios from the sums - see flood_agreement.py
for why ratios are only ever computed on SUMMED counts, never per-tile.

Usage:
    python compute_calibration_metrics.py
    python compute_calibration_metrics.py --base-dir-name validation_sfincs_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flood_agreement import metrics_from_counts  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

COUNT_FIELDS = ("matched_km2", "only_km2", "sfincs_only_km2")


def _load_tile_row(summary_path: Path) -> dict | None:
    with retry_transient_io(open, summary_path) as f:
        data = json.load(f)
    row = {"tile_id": data.get("tile_id", summary_path.parent.parent.name)}
    for prefix in ("bathtub", "eikonal"):
        for field in COUNT_FIELDS:
            row[f"{prefix}_{field}"] = data.get(f"{prefix}_{field}")
    return row


def pooled_metrics(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    matched = df[f"{prefix}_matched_km2"].sum()
    model_only = df[f"{prefix}_only_km2"].sum()
    sfincs_only = df[f"{prefix}_sfincs_only_km2"].sum()
    return metrics_from_counts(matched, model_only, sfincs_only)


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name

    summary_paths = sorted(base_dir.glob("*/outputs/summary.json"), key=lambda p: int(p.parent.parent.name))
    print(f"{len(summary_paths)} tile(s) with a summary.json found under {base_dir}")

    rows = [_load_tile_row(p) for p in summary_paths]
    if not rows:
        print("No summary.json files found - nothing to aggregate.")
        return

    df = pd.DataFrame(rows)
    out_path = base_dir / "calibration_metrics_per_tile.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} tile(s))")

    for prefix in ("bathtub", "eikonal"):
        n_valid = df[f"{prefix}_matched_km2"].notna().sum()
        n_missing = len(df) - n_valid
        pooled = pooled_metrics(df.dropna(subset=[f"{prefix}_matched_km2"]), prefix)
        print(f"\nSFINCS vs {prefix} - pooled across {n_valid} tile(s) ({n_missing} missing counts, skipped):")
        for k, v in pooled.items():
            print(f"  {k}: {v:.3f}" if not np.isnan(v) else f"  {k}: NaN")


if __name__ == "__main__":
    main()
