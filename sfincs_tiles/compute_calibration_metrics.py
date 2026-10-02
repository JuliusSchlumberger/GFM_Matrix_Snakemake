"""Aggregates the per-tile flood-extent agreement counts
postprocess_tile_summary.py writes into each tile's
outputs/summary_bathtub.json/summary_eikonal.json into pooled HT/FAR/CSI/bias
for SFINCS-vs-HC-bathtub and SFINCS-vs-EA-bathtub (display names only, see
DISPLAY_LABEL - JSON field prefixes stay "bathtub"/"eikonal").

Pure JSON aggregation, no raster I/O.

Usage:
    python compute_calibration_metrics.py --base-dir-name validation_sfincs_v4
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
DISPLAY_LABEL = {"bathtub": "HC-bathtub", "eikonal": "EA-bathtub"}


def _load_tile_rows(base_dir: Path) -> list[dict]:
    """Merges each tile's separate summary_bathtub.json/summary_eikonal.json
    (postprocess_tile_summary.py) into one row per tile."""
    by_tile: dict[str, dict] = {}
    for prefix in ("bathtub", "eikonal"):
        for summary_path in sorted(base_dir.glob(f"*/outputs/summary_{prefix}.json")):
            with retry_transient_io(open, summary_path) as f:
                data = json.load(f)
            row = by_tile.setdefault(data["tile_id"], {"tile_id": data["tile_id"]})
            for field in COUNT_FIELDS:
                row[f"{prefix}_{field}"] = data.get(f"{prefix}_{field}")
    return list(by_tile.values())


def pooled_metrics(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    matched = df[f"{prefix}_matched_km2"].sum()
    model_only = df[f"{prefix}_only_km2"].sum()
    sfincs_only = df[f"{prefix}_sfincs_only_km2"].sum()
    return metrics_from_counts(matched, model_only, sfincs_only)


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name

    rows = _load_tile_rows(base_dir)
    print(f"{len(rows)} tile(s) with a summary_bathtub.json/summary_eikonal.json found under {base_dir}")
    if not rows:
        print("No summary files found - nothing to aggregate.")
        return

    df = pd.DataFrame(rows)
    out_path = base_dir / "calibration_metrics_per_tile.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} tile(s))")

    for prefix in ("bathtub", "eikonal"):
        n_valid = df[f"{prefix}_matched_km2"].notna().sum()
        n_missing = len(df) - n_valid
        pooled = pooled_metrics(df.dropna(subset=[f"{prefix}_matched_km2"]), prefix)
        print(f"\nSFINCS vs {DISPLAY_LABEL[prefix]} - pooled across {n_valid} tile(s) ({n_missing} missing counts, skipped):")
        for k, v in pooled.items():
            print(f"  {k}: {v:.3f}" if not np.isnan(v) else f"  {k}: NaN")


if __name__ == "__main__":
    main()
