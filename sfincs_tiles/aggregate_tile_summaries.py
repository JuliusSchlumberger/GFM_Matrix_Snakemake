"""Glob every {base_dir_name}/{tile_id}/outputs/summary.json written by
postprocess_tile_summary.py and concatenate into one master CSV - the direct
input for the correlation/calibration analysis.

Usage:
    python aggregate_tile_summaries.py
    python aggregate_tile_summaries.py --base-dir-name validation_sfincs_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v2")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name

    rows = []
    errors = []
    for summary_path in sorted(base_dir.glob("*/outputs/summary.json")):
        try:
            with open(summary_path) as f:
                rows.append(json.load(f))
        except Exception as e:
            errors.append((summary_path, str(e)))

    if not rows:
        print(f"No summary.json files found under {base_dir}/*/outputs/")
        return

    df = pd.DataFrame(rows)
    out_path = base_dir / "all_tiles_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Aggregated {len(df)} tile(s) ({(df['set'] == 'A').sum()} Set A, {(df['set'] == 'B').sum()} Set B)")
    print(f"Wrote {out_path}")
    if errors:
        print(f"\n{len(errors)} file(s) failed to parse:")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
