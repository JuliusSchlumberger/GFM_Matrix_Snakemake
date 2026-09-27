"""Glob every {base_dir_name}/{tile_id}/outputs/summary_{model}.json written
by postprocess_tile_summary.py (one independent file per bathtub/eikonal/
sfincs) and merge them into one master CSV, one row per tile - the direct
input for the correlation/calibration analysis.

Usage:
    python aggregate_tile_summaries.py --base-dir-name validation_sfincs_v4
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
    parser.add_argument("--base-dir-name", required=True)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name

    by_tile: dict[str, dict] = {}
    errors = []
    for model in ("bathtub", "eikonal", "sfincs"):
        for summary_path in sorted(base_dir.glob(f"*/outputs/summary_{model}.json")):
            try:
                with open(summary_path) as f:
                    data = json.load(f)
            except Exception as e:
                errors.append((summary_path, str(e)))
                continue
            by_tile.setdefault(data["tile_id"], {}).update(data)

    if not by_tile:
        print(f"No summary_{{model}}.json files found under {base_dir}/*/outputs/")
        return

    df = pd.DataFrame(list(by_tile.values()))
    out_path = base_dir / "all_tiles_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Aggregated {len(df)} tile(s)")
    print(f"Wrote {out_path}")
    if errors:
        print(f"\n{len(errors)} file(s) failed to parse:")
        for path, err in errors:
            print(f"  {path}: {err}")


if __name__ == "__main__":
    main()
