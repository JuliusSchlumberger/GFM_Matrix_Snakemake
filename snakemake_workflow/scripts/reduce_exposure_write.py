"""Sum pass-2 exposure batches and write the File 1/2/3 exposure CSVs.

Partial per-task EAI DataFrames are combined with `.add(fill_value=0.0)`;
the output matches analysis/compute_exposure_analysis.py's single-machine main().

Usage:
    python reduce_exposure_write.py --config path/to/config.yml \\
        --parts-glob ".../exposure/pass2_batch_*.pkl" \\
        --shares .../exposure/shares_by_intensity.json \\
        --out-dir D:/GFM/merged_results/exposure
"""

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from config_utils import load_config  # noqa: E402
from compute_exposure_analysis import (  # noqa: E402
    build_exposure_tasks, load_analysis_context, write_exposure_outputs,
)


def main() -> None:
    """Write the final merged exposure CSVs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parts-glob", required=True, help="glob pattern matching every pass-2 batch's output pickle")
    parser.add_argument("--shares", required=True, help="reduce_exposure_shares.py's output JSON")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ctx = load_analysis_context(cfg)
    with open(args.shares, encoding="utf-8") as f:
        shares_by_intensity = json.load(f)
    tasks = build_exposure_tasks(ctx, shares_by_intensity)

    part_paths = sorted(glob.glob(args.parts_glob))
    if not part_paths:
        raise SystemExit(f"No pass-2 part files matched {args.parts_glob!r}")

    totals: dict[str, pd.DataFrame] = {}
    for p in part_paths:
        with open(p, "rb") as f:
            part = pickle.load(f)
        for key, df in part.items():
            totals[key] = df if key not in totals else totals[key].add(df, fill_value=0.0)

    print(f"Reduced {len(part_paths)} pass-2 parts -> {len(totals)} task totals. Writing CSVs…")
    write_exposure_outputs(ctx, tasks, totals, Path(args.out_dir))
    print(f"All exposure CSVs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
