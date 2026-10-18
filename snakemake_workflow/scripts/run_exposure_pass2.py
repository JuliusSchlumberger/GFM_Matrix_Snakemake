"""HPC batch worker: exposure pass 2 (per-country EAI for every adaptation task).

Requires reduce_exposure_shares.py's globally-reduced shares, not a single
pass-1 batch's partial contribution.

Usage:
    python run_exposure_pass2.py --config path/to/config.yml \\
        --chunks-file batch_000_chunks.txt \\
        --shares shares_by_intensity.json --out pass2_batch_000.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402
from compute_exposure_analysis import (  # noqa: E402
    build_exposure_tasks, load_analysis_context, pass2_all_tasks,
)


def main() -> None:
    """Run pass 2 for this batch's chunk subset and pickle the task results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--chunks-file", required=True, help="one chunk_id per line")
    parser.add_argument("--shares", required=True, help="reduce_exposure_shares.py's output JSON")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ctx = load_analysis_context(cfg)
    chunk_ids = [
        line.strip() for line in Path(args.chunks_file).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    with open(args.shares, encoding="utf-8") as f:
        shares_by_intensity = json.load(f)

    tasks = build_exposure_tasks(ctx, shares_by_intensity)
    print(f"Pass 2: {len(chunk_ids)} chunks x {len(tasks)} tasks…")
    results = pass2_all_tasks(
        chunk_ids, ctx.chunks_dir, ctx.flood_frac_dir, ctx.return_periods, ctx.slr_scenarios,
        ctx.rp_applied, ctx.iso_lookup, tasks,
    )

    out_path = Path(args.out)
    retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
    # Atomic: a --resume dispatch treats the file's existence as "batch done".
    atomic_write(out_path, lambda f: pickle.dump(results, f), mode="wb")
    print(f"Written: {out_path} ({len(results)} task results)")


if __name__ == "__main__":
    main()
