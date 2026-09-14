"""One-off diagnostic: trace why exposure_avoid_*_ssp.csv comes out ~0 for
most countries (e.g. BGD) while a real value survives for a handful of
others (RUS/CAN/USA/NZL/small Pacific islands).

Inspects each pass2_batch_*.pkl individually (BEFORE reduce_exposure_write's
summation) for one avoid_ssp task key, plus the matching retreat task (whose
share_retreat avoid's redirected term is derived from) - to see whether the
zero enters at the per-batch compute step or only after reduction.

Usage:
    python snakemake_workflow/scripts/debug_avoid_ssp.py --config snakemake_workflow/config/config.yml \\
        --jobs-dir "P:\\11212688-004-global-floodmaps\\modelling\\model_outputs\\hpc_jobs\\exposure" \\
        --shares "P:\\11212688-004-global-floodmaps\\modelling\\model_outputs\\hpc_jobs\\exposure\\shares_by_intensity.json" \\
        --iso BGD --slr-int SLR_500 --ssp SSP1 --year 2050
"""

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))

from config_utils import load_config  # noqa: E402
from compute_exposure_analysis import build_exposure_tasks, load_analysis_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs-dir", required=True, help="hpc_jobs/exposure directory")
    parser.add_argument("--shares", required=True)
    parser.add_argument("--iso", default="BGD")
    parser.add_argument("--slr-int", default="SLR_500")
    parser.add_argument("--ssp", default="SSP1")
    parser.add_argument("--year", type=int, default=2050)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ctx = load_analysis_context(cfg)
    with open(args.shares, encoding="utf-8") as f:
        shares_by_intensity = json.load(f)
    tasks = build_exposure_tasks(ctx, shares_by_intensity)

    avoid_key = f"avoid_{args.slr_int}_ssp_{args.ssp}_{args.year}"
    retreat_key = f"retreat_{args.slr_int}"

    avoid_task = next((t for t in tasks if t.key == avoid_key), None)
    retreat_task = next((t for t in tasks if t.key == retreat_key), None)
    print(f"avoid task found: {avoid_task is not None}")
    if avoid_task is not None:
        print(f"  growth_by_iso[{args.iso}] = {avoid_task.growth_by_iso.get(args.iso, 'MISSING')}")
        print(f"  redirected_share has {args.iso}: {args.iso in avoid_task.redirected_share}"
              f" (value={avoid_task.redirected_share.get(args.iso, 'MISSING')})")
        print(f"  redirected_share total entries: {len(avoid_task.redirected_share)}")
    print(f"retreat task found: {retreat_task is not None}")
    if retreat_task is not None:
        print(f"  share_retreat has {args.iso}: {args.iso in retreat_task.share_retreat}"
              f" (value={retreat_task.share_retreat.get(args.iso, 'MISSING')})")
        print(f"  share_retreat total entries: {len(retreat_task.share_retreat)}")

    print(f"\nctx.slr_mm_sorted = {ctx.slr_mm_sorted}")
    print(f"ctx.slr_traj columns = {list(ctx.slr_traj.columns) if ctx.slr_traj is not None else None}")
    if ctx.slr_traj is not None and args.ssp in ctx.slr_traj.columns:
        print(f"ctx.slr_traj[{args.ssp}] at year {args.year} (interpolated separately below)")

    print(f"\n=== per-batch pickle inspection for {args.iso} ===")
    parts = sorted(glob.glob(str(Path(args.jobs_dir) / "pass2_batch_*.pkl")))
    total_avoid = None
    total_retreat = None
    for p in parts:
        with open(p, "rb") as f:
            part = pickle.load(f)
        avoid_df = part.get(avoid_key)
        retreat_df = part.get(retreat_key)
        avoid_row = avoid_df.loc[args.iso].to_dict() if (avoid_df is not None and args.iso in avoid_df.index) else None
        retreat_row = (
            retreat_df.loc[args.iso].to_dict() if (retreat_df is not None and args.iso in retreat_df.index) else None
        )
        if avoid_row is not None or retreat_row is not None:
            print(f"{Path(p).name}: avoid={avoid_row}  retreat={retreat_row}")
        if avoid_df is not None and args.iso in avoid_df.index:
            total_avoid = avoid_df.loc[[args.iso]] if total_avoid is None else total_avoid.add(
                avoid_df.loc[[args.iso]], fill_value=0.0
            )
        if retreat_df is not None and args.iso in retreat_df.index:
            total_retreat = retreat_df.loc[[args.iso]] if total_retreat is None else total_retreat.add(
                retreat_df.loc[[args.iso]], fill_value=0.0
            )

    print(f"\n=== reduced (summed across batches) for {args.iso} ===")
    print(f"avoid total:\n{total_avoid}")
    print(f"retreat total:\n{total_retreat}")


if __name__ == "__main__":
    main()
