"""Per-tile completion report for a friction-sweep calibration study, and
the direct fix for the "sweep is incomplete, and unevenly so across
friction_scale_factor values" symptom.

THE canonical way to check completeness (2026-10-07, replacing
run_one_tile.sh's own run_one_tile_failures.txt, removed the same day):
that file only ever recorded that a tile failed AT SOME POINT in the
study's history, never whether it's still failing now - idempotent re-runs
regularly fix a tile on a later attempt, but nothing ever cleared its
earlier entry (found ~75% stale in practice: of every tile that ever
logged a failure, 3/4 had since succeeded). This script instead reads
real file presence on disk right now, so it's never stale, and its two
outputs (missing_eikonal_pairs.csv, tile_ids_needs_rebuild.txt - see
below) are exactly what a resume/resubmission needs, with nothing further
to derive.

Root cause (2026-10-07): check_sfincs_calibration_progress.py's own
per-fsf counts form a staircase, e.g.:
    fsf=3  980/1126   fsf=6..21  957/1126 (flat)   fsf=24  950   fsf=27  921   default(=30)  905
not all 10 points having the same count. generate_validation_batch_jobs.py's
own --defer-eikonal pairs file is written fsf-MAJOR (`for fsf in factors
for tile_id in tiles` - see that script's own eikonal_pairs.csv-writing
lines): every tile in a batch gets its fsf=3 point before ANY of them get
fsf=6, and so on up to fsf=30 last. A batch that runs out of its own
--time budget (or gets killed/preempted) partway through therefore leaves
every LATER fsf value under-represented for that batch's tiles, in a
strict first-assigned-least-affected order - exactly the staircase
observed. This is not data corruption or a bug in the eikonal solve
itself - it's expected fallout from several batches not finishing their
own fsf sweep before being cut off (or still being mid-sweep right now).

This script's own output is the precise, minimal fix, covering BOTH repair
paths a resume might need:
  - `missing_eikonal_pairs.csv`: tiles that (a) have a real COAST-RP
    station (can ever be forced at all) and (b) already have a built
    SFINCS model (hmax_subgrid.tif exists), but are missing one or more
    sweep points - the exact MISSING (tile_id, friction_scale_factor)
    pairs, feed straight into run_friction_sweep_batch.py directly
    (bypassing generate_validation_batch_jobs.py's whole per-tile
    orchestration, since preprocessing/SFINCS build/run are already done
    for these tiles) - by far the cheapest way to mop up a partial sweep.
  - `tile_ids_needs_rebuild.txt`: tiles with a real station but NO SFINCS
    model yet - the eikonal solve needs hmax_subgrid.tif, so these need
    the FULL per-tile pipeline rerun (generate_validation_batch_jobs.py
    --tile-ids-file tile_ids_needs_rebuild.txt), not a sweep-only fix.
A tile with no COAST-RP station at all is excluded from both - it can
never be forced, resubmitting it in any form is pure waste.

Usage:
    python report_calibration_tile_status.py --base-dir-name sfincs_calibration
    python report_calibration_tile_status.py --base-dir-name sfincs_calibration --max-outer-iterations 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_sfincs_calibration_progress import (  # noqa: E402
    FRICTION_SCALE_FACTORS_DEFAULT, MAX_OUTER_ITERATIONS_SWEEP_DEFAULT, _entries, _eikonal_filename,
)
from gfm_config import read_root  # noqa: E402
from tile_status import read_tile_status, tile_has_station  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="sfincs_calibration")
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", default=FRICTION_SCALE_FACTORS_DEFAULT)
    parser.add_argument("--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS_SWEEP_DEFAULT)
    parser.add_argument("--out-csv", default=None, help="default: {base_dir_name}/calibration_tile_status.csv")
    parser.add_argument("--out-pairs", default=None, help="default: {base_dir_name}/missing_eikonal_pairs.csv")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    tile_ids = [line.strip() for line in (base_dir / "tile_ids.txt").read_text().splitlines() if line.strip()]
    print(f"{len(tile_ids)} tile(s) from {base_dir / 'tile_ids.txt'}")

    sweep_filenames = {fsf: _eikonal_filename(fsf, args.max_outer_iterations) for fsf in args.friction_scale_factors}

    rows = []
    missing_pairs: list[tuple[str, float]] = []
    for i, tid in enumerate(tile_ids, start=1):
        tile_dir = base_dir / tid
        has_station = tile_has_station(tile_dir / "inputs")
        sfincs_ready = (tile_dir / "sfincs_model" / "hmax_subgrid.tif").exists()
        output_entries = _entries(tile_dir / "outputs")
        present = {fsf: fname in output_entries for fsf, fname in sweep_filenames.items()}
        n_present = sum(present.values())
        fully_done = n_present == len(args.friction_scale_factors)
        status_record = read_tile_status(root, args.base_dir_name, tid)

        row = {
            "tile_id": tid, "has_station": has_station, "sfincs_ready": sfincs_ready,
            "n_sweep_points_present": n_present, "n_sweep_points_total": len(args.friction_scale_factors),
            "fully_done": fully_done,
            "last_status": status_record.get("status") if status_record else None,
            "last_status_stage": status_record.get("stage") if status_record else None,
            "last_status_message": status_record.get("message") if status_record else None,
        }
        for fsf, is_present in present.items():
            row[f"fsf_{fsf:g}"] = is_present
        rows.append(row)

        if has_station and sfincs_ready and not fully_done:
            missing_pairs.extend((tid, fsf) for fsf, is_present in present.items() if not is_present)

        if i % 100 == 0 or i == len(tile_ids):
            print(f"  [{i}/{len(tile_ids)}] scanned", flush=True)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv) if args.out_csv else base_dir / "calibration_tile_status.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} row(s))")

    n_total = len(df)
    n_fully_done = int(df["fully_done"].sum())
    n_no_station = int((~df["has_station"]).sum())
    n_not_sfincs_ready = int((df["has_station"] & ~df["sfincs_ready"]).sum())
    n_partial_sweep = int((df["has_station"] & df["sfincs_ready"] & ~df["fully_done"]).sum())

    print("\n=== Summary ===")
    print(f"  fully done (all {len(args.friction_scale_factors)} sweep points present): {n_fully_done}/{n_total}")
    print(f"  no COAST-RP station (can NEVER complete, not a bug):                      {n_no_station}/{n_total}")
    print(f"  has station but SFINCS model not built yet (needs FULL tile rebuild):      {n_not_sfincs_ready}/{n_total}")
    print(f"  has station + SFINCS built, but sweep partial (fixable via pairs-only resubmit): {n_partial_sweep}/{n_total}")

    print("\n=== Per-fsf staircase (confirms the time-limit-cutoff diagnosis) ===")
    for fsf in args.friction_scale_factors:
        n = int(df[f"fsf_{fsf:g}"].sum())
        print(f"  fsf={fsf:g}: {n}/{n_total} ({100 * n / n_total:.1f}%)")

    not_ready_df = df[df["has_station"] & ~df["sfincs_ready"]]
    if len(not_ready_df):
        print("\n=== Needs-rebuild tiles, by last known failure category (tile_status.json) ===")
        for status, grp in not_ready_df.groupby(not_ready_df["last_status"].fillna("unknown (never attempted, or pre-dates tile_status.json)")):
            permanent_note = " - PERMANENT, will be skipped immediately on resubmit" if status in ("no_station", "no_boundary_cells", "antimeridian") else ""
            print(f"  {status}: {len(grp)} tile(s){permanent_note}")

    out_pairs = Path(args.out_pairs) if args.out_pairs else base_dir / "missing_eikonal_pairs.csv"
    with open(out_pairs, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(f"{tid},{fsf:g}" for tid, fsf in missing_pairs) + ("\n" if missing_pairs else ""))
    print(f"\nWrote {out_pairs} ({len(missing_pairs)} missing (tile, fsf) pair(s) across {n_partial_sweep} tile(s))")

    # Companion resubmission target for the OTHER repair path (full per-tile pipeline rerun,
    # via generate_validation_batch_jobs.py --tile-ids-file, not run_friction_sweep_batch.py
    # directly - these tiles have no SFINCS model yet, so there is no sweep to resume).
    not_ready = df[df["has_station"] & ~df["sfincs_ready"]]["tile_id"].tolist()
    out_rebuild = base_dir / "tile_ids_needs_rebuild.txt"
    with open(out_rebuild, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(not_ready) + ("\n" if not_ready else ""))
    print(f"Wrote {out_rebuild} ({len(not_ready)} tile(s) needing a full rebuild - SFINCS model missing, "
          f"not in missing_eikonal_pairs.csv)")


if __name__ == "__main__":
    main()
