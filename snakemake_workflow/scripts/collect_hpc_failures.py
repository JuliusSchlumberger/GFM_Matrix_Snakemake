"""Compile a single report of every tile/scenario that did NOT produce a
real simulated waterdepth after an HPC run.

Two genuinely different categories, kept separate rather than merged into
one "failed" bucket:

  - JOB ERRORS: the (tile_id, return_period, waterlevel_name) lines
    appended by each generated sbatch script's `run_job()` wrapper (see
    generate_aqueduct_jobs.py) to `{jobs_dir}/logs/wave*_batch_*_failures.
    txt` when `run_aqueduct_cli.py` exits non-zero - a genuine, unresolved
    unknown (crash, timeout, bad input) that needs investigating and
    re-running. `run_job()` already logs the failure and moves on to the
    next job in its batch instead of aborting (`set -uo pipefail`, not
    `-e`), so one bad tile never takes down the rest of a node's batch or
    a wave.
  - CONFIDENTLY-RESOLVED SKIPS: `model_outputs/skipped_tiles/*.txt`
    markers written DURING a normal run (no boundary stations, no upstream
    flooding found yet, or OOM/too-large) - these are real, deliberate
    results (a genuine zero or a genuine "unknown, too large"), not
    failures. Listed here for visibility only; re-running them the same
    way as a job error would be wrong for the first two (same outcome
    every time) and pointless for OOM (same outcome unless the tile is
    re-chunked).

Usage:
    python collect_hpc_failures.py [--config path/to/config.yml] [--out report.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import load_config  # noqa: E402

_OOM_MARKERS = ("MemoryError", "tile too large")
_NO_STATIONS_MARKER = "no water level boundary stations"
_NO_UPSTREAM_MARKER = "no non-zero water level found"


def _classify_skip_reason(reason: str) -> str:
    if _NO_STATIONS_MARKER in reason:
        return "no_boundary_stations"
    if _NO_UPSTREAM_MARKER in reason:
        return "no_upstream_flooding"
    if any(marker in reason for marker in _OOM_MARKERS):
        return "oom_too_large"
    return "skipped_other"


def collect_job_errors(jobs_dir: Path) -> list[dict]:
    rows = []
    logs_dir = jobs_dir / "logs"
    for fail_file in sorted(logs_dir.glob("wave*_batch_*_failures.txt")):
        for line in fail_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                print(f"WARNING: unparsable line in {fail_file.name}: {line!r}", file=sys.stderr)
                continue
            tile_id, return_period, waterlevel_name = parts
            rows.append({
                "category": "job_error",
                "tile_id": tile_id,
                "return_period": return_period,
                "waterlevel_name": waterlevel_name,
                "detail": "non-zero exit from run_aqueduct_cli.py",
                "source": str(fail_file.relative_to(jobs_dir)),
            })
    return rows


def collect_skipped_tiles(skipped_dir: Path) -> list[dict]:
    rows = []
    for marker in sorted(skipped_dir.glob("*.txt")):
        # {tile_id}_{return_period}_{waterlevel_name}.txt - return_period
        # (e.g. "RP100") never contains "_", but waterlevel_name (e.g.
        # "SLR_0") often does, so split on the FIRST two underscores only.
        tile_id, scenario_name = marker.stem.split("_", 1)
        return_period, _, waterlevel_name = scenario_name.partition("_")
        reason = marker.read_text(encoding="utf-8-sig").strip()
        rows.append({
            "category": _classify_skip_reason(reason),
            "tile_id": tile_id,
            "return_period": return_period,
            "waterlevel_name": waterlevel_name,
            "detail": reason,
            "source": str(marker.relative_to(skipped_dir.parent)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--out", default=None, help="Output CSV path (default: {jobs_dir}/failure_report.csv)")
    args = parser.parse_args()

    config = load_config(args.config)
    model_outputs = Path(config["simulation"]["model_outputs"])
    jobs_dir = Path(config["hpc"]["jobs_dir"])
    out_path = Path(args.out) if args.out else jobs_dir / "failure_report.csv"

    rows = collect_job_errors(jobs_dir) + collect_skipped_tiles(model_outputs / "skipped_tiles")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "tile_id", "return_period", "waterlevel_name", "detail", "source"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["category"], r["tile_id"], r["return_period"], r["waterlevel_name"])))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1

    print(f"Wrote {len(rows)} row(s) to {out_path}\n")
    print("By category:")
    for category, n in sorted(counts.items()):
        print(f"  {category:22s} {n}")
    if counts.get("job_error", 0):
        print(
            f"\n{counts['job_error']} job_error row(s) need investigating and re-running "
            "(retry with run_aqueduct_cli.py directly - see hpc.md step 2)."
        )


if __name__ == "__main__":
    main()
