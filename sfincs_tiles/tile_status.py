"""Per-tile status record - one small JSON file per tile
(`{base_dir_name}/{tile_id}/tile_status.json`), OVERWRITTEN on every
attempt, never appended - so it always reflects the tile's most recent
outcome instead of accumulating a growing, increasingly-stale history (see
run_one_tile.sh's own module docstring for why the old
run_one_tile_failures.txt was retired in favor of this - that file, once
a tile started succeeding on a later retry, kept its old failure entry
forever; cross-checked 2026-10-07 against real on-disk state and found
~75% of everything it had ever logged had since succeeded).

Status categories:
  ok                - tile fully processed through postprocess_tile_summary.py.
  no_station        - boundaries file is empty (no COAST-RP station nearby
                       at all) - PERMANENT: geometry-determined, re-running
                       changes nothing.
  no_boundary_cells - a real station exists, but the ocean polygon never
                       touches an active-domain edge cell on this tile's
                       grid (build_sfincs_tile.py's own create_boundary()
                       step) - PERMANENT, same reasoning as no_station
                       (no weir/discharge fallback exists in this
                       pipeline), different mechanism.
  antimeridian      - hydromt_sfincs's own water_level.create() masking
                       fails near +-180 deg longitude (see
                       build_sfincs_tile.py's own
                       _classify_water_level_create_error, which verifies
                       the error's own coordinate is actually near +-180 -
                       an earlier version matched on exception TEXT alone
                       and mislabeled unrelated TopologyExceptions) -
                       PERMANENT, not fixable via buffer tuning.
  other_error       - anything else - NOT assumed permanent (could be
                       transient: a P:\\ mount hiccup, resource contention,
                       a since-fixed bug) - a resume should keep retrying
                       these rather than silently skip them forever.

PERMANENT_STATUSES tiles are the ones a batch run should skip immediately,
before attempting (or re-attempting) any real work - see
is_permanently_unsolvable() below, called early in run_one_tile.sh.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PERMANENT_STATUSES = {"no_station", "no_boundary_cells", "antimeridian"}


def status_path(root: Path, base_dir_name: str, tile_id: str) -> Path:
    return root / base_dir_name / str(tile_id) / "tile_status.json"


def write_tile_status(root: Path, base_dir_name: str, tile_id: str, status: str, stage: str, message: str = "") -> None:
    p = status_path(root, base_dir_name, tile_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "tile_id": str(tile_id),
        "status": status,
        "stage": stage,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2), encoding="utf-8")


def read_tile_status(root: Path, base_dir_name: str, tile_id: str) -> dict | None:
    p = status_path(root, base_dir_name, tile_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_permanently_unsolvable(root: Path, base_dir_name: str, tile_id: str) -> tuple[bool, dict | None]:
    """(True, status_dict) if this tile's own last-recorded status is a
    PERMANENT category - the caller should skip immediately, no retry."""
    status = read_tile_status(root, base_dir_name, tile_id)
    if status and status.get("status") in PERMANENT_STATUSES:
        return True, status
    return False, status


def tile_has_station(inputs_dir: Path) -> bool:
    """True if boundaries_RP100_SLR_0.gpkg exists and has >=1 feature (a
    real COAST-RP station nearby). Used both by run_one_tile.sh's own early
    no_station short-circuit (BEFORE regenerate_dem_mask.py/build_elevation.py/
    build_roughness.py run - build_boundary_forcing.py itself only runs
    AFTER those, so without this check a no-station tile still wastes that
    work every single attempt) and by report_calibration_tile_status.py's
    own per-tile report."""
    p = inputs_dir / "boundaries_RP100_SLR_0.gpkg"
    if not p.exists():
        return False
    import geopandas as gpd
    try:
        return len(gpd.read_file(p)) > 0
    except Exception:
        return False


def _cli() -> None:
    """Thin CLI so run_one_tile.sh (bash) can call this without importing
    the full config stack - --root is passed directly (run_one_tile.sh
    already has DATA_ROOT as a plain bash variable, no YAML parsing needed)."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_write = sub.add_parser("write")
    p_write.add_argument("--root", required=True)
    p_write.add_argument("--base-dir-name", required=True)
    p_write.add_argument("--tile-id", required=True)
    p_write.add_argument("--status", required=True)
    p_write.add_argument("--stage", required=True)
    p_write.add_argument("--message", default="")

    p_check = sub.add_parser("check")
    p_check.add_argument("--root", required=True)
    p_check.add_argument("--base-dir-name", required=True)
    p_check.add_argument("--tile-id", required=True)

    p_station = sub.add_parser("check-station")
    p_station.add_argument("--inputs-dir", required=True)

    args = parser.parse_args()
    if args.cmd == "write":
        write_tile_status(Path(args.root), args.base_dir_name, args.tile_id, args.status, args.stage, args.message)
    elif args.cmd == "check":
        # Exit 0 (permanent, skip) / 1 (not permanent, proceed) - stdout prints
        # "status stage" for the caller's own log line, or "none" if never attempted.
        permanent, status = is_permanently_unsolvable(Path(args.root), args.base_dir_name, args.tile_id)
        if status:
            print(f"{status.get('status')} {status.get('stage')}")
        else:
            print("none")
        raise SystemExit(0 if permanent else 1)
    elif args.cmd == "check-station":
        # Exit 0 (has a real station) / 1 (empty boundaries file - no station).
        raise SystemExit(0 if tile_has_station(Path(args.inputs_dir)) else 1)


if __name__ == "__main__":
    _cli()
