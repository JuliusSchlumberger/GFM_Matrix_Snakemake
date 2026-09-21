"""Validate that build_sfincs_tile.py's TRUNCATE_WINDOW_HR_DEFAULT actually
covers the real, current COAST-HG corrected-hydrograph peak, for EVERY
tile's own corrected_hydrographs.csv already on disk under
validation_sfincs/ - not just the 4 tiles (2335, 1573, 1907, 929) whose
agreement on this originally justified the constant.

build_sfincs_tile.py's own comment on TRUNCATE_WINDOW_HR_DEFAULT=(40, 110)
states this window is "a property of the dataset" (every COAST-HG
hydrograph shares the same synthetic time axis, peak always at t=74.5h),
confirmed only across 4 real tiles at the time. Now that up to 255 tiles
are being built at production scale, resting on n=4 confirmation is a real,
if modest, extrapolation risk - a hydrograph whose peak falls outside the
window would be silently truncated to a scenario that never reaches its
own real storm peak, without any error or warning.

This is a DATA-validation test, not a pure-logic one: it needs real
corrected_hydrographs.csv files on disk (from a real build_boundary_forcing.py
run) to check anything meaningful, so it's the one test in this batch that
degrades to a documented skip rather than a hard failure when no tiles have
been built yet (e.g. a fresh checkout, before any local --build-only run).

Usage:
    python validate_truncation_window.py
    python validate_truncation_window.py --root "P:\\...\\modelling"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_sfincs_tile import TRUNCATE_WINDOW_HR_DEFAULT  # noqa: E402


def find_corrected_hydrograph_files(root: Path) -> list[Path]:
    validation_dir = root / "validation_sfincs"
    if not validation_dir.exists():
        return []
    return sorted(validation_dir.glob("*/sfincs_model/corrected_hydrographs.csv"))


def check_peak_within_window(csv_path: Path, window_hr: tuple[float, float]) -> tuple[bool, str]:
    """Returns (ok, detail) - ok=False means at least one station's own
    peak falls outside window_hr (the truncated series would silently
    never reach that station's real storm peak)."""
    df = pd.read_csv(csv_path)
    station_cols = [c for c in df.columns if c != "elapsed_hr"]
    if not station_cols:
        return True, "no station columns (empty file, nothing to check)"

    t_start, t_end = window_hr
    problems = []
    for col in station_cols:
        peak_idx = df[col].idxmax()
        peak_hr = float(df.loc[peak_idx, "elapsed_hr"])
        if not (t_start <= peak_hr <= t_end):
            problems.append(f"station {col}: peak at t={peak_hr:.1f}h, outside [{t_start},{t_end}]h")
    if problems:
        return False, "; ".join(problems)
    return True, f"{len(station_cols)} station(s), all peaks within [{t_start},{t_end}]h"


def test_all_available_tiles_peak_within_truncation_window(root: Path | None = None) -> None:
    print("=== TRUNCATE_WINDOW_HR_DEFAULT: real on-disk hydrographs peak inside the window ===")
    if root is None:
        candidates = [
            Path(r"P:\11212688-004-global-floodmaps\modelling"),
        ]
        root = next((c for c in candidates if c.exists()), None)

    if root is None or not (root / "validation_sfincs").exists():
        print("SKIP: no validation_sfincs/ data available at the expected root - "
              "this check needs at least one real build_boundary_forcing.py output "
              "(corrected_hydrographs.csv) to validate against. Not a failure, just "
              "nothing to check in this environment.")
        print()
        return

    files = find_corrected_hydrograph_files(root)
    if not files:
        print(f"SKIP: {root / 'validation_sfincs'} exists but no corrected_hydrographs.csv found under it yet.")
        print()
        return

    n_checked = 0
    failures: list[str] = []
    for f in files:
        try:
            ok, detail = check_peak_within_window(f, TRUNCATE_WINDOW_HR_DEFAULT)
        except Exception as e:  # a malformed/partial file from an interrupted build - report, don't crash the batch
            failures.append(f"{f.parent.parent.name}: could not check ({e})")
            continue
        n_checked += 1
        if not ok:
            failures.append(f"{f.parent.parent.name}: {detail}")

    print(f"Checked {n_checked} tile(s) under {root / 'validation_sfincs'}")
    assert not failures, (
        f"{len(failures)} tile(s) have a hydrograph peak outside "
        f"TRUNCATE_WINDOW_HR_DEFAULT={TRUNCATE_WINDOW_HR_DEFAULT} - the truncation window "
        f"may need widening, or these tiles' forcing needs investigating:\n  " + "\n  ".join(failures)
    )
    print(f"PASS: all {n_checked} checked tile(s)' hydrograph peaks fall within "
          f"TRUNCATE_WINDOW_HR_DEFAULT={TRUNCATE_WINDOW_HR_DEFAULT}")
    print()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="modelling root (default: auto-detect the real P:\\ share)")
    args = parser.parse_args()
    test_all_available_tiles_peak_within_truncation_window(Path(args.root) if args.root else None)
    print("Hydrograph truncation-window coverage check complete.")


if __name__ == "__main__":
    main()
