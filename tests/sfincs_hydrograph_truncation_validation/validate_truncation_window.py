"""Check every built tile's COAST-HG hydrograph peaks inside TRUNCATE_WINDOW_HR_DEFAULT.

COAST-HG shares one synthetic time axis (peak at t=74.5 h); a peak outside the
window would be silently truncated away. Skips when no built tiles are on disk.

Usage:
    python validate_truncation_window.py
    python validate_truncation_window.py --root "P:\\...\\modelling"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

from build_sfincs_tile import TRUNCATE_WINDOW_HR_DEFAULT  # noqa: E402

BASE_DIR_NAME = "calibration_roughness_SFINCS"


def find_corrected_hydrograph_files(root: Path) -> list[Path]:
    validation_dir = root / BASE_DIR_NAME
    if not validation_dir.exists():
        return []
    return sorted(validation_dir.glob("*/sfincs_model/corrected_hydrographs.csv"))


def check_peak_within_window(csv_path: Path, window_hr: tuple[float, float]) -> tuple[bool, str]:
    """Return (ok, detail); ok is False if any station peaks outside window_hr."""
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

    if root is None or not (root / BASE_DIR_NAME).exists():
        pytest.skip(f"no {BASE_DIR_NAME}/ data available at the expected root - "
                    "this check needs at least one real build_line_forcing.py output "
                    "(corrected_hydrographs.csv) to validate against")

    files = find_corrected_hydrograph_files(root)
    if not files:
        pytest.skip(f"{root / BASE_DIR_NAME} exists but no corrected_hydrographs.csv found under it yet")

    n_checked = 0
    failures: list[str] = []
    for f in files:
        try:
            ok, detail = check_peak_within_window(f, TRUNCATE_WINDOW_HR_DEFAULT)
        except Exception as e:  # partial file from an interrupted build
            failures.append(f"{f.parent.parent.name}: could not check ({e})")
            continue
        n_checked += 1
        if not ok:
            failures.append(f"{f.parent.parent.name}: {detail}")

    print(f"Checked {n_checked} tile(s) under {root / BASE_DIR_NAME}")
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
    try:
        test_all_available_tiles_peak_within_truncation_window(Path(args.root) if args.root else None)
    except pytest.skip.Exception as exc:
        print(f"SKIP: {exc}\n")
    print("Hydrograph truncation-window coverage check complete.")


if __name__ == "__main__":
    main()
