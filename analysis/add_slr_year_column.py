"""Adds a `year` column to `compute_delta_flood_extent.py`'s own output CSV:
the year at which each row's `ssp_scenario` median AR6 SLR trajectory first
reaches that row's own `slr_scenario` magnitude (relative to a 2020
baseline) - i.e. "under ssp126, when does global SLR reach 0.5m?".

Reads the ALREADY-PROCESSED trajectory CSV `extract_slr_trajectories.py`
writes (`visualization.slr_trajectories_csv`, confirmed present on disk -
no need to regenerate it), reusing that script's own `{SSP}_p50`-in-mm
convention directly rather than re-deriving anything from the raw AR6 files.

This is a standalone enrichment step, not folded into
`compute_delta_flood_extent.py` itself - re-running that script's own
expensive per-delta raster merge just to add one derived column would be
wasteful; this reads/writes the CSV only.

Reverse lookup (SLR magnitude -> year) is `np.interp` on the trajectory's
own (year, SLR) points, SWAPPED from the forward (year -> SLR) direction
the rest of the codebase uses (e.g. compute_exposure_analysis.py) - valid
because AR6 median SLR trajectories are monotonically increasing in time.
Explicitly NaN's (never silently clamps) any row whose target SLR exceeds
the trajectory's own last data point (2150) - a real, expected case: under
ssp126/ssp245 MEDIAN, SLR never reaches 1.0m within the dataset's own
2020-2150 range (ssp126 tops out at 632mm, ssp245 at 870mm by 2150), so
every SLR_1000/SLR_1500/SLR_2000 row under either scenario gets NaN here,
not a silently-wrong clamped year.

Usage:
    python add_slr_year_column.py
    python add_slr_year_column.py --csv path/to/delta_flood_extent.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
SSP_RCP_CODES_DEFAULT = {"SSP1": 126, "SSP2": 245, "SSP5": 585}


def year_for_slr(traj_df: pd.DataFrame, ssp_scenario: str, target_mm: float, ssp_rcp_codes: dict[str, int]) -> float:
    """Year at which `ssp_scenario`'s median AR6 trajectory first reaches
    `target_mm` (mm, relative to 2020) - NaN if that's beyond the
    trajectory's own last data point (out of range, not silently clamped)."""
    code = int(ssp_scenario.replace("ssp", ""))
    matches = [label for label, c in ssp_rcp_codes.items() if c == code]
    if not matches:
        return float("nan")
    col = f"{matches[0]}_p50"
    if col not in traj_df.columns:
        return float("nan")
    traj_years = traj_df.index.to_numpy(dtype=float)
    traj_mm = traj_df[col].to_numpy(dtype=float)
    if target_mm > traj_mm.max():
        return float("nan")
    return float(np.interp(target_mm, traj_mm, traj_years))


def add_year_column(df: pd.DataFrame, traj_df: pd.DataFrame, ssp_rcp_codes: dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    target_mm = df["slr_scenario"].str.split("_").str[1].astype(float)
    df["year_slr_reached_median"] = [
        year_for_slr(traj_df, ssp, mm, ssp_rcp_codes)
        for ssp, mm in zip(df["ssp_scenario"], target_mm)
    ]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--csv", default=None, help="default: {root}/deltas_floodmaps/delta_flood_extent.csv")
    parser.add_argument("--slr-trajectories-csv", default=None,
                         help="default: visualization.slr_trajectories_csv from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    csv_path = Path(args.csv) if args.csv else root / "deltas_floodmaps" / "delta_flood_extent.csv"
    viz = cfg.get("visualization", {})
    traj_path = Path(args.slr_trajectories_csv) if args.slr_trajectories_csv else Path(
        viz.get("slr_trajectories_csv", root / "processed_inputs" / "slr_trajectories_global_median.csv")
    )
    ssp_rcp_codes = viz.get("ssp_rcp_codes", SSP_RCP_CODES_DEFAULT)

    df = pd.read_csv(csv_path)
    traj_df = pd.read_csv(traj_path, index_col="year")
    print(f"{len(df)} row(s) from {csv_path}")
    print(f"SLR trajectory data: {traj_path} (years {traj_df.index.min()}-{traj_df.index.max()})")

    df = add_year_column(df, traj_df, ssp_rcp_codes)
    n_out_of_range = df["year_slr_reached_median"].isna().sum()
    print(f"{n_out_of_range}/{len(df)} row(s) have a target SLR beyond the trajectory's "
          f"{traj_df.index.max()} end-year under their own ssp_scenario median - left NaN (not clamped)")

    retry_transient_io(df.to_csv, csv_path, index=False)
    print(f"Wrote {csv_path} (added year_slr_reached_median column)")


if __name__ == "__main__":
    main()
