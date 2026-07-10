"""Extract global-mean SLR trajectories from pre-computed per-SSP CSV files.

Reads the SLR_{code}_wg1.csv files in the `ipcc_ar6_slr_wg1_csv` catalog
source's directory, normalises each series so the 2020 value is zero, and
writes a combined CSV with one column per SSP (median p50) plus optional
uncertainty bands (p17, p83).

The SSP->RCP-code mapping is read from `visualization.ssp_rcp_codes` in
config.yml (e.g. SSP1->126, SSP2->245, SSP5->585).

Output CSV columns:
  year        — integer year
  {SSP}_p17   — 17th-percentile SLR in mm (relative to 2020 baseline)
  {SSP}_p50   — median SLR in mm
  {SSP}_p83   — 83rd-percentile SLR in mm

Usage:
    python snakemake_workflow/analysis/extract_slr_trajectories.py \\
        [--config snakemake_workflow/config/config.yml] \\
        [--output D:/GFM/processed_inputs/slr_trajectories_global_median.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_slr_csv(csv_path: Path) -> pd.DataFrame:
    """Read one SLR_XXX_wg1.csv file.

    Expected format:
        index column: years (integer)
        data columns: '0.17', '0.5', '0.83'  — percentile SLR in metres
                      (some files may include more percentiles; we keep those three).

    Returns a DataFrame indexed by year with columns p17, p50, p83 in millimetres,
    normalised so the 2020 median is 0.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "year"
    df.index = df.index.astype(int)

    # Rename percentile columns robustly
    rename = {}
    for col in df.columns:
        col_str = str(col).strip()
        if col_str in ("0.17", ".17"):
            rename[col] = "p17"
        elif col_str in ("0.5", ".5", "0.50"):
            rename[col] = "p50"
        elif col_str in ("0.83", ".83"):
            rename[col] = "p83"
    df = df.rename(columns=rename)

    # Keep only the three quantile columns
    keep = [c for c in ("p17", "p50", "p83") if c in df.columns]
    df = df[keep].copy()

    # Normalise: subtract 2020 baseline so SLR starts at 0
    if 2020 in df.index:
        df = df - df.loc[2020]
    else:
        # Use the first available year as baseline
        df = df - df.iloc[0]

    # Convert metres -> millimetres if the values look like metres
    if df["p50"].abs().max() < 10:
        df = df * 1000.0

    return df


def main() -> None:
    _default_cfg = str(Path(__file__).resolve().parent.parent / "config" / "config.yml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_default_cfg, help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument(
        "--output", default=None,
        help="output CSV path (default: visualization.slr_trajectories_csv from config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    viz = cfg.get("visualization", {})

    if args.output is None:
        args.output = viz.get(
            "slr_trajectories_csv",
            f"{cfg['paths']['root']}/processed_inputs/slr_trajectories_global_median.csv",
        )
    ssp_rcp_codes: dict[str, int] = viz.get("ssp_rcp_codes", {
        "SSP1": 126, "SSP2": 245, "SSP5": 585,
    })
    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"])
    slr_csv_dir = Path(catalog.get_source("ipcc_ar6_slr_wg1_csv").path)  # catalog key (data_catalog_gfm.yml)

    if not slr_csv_dir.exists():
        print(f"ERROR: SLR CSV directory not found: {slr_csv_dir}")
        sys.exit(1)

    frames: dict[str, pd.DataFrame] = {}
    for ssp, code in sorted(ssp_rcp_codes.items()):
        csv_path = slr_csv_dir / f"SLR_{code}_wg1.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path.name} not found — skipping {ssp}.")
            continue
        print(f"  Reading {csv_path.name} -> {ssp}...")
        df_ssp = load_slr_csv(csv_path)
        # Prefix each column with SSP label
        df_ssp.columns = [f"{ssp}_{col}" for col in df_ssp.columns]
        frames[ssp] = df_ssp

    if not frames:
        print("ERROR: no SLR CSV files were loaded.")
        sys.exit(1)

    result = pd.concat(frames.values(), axis=1)
    result.index.name = "year"
    result = result.sort_index()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path)
    print(f"\nWritten: {out_path}")

    # Quick summary
    print("\nMedian SLR (mm, relative to 2020 baseline):  [selected years]")
    p50_cols = [c for c in result.columns if c.endswith("_p50")]
    sample_years = [yr for yr in [2050, 2100, 2150] if yr in result.index]
    print(result.loc[sample_years, p50_cols].round(1).to_string())


if __name__ == "__main__":
    main()
