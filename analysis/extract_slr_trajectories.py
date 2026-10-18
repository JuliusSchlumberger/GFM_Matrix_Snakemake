"""Extract global-mean AR6 SLR trajectories (mm, relative to 2020) per SSP.

Reads SLR_{code}_wg1.csv from the `ipcc_ar6_slr_wg1_csv` source, with codes from
visualization.ssp_rcp_codes, and writes columns {SSP}_p17/_p50/_p83 indexed by year.

Usage:
    python analysis/extract_slr_trajectories.py \\
        [--config snakemake_workflow/config/config.yml] \\
        [--out <processed_inputs>/slr_trajectories_global_median.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_slr_csv(csv_path: Path) -> pd.DataFrame:
    """Read one SLR_XXX_wg1.csv as p17/p50/p83 in mm, relative to 2020.

    Input: year index, percentile columns '0.17'/'0.5'/'0.83' in metres. Falls back to
    the first year as baseline if 2020 is missing; values with max |p50| < 10 are taken
    to be metres.
    """
    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = "year"
    df.index = df.index.astype(int)

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

    keep = [c for c in ("p17", "p50", "p83") if c in df.columns]
    df = df[keep].copy()

    if 2020 in df.index:
        df = df - df.loc[2020]
    else:
        df = df - df.iloc[0]

    if df["p50"].abs().max() < 10:
        df = df * 1000.0

    return df


def main() -> None:
    _default_cfg = str(Path(__file__).resolve().parent.parent / "snakemake_workflow" / "config" / "config.yml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_default_cfg, help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument(
        "--out", default=None,
        help="output CSV path (default: visualization.slr_trajectories_csv from config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    viz = cfg["visualization"]

    if args.out is None:
        args.out = viz["slr_trajectories_csv"]
    ssp_rcp_codes: dict[str, int] = viz["ssp_rcp_codes"]
    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    slr_csv_dir = Path(catalog.get_source("ipcc_ar6_slr_wg1_csv").path)

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
        df_ssp.columns = [f"{ssp}_{col}" for col in df_ssp.columns]
        frames[ssp] = df_ssp

    if not frames:
        print("ERROR: no SLR CSV files were loaded.")
        sys.exit(1)

    result = pd.concat(frames.values(), axis=1)
    result.index.name = "year"
    result = result.sort_index()

    out_path = Path(args.out)
    retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
    result.to_csv(out_path)
    print(f"\nWritten: {out_path}")

    print("\nMedian SLR (mm, relative to 2020 baseline):  [selected years]")
    p50_cols = [c for c in result.columns if c.endswith("_p50")]
    sample_years = [yr for yr in [2050, 2100, 2150] if yr in result.index]
    print(result.loc[sample_years, p50_cols].round(1).to_string())


if __name__ == "__main__":
    main()
