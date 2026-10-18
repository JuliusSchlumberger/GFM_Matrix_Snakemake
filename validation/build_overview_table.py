"""Build overview tables of our HR/FAR/bias/CSI next to published CSI per region.

Also writes a SF Bay RMSE/MAE table against Kasmalkar et al. (2024). Only reshapes
CSVs written by the other validation scripts; no new scoring.

Usage:
    python validation/build_overview_table.py [--config path/to/config.yml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config_utils import load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_against_published import (  # noqa: E402
    KASMALKAR_SF_BAY,
    PUBLISHED_CSI,
    england_fz2_band_path,
    england_fz3_results,
    sf_bay_results_path,
)

OUT_DIR = Path(__file__).resolve().parent

# (label, (source, region), ranged). ranged=True: the benchmark's native
# ~200yr RP is bracketed by RP100/RP250, so one row per RP. New Brunswick has no RP250
# run. England comes from its rescore files: metrics_GBR_{RP}_{SLR}.csv "england" rows are stale.
_REGIONS = [
    ("Wales (whole)", ("gbr", "wales"), True),
    ("Wales - Severn Estuary", ("gbr", "severn_estuary"), True),
    ("Wales - Menai Strait", ("gbr", "menai_strait"), True),
    ("Scotland (whole)", ("gbr", "scotland"), True),
    ("Scotland - Firth of Forth", ("gbr", "firth_of_forth"), True),
    ("New Brunswick - Cumberland Basin", ("can", "cumberland_basin"), False),
    ("New Brunswick - Petitcodiac", ("can", "petitcodiac"), False),
]
_NB_NOTE = "known COAST-RP/GTSM tidal-resonance limitation at the Bay of Fundy head, not a pipeline bug"
_FZ2_NOTE = "thin incremental band (0.1-0.5% AEP) - positional sensitivity, not a source-filter issue"
def _published(region: str) -> list[tuple[str, float]]:
    return [(study, csi) for study, r, csi in PUBLISHED_CSI if r == region]


def _fmt_published(pairs: list[tuple[str, float]]) -> str:
    if not pairs:
        return "-"
    return "; ".join(f"{study}: {val:.2f}" for study, val in pairs)


def build_extent_table(
    gbr_metrics_rp100: Path, gbr_metrics_rp250: Path, can_metrics: Path,
    fz3: pd.DataFrame, england_fz2_band: Path, sf_bay_csv: Path,
) -> pd.DataFrame:
    """Build the HR/FAR/bias/CSI (ours) + published-CSI table, one row per region/RP."""
    gbr_rp100 = pd.read_csv(gbr_metrics_rp100).set_index("region")
    gbr_rp250 = pd.read_csv(gbr_metrics_rp250).set_index("region")
    can = pd.read_csv(can_metrics).set_index("region")
    fz2 = pd.read_csv(england_fz2_band)
    sf_bay = pd.read_csv(sf_bay_csv, index_col=0)

    rows = []
    for label, (src, region), ranged in _REGIONS:
        published = _published(region)
        if src == "can":
            r = can.loc[region]
            rows.append({
                "Region": label, "HR": r["HR"], "FAR": r["FAR"], "bias": r["bias"], "CSI": r["CSI"],
                "Published CSI": _fmt_published(published),
                "Note": _NB_NOTE if region in ("cumberland_basin", "petitcodiac") else "",
            })
            continue
        for rp_label, df in [("RP100", gbr_rp100), ("RP250", gbr_rp250)]:
            r = df.loc[region]
            rows.append({
                "Region": f"{label} ({rp_label})" if ranged else label,
                "HR": r["HR"], "FAR": r["FAR"], "bias": r["bias"], "CSI": r["CSI"],
                "Published CSI": _fmt_published(published), "Note": "",
            })

    for _, r in fz3.iterrows():
        rows.append({
            "Region": f"England - Flood Zone 3 ({r['return_period']})",
            "HR": r["HR"], "FAR": r["FAR"], "bias": r["bias"], "CSI": r["CSI"],
            "Published CSI": _fmt_published(_published("england")), "Note": "",
        })
    r = fz2.iloc[0]
    rows.append({
        "Region": f"England - Flood Zone 2 ({r['band_inner_rp']}-{r['band_outer_rp']} band)",
        "HR": r["HR"], "FAR": r["FAR"], "bias": r["bias"], "CSI": r["CSI"],
        "Published CSI": _fmt_published(_published("england")), "Note": _FZ2_NOTE,
    })

    for direction in sf_bay.index:
        r = sf_bay.loc[direction]
        rows.append({
            "Region": f"San Francisco Bay ({direction})",
            "HR": r["HT"], "FAR": r["FAR"], "bias": r["bias"], "CSI": r["CSI"],
            "Published CSI": "-", "Note": "",
        })

    return pd.DataFrame(rows)


def build_sf_bay_depth_table(sf_bay_csv: Path) -> pd.DataFrame:
    """Build the SF Bay RMSE/MAE (m) table against Kasmalkar et al. (2024)."""
    sf_bay = pd.read_csv(sf_bay_csv, index_col=0)
    published = KASMALKAR_SF_BAY
    rows = []
    for direction in sf_bay.index:
        r = sf_bay.loc[direction]
        rows.append({
            "Direction": direction,
            "RMSE all (ours)": r["RMSE_all_m"], "RMSE all (Kasmalkar 2024)": published["RMSE_all_m"],
            "MAE all (ours)": r["MAE_all_m"], "MAE all (Kasmalkar 2024)": published["MAE_all_m"],
            "RMSE flooded (ours)": r["RMSE_flooded_m"], "RMSE flooded (Kasmalkar 2024)": published["RMSE_flooded_m"],
            "MAE flooded (ours)": r["MAE_flooded_m"], "MAE flooded (Kasmalkar 2024)": published["MAE_flooded_m"],
        })
    return pd.DataFrame(rows)


def main(config_path: str | Path | None = None) -> None:
    """Print both overview tables and write them as CSV next to this script."""
    cfg = load_config(config_path or _REPO_ROOT / "snakemake_workflow" / "config" / "config.yml")
    validation_dir = Path(cfg["validation"]["output_dir"])
    slr = cfg["validation"]["waterlevel_name"]
    gbr_metrics_rp100 = validation_dir / "GBR" / "metrics_GBR_RP100_SLR_0.csv"
    gbr_metrics_rp250 = validation_dir / "GBR" / "metrics_GBR_RP250_SLR_0.csv"
    can_metrics = validation_dir / "CAN" / "metrics_CAN_RP100_SLR_0.csv"
    fz3 = england_fz3_results(validation_dir, slr)
    england_fz2_band = england_fz2_band_path(validation_dir, slr)
    sf_bay_csv = sf_bay_results_path(validation_dir, slr)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)

    extent_df = build_extent_table(
        gbr_metrics_rp100, gbr_metrics_rp250, can_metrics, fz3, england_fz2_band, sf_bay_csv,
    )
    print("=== HR / FAR / bias / CSI (ours), CSI (published where available) ===")
    print(extent_df.drop(columns="Note").round(3).to_string(index=False))
    notes = extent_df[extent_df["Note"] != ""][["Region", "Note"]]
    if not notes.empty:
        print("\nNotes:")
        for _, r in notes.iterrows():
            print(f"  - {r['Region']}: {r['Note']}")
    extent_out = OUT_DIR / "overview_extent_metrics.csv"
    extent_df.to_csv(extent_out, index=False)
    print(f"\nwrote {extent_out}")

    sf_bay_df = build_sf_bay_depth_table(sf_bay_csv)
    print("\n=== San Francisco Bay depth accuracy (RMSE/MAE, m) ===")
    print(sf_bay_df.round(3).to_string(index=False))
    sf_bay_out = OUT_DIR / "overview_sf_bay_depth.csv"
    sf_bay_df.to_csv(sf_bay_out, index=False)
    print(f"\nwrote {sf_bay_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()
    main(config_path=args.config)
