"""Compare our validation metrics against published studies as a long/tidy CSV.

CSI vs Bates et al. (2023), Wing et al. (2024), Chaudhuri et al. (2026); SF Bay
RMSE/MAE vs Kasmalkar et al. (2024). Published values are transcribed, not re-derived.

Usage:
    python validation/compare_against_published.py [--config path/to/config.yml] [--out path.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config_utils import load_config  # noqa: E402

SF_BAY_RESULTS = Path(__file__).resolve().parent / "sf_bay_results.csv"

# "wales"/"england" = whole territory; other regions are Chaudhuri et al.'s study areas.
PUBLISHED_CSI = [
    ("Bates et al. (2023)", "wales", 0.76),
    ("Wing et al. (2024)", "wales", 0.82),
    ("Chaudhuri et al. (2026)", "severn_estuary", 0.70),
    ("Chaudhuri et al. (2026)", "menai_strait", 0.49),
    ("Chaudhuri et al. (2026)", "firth_of_forth", 0.75),
    ("Chaudhuri et al. (2026)", "cumberland_basin", 0.87),
    ("Chaudhuri et al. (2026)", "petitcodiac", 0.56),
    ("Wing et al. (2024)", "england", 0.62),
]
NB_CAVEAT = (
    "Known limitation: the COAST-RP/GTSM boundary forcing does not capture the Bay of Fundy's "
    "extreme tidal resonance; the near-zero CSI here is attributed to that, not to a pipeline bug."
)
ENGLAND_FZ2_CAVEAT = (
    "Flood Zone 2 (sea) is the incremental 0.1%-0.5% AEP band (1-in-1000 to 1-in-200), not the "
    "cumulative 1-in-1000 extent, so it is compared against the model's RP1000-minus-RP100 band "
    "(RP100 is the nearest native RP below the ~RP200 inner edge). CSI stays low, plausibly "
    "because a thin band is highly sensitive to small positional offsets between model and benchmark."
)

KASMALKAR_SF_BAY = {
    "RMSE_all_m": 0.138, "MAE_all_m": 0.017,
    "RMSE_flooded_m": 0.525, "MAE_flooded_m": 0.319,
}


def _csi_rows(gbr_metrics: Path, can_metrics: Path, england_fz3_rescore: Path, england_fz2_band: Path) -> list[dict]:
    gbr_df = pd.read_csv(gbr_metrics)
    can_df = pd.read_csv(can_metrics)
    fz3_df = pd.read_csv(england_fz3_rescore)
    fz2_band_df = pd.read_csv(england_fz2_band) if england_fz2_band.exists() else None
    rows = []
    for study, region, published in PUBLISHED_CSI:
        if region == "england":
            # FZ3 (~200yr) is bracketed by RP100/RP250; both reported. FZ3 comes from
            # rescore_england_fz3.py: multi_rp_summary.csv FZ3 rows use the stale
            # sea-only benchmark. FZ2 is the incremental RP100-RP1000 band.
            fz3 = fz3_df[fz3_df["benchmark_key"] == "england_ea_floodzone3_sea"]
            for _, r in fz3.iterrows():
                rows.append({
                    "study": study, "country_iso": "GBR", "region": f"england ({r['return_period']}, FZ3)",
                    "metric": "CSI", "ours": float(r["CSI"]), "published": published,
                    "diff": float(r["CSI"]) - published, "note": "",
                })
            if fz2_band_df is not None and not fz2_band_df.empty:
                r = fz2_band_df.iloc[0]
                rows.append({
                    "study": study, "country_iso": "GBR",
                    "region": f"england ({r['band_inner_rp']}-{r['band_outer_rp']} band, FZ2)",
                    "metric": "CSI", "ours": float(r["CSI"]), "published": published,
                    "diff": float(r["CSI"]) - published, "note": ENGLAND_FZ2_CAVEAT,
                })
            continue
        df, iso = (can_df, "CAN") if region in ("cumberland_basin", "petitcodiac") else (gbr_df, "GBR")
        match = df[df["region"] == region]
        if match.empty:
            continue
        ours = float(match.iloc[0]["CSI"])
        rows.append({
            "study": study, "country_iso": iso, "region": region, "metric": "CSI",
            "ours": ours, "published": published, "diff": ours - published,
            "note": NB_CAVEAT if region in ("cumberland_basin", "petitcodiac") else "",
        })
    return rows


def _sf_bay_rows() -> list[dict]:
    df = pd.read_csv(SF_BAY_RESULTS, index_col=0)
    rows = []
    for direction in df.index:
        for metric, published in KASMALKAR_SF_BAY.items():
            ours = float(df.loc[direction, metric])
            rows.append({
                "study": "Kasmalkar et al. (2024)",
                "country_iso": "USA",
                "region": f"san_francisco_bay ({direction})",
                "metric": metric,
                "ours": ours,
                "published": published,
                "diff": ours - published,
                "note": "",
            })
        for metric, col in [("HR", "HT"), ("FAR", "FAR"), ("CSI", "CSI"), ("bias", "bias")]:
            rows.append({
                "study": "(this model only - no published value given)",
                "country_iso": "USA",
                "region": f"san_francisco_bay ({direction})",
                "metric": metric,
                "ours": float(df.loc[direction, col]),
                "published": float("nan"),
                "diff": float("nan"),
                "note": "",
            })
    return rows


def main(config_path: str | Path | None = None, out_csv: str | Path | None = None) -> None:
    """Print and write the published-comparison table."""
    cfg = load_config(config_path or _REPO_ROOT / "snakemake_workflow" / "config" / "config.yml")
    validation_dir = Path(cfg["validation"]["output_dir"])
    gbr_metrics = validation_dir / "GBR" / "metrics_GBR_RP100_SLR_0.csv"
    can_metrics = validation_dir / "CAN" / "metrics_CAN_RP100_SLR_0.csv"
    england_fz3_rescore = validation_dir / "england_fz3_rescore.csv"
    england_fz2_band = validation_dir / "GBR" / "england_fz2_band_result.csv"
    if out_csv is None:
        out_csv = Path(__file__).resolve().parent / "comparison_vs_published.csv"

    df = pd.DataFrame(_csi_rows(gbr_metrics, can_metrics, england_fz3_rescore, england_fz2_band) + _sf_bay_rows())
    df["ratio_ours_over_published"] = df["ours"] / df["published"]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print(df.drop(columns="note").to_string(index=False))

    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--out", "--out-csv", dest="out_csv", default=str(Path(__file__).resolve().parent / "comparison_vs_published.csv"))
    args = parser.parse_args()
    main(config_path=args.config, out_csv=args.out_csv)
