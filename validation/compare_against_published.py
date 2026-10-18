"""Consolidate this model's own validation output against two published
benchmark studies:

  Chaudhuri et al. (2026) - per-sub-region CSI for Wales/Scotland/New
    Brunswick, from validate_country.py's own metrics_{ISO}_{RP}_{SLR}.csv
    (severn_estuary, menai_strait, firth_of_forth, cumberland_basin,
    petitcodiac - see data_catalog_validation.yml's own `regions:` blocks).
  Kasmalkar et al. (2024) - San Francisco Bay depth-accuracy (RMSE/MAE, all
    pixels and flooded-pixels-only), from validate_sf_bay.py's own
    sf_bay_results.csv (both the "coarsen to model grid" and "refine to 2m"
    directions - see that script's own docstring).

Published numbers below are transcribed from the user directly (2026-10-15),
not re-derived from either paper - if either paper is revised, update here.

Usage:
    python validation/compare_against_published.py [--config path/to/config.yml] [--out-csv path.csv]
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

# Published CSI, per (study, region). Region "wales"/"scotland" (no suffix) is
# the WHOLE-TERRITORY row; the named sub-regions are Chaudhuri et al.'s own
# study-area subsets within each territory (see data_catalog_validation.yml's
# `regions:` blocks). One region can have more than one published study to
# compare against (Wales as a whole: both Bates et al. and Wing et al.).
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
    "Known, already-documented limitation (not a new finding) - data_catalog_validation.yml's "
    "own comments attribute Bay of Fundy's near-zero CSI to a COAST-RP/GTSM boundary-forcing "
    "limitation capturing the Fundy's extreme tidal resonance, not a pipeline bug."
)
ENGLAND_FZ2_CAVEAT = (
    "CONFIRMED (2026-10-16, EA Flood Zones Product Description via environment.data.gov.uk FAQ "
    "778338325): Flood Zone 2 (sea) IS the INCREMENTAL 0.1%-0.5% AEP band (1-in-1000 to 1-in-200), "
    "not the cumulative 1-in-1000 extent - comparing it against the model's cumulative RP1000 extent "
    "(what the naive comparison below does) gave bias=26.2x/CSI=0.023, a benchmark-definition "
    "mismatch, not a real model failure. validate_england_fz2_band.py re-scores against the model's "
    "own matching incremental band (RP1000 minus RP100 - RP100 is the model's nearest native RP below "
    "the benchmark's real ~RP200 inner edge) instead: bias drops to a sane 2.4x (confirming the "
    "cumulative-vs-incremental mismatch was the dominant driver of the 26x figure), but CSI stays low "
    "(0.020, HR=0.066) - plausibly because a thin coastal band is highly sensitive to small positional "
    "offsets between model and benchmark, but this is NOT independently confirmed; treat the band "
    "result as a real, methodologically-sound number, not yet a fully-explained one."
)

# Kasmalkar et al. (2024) - published San Francisco Bay depth-accuracy.
KASMALKAR_SF_BAY = {
    "RMSE_all_m": 0.138, "MAE_all_m": 0.017,
    "RMSE_flooded_m": 0.525, "MAE_flooded_m": 0.319,
}


def _csi_rows(gbr_metrics: Path, can_metrics: Path, england_fz3_rescore: Path, england_fz2_band: Path) -> list[dict]:
    """Build the per-(study, region) CSI comparison rows.

    Parameters
    ----------
    gbr_metrics : Path
        ``metrics_GBR_RP100_SLR_0.csv`` from ``validate_country.py``.
    can_metrics : Path
        ``metrics_CAN_RP100_SLR_0.csv`` from ``validate_country.py``.
    england_fz3_rescore : Path
        ``england_fz3_rescore.csv`` from ``_rescore_england_fz3.py``.
    england_fz2_band : Path
        ``england_fz2_band_result.csv`` from ``validate_england_fz2_band.py``.

    Returns
    -------
    list of dict
        One row per (study, region) CSI comparison.
    """
    gbr_df = pd.read_csv(gbr_metrics)
    can_df = pd.read_csv(can_metrics)
    fz3_df = pd.read_csv(england_fz3_rescore)
    fz2_band_df = pd.read_csv(england_fz2_band) if england_fz2_band.exists() else None
    rows = []
    for study, region, published in PUBLISHED_CSI:
        if region == "england":
            # England has no exact native-RP match for FZ3 (~200yr, bracketed
            # RP100/RP250, same as Wales/Scotland) - report both bracket
            # points against Wing et al.'s single published value. FZ3 comes
            # from _rescore_england_fz3.py's own output, NOT
            # multi_rp_summary.csv - that file's own FZ3 rows are from the
            # pre-correction (sea-only, no estuarine 'river and sea') benchmark
            # and were never regenerated (the run that would have refreshed
            # them was killed for stalling/memory-spiking - see
            # _rescore_england_fz3.py's own docstring). FZ2 is reported as the
            # model's own incremental RP100-RP1000 band vs the benchmark's own
            # incremental band (validate_england_fz2_band.py's output), NOT a
            # naive cumulative-RP1000-vs-benchmark comparison - see
            # ENGLAND_FZ2_CAVEAT.
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
        # HR/FAR/CSI/bias - no published counterpart given, reported as this
        # model's own confusion-matrix result at the honest surveyed-area
        # domain (see validate_sf_bay.py's own domain-construction note).
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
    """Build, print, and write the published-comparison table.

    Parameters
    ----------
    config_path : str, Path, or None, optional
        Path to config.yml, used to resolve the GBR/CAN/England input CSVs
        under ``validation.output_dir``. Defaults to this repo's
        ``snakemake_workflow/config/config.yml``.
    out_csv : str, Path, or None, optional
        Destination for the comparison CSV. Defaults to
        ``comparison_vs_published.csv`` next to this script.
    """
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
    parser.add_argument("--out-csv", default=str(Path(__file__).resolve().parent / "comparison_vs_published.csv"))
    args = parser.parse_args()
    main(config_path=args.config, out_csv=args.out_csv)
