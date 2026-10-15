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
    python validation/compare_against_published.py [--out-csv path.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GBR_METRICS = Path("P:/11212688-004-global-floodmaps/modelling/validation/GBR/metrics_GBR_RP100_SLR_0.csv")
CAN_METRICS = Path("P:/11212688-004-global-floodmaps/modelling/validation/CAN/metrics_CAN_RP100_SLR_0.csv")
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
ENGLAND_CAVEAT = (
    "No England benchmark dataset exists yet in this repo (checked inputs/validation/ - only "
    "Wales/Scotland have NRW/SEPA data) - 'ours' is not yet computable; needs a new Environment "
    "Agency (or equivalent) dataset, the same kind of setup San Francisco Bay just went through."
)

# Kasmalkar et al. (2024) - published San Francisco Bay depth-accuracy.
KASMALKAR_SF_BAY = {
    "RMSE_all_m": 0.138, "MAE_all_m": 0.017,
    "RMSE_flooded_m": 0.525, "MAE_flooded_m": 0.319,
}


def _csi_rows() -> list[dict]:
    gbr_df = pd.read_csv(GBR_METRICS)
    can_df = pd.read_csv(CAN_METRICS)
    rows = []
    for study, region, published in PUBLISHED_CSI:
        if region == "england":
            rows.append({
                "study": study, "country_iso": "GBR", "region": region, "metric": "CSI",
                "ours": float("nan"), "published": published, "diff": float("nan"),
                "note": ENGLAND_CAVEAT,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-csv", default=str(Path(__file__).resolve().parent / "comparison_vs_published.csv"))
    args = parser.parse_args()

    df = pd.DataFrame(_csi_rows() + _sf_bay_rows())
    df["ratio_ours_over_published"] = df["ours"] / df["published"]

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print(df.drop(columns="note").to_string(index=False))

    df.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
