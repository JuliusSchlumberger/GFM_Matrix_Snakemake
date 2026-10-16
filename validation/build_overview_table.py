"""Final overview table: this model's own HR/FAR/bias/CSI per region/benchmark,
alongside each published study's CSI where one exists - plus a second,
San-Francisco-Bay-only table with RMSE/MAE (all-pixels and flooded-pixels,
both comparison directions) against Kasmalkar et al. (2024).

Reuses every number already computed by validate_country.py (GBR/CAN
metrics_*.csv), _rescore_england_fz3.py, validate_england_fz2_band.py and
validate_sf_bay.py - no new scoring, just reshaped for reporting. See
compare_against_published.py for the long/tidy equivalent this draws the
same underlying numbers from (published values transcribed from the user,
2026-10-15/16 - update PUBLISHED_CSI/KASMALKAR_SF_BAY if either paper is
revised).

Usage:
    python validation/build_overview_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

GBR_METRICS_RP100 = Path("P:/11212688-004-global-floodmaps/modelling/validation/GBR/metrics_GBR_RP100_SLR_0.csv")
GBR_METRICS_RP250 = Path("P:/11212688-004-global-floodmaps/modelling/validation/GBR/metrics_GBR_RP250_SLR_0.csv")
CAN_METRICS = Path("P:/11212688-004-global-floodmaps/modelling/validation/CAN/metrics_CAN_RP100_SLR_0.csv")
ENGLAND_FZ3_RESCORE = Path("P:/11212688-004-global-floodmaps/modelling/validation/england_fz3_rescore.csv")
ENGLAND_FZ2_BAND = Path("P:/11212688-004-global-floodmaps/modelling/validation/GBR/england_fz2_band_result.csv")
SF_BAY_RESULTS = Path(__file__).resolve().parent / "sf_bay_results.csv"
OUT_DIR = Path(__file__).resolve().parent

# (label, lookup, published, ranged) - lookup is (source_df_key, region_value).
# ranged=True regions have a native RP100/RP250 bracket (same ~200yr-vs-
# native-RP mismatch as England's Flood Zone 3 - see data_catalog_validation.yml's
# own comparison_return_periods on each of these benchmarks) and get TWO rows,
# one per RP, same convention as England's own FZ3 rows below. New Brunswick's
# benchmark has no comparison_return_periods configured (no RP250 run exists),
# so it stays single-RP100. England itself is built separately from its own
# dedicated rescore files - see compare_against_published.py's own note on why
# metrics_GBR_*.csv's "england" rows are stale.
_REGIONS = [
    ("Wales (whole)", ("gbr", "wales"), [("Bates et al. (2023)", 0.76), ("Wing et al. (2024)", 0.82)], True),
    ("Wales - Severn Estuary", ("gbr", "severn_estuary"), [("Chaudhuri et al. (2026)", 0.70)], True),
    ("Wales - Menai Strait", ("gbr", "menai_strait"), [("Chaudhuri et al. (2026)", 0.49)], True),
    ("Scotland (whole)", ("gbr", "scotland"), [], True),
    ("Scotland - Firth of Forth", ("gbr", "firth_of_forth"), [("Chaudhuri et al. (2026)", 0.75)], True),
    ("New Brunswick - Cumberland Basin", ("can", "cumberland_basin"), [("Chaudhuri et al. (2026)", 0.87)], False),
    ("New Brunswick - Petitcodiac", ("can", "petitcodiac"), [("Chaudhuri et al. (2026)", 0.56)], False),
]
_NB_NOTE = "known COAST-RP/GTSM tidal-resonance limitation at the Bay of Fundy head, not a pipeline bug"
_FZ2_NOTE = "thin incremental band (0.1-0.5% AEP) - positional sensitivity, not a source-filter issue (see session notes)"
_ENGLAND_PUBLISHED = ("Wing et al. (2024)", 0.62)


def _fmt_published(pairs: list[tuple[str, float]]) -> str:
    if not pairs:
        return "-"
    return "; ".join(f"{study}: {val:.2f}" for study, val in pairs)


def build_extent_table() -> pd.DataFrame:
    gbr_rp100 = pd.read_csv(GBR_METRICS_RP100).set_index("region")
    gbr_rp250 = pd.read_csv(GBR_METRICS_RP250).set_index("region")
    can = pd.read_csv(CAN_METRICS).set_index("region")
    fz3 = pd.read_csv(ENGLAND_FZ3_RESCORE)
    fz2 = pd.read_csv(ENGLAND_FZ2_BAND)
    sf_bay = pd.read_csv(SF_BAY_RESULTS, index_col=0)

    rows = []
    for label, (src, region), published, ranged in _REGIONS:
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
            "Published CSI": _fmt_published([_ENGLAND_PUBLISHED]), "Note": "",
        })
    r = fz2.iloc[0]
    rows.append({
        "Region": f"England - Flood Zone 2 ({r['band_inner_rp']}-{r['band_outer_rp']} band)",
        "HR": r["HR"], "FAR": r["FAR"], "bias": r["bias"], "CSI": r["CSI"],
        "Published CSI": _fmt_published([_ENGLAND_PUBLISHED]), "Note": _FZ2_NOTE,
    })

    for direction in sf_bay.index:
        r = sf_bay.loc[direction]
        rows.append({
            "Region": f"San Francisco Bay ({direction})",
            "HR": r["HT"], "FAR": r["FAR"], "bias": r["bias"], "CSI": r["CSI"],
            "Published CSI": "-", "Note": "",
        })

    return pd.DataFrame(rows)


def build_sf_bay_depth_table() -> pd.DataFrame:
    sf_bay = pd.read_csv(SF_BAY_RESULTS, index_col=0)
    published = {"RMSE_all_m": 0.138, "MAE_all_m": 0.017, "RMSE_flooded_m": 0.525, "MAE_flooded_m": 0.319}
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


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)

    extent_df = build_extent_table()
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

    sf_bay_df = build_sf_bay_depth_table()
    print("\n=== San Francisco Bay depth accuracy (RMSE/MAE, m) ===")
    print(sf_bay_df.round(3).to_string(index=False))
    sf_bay_out = OUT_DIR / "overview_sf_bay_depth.csv"
    sf_bay_df.to_csv(sf_bay_out, index=False)
    print(f"\nwrote {sf_bay_out}")


if __name__ == "__main__":
    main()
