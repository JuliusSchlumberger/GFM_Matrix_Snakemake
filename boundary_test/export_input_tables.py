"""Export the per-site data behind INPUT_ANALYSIS.md as CSVs, plus markdown sample tables.

Quality sites only. Files (in --dest, default boundary_test/input_analysis_data/):
  sites_levels.csv    MHWS / spring range of GESLA, GTSM and FES2022 and their ratios
                      (the data of levels_scatter.png and levels_ratio_maps.png)
  sites_extremes.csv  GEV return levels (GESLA, GTSM) with 90% intervals, COAST-RP and its
                      ETC / TC parts, station class
  sites_shapes.csv    profile metrics of the 3 largest events per source (compare_top_events.csv)
  tables.md           regional summary, a stratified sample of sites_levels and the FES misses

Usage:
    python boundary_test/export_input_tables.py --out-dir .cleanup_baseline/level_sources
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DEST = Path(__file__).resolve().parent / "input_analysis_data"
SAMPLE_PER_REGION = 3
FES_MISS_M = 0.25
SEED = 0


def levels(sites: pd.DataFrame) -> pd.DataFrame:
    q = sites[sites.quality]
    t = pd.DataFrame({
        "site": q.site_name, "country": q.country, "region": q.region, "lon": q.lon.round(3), "lat": q.lat.round(3),
        "gtsm_dist_km": q.gtsm_dist_km.round(1), "fit_years": q.fit_years.round(1),
        "spring_range_gesla_m": q.gesla_harm__spring_range_m, "mhws_gesla_m": q.gesla_harm__mhws_m,
        "mhws_gtsm_m": q.gtsm_harm__mhws_m, "mhws_fes_gauge_m": q.fes_gauge__mhws_m,
        "mhws_fes_station_m": q.fes_station__mhws_m})
    t["ratio_gtsm_gesla"] = t.mhws_gtsm_m / t.mhws_gesla_m
    t["ratio_fes_gesla"] = t.mhws_fes_gauge_m / t.mhws_gesla_m
    t["ratio_gtsm_fes"] = t.mhws_gtsm_m / t.mhws_fes_station_m
    return t.round(3)


def extremes(ex: pd.DataFrame, keep: pd.Index) -> pd.DataFrame:
    g = ex[ex.source == "gesla"].set_index("file_name")
    t = ex[ex.source == "gtsm"].set_index("file_name")
    idx = keep.intersection(g.index)
    out = pd.DataFrame(index=idx)
    out["n_annual_maxima_gesla"] = g.loc[idx, "n_annual_maxima"]
    for rp in (10, 100):
        for c in ("", "_p05", "_p95"):
            out[f"gesla_rp{rp}{c}_m"] = g.loc[idx, f"rp{rp}{c}_m"]
        out[f"gtsm_rp{rp}_m"] = t[f"rp{rp}_m"].reindex(idx)
        for part in ("coast_rp", "coast_rp_etc", "coast_rp_tc"):
            out[f"{part}_rp{rp}_m"] = g.loc[idx, f"{part}_rp{rp}_m"]
    tc, etc = out.coast_rp_tc_rp100_m, out.coast_rp_etc_rp100_m
    out["station_class"] = np.where(tc.isna(), "no_tc", np.where(tc > etc, "tc", "etc"))
    return out.round(3)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    args = parser.parse_args()
    out, dest = Path(args.out_dir), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(out / "analysis" / "levels_sites.csv", index_col=0)
    lv = levels(sites)
    lv.to_csv(dest / "sites_levels.csv")
    extremes(pd.read_parquet(out / "extremes.parquet"), lv.index).to_csv(dest / "sites_extremes.csv")
    shapes = pd.read_csv(out / "analysis" / "compare_top_events.csv")
    shapes[shapes.file_name.isin(lv.index)].round(3).to_csv(dest / "sites_shapes.csv", index=False)

    region = lv.groupby("region").agg(n=("site", "size"), gtsm_gesla=("ratio_gtsm_gesla", "median"),
                                      fes_gesla=("ratio_fes_gesla", "median"), gtsm_fes=("ratio_gtsm_fes", "median"),
                                      spring_range_m=("spring_range_gesla_m", "median")).reset_index()
    cols = ["site", "country", "region", "lon", "lat", "spring_range_gesla_m", "mhws_gesla_m", "mhws_gtsm_m",
            "mhws_fes_gauge_m", "ratio_gtsm_gesla", "ratio_fes_gesla", "ratio_gtsm_fes"]
    sample = (lv.groupby("region", group_keys=False)
              .apply(lambda g: g.sample(min(SAMPLE_PER_REGION, len(g)), random_state=SEED))[cols]
              .sort_values(["region", "site"]))
    miss = lv[(lv.mhws_fes_gauge_m - lv.mhws_gesla_m).abs() > FES_MISS_M][cols].sort_values("ratio_fes_gesla")
    (dest / "tables.md").write_text("\n\n".join([md_table(region), md_table(sample), md_table(miss)]) + "\n",
                                    encoding="utf-8")
    print(f"{len(lv)} sites -> {dest}")


if __name__ == "__main__":
    main()
