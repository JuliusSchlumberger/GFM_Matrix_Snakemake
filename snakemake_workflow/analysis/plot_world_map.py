"""World choropleth maps of EAI at selected (SSP, year) snapshots.

Reads File 3 (exposure_{label}_ssp.csv) from compute_exposure_analysis.py -
one column per (SSP, year), already resolved to that SSP's real projected
SLR and population growth factor at file-creation time, so this reflects
real SSP population growth (an earlier version only ever used the unscaled
2020-population base EAI here, ignoring growth). Avoid is included too,
since File 3 exists for it (unlike File 1).

Uses the WRI geogunit_109 country raster for drawing — no external vector
shapefile needed.  The geogunit_109→ISO lookup is built once and cached.

Usage:
    python snakemake_workflow/analysis/plot_world_map.py \\
        --config  snakemake_workflow/config/config.yml \\
        --expdir  D:/GFM/merged_results/exposure \\
        [--outdir D:/GFM/figures/world_maps] \\
        [--years 2050 2100] \\
        [--ssps  SSP1 SSP2 SSP5]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visualization import (  # noqa: E402
    build_geo109_to_iso_lookup,
    plot_world_map,
)


def _expand(s, root, code_root=""):
    return str(s).replace("{root}", root).replace("{code_root}", code_root or root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--years", nargs="+", type=int, default=[2050, 2100])
    parser.add_argument("--ssps", nargs="+", default=["SSP1", "SSP2", "SSP5"])
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    root = cfg.get("paths", {}).get("root", "")
    code_root = cfg.get("paths", {}).get("code_root", root)
    ex = lambda s: _expand(s, root, code_root)

    viz = cfg.get("visualization", {})

    exp_dir = Path(ex(args.expdir))
    out_dir = Path(ex(args.outdir or str(
        Path(ex(viz.get("output_dir", "{root}/figures"))) / "world_maps"
    )))
    out_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(viz.get("dpi", 200))
    cmap = viz.get("cmap_ember", "YlOrRd")

    # Geogunit_109 → ISO lookup (built once, cached)
    geo109_nc = ex("{root}/inputs/WRI/geogunit_109_all.nc")
    geo107_nc = ex("{root}/inputs/WRI/geogunit_107_all.nc")
    flopros_xlsx = ex("{root}/inputs/protection_levels/FLOPROS_NL_geogunit_107.xlsx")
    lookup_cache = out_dir / "geo109_iso_lookup.csv"

    if lookup_cache.exists():
        cache_df = pd.read_csv(lookup_cache, index_col=0)
        geo109_to_iso = dict(zip(cache_df.index.astype(int), cache_df["ISO"].astype(str)))
    else:
        print("Building geogunit_109→ISO lookup (one-time)…")
        geo109_to_iso = build_geo109_to_iso_lookup(geo109_nc, geo107_nc, flopros_xlsx, subsample=10)
        pd.DataFrame({"ISO": geo109_to_iso}).to_csv(lookup_cache)

    eai_by_label: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(exp_dir.glob("exposure_*_ssp.csv")):
        label = csv_path.stem.replace("exposure_", "").replace("_ssp", "")
        eai_by_label[label] = pd.read_csv(csv_path, index_col=0)

    if not eai_by_label:
        print(f"No exposure_*_ssp.csv files found in {exp_dir}")
        sys.exit(0)

    # Compute global vmax for consistent colour scale
    global_vmax = 0.0
    for eai_df in eai_by_label.values():
        for ssp in args.ssps:
            for yr in args.years:
                col = f"EAI_{ssp}_{yr}"
                if col in eai_df.columns:
                    global_vmax = max(global_vmax, float(eai_df[col].max()))

    # Generate maps
    for label, eai_df in eai_by_label.items():
        for ssp in args.ssps:
            for yr in args.years:
                col = f"EAI_{ssp}_{yr}"
                if col not in eai_df.columns:
                    print(f"  {label}: {col} not available — skipping.")
                    continue
                title = f"{label} | {ssp} | {yr}"
                fig = plot_world_map(
                    eai_df[col], geo109_nc, geo109_to_iso,
                    title=title, cmap=cmap, vmax=global_vmax, dpi=dpi,
                )
                fname = out_dir / f"world_map_{label}_{ssp}_{yr}.png"
                fig.savefig(fname, bbox_inches="tight")
                plt.close(fig)
                print(f"  {fname.name}")

    print(f"\nAll world maps written to: {out_dir}")


if __name__ == "__main__":
    main()
