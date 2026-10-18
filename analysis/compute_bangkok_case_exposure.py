"""Compute flooded area and exposed population inside a case polygon per (RP, SLR).

Streams the postprocess chunk outputs like compute_flood_totals.py, but scoped to the
Bangkok / Chao Phraya case polygon. exposed_population = flood_fraction * pop assumes
population is uniform within each population-grid cell.

Usage:
    python compute_bangkok_case_exposure.py \\
        --config snakemake_workflow/config/bangkok_chao_phraya_materialized.yml \\
        --case-polygon-gpkg P:/.../bangkok_chao_phraya/bangkok_chao_phraya_domain.gpkg \\
        --case-polygon-layer bangkok_tile \\
        --out-dir P:/.../bangkok_chao_phraya/merged_results/exposure
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import _read_band, _read_band_and_transform, _round_val, discover_chunk_ids  # noqa: E402
from config_utils import atomic_write, load_config, merged_waterlevel_names, retry_transient_io, return_period_label  # noqa: E402
from exposure_analysis import _safe  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
from rasters import polygon_domain_mask  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    _default_cfg = str(_REPO_ROOT / "snakemake_workflow" / "config" / "bangkok_chao_phraya_materialized.yml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--case-polygon-gpkg", required=True)
    parser.add_argument("--case-polygon-layer", default="bangkok_tile")
    parser.add_argument("--out-dir", default=None, help="default: {merged_outputs}/exposure")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bc = cfg["boundary_conditions"]
    return_periods = bc["return_periods"]
    slr_scenarios = merged_waterlevel_names(bc, cfg["adaptation"])

    merged_dir = Path(cfg["postprocessing"]["merged_outputs"])
    flood_frac_dir = merged_dir / "chunks" / "flood_fraction"
    chunks_dir = merged_dir / "chunks"
    out_dir = Path(args.out_dir) if args.out_dir else merged_dir / "exposure"
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    case_poly = gpd.read_file(args.case_polygon_gpkg, layer=args.case_polygon_layer)

    chunk_ids, all_chunk_ids = discover_chunk_ids(flood_frac_dir, chunks_dir)
    if not all_chunk_ids:
        print(f"ERROR: no flood fraction files in {flood_frac_dir} - run the Snakemake postprocess target first.")
        sys.exit(1)
    if not chunk_ids:
        print("ERROR: no population chunk files found - run the Snakemake postprocess target first.")
        sys.exit(1)
    print(f"Using {len(chunk_ids)} chunk(s), {len(return_periods)} RPs, {len(slr_scenarios)} SLR scenarios.")

    totals: dict[tuple[int, str], dict[str, float]] = {}
    polygon_area_km2 = 0.0
    chunks_with_overlap = 0

    for i, cid in enumerate(chunk_ids, 1):
        pop_path = chunks_dir / f"exposure_population_grid_{cid}.tif"
        pop, transform = _read_band_and_transform(pop_path, "float64", 0.0)
        pop[~np.isfinite(pop)] = 0.0
        height, width = pop.shape

        mask = polygon_domain_mask(case_poly, transform, "EPSG:4326", (height, width))
        if not mask.any():
            continue
        chunks_with_overlap += 1

        area_km2 = pixel_area_km2_grid(transform, width, height)
        polygon_area_km2 += float(area_km2[mask].sum())

        for rp in return_periods:
            for slr in slr_scenarios:
                p = flood_frac_dir / f"flood_fraction_{cid}_RP{rp}_{slr}.tif"
                if not (p.exists() and p.stat().st_size > 0):
                    continue
                ff = _safe(_read_band(p, "float64", -1.0))
                flooded_km2 = float((ff * area_km2)[mask].sum())
                exposed_population = float((ff * pop)[mask].sum())
                key = (int(rp), slr)
                cur = totals.setdefault(key, {"flooded_km2": 0.0, "exposed_population": 0.0})
                cur["flooded_km2"] += flooded_km2
                cur["exposed_population"] += exposed_population

        print(f"  {i}/{len(chunk_ids)} chunk(s) processed ({cid}: overlaps case polygon)…")

    if chunks_with_overlap == 0:
        print("ERROR: no chunk overlaps the case polygon - check --case-polygon-gpkg/--case-polygon-layer.")
        sys.exit(1)
    print(f"\nCase polygon spans {chunks_with_overlap} chunk(s), total area {polygon_area_km2:.2f} km2.")

    rows = [
        {
            "return_period": return_period_label(rp), "waterlevel_name": slr,
            "flooded_km2": _round_val(v["flooded_km2"]),
            "flooded_pct_of_polygon": _round_val(100.0 * v["flooded_km2"] / polygon_area_km2),
            "exposed_population": _round_val(v["exposed_population"]),
        }
        for (rp, slr), v in sorted(totals.items())
    ]
    df = pd.DataFrame(rows)
    out_path = out_dir / "bangkok_case_exposure.csv"
    atomic_write(out_path, lambda f: df.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
    print(f"Wrote {out_path} ({len(df)} row(s))")


if __name__ == "__main__":
    main()
