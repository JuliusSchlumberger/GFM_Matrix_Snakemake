"""Score every benchmark with `comparison_return_periods` at each listed RP.

Writes multi_rp_summary.csv (the blended "ALL" row per country/benchmark/RP, else
the single region row), metrics_{ISO}_{RP}_{SLR}.csv per RP, and one
plot_country() pass per country. Benchmarks without comparison_return_periods
are skipped. Listed RPs must be native (merged chunks on disk); a non-native RP
finds no chunks and yields no row.

Usage:
    python validation/run_multi_rp_summary.py \\
        --config snakemake_workflow/config/config.yml [--countries GBR CAN]
"""

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
import validation as v  # noqa: E402
from plot_agreement_map import plot_country  # noqa: E402
from validate_country import (  # noqa: E402
    _BLENDED_REGION,
    _find_benchmark_keys,
    validate_country_national_coverage,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the per-RP scoring sweep, write the CSVs and plot each country."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--countries", nargs="+", default=None, help="ISO-3 codes to restrict to (default: every country with at least one comparison_return_periods benchmark)")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    val_cfg = base_cfg["validation"]

    gfm_catalog = get_data_catalog(_REPO_ROOT / base_cfg["paths"]["hydromt_data_catalog"], root=base_cfg["paths"]["root"])
    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])

    all_isos = {
        meta["country_iso"] for key in bench_catalog.sources.keys()
        if (meta := bench_catalog.get_source(key).meta or {}).get("hazard_type") == "coastal"
    }
    countries = args.countries or sorted(all_isos)

    summary_rows: list[dict] = []
    # plot_country() finds every RP itself, so it runs once per country after scoring.
    plotted_countries: set[str] = set()
    written_metrics_csvs: set[tuple[str, str]] = set()

    for country_iso in countries:
        for benchmark_key in _find_benchmark_keys(bench_catalog, country_iso):
            spec = v.load_benchmark_spec(bench_catalog, benchmark_key)
            if not spec.comparison_return_periods:
                continue
            print(f"=== {country_iso} / {benchmark_key}: RPs {spec.comparison_return_periods} ===")

            for rp in spec.comparison_return_periods:
                rp_label = f"RP{rp}"
                cfg = copy.deepcopy(base_cfg)
                cfg["validation"]["return_period"] = rp_label

                df = validate_country_national_coverage(country_iso, cfg, gfm_catalog, bench_catalog)
                if df.empty or "benchmark_key" not in df.columns:
                    print(f"  RP{rp}: no rows produced (no chunks/regions had real data) - skipping.")
                    continue

                # df already covers every benchmark of the country: write once per RP.
                # plot_country() reads this file for its panel annotations.
                if (country_iso, rp_label) not in written_metrics_csvs:
                    metrics_out_dir = Path(val_cfg["output_dir"]) / country_iso
                    retry_transient_io(metrics_out_dir.mkdir, parents=True, exist_ok=True)
                    metrics_path = metrics_out_dir / f"metrics_{country_iso}_{rp_label}_{val_cfg['waterlevel_name']}.csv"
                    atomic_write(metrics_path, lambda f: df.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
                    written_metrics_csvs.add((country_iso, rp_label))
                plotted_countries.add(country_iso)

                df_this = df[df["benchmark_key"] == benchmark_key]
                if df_this.empty:
                    print(f"  RP{rp}: no rows produced for this benchmark specifically - skipping.")
                    continue

                row = df_this[df_this["region"] == _BLENDED_REGION]
                if row.empty:
                    row = df_this
                row = row.iloc[0]
                summary_rows.append({
                    "country": country_iso, "benchmark_key": benchmark_key, "return_period": rp_label,
                    "waterlevel_name": val_cfg["waterlevel_name"], "region": row["region"],
                    "HR": row["HR"], "FAR": row["FAR"], "CSI": row["CSI"],
                    "EB": row["EB"], "EB_ratio": row["EB_ratio"], "bias": row["bias"],
                    "tp_km2": row["tp_km2"], "fp_km2": row["fp_km2"], "fn_km2": row["fn_km2"],
                    "benchmark_wet_outside_model_domain_pct": row["benchmark_wet_outside_model_domain_pct"],
                })
                print(f"  RP{rp}: HR={row['HR']:.3f} FAR={row['FAR']:.3f} CSI={row['CSI']:.3f}")

    for country_iso in sorted(plotted_countries):
        cfg = copy.deepcopy(base_cfg)
        print(f"=== {country_iso}: plotting ===")
        plot_country(cfg, country_iso)

    if not summary_rows:
        print("\nNo comparison_return_periods benchmarks found (or none produced data) - nothing to write.")
        return

    out_path = Path(val_cfg["output_dir"]) / "multi_rp_summary.csv"
    summary_df = pd.DataFrame(summary_rows)
    retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
    atomic_write(out_path, lambda f: summary_df.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
    print(f"\nMulti-RP summary written: {out_path} ({len(summary_df)} row(s))")


if __name__ == "__main__":
    main()
