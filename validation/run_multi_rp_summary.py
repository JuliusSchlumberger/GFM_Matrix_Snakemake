"""Multi-return-period country-level summary: for every benchmark with
`meta.comparison_return_periods` set (e.g. Norway/Wales/Scotland: `[100, 250]`),
runs the full extent-path scoring (`validate_country`/`validate_country_national_coverage`)
once per listed RP and writes one tidy row per `(country, benchmark_key, RP)`
to `{validation.output_dir}/multi_rp_summary.csv` - HR/FAR/CSI/bias/EB only,
the country-level blended row (`region == "ALL"`, `validate_country.py`'s own
`_blend_regions`) where one exists, else the single region row.

Also writes the full (every-benchmark) `metrics_{country}_{RP}_{SLR}.csv` for
every RP it scores (not just the one `validate_country.py`'s own `main()`
would write at the single global RP) and plots every country it touches
once, after all its RPs are scored (`plot_agreement_map.plot_country` -
that function discovers each benchmark's own RPs/rasters itself and combines
them into one side-by-side figure per benchmark, so this script never calls
it more than once per country). Without this, RP250's own agreement maps for
Norway/Wales/Scotland would never exist anywhere, even though RP250's own
category rasters are written by the scoring step above regardless - the
normal `run_validation.py`/`validate_country.py` + `plot_agreement_map.py`
pairing only ever produces plots at the single global `validation.return_period`.

A benchmark with no `comparison_return_periods` configured is scored once,
at the single global `validation.return_period` - same as every other
validation entry point. This script does not replace `validate_country.py`/
`run_validation.py` - it is an ADDITIONAL sweep over RP, for the small
number of benchmarks that specifically need one (today: Norway, Wales, Scotland).

Every listed RP must be NATIVE - a real merged chunk already on disk
(`merged_results/chunks/waterdepth_{chunk_id}_{RP}_{SLR}.tif`, i.e. `RP` is
one of `boundary_conditions.return_periods` in config.yml). This just
re-runs the existing scoring functions with `validation.return_period`
overridden - no new code path, same result a manual `--country NOR` run at
that RP would give. (2026-10 - an earlier version also accepted a non-native
RP, e.g. Norway's real ~200yr benchmark class between the model's native
RP100/RP250, and log-linear-interpolated model depth to fill the gap -
removed by user decision: report the model's own real, simulated return
periods only, not an estimated intermediate one. A non-native RP in
`comparison_return_periods` today just finds no chunks and produces no row
for it, same as any other benchmark key with nothing on disk - see the
empty-dataframe handling below.)

Usage:
    python validation/run_multi_rp_summary.py \\
        --config snakemake_workflow/config/config.yml [--countries NOR]
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
    validate_country,
    validate_country_national_coverage,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--countries", nargs="+", default=None, help="ISO-3 codes to restrict to (default: every country with at least one comparison_return_periods benchmark)")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    val_cfg = base_cfg["validation"]

    gfm_catalog = get_data_catalog(_REPO_ROOT / base_cfg["paths"]["hydromt_data_catalog"], root=base_cfg["paths"]["root"])
    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])

    # Every (country_iso, benchmark_key, [rps]) with a real comparison_return_periods list.
    all_isos = {
        meta["country_iso"] for key in bench_catalog.sources.keys()
        if (meta := bench_catalog.get_source(key).meta or {}).get("hazard_type") == "coastal"
    }
    countries = args.countries or sorted(all_isos)

    summary_rows: list[dict] = []
    # Countries actually scored, for the single plot_country() pass after
    # this loop - not called inline per (country, rp, benchmark): a country
    # can have more than one comparison_return_periods benchmark (GBR:
    # Wales + Scotland, both [100, 250]), and plot_country() itself
    # discovers every RP a benchmark needs straight from the catalog/disk
    # (its own docstring) - it only needs to run once per country, AFTER
    # every RP below has been scored and written.
    plotted_countries: set[str] = set()
    written_metrics_csvs: set[tuple[str, str]] = set()  # (country_iso, rp_label) already written

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

                df_partial = validate_country(country_iso, cfg, gfm_catalog, bench_catalog)
                df_national = validate_country_national_coverage(country_iso, cfg, gfm_catalog, bench_catalog)
                df = pd.concat([df_partial, df_national], ignore_index=True)
                if df.empty or "benchmark_key" not in df.columns:
                    print(f"  RP{rp}: no rows produced (no chunks/regions had real data) - skipping.")
                    continue

                # The full (every-benchmark) metrics CSV for this (country, RP) -
                # same file/schema validate_country.py's own main() writes, and
                # what plot_country()'s own per-panel HR/FAR/CSI/bias/CSI_tol
                # annotation reads. Written once per (country, RP), not once per
                # benchmark - validate_country_national_coverage already scores
                # every national benchmark for this country in one call, so a
                # later benchmark_key's own df here would just be a duplicate.
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
                    row = df_this  # single-region benchmark - no "ALL" row, use its one real row
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

    # Plots (main map + subregions grid per benchmark, tile coverage, CSI
    # dots) - once per country, AFTER every RP above has been scored and
    # written, not once per (country, rp): plot_country() discovers each
    # benchmark's own RPs and their on-disk rasters itself (its own
    # docstring), combining e.g. RP100+RP250 into one side-by-side figure
    # per benchmark rather than writing one independent file per RP.
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
