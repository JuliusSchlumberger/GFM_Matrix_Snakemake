"""Compare HR/FAR/CSI/EB across the ESP/FRA/NOR calibration sweep's parameter
choices, at one representative (region, domain, threshold) per country.

Reads `calibration_results.csv` (written by aggregate_calibration_results.py -
run that first, or pass --no-aggregate to reuse an existing file as-is).
Filters down to:
  - `threshold_m == validation.primary_threshold_m` (0.10 by default - the same
    single threshold validate_country.py's own console summary highlights).
  - one MAIN_REGION per country (mainland/metropole - see MAIN_REGION below),
    not every overseas territory - keeps the comparison to one clean line per
    country rather than 6 for France alone.
  - `domain == "buffered"` where that domain exists for a country, falling back
    to `domain == "model_only"` otherwise (Norway's national-coverage benchmark
    never produces a "buffered" row at all - see validate_country_national_
    coverage's own docstring) - same preferred-domain logic validate_country.py's
    own main() already uses for its console summary, reused here rather than
    reinvented.

Writes:
  - `calibration_comparison_{threshold}.csv` - the tidy table (one row per
    group/sweep_point/country) this figure is drawn from.
  - `calibration_sensitivity.png` - one subplot per indicator (HR, FAR, CSI,
    EB), x-axis = sweep point (categorical, baseline first then grouped by
    swept_param), colour/marker = country. A missing (sweep_point, country)
    combination - e.g. a run the HPC connection dropped before finishing - is
    simply absent, not interpolated or zero-filled.

Usage:
    python plot_calibration_sensitivity.py [--config <config.yml>] [--no-aggregate]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from config_utils import load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]

# One "main" territory per country - not every catalog region (France alone has
# 6) - see this file's own docstring for why. Explicit, not guessed: matches
# each country's own data_catalog_validation.yml regions block.
MAIN_REGION = {"ESP": "mainland", "FRA": "metropole", "NOR": "mainland"}

INDICATORS = ["HR", "FAR", "CSI", "EB"]

# Sweep-point display order: baseline first, then grouped by the parameter
# family it belongs to (matches aggregate_calibration_results.py's own
# SWEEP_POINTS dict ordering) - NOT alphabetical, which would interleave
# unrelated parameters and make the x-axis unreadable.
SWEEP_POINT_ORDER = [
    "baseline",
    "friction_0.5", "friction_2.0",
    "max_rounds_4", "max_rounds_8", "max_rounds_20",
    "obstacle_coupling_off", "obstacle_coupling_iter1", "obstacle_coupling_iter3", "obstacle_coupling_iter10",
    "waterlevel_eps_0.01", "waterlevel_eps_0.10",
    "exceedance_threshold_0.05", "exceedance_threshold_0.20",
]

COUNTRY_STYLE = {
    "ESP": dict(color="#d62728", marker="o"),
    "FRA": dict(color="#1f77b4", marker="s"),
    "NOR": dict(color="#2ca02c", marker="^"),
}


def _select_comparison_rows(combined: pd.DataFrame, primary_threshold_m: float) -> pd.DataFrame:
    df = combined[combined["threshold_m"] == primary_threshold_m].copy()

    keep = []
    for country, region in MAIN_REGION.items():
        keep.append(df[(df["country"] == country) & (df["region"] == region)])
    df = pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]

    # Prefer domain="buffered" per (run_tag, country); fall back to whatever
    # that combination actually has (Norway: always "model_only") - same rule
    # validate_country.py's own main() uses for its console summary.
    df["_domain_rank"] = (df["domain"] == "buffered").astype(int)
    df = df.sort_values("_domain_rank", ascending=False).drop_duplicates(
        subset=["run_tag", "country"], keep="first",
    ).drop(columns="_domain_rank")

    df["sweep_point"] = pd.Categorical(df["sweep_point"], categories=SWEEP_POINT_ORDER, ordered=True)
    df = df.sort_values(["sweep_point", "country"]).reset_index(drop=True)
    return df[[
        "group", "sweep_point", "swept_param", "swept_value", "country", "region", "domain",
        "threshold_m", *INDICATORS, "bias", "benchmark_wet_outside_model_domain_pct",
    ]]


def _plot(table: pd.DataFrame, out_path: Path, primary_threshold_m: float) -> None:
    sweep_points = [s for s in SWEEP_POINT_ORDER if s in table["sweep_point"].unique()]
    x_pos = {s: i for i, s in enumerate(sweep_points)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, indicator in zip(axes.flat, INDICATORS):
        for country, style in COUNTRY_STYLE.items():
            sub = table[table["country"] == country]
            if sub.empty:
                continue
            xs = [x_pos[s] for s in sub["sweep_point"]]
            ax.plot(
                xs, sub[indicator], linestyle="none",
                marker=style["marker"], color=style["color"], markersize=8,
                label=country,
            )
            baseline = sub[sub["sweep_point"] == "baseline"]
            if not baseline.empty:
                ax.axhline(
                    baseline[indicator].iloc[0], color=style["color"], linestyle=":", linewidth=1, alpha=0.5,
                )
        ax.set_title(indicator)
        ax.set_ylabel(indicator)
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xticks(range(len(sweep_points)))
        ax.set_xticklabels(sweep_points, rotation=45, ha="right")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if not handles:  # every panel might have skipped a country with no data - grab from any panel that has some
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Calibration sweep sensitivity - primary threshold {primary_threshold_m} m, "
        f"main region per country ({', '.join(f'{c}={r}' for c, r in MAIN_REGION.items())})",
        y=1.08,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = _REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--no-aggregate", action="store_true",
        help="Skip re-running aggregate_calibration_results.py - reuse whatever calibration_results.csv "
             "already exists (faster, but may be stale relative to the real P: drive results).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sweep_root = Path(cfg["paths"]["root"]) / "calibration_esp_fra_nor"
    primary_threshold_m = float(cfg["validation"]["primary_threshold_m"])

    if not args.no_aggregate:
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("aggregate_calibration_results.py")), "--config", args.config],
            check=True,
        )

    combined_path = sweep_root / "calibration_results.csv"
    combined = pd.read_csv(combined_path)

    table = _select_comparison_rows(combined, primary_threshold_m)
    if table.empty:
        print(f"No rows left after filtering to threshold_m={primary_threshold_m} and MAIN_REGION={MAIN_REGION} "
              f"- nothing to plot.")
        return

    table_path = sweep_root / f"calibration_comparison_{primary_threshold_m}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {table_path} ({len(table)} row(s))")

    all_expected = [(sp, c) for sp in SWEEP_POINT_ORDER for c in MAIN_REGION]
    present = set(zip(table["sweep_point"].astype(str), table["country"]))
    truly_missing = [pair for pair in all_expected if pair not in present]
    if truly_missing:
        print(f"Missing (sweep_point, country) combinations - not yet run or not yet validated: {truly_missing}")

    fig_path = sweep_root / "calibration_sensitivity.png"
    _plot(table, fig_path, primary_threshold_m)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
