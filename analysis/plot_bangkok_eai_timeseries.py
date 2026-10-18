"""Plot the Bangkok / Chao Phraya EAI (people) time series, one panel per SSP.

No sub-national SSP growth projection exists, so the case-polygon exposure is scaled
by Thailand's national growth factor.

Usage:
    python plot_bangkok_eai_timeseries.py \\
        --config snakemake_workflow/config/bangkok_chao_phraya_materialized.yml \\
        --exposure-csv P:/.../bangkok_chao_phraya/merged_results/exposure/bangkok_case_exposure.csv \\
        --out-dir P:/.../bangkok_chao_phraya/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, merged_waterlevel_names, return_period_years  # noqa: E402
from exposure_analysis import _trapezoid_eai, resolve_ssp_scenario_eai  # noqa: E402
from population_growth import load_ssp_growth_factors  # noqa: E402
from analysis_figures import load_slr_trajectories, ssp_rcp_label  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CASE_ISO = "THA"


def main() -> None:
    _default_cfg = str(_REPO_ROOT / "snakemake_workflow" / "config" / "bangkok_chao_phraya_materialized.yml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--exposure-csv", required=True)
    parser.add_argument("--out-dir", default=None, help="default: visualization.output_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz = cfg["visualization"]
    bc = cfg["boundary_conditions"]
    return_periods = bc["return_periods"]
    slr_scenarios = merged_waterlevel_names(bc, cfg["adaptation"])
    slr_mm_values = sorted(int(s.split("_")[1]) for s in slr_scenarios)
    slr_order = [f"SLR_{mm}" for mm in slr_mm_values]

    df = pd.read_csv(args.exposure_csv)
    df["return_period"] = df["return_period"].map(return_period_years)  # "RP100" -> 100

    # Trapezoidal integration over 1/RP; columns must follow slr_mm_values order.
    eai_by_slr = {}
    for slr in slr_order:
        sub = df[df["waterlevel_name"] == slr].set_index("return_period").reindex(return_periods)
        exposures = sub["exposed_population"].to_numpy(dtype=float).reshape(1, -1)
        eai_by_slr[slr] = float(_trapezoid_eai(exposures, return_periods)[0])
    eai_df = pd.DataFrame([eai_by_slr], index=[_CASE_ISO])[slr_order]
    print("EAI-vs-SLR (discrete, simulated points):")
    print(eai_df.to_string())

    traj_path = viz["slr_trajectories_csv"]
    traj_full = load_slr_trajectories(traj_path)
    p50_cols = {c: c.replace("_p50", "") for c in traj_full.columns if c.endswith("_p50")}
    slr_traj = traj_full[list(p50_cols)].rename(columns=p50_cols)

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    growth_df = load_ssp_growth_factors(catalog.get_source("ssp_population_growth_factors").path)

    ssps = cfg["population_growth"]["ssps"]
    years = cfg["population_growth"]["output_years"]
    resolved = resolve_ssp_scenario_eai(eai_df, slr_mm_values, growth_df, ssps, years, slr_traj)
    row = resolved.loc[_CASE_ISO]

    out_dir = Path(args.out_dir) if args.out_dir else Path(viz["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    ssp_colors = viz["ssp_colors"]
    ssp_rcp_codes = viz["ssp_rcp_codes"]
    ssps_avail = [s for s in ssps if any(c.startswith(f"EAI_{s}_") for c in resolved.columns)]

    fig, axes = plt.subplots(1, len(ssps_avail), figsize=(4 * len(ssps_avail) + 2, 5), sharey=True)
    if len(ssps_avail) == 1:
        axes = [axes]

    for ax, ssp in zip(axes, ssps_avail):
        col = ssp_colors.get(ssp, "#444444")
        yrs = [yr for yr in years if f"EAI_{ssp}_{yr}" in resolved.columns]
        vals = [float(row[f"EAI_{ssp}_{yr}"]) for yr in yrs]
        ax.plot(yrs, vals, marker="o", markersize=4, linewidth=2, color=col)
        ax.set_title(ssp_rcp_label(ssp, ssp_rcp_codes), color=col, fontsize=10)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Expected annual impacted population (people)")
    fig.suptitle("Bangkok / Chao Phraya case study: EAI over time", y=1.02)
    fig.tight_layout()

    out_path = out_dir / "bangkok_eai_timeseries.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
