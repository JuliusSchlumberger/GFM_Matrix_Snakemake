"""Tipping-point figure: when does a country's expected annual impact (EAI)
exceed a fixed number of people, under each SSP-RCP scenario and adaptation
strategy?

One row per country (visualization.tipping_points.countries), two columns:
  - left:  the country's scenario-neutral SLR x population-change space with
           one tipping isoline per adaptation strategy (EAI = threshold, from
           each strategy's growth-matrix CSV at the configured design
           intensity) and each SSP's pathway through that space (global
           median SLR vs the country's projected population change). Where a
           pathway crosses a strategy's isoline, that strategy tips under
           that scenario - marked with a dot. Colour = SSP throughout,
           line style = adaptation strategy.
  - right: the year of that crossing per strategy (rows) and SSP (colour):
           dot = median (p50) SLR pathway, bar = the p17-p83 SLR range (the
           higher SLR path tips earlier); population change is the SSP's
           projection in all three. A triangle at the right edge means not
           reached by 2100 (for the median, or for the p17 end of the bar).

EAI along a pathway is read from the growth matrix by bilinear interpolation
(linear extrapolation in the population direction where a pathway leaves the
matrix's growth range; the isolines are drawn on the same extended grid).
Writes tipping_points.png and tipping_points.csv (every tipping year) to
--outdir.

Config (visualization.tipping_points):
    countries:                [VNM, NLD, NGA]
    eai_threshold:            1000000      # people per year
    per_country_eai_threshold: {}           # e.g. {NLD: 100000}
    design_intensity:         SLR_1000     # adaptation design the strategies use
    strategies:               [baseline, protect, avoid, retreat, protect_avoid]

Usage:
    python analysis/plot_tipping_points.py --config snakemake_workflow/config/config.yml \\
        --expdir <merged_results>/exposure [--outdir <figures>/tipping_points]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from population_growth import _build_name_to_iso, interpolate_growth_factor, load_ssp_growth_factors  # noqa: E402
from visualization import (  # noqa: E402
    _DEFAULT_SLR_COLORS, build_growth_matrix_from_grid, load_growth_matrix_csv,
    load_slr_trajectories, ssp_rcp_label,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_NAMES = {"baseline": "No adaptation", "protect": "Protect", "avoid": "Avoid", "retreat": "Retreat",
                  "protect_avoid": "Protect + Avoid"}
# isolines: colour is reserved for the SSPs, so strategies differ by line style
STRATEGY_STYLES = {"baseline": "-", "protect": "--", "avoid": ":", "retreat": "-.",
                   "protect_avoid": (0, (8, 2, 1, 2, 1, 2))}
YEARS = np.arange(2020, 2101)


def _growth_matrix_path(exp_dir: Path, strategy: str, design: str) -> Path:
    label = "baseline" if strategy == "baseline" else f"{strategy}_{design}"
    return exp_dir / f"exposure_{label}_growth_matrix.csv"


def _first_crossing(eai: np.ndarray, threshold: float) -> float | None:
    """First (fractional) year in YEARS where `eai` reaches `threshold`, or None."""
    above = np.nonzero(eai >= threshold)[0]
    if len(above) == 0:
        return None
    k = int(above[0])
    if k == 0:
        return float(YEARS[0])
    e0, e1 = eai[k - 1], eai[k]
    return float(YEARS[k - 1] + (threshold - e0) / (e1 - e0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True, help="directory with exposure_*_growth_matrix.csv files")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz = cfg.get("visualization", {})
    tp = viz.get("tipping_points", {})
    countries = tp.get("countries", ["VNM", "NLD", "NGA"])
    default_thr = float(tp.get("eai_threshold", 1_000_000))
    thr_by_iso = {k: float(v) for k, v in (tp.get("per_country_eai_threshold") or {}).items()}
    design = tp.get("design_intensity", "SLR_1000")
    exp_dir = Path(args.expdir)
    strategies = [s for s in tp.get("strategies", list(STRATEGY_NAMES))
                  if _growth_matrix_path(exp_dir, s, design).exists()]
    out_dir = Path(args.outdir or Path(viz.get("output_dir", f"{cfg['paths']['root']}/figures")) / "tipping_points")
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    s_colors = viz.get("ssp_colors") or _DEFAULT_SLR_COLORS
    ssp_rcp_codes = viz.get("ssp_rcp_codes", {})

    traj = load_slr_trajectories(viz["slr_trajectories_csv"])
    ssps = [s for s in cfg.get("population_growth", {}).get("ssps", list(s_colors))
            if all(f"{s}_{p}" in traj.columns for p in ("p17", "p50", "p83"))]
    slr_paths = {(s, p): np.interp(YEARS, traj.index.to_numpy(dtype=float), traj[f"{s}_{p}"].to_numpy(dtype=float))
                 for s in ssps for p in ("p17", "p50", "p83")}  # mm

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    growth_df = load_ssp_growth_factors(Path(catalog.get_source("ssp_population_growth_factors").path))
    iso_to_name = {v: k for k, v in _build_name_to_iso().items()}

    grids = {s: load_growth_matrix_csv(_growth_matrix_path(exp_dir, s, design)) for s in strategies}

    fs = 11
    fig, axes = plt.subplots(len(countries), 2, figsize=(15, 4.4 * len(countries) + 2.0),
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.25, "hspace": 0.35}, squeeze=False)
    fig.subplots_adjust(top=1 - 0.6 / (4.4 * len(countries) + 2.0), bottom=1.7 / (4.4 * len(countries) + 2.0))
    rows = []
    for r, iso in enumerate(countries):
        thr = thr_by_iso.get(iso, default_thr)
        ax_l, ax_r = axes[r]
        growth_paths = {s: np.array([interpolate_growth_factor(growth_df, s, iso, int(y), 1.0) for y in YEARS]) - 1.0
                        for s in ssps}  # fraction
        g_lo = min(-0.5, min(p.min() for p in growth_paths.values()) - 0.05)
        g_hi = max(1.5, max(p.max() for p in growth_paths.values()) + 0.05)
        g_dense = np.linspace(g_lo, g_hi, 121)

        crossings: dict[tuple[str, str], dict[str, float | None]] = {}
        for k, strat in enumerate(strategies):
            eai_grid, slr_mm, growth_fracs = grids[strat]
            if iso not in eai_grid.index:
                continue
            matrix = build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, growth_fracs)
            interp = RegularGridInterpolator((np.array(growth_fracs), np.array(slr_mm)), matrix,
                                             bounds_error=False, fill_value=None)
            gg, ss = np.meshgrid(g_dense, slr_mm, indexing="ij")
            dense = np.clip(interp(np.column_stack([gg.ravel(), ss.ravel()])).reshape(gg.shape), 0.0, None)
            if dense.max() >= thr:
                ax_l.contour(np.array(slr_mm) / 1000.0, g_dense * 100.0, dense, levels=[thr],
                             colors="k", linewidths=1.8, linestyles=[STRATEGY_STYLES.get(strat, "-")])
            for ssp in ssps:
                years = {}
                for p in ("p17", "p50", "p83"):
                    eai = np.clip(interp(np.column_stack([growth_paths[ssp], slr_paths[(ssp, p)]])), 0.0, None)
                    years[p] = _first_crossing(eai, thr)
                crossings[(strat, ssp)] = years
                rows.append({"ISO": iso, "threshold": thr, "strategy": strat, "design_intensity": design,
                             "ssp": ssp, **{f"year_{p}": years[p] for p in ("p17", "p50", "p83")}})
                if years["p50"] is not None:
                    t = years["p50"]
                    ax_l.plot(np.interp(t, YEARS, slr_paths[(ssp, "p50")]) / 1000.0,
                              np.interp(t, YEARS, growth_paths[ssp]) * 100.0, "o", ms=8,
                              mfc="w", mec=s_colors.get(ssp, "grey"), mew=2.0, zorder=7)

        # SSP pathways (median SLR), years marked
        for ssp in ssps:
            x = slr_paths[(ssp, "p50")] / 1000.0
            y = growth_paths[ssp] * 100.0
            ax_l.plot(x, y, "-", color=s_colors.get(ssp, "grey"), lw=2.5, zorder=5)
            for yr in (2050, 2100):
                i = int(yr - YEARS[0])
                ax_l.plot(x[i], y[i], "s", color=s_colors.get(ssp, "grey"), ms=5, zorder=6)
            ax_l.annotate("2100", (x[-1], y[-1]), textcoords="offset points", xytext=(4, -10),
                          fontsize=fs - 3, color=s_colors.get(ssp, "grey"))
        ax_l.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax_l.set_xlim(0, 1.5)
        ax_l.set_ylim(g_lo * 100.0, g_hi * 100.0)
        ax_l.grid(alpha=0.25)
        ax_l.set_ylabel("Population change\nrelative to 2020 (%)", fontsize=fs)
        ax_l.set_title(f"{iso_to_name.get(iso, iso)}: EAI = {thr:,.0f} people/yr", fontsize=fs + 1, loc="left")
        if r == len(countries) - 1:
            ax_l.set_xlabel("Global mean sea-level rise (m)", fontsize=fs)

        # tipping years: one row per strategy, SSPs offset within the row
        offsets = np.linspace(-0.25, 0.25, len(ssps)) if len(ssps) > 1 else np.zeros(1)
        x_end = 2100
        for k, strat in enumerate(strategies):
            for ssp, off in zip(ssps, offsets):
                years = crossings.get((strat, ssp))
                if not years:
                    continue
                c = s_colors.get(ssp, "grey")
                yk = k + off
                early, late = years["p83"], years["p17"]
                if early is not None:
                    ax_r.plot([early, late if late is not None else x_end], [yk, yk], "-", color=c, lw=2.5,
                              solid_capstyle="butt")
                    if late is None:
                        ax_r.plot(x_end, yk, ">", color=c, ms=6)
                if years["p50"] is not None:
                    ax_r.plot(years["p50"], yk, "o", color=c, ms=7, mec="k", mew=0.6, zorder=5)
                elif early is None:
                    ax_r.plot(x_end, yk, ">", mfc="none", mec=c, ms=7)
        ax_r.set_yticks(range(len(strategies)))
        ax_r.set_yticklabels([STRATEGY_NAMES.get(s, s) for s in strategies], fontsize=fs)
        ax_r.set_ylim(len(strategies) - 0.5, -0.5)
        ax_r.set_xlim(2020, 2103)
        ax_r.grid(axis="x", alpha=0.3)
        ax_r.set_title(f"Year EAI first exceeds {thr:,.0f} people/yr", fontsize=fs + 1, loc="left")
        if r == len(countries) - 1:
            ax_r.set_xlabel("Year", fontsize=fs)

    handles = [Line2D([], [], color="k", lw=1.8, ls=STRATEGY_STYLES.get(s, "-"),
                      label=f"{STRATEGY_NAMES.get(s, s)} tipping isoline") for s in strategies]
    handles += [Line2D([], [], color=s_colors.get(s, "grey"), lw=2.5, label=ssp_rcp_label(s, ssp_rcp_codes))
                for s in ssps]
    handles += [Line2D([], [], ls="none", marker="s", color="k", ms=5, label="pathway in 2050 and 2100"),
                Line2D([], [], ls="none", marker="o", mfc="w", mec="k", mew=2.0, label="tipping point (median SLR)"),
                Line2D([], [], color="k", lw=2.5, label="tipping year, SLR p17-p83 range"),
                Line2D([], [], ls="none", marker=">", mfc="none", mec="k", label="not reached by 2100")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=fs - 1, frameon=False, bbox_to_anchor=(0.5, 0.0))
    design_m = f"{int(design.split('_')[1]) / 1000:g} m"
    fig.suptitle(f"Tipping points: expected annual impact under SSP-RCP scenarios and adaptation strategies "
                 f"(design for {design_m} SLR)", fontsize=fs + 3, y=1.0)
    out_png = out_dir / "tipping_points.png"
    fig.savefig(out_png, dpi=int(viz.get("dpi", 200)), bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "tipping_points.csv", index=False)
    print(f"Written: {out_png} and tipping_points.csv")


if __name__ == "__main__":
    main()
