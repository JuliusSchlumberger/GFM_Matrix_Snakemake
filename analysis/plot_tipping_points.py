"""Tipping-point figure: when does a country's expected annual impact (EAI)
exceed a fixed number of people, under each SSP-RCP scenario and adaptation
strategy?

One row per country (visualization.tipping_points.countries), two columns:
  - left:  the country's scenario-neutral SLR x population-change space with
           one tipping isoline per adaptation strategy (EAI = threshold,
           from each strategy's growth-matrix CSV at the configured design
           intensity) and each SSP's pathway through that space (global
           median SLR vs the country's projected population change). Where
           a pathway crosses a strategy's isoline, that strategy tips under
           that scenario - marked with a marker shaped by strategy
           (STRATEGY_MARKERS) and coloured by SSP. Line style = adaptation
           strategy (STRATEGY_STYLES); colour = SSP throughout.
  - right: the year of that crossing per strategy (rows) and SSP (colour):
           marker (same per-strategy shape as the left panel) = median
           (p50) SLR pathway, bar = the p17-p83 SLR range (the higher SLR
           path tips earlier); population change is the SSP's projection in
           all three. A triangle at the right edge means not reached by
           2100 (for the median, or for the p17 end of the bar).

No legend - STRATEGY_ABBREV/STRATEGY_MARKERS/STRATEGY_STYLES/SSP colours are
meant to be read directly off the right panel's row labels and the figure
caption, not re-derived from a legend.

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
        --expdir <merged_results>/exposure [--outdir <figures>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # analysis/ (for _common)
from _common import _growth_matrix_path  # noqa: E402
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pathways import eai_along_pathway, matrix_interpolator, slr_pathway  # noqa: E402
from population_extension import extend_growth_factors  # noqa: E402
from population_growth import _build_name_to_iso, interpolate_growth_factor, load_ssp_growth_factors  # noqa: E402
from analysis_figures import (  # noqa: E402
    _DEFAULT_SLR_COLORS, build_growth_matrix_from_grid, load_growth_matrix_csv, load_slr_trajectories,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
# right-column row labels (A + R: a future avoid + retreat strategy)
STRATEGY_ABBREV = {"baseline": "B", "avoid": "A", "protect": "P", "retreat": "R", "protect_avoid": "P + A",
                   "avoid_retreat": "A + R"}
STRATEGY_NAMES = {"baseline": "No adaptation", "protect": "Protect", "avoid": "Avoid", "retreat": "Retreat",
                  "protect_avoid": "Protect + Avoid"}
# isolines: colour is reserved for the SSPs, so strategies differ by line style
STRATEGY_STYLES = {"baseline": "-", "protect": "--", "avoid": ":", "retreat": "-.",
                   "protect_avoid": (0, (8, 2, 1, 2, 1, 2))}
# tipping-point markers: one shape per strategy (independent of the SSP colour
# already used for the isolines/pathways), so the SAME marker identifies a
# strategy's tipping point in both the matrix (left) and the tipping-year
# panel (right) without having to trace which isoline a dot sits on.
STRATEGY_MARKERS = {"baseline": "o", "protect": "s", "avoid": "^", "retreat": "D",
                    "protect_avoid": "P", "avoid_retreat": "X"}
YEARS = np.arange(2020, 2101)
LAST_SSP_YEAR = 2100  # IIASA SSP population projections end here


def _first_crossing(eai: np.ndarray, threshold: float, years: np.ndarray = YEARS) -> float | None:
    """First (fractional) year in `years` where `eai` reaches `threshold`, or None."""
    above = np.nonzero(eai >= threshold)[0]
    if len(above) == 0:
        return None
    k = int(above[0])
    if k == 0:
        return float(years[0])
    e0, e1 = eai[k - 1], eai[k]
    return float(years[k - 1] + (threshold - e0) / (e1 - e0))


def main() -> None:
    """Parse CLI arguments and plot each country's adaptation tipping points by scenario.

    For every country in visualization.tipping_points.countries, draws a two-panel figure:
    the SLR-vs-population tipping isolines per adaptation strategy with each SSP's pathway
    overlaid, and the year each strategy's EAI first crosses the configured threshold.
    Writes tipping_points.png and tipping_points.csv (every tipping year) to --outdir.
    """
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
    out_dir = Path(args.outdir or viz.get("output_dir", f"{cfg['paths']['root']}/figures"))
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    s_colors = viz.get("ssp_colors") or _DEFAULT_SLR_COLORS

    traj = load_slr_trajectories(viz["slr_trajectories_csv"])
    ssps = [s for s in tp.get("ssps", cfg.get("population_growth", {}).get("ssps", list(s_colors)))
            if all(f"{s}_{p}" in traj.columns for p in ("p17", "p50", "p83"))]
    to_year = min(int(tp.get("to_year", LAST_SSP_YEAR)), int(traj.index.max()))
    years = np.arange(2020, to_year + 1)
    i2100 = int(min(LAST_SSP_YEAR, to_year) - years[0])
    slr_paths = {(s, p): slr_pathway(traj, f"{s}_{p}", years) for s in ssps for p in ("p17", "p50", "p83")}  # mm

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    growth_df = load_ssp_growth_factors(Path(catalog.get_source("ssp_population_growth_factors").path))
    # population after 2100: the illustrative continuation variants (visualization.extended_pathways)
    variants = (viz.get("extended_pathways", {}).get("variants") if to_year > LAST_SSP_YEAR else None) \
        or {"ssp": {"mode": "constant", "label": "SSP"}}
    ext = {v: extend_growth_factors(growth_df, spec, to_year=to_year) for v, spec in variants.items()}
    v0 = next(iter(variants))  # reference variant: dot and thick bar
    iso_to_name = {v: k for k, v in _build_name_to_iso().items()}

    grids = {s: load_growth_matrix_csv(_growth_matrix_path(exp_dir, s, design)) for s in strategies}

    fs = 11
    height = 4.0 * len(countries) + 1.6
    fig, axes = plt.subplots(len(countries), 2, figsize=(15, height), sharex="col",
                             gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.12, "hspace": 0.05}, squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=1 - 0.1 / height, bottom=0.5 / height)
    rows = []
    for r, iso in enumerate(countries):
        thr = thr_by_iso.get(iso, default_thr)
        ax_l, ax_r = axes[r]
        growth_paths = {(s, v): np.array([interpolate_growth_factor(ext[v], s, iso, int(y), 1.0) for y in years]) - 1.0
                        for s in ssps for v in variants}  # fraction
        g_lo = min(-0.5, min(p.min() for p in growth_paths.values()) - 0.05)
        g_hi = max(1.5, max(p.max() for p in growth_paths.values()) + 0.05)
        g_dense = np.linspace(g_lo, g_hi, 121)

        crossings: dict[tuple[str, str], dict[tuple[str, str], float | None]] = {}
        for k, strat in enumerate(strategies):
            eai_grid, slr_mm, growth_fracs = grids[strat]
            if iso not in eai_grid.index:
                continue
            matrix = build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, growth_fracs)
            interp = matrix_interpolator(matrix, slr_mm, growth_fracs)
            gg, ss = np.meshgrid(g_dense, slr_mm, indexing="ij")
            dense = np.clip(interp(np.column_stack([gg.ravel(), ss.ravel()])).reshape(gg.shape), 0.0, None)
            if dense.max() >= thr:
                ax_l.contour(np.array(slr_mm) / 1000.0, g_dense * 100.0, dense, levels=[thr],
                             colors="k", linewidths=1.8, linestyles=[STRATEGY_STYLES.get(strat, "-")])
            for ssp in ssps:
                yrs = {}
                for v in variants:
                    for p in ("p17", "p50", "p83"):
                        eai = eai_along_pathway(matrix, slr_mm, growth_fracs, slr_paths[(ssp, p)], growth_paths[(ssp, v)])
                        yrs[(v, p)] = _first_crossing(eai, thr, years)
                    rows.append({"ISO": iso, "threshold": thr, "strategy": strat, "design_intensity": design,
                                 "ssp": ssp, "population_after_2100": v,
                                 **{f"year_{p}": yrs[(v, p)] for p in ("p17", "p50", "p83")}})
                    t = yrs[(v, "p50")]
                    if t is not None and v == v0:  # tipping point on the reference pathway
                        ax_l.plot(np.interp(t, years, slr_paths[(ssp, "p50")]) / 1000.0,
                                  np.interp(t, years, growth_paths[(ssp, v)]) * 100.0,
                                  STRATEGY_MARKERS.get(strat, "o"), ms=14,
                                  mfc="w", mec=s_colors.get(ssp, "grey"), mew=2.6, zorder=7)
                crossings[(strat, ssp)] = yrs

        # SSP pathways (median SLR): solid to 2100, then one branch per population variant
        for ssp in ssps:
            c = s_colors.get(ssp, "grey")
            x = slr_paths[(ssp, "p50")] / 1000.0
            y = growth_paths[(ssp, v0)] * 100.0
            ax_l.plot(x[:i2100 + 1], y[:i2100 + 1], "-", color=c, lw=2.5, zorder=5)
            if to_year > LAST_SSP_YEAR:  # envelope over the population variants after 2100
                yv = np.vstack([growth_paths[(ssp, v)][i2100:] for v in variants]) * 100.0
                ax_l.fill_between(x[i2100:], yv.min(axis=0), yv.max(axis=0), color=c, alpha=0.25, lw=0, zorder=4)
                ax_l.plot(x[i2100:], yv.min(axis=0), "-", color=c, lw=0.8, zorder=4)
                ax_l.plot(x[i2100:], yv.max(axis=0), "-", color=c, lw=0.8, zorder=4)
        ax_l.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax_l.set_xlim(0, 1.5)
        ax_l.set_ylim(g_lo * 100.0, g_hi * 100.0)
        ax_l.grid(alpha=0.25)
        ax_l.set_ylabel("Population change\nrelative to 2020 (%)", fontsize=fs)
        ax_l.annotate(iso_to_name.get(iso, iso), xy=(0, 0.5), xycoords="axes fraction", xytext=(-72, 0),
                      textcoords="offset points", rotation=90, ha="center", va="center", fontsize=fs + 3,
                      fontweight="bold")
        ax_l.text(0.98, 0.96, f"tipping point: EAI = {thr:,.0f} people/yr", transform=ax_l.transAxes,
                  ha="right", va="top", fontsize=fs, bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9, pad=3))
        if r == len(countries) - 1:
            ax_l.set_xlabel("Global mean sea-level rise (m)", fontsize=fs)

        # tipping years: one row per strategy, SSPs offset within the row
        offsets = np.linspace(-0.25, 0.25, len(ssps)) if len(ssps) > 1 else np.zeros(1)
        x_end = float(to_year)
        for k, strat in enumerate(strategies):
            for ssp, off in zip(ssps, offsets):
                yrs = crossings.get((strat, ssp))
                if not yrs:
                    continue
                c = s_colors.get(ssp, "grey")
                yk = k + off
                # one bar over the whole uncertainty: SLR p17-p83 x population variants after 2100
                all_y = [t for t in yrs.values() if t is not None]
                if all_y:
                    open_end = any(t is None for t in yrs.values())
                    ax_r.plot([min(all_y), x_end if open_end else max(all_y)], [yk, yk], "-", color=c, lw=2.5,
                              solid_capstyle="butt")
                    if open_end:
                        ax_r.plot(x_end, yk, ">", color=c, ms=9)
                if yrs[(v0, "p50")] is not None:
                    ax_r.plot(yrs[(v0, "p50")], yk, STRATEGY_MARKERS.get(strat, "o"), color=c, ms=12,
                              mec="k", mew=0.8, zorder=5)
                elif not all_y:
                    ax_r.plot(x_end, yk, ">", mfc="none", mec=c, ms=10)
        if to_year > LAST_SSP_YEAR:
            ax_r.axvline(LAST_SSP_YEAR, color="0.4", lw=0.8, ls=":")
        ax_r.set_yticks(range(len(strategies)))
        ax_r.set_yticklabels([STRATEGY_ABBREV.get(s, s) for s in strategies], fontsize=fs + 2, fontweight="bold")
        ax_r.set_ylim(len(strategies) - 0.5, -0.5)
        ax_r.set_xlim(2020, to_year + 3)
        ax_r.grid(axis="x", alpha=0.3)
        if r == len(countries) - 1:
            ax_r.set_xlabel("Year EAI first exceeds the tipping point", fontsize=fs)

    out_png = out_dir / "tipping_points.png"
    fig.savefig(out_png, dpi=int(viz.get("dpi", 200)), bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "tipping_points.csv", index=False)
    print(f"Written: {out_png} and tipping_points.csv")


if __name__ == "__main__":
    main()
