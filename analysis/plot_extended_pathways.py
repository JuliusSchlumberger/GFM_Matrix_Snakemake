"""SSP pathways continued beyond 2100 through the scenario-neutral growth matrices.

The growth matrices (exposure_*_growth_matrix.csv) give EAI for any combination of global SLR
and population change, independent of any scenario. This script follows each SSP-RCP pathway
through them up to visualization.extended_pathways.to_year (default 2150): SLR from the AR6
trajectories (median, which run to 2150), population from the IIASA SSP projections to 2100 and
after that from illustrative continuation variants (src/population_extension.py - assumptions,
not projections). EAI along a pathway: src/pathways.py (bilinear interpolation; linear
extrapolation where a pathway leaves the matrix).

Writes to --outdir (default: visualization.output_dir):
  timeseries_extended_continents.png  EAI 2020-to_year per continent (rows, by EAI in 2100) and
                                      SSP (columns), per strategy in extended_pathways.strategies:
                                      one line to 2100, then one per variant with the range shaded.
  extended_pathways.csv               EAI per (strategy, ISO, SSP, variant, year), every 5 years.
  ember_extended_pathways.png         only with extended_pathways.ember: true - no-adaptation ember
                                      panels for extended_pathways.ember_countries, each SSP's
                                      pathway branching into the population variants after 2100.

Usage:
    python analysis/plot_extended_pathways.py --config snakemake_workflow/config/config.yml \\
        --expdir <merged_results>/exposure [--outdir <figures>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pathways import eai_along_pathway, slr_pathway  # noqa: E402
from population_extension import extend_growth_factors  # noqa: E402
from population_growth import _build_name_to_iso, interpolate_growth_factor, load_ssp_growth_factors  # noqa: E402
from visualization import (  # noqa: E402
    CONTINENTS, build_growth_matrix_from_grid, draw_ember_panel, iso_to_continent, load_growth_matrix_csv,
    load_slr_trajectories, ssp_rcp_label,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
LAST_SSP_YEAR = 2100
VARIANT_STYLES = ["--", ":", "-."]
PROJECTION_LINESTYLE = "--"  # timeseries_extended_continents.png's own single "this is a projection" style
STRATEGY_NAMES = {"baseline": "No adaptation", "protect": "Protect", "avoid": "Avoid", "retreat": "Retreat",
                  "protect_avoid": "Protect + Avoid"}


def _people(v: float, _=None) -> str:
    return f"{v / 1e6:g} M" if v >= 1e6 else (f"{v / 1e3:g} k" if v >= 1e3 else f"{v:g}")


def _nice_vmax(top: float, n: int) -> float:
    """Smallest n x (1, 2, 2.5 or 5 x 10^k) >= top, so n evenly spaced contours fall on round values."""
    raw = max(top, 1.0) / n
    mag = 10.0 ** np.floor(np.log10(raw))
    return n * min(k * mag for k in (1, 2, 2.5, 5, 10) if k * mag >= raw)


def _strategy_name(label: str) -> str:
    measure, _, level = label.partition("_SLR_")
    return STRATEGY_NAMES.get(measure, measure) + (f" ({int(level) / 1000:g} m)" if level else "")


def main() -> None:
    """Parse CLI arguments and plot SSP pathways continued beyond 2100 through the growth matrices.

    Follows each strategy's SSP-RCP pathway through the scenario-neutral growth matrices up
    to extended_pathways.to_year, using illustrative population continuations after 2100.
    Writes timeseries_extended_continents.png and extended_pathways.csv (and, if
    extended_pathways.ember is set, ember_extended_pathways.png) to --outdir.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True, help="directory with exposure_*_growth_matrix.csv files")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz = cfg.get("visualization", {})
    ep = viz.get("extended_pathways", {})
    to_year = int(ep.get("to_year", 2150))
    ssps = list(ep.get("ssps", ["SSP2", "SSP5"]))
    variants: dict[str, dict] = ep.get("variants") or {"stabilise": {"mode": "constant"}}
    labels = [l for l in ep.get("strategies", ["baseline"])
              if (Path(args.expdir) / f"exposure_{l}_growth_matrix.csv").exists()]
    out_dir = Path(args.outdir or viz.get("output_dir", f"{cfg['paths']['root']}/figures"))
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    s_colors = viz.get("ssp_colors", {})
    m_colors = viz.get("measure_colors", {})
    rcp = viz.get("ssp_rcp_codes", {})
    dpi = int(viz.get("dpi", 200))

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    growth_df = load_ssp_growth_factors(Path(catalog.get_source("ssp_population_growth_factors").path))
    growth_ext = {v: extend_growth_factors(growth_df, spec, to_year=to_year) for v, spec in variants.items()}
    traj = load_slr_trajectories(viz["slr_trajectories_csv"])
    if traj.index.max() < to_year:
        raise SystemExit(f"SLR trajectories end in {traj.index.max()}, before to_year {to_year}")
    years = np.arange(2020, to_year + 1)
    slr_paths = {s: slr_pathway(traj, f"{s}_p50", years) for s in ssps}
    names = {v: k for k, v in _build_name_to_iso().items()}

    grids = {l: load_growth_matrix_csv(Path(args.expdir) / f"exposure_{l}_growth_matrix.csv") for l in labels}
    isos = grids[labels[0]][0].index.tolist()
    print(f"Strategies: {labels}; matrix SLR 0-{max(grids[labels[0]][1]) / 1000:g} m, "
          f"population {min(grids[labels[0]][2]) * 100:+.0f} to {max(grids[labels[0]][2]) * 100:+.0f} %")

    def growth_path(iso: str, ssp: str, variant: str) -> np.ndarray:
        g = growth_ext[variant]
        return np.array([interpolate_growth_factor(g, ssp, iso, int(y), 1.0) for y in years]) - 1.0

    # EAI along every (strategy, country, SSP, variant) pathway
    eai: dict[tuple[str, str, str, str], np.ndarray] = {}
    for label in labels:
        eai_grid, slr_mm, gfr = grids[label]
        for iso in isos:
            m = build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, gfr)
            for ssp in ssps:
                for v in variants:
                    eai[(label, iso, ssp, v)] = eai_along_pathway(m, slr_mm, gfr, slr_paths[ssp], growth_path(iso, ssp, v))
    step = years % 5 == 0
    rows = [{"strategy": l, "ISO": i, "ssp": s, "variant": v, "year": int(y), "slr_mm": slr_paths[s][k], "EAI": e[k]}
            for (l, i, s, v), e in eai.items() for k, y in enumerate(years) if step[k]]
    pd.DataFrame(rows).to_csv(out_dir / "extended_pathways.csv", index=False)

    var_style = {v: VARIANT_STYLES[k % len(VARIANT_STYLES)] for k, v in enumerate(variants)}
    i_end = int(LAST_SSP_YEAR - years[0])

    # ── ember panels with branching pathways ──────────────────────────────────────────────
    countries = [c for c in ep.get("ember_countries", []) if c in isos] if ep.get("ember", False) else []
    if countries and "baseline" in grids:
        eai_grid, slr_mm, gfr = grids["baseline"]
        ncols = 3
        nrows = int(np.ceil(len(countries) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.6 * nrows + 0.8), squeeze=False)
        halo = [pe.Stroke(linewidth=4.0, foreground="w"), pe.Normal()]
        x_hi = max(max(slr_mm), max(p.max() for p in slr_paths.values())) / 1000.0
        for ax, iso in zip(axes.flat, countries):
            m = build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, gfr)
            im = draw_ember_panel(ax, m, slr_mm, np.array(gfr), cmap=viz.get("cmap_ember", "YlOrRd"),
                                  vmax=_nice_vmax(float(m.max()), 5), n_contours=5, contour_labels=True,
                                  fontsize=11, label_fontsize=9)
            cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
            cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(_people))
            cb.ax.tick_params(labelsize=9)
            y_lo, y_hi = min(gfr) * 100, max(gfr) * 100
            for ssp in ssps:
                c = s_colors.get(ssp, "k")
                x = slr_paths[ssp] / 1000.0
                for v in variants:
                    y = growth_path(iso, ssp, v) * 100.0
                    y_lo, y_hi = min(y_lo, y.min()), max(y_hi, y.max())
                    ax.plot(x[i_end:], y[i_end:], linestyle=var_style[v], color=c, lw=2.0, zorder=5, path_effects=halo)
                    ax.plot(x[-1], y[-1], "s", ms=5, color=c, mec="w", mew=0.8, zorder=6)
                y = growth_path(iso, ssp, next(iter(variants))) * 100.0
                ax.plot(x[:i_end + 1], y[:i_end + 1], "-", color=c, lw=2.4, zorder=5, path_effects=halo)
                ax.plot(x[i_end], y[i_end], "o", ms=6, mfc="w", mec=c, mew=1.8, zorder=7)
            ax.set_xlim(0, x_hi * 1.02)
            ax.set_ylim(y_lo - 5, y_hi + 5)
            ax.set_title(names.get(iso, iso), fontsize=13)
            ax.set_xlabel("Global mean SLR (m)", fontsize=11)
            ax.set_ylabel("Population change\nrelative to 2020 (%)", fontsize=11)
        for ax in axes.flat[len(countries):]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=s_colors.get(s, "k"), lw=2.4, label=f"{ssp_rcp_label(s, rcp)} to {LAST_SSP_YEAR}")
                   for s in ssps]
        handles += [Line2D([], [], color="0.3", lw=2.0, ls=var_style[v],
                           label=f"after {LAST_SSP_YEAR}: population {spec.get('label', v)}")
                    for v, spec in variants.items()]
        handles += [Line2D([], [], ls="none", marker="o", mfc="w", mec="0.3", mew=1.8, label=str(LAST_SSP_YEAR)),
                    Line2D([], [], ls="none", marker="s", color="0.3", label=str(to_year))]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(out_dir / "ember_extended_pathways.png", dpi=dpi)
        plt.close(fig)
        print("Written: ember_extended_pathways.png")

    # ── time series to to_year: continents (rows, by EAI in 2100) x SSPs ─────────────────
    cont_of = iso_to_continent(isos)
    groups = {c: [i for i, k in cont_of.items() if k == c] for c in CONTINENTS}
    groups = {g: v for g, v in groups.items() if v}
    sort_label = "baseline" if "baseline" in labels else labels[0]
    end_val = {g: max(sum(eai[(sort_label, i, s, v)][i_end] for i in members) for s in ssps for v in variants)
               for g, members in groups.items()}
    order = sorted(end_val, key=end_val.get, reverse=True)
    fig, axes = plt.subplots(len(order), len(ssps), figsize=(5.6 * len(ssps) + 0.8, 2.2 * len(order) + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for row, g in zip(axes, order):
        top = 0.0
        for ax, ssp in zip(row, ssps):
            for label in labels:
                c = m_colors.get(label.partition("_SLR_")[0], "#444444")
                series = {v: sum(eai[(label, i, ssp, v)] for i in groups[g]) for v in variants}
                stacked = np.vstack(list(series.values()))
                ax.plot(years[:i_end + 1], stacked[0, :i_end + 1], "-", color=c, lw=2.0)
                ax.fill_between(years[i_end:], stacked[:, i_end:].min(axis=0), stacked[:, i_end:].max(axis=0),
                                color=c, alpha=0.15, lw=0)
                for s in series.values():
                    ax.plot(years[i_end:], s[i_end:], linestyle=PROJECTION_LINESTYLE, color=c, lw=1.8)
                top = max(top, stacked.max())
            ax.grid(alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_people))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.tick_params(labelsize=12)
        # extra headroom in the first row, which holds the legend in its upper-left corner
        row[0].set_ylim(0, (1.55 if g == order[0] else 1.05) * top or 1.0)
        row[0].set_ylabel("EAI (people)", fontsize=12)
        row[0].annotate(g, xy=(0, 0.5), xycoords="axes fraction", xytext=(-78, 0), textcoords="offset points",
                        rotation=90, ha="center", va="center", fontsize=15, fontweight="bold")
    for ax, ssp in zip(axes[0], ssps):
        ax.set_title(ssp_rcp_label(ssp, rcp), fontsize=15)
    for ax in axes[-1]:
        ax.set_xlabel("Year", fontsize=13)
        ax.set_xticks(list(range(2025, to_year + 1, 25)))
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    handles = [Patch(color=m_colors.get(l.partition("_SLR_")[0], "#444444"), label=_strategy_name(l)) for l in labels]
    handles += [Line2D([], [], color="0.3", lw=2.0, label=f"to {LAST_SSP_YEAR}: SSP population")]
    handles += [Line2D([], [], color="0.3", lw=1.8, ls=PROJECTION_LINESTYLE, label=f"after {LAST_SSP_YEAR}: projection")]
    # inside the first panel's upper-left corner, where the series are still low
    axes[0, 0].legend(handles=handles, loc="upper left", ncol=2, fontsize=10.5, frameon=True, framealpha=0.9,
                      handlelength=2.2, columnspacing=1.0, borderaxespad=0.3, labelspacing=0.3)
    fig.savefig(out_dir / "timeseries_extended_continents.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: timeseries_extended_continents.png and extended_pathways.csv -> {out_dir}")


if __name__ == "__main__":
    main()
