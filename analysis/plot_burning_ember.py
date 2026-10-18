"""Plot burning-ember (SLR x population growth -> EAI) figures from exposure CSVs.

Background: exposure_*_growth_matrix.csv (File 2); SSP overlays: SSP growth
factors + slr_trajectories_csv (p50 lines, p17-p83 uncertainty subplot).

Usage:
    python analysis/plot_burning_ember.py \\
        --config  snakemake_workflow/config/config.yml \\
        --expdir  D:/GFM/merged_results/exposure \\
        [--out-dir D:/GFM/figures]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pathways import slr_pathway  # noqa: E402
from population_extension import extend_growth_factors  # noqa: E402
from analysis_figures import (  # noqa: E402
    load_growth_matrix_csv,
    build_growth_matrix_from_grid,
    draw_ember_panel,
    plot_burning_ember,
    load_slr_trajectories,
    ssp_rcp_label,
)
from population_growth import (  # noqa: E402
    load_ssp_growth_factors,
    interpolate_growth_factor,
    _build_name_to_iso,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _ssp_growth_for_isos(growth_df, iso_codes: list[str], ssp: str, year: int) -> float:
    factors = [interpolate_growth_factor(growth_df, ssp, iso, year, 1.0) for iso in iso_codes]
    return float(np.mean(factors)) if factors else 1.0


def _build_ssp_growth(
    growth_df, slr_traj, ssps: list[str], isos: list[str], mean: bool,
) -> dict[str, pd.Series]:
    # mean=True averages over isos (aggregates); False uses isos[0].
    result: dict[str, pd.Series] = {}
    if slr_traj is None or growth_df is None:
        return result
    for ssp in ssps:
        if ssp not in slr_traj.columns:
            continue
        yrs = sorted(c for c in slr_traj.index if isinstance(c, (int, float)))
        if mean:
            result[ssp] = pd.Series(
                {yr: _ssp_growth_for_isos(growth_df, isos, ssp, int(yr)) for yr in yrs}
            )
        else:
            iso = isos[0]
            result[ssp] = pd.Series(
                {yr: interpolate_growth_factor(growth_df, ssp, iso, int(yr), 1.0) for yr in yrs}
            )
    return result


PATHWAY_YEARS = np.arange(2020, 2101)


def _draw_ssp_pathways(ax, growth_df, iso: str, slr_paths: dict[str, np.ndarray], colors: dict) -> None:
    # x = median global SLR (m), y = country's SSP population change (%).
    halo = [pe.Stroke(linewidth=4.0, foreground="w"), pe.Normal()]
    for ssp, slr in slr_paths.items():
        g = np.array([interpolate_growth_factor(growth_df, ssp, iso, int(y), 1.0) for y in PATHWAY_YEARS]) - 1.0
        x, y = slr / 1000.0, g * 100.0
        c = colors.get(ssp, "k")
        ax.plot(x, y, "-", color=c, lw=2.2, zorder=5, path_effects=halo)
        for yr in (2050, 2100):
            i = int(yr - PATHWAY_YEARS[0])
            ax.plot(x[i], y[i], "o", ms=6, mfc="w", mec=c, mew=1.8, zorder=6)


def plot_top_countries_ember(
    growth_matrix_path: Path, iso_to_name: dict[str, str], viz: dict, out_path: Path,
    n: int = 12, ncols: int = 4, vmax: float = 1e7, rank_slr_mm: float = 1400.0,
    growth_df=None, slr_traj_full: pd.DataFrame | None = None, ssps: list[str] | None = None,
    ssp_rcp_codes: dict | None = None,
) -> None:
    """Plot no-adaptation ember panels for the top-N countries by EAI.

    Ranked at the grid SLR nearest `rank_slr_mm` (mm) and growth 0%. Shared
    linear scale 0..`vmax` people (saturating), contour every vmax/10. SSP
    pathways to 2100 are overlaid only if growth_df and slr_traj_full are given.
    """
    eai_grid, slr_mm, growth_fracs = load_growth_matrix_csv(growth_matrix_path, window=viz.get("ember_display"))
    g0 = min(growth_fracs, key=abs)
    slr_rank = min(slr_mm, key=lambda v: abs(v - rank_slr_mm))
    ranked = eai_grid[(slr_rank, g0)].sort_values(ascending=False)
    top = [iso for iso in ranked.index[:n] if ranked[iso] > 0]
    matrices = {iso: build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, growth_fracs) for iso in top}
    colors = viz.get("ssp_colors") or {}
    paths = {}  # {ssp: median SLR (mm) per PATHWAY_YEARS}
    if growth_df is not None and slr_traj_full is not None:
        yrs = slr_traj_full.index.to_numpy(dtype=float)
        paths = {s: np.interp(PATHWAY_YEARS, yrs, slr_traj_full[f"{s}_p50"].to_numpy(dtype=float))
                 for s in (ssps or []) if f"{s}_p50" in slr_traj_full.columns}
    nrows = int(np.ceil(len(top) / ncols))
    fs = 11
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols + 1.2, 3.9 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.08, top=0.95, wspace=0.08, hspace=0.25)
    im = None
    for ax, iso in zip(axes.flat, top):
        im = draw_ember_panel(ax, matrices[iso], slr_mm, np.array(growth_fracs), cmap=viz.get("cmap_ember", "YlOrRd"),
                              vmax=vmax, n_contours=10, contour_labels=False, fontsize=fs)
        ax.set_title(iso_to_name.get(iso, iso), fontsize=fs + 1)
        if paths:
            _draw_ssp_pathways(ax, growth_df, iso, paths, colors)
            ax.set_ylim(min(growth_fracs) * 100.0, max(growth_fracs) * 100.0)  # clip pathways to the matrix
    for ax in axes.flat[len(top):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Global mean SLR (m)", fontsize=fs)
    if paths:
        handles = [Line2D([], [], color=colors.get(s, "k"), lw=2.2, label=ssp_rcp_label(s, ssp_rcp_codes or {}))
                   for s in paths]
        handles.append(Line2D([], [], ls="none", marker="o", mfc="w", mec="k", label="2050 and 2100 (median SLR)"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=fs, frameon=False,
                   bbox_to_anchor=(0.48, -0.01))
    for ax in axes[:, 0]:
        ax.set_ylabel("Population change\nrelative to 2020 (%)", fontsize=fs)
    cax = fig.add_axes((0.90, 0.08, 0.015, 0.87))
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.ax.tick_params(labelsize=fs - 1)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v / 1e6:g} M" if v >= 1e6 else (f"{v / 1e3:g} k" if v >= 1e3 else f"{v:g}")))
    cb.set_label("Expected annual impact (people)", fontsize=fs + 1)
    fig.savefig(out_path, dpi=int(viz.get("dpi", 200)), bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out_path.name}")


def plot_country_set_ember(
    growth_matrix_path: Path, iso_to_name: dict[str, str], viz: dict, out_path: Path,
    countries: list[str], growth_df, slr_traj_full: pd.DataFrame, ssps: list[str],
    variants: dict[str, dict], to_year: int = 2150, ncols: int = 3, vmax: float = 1e7,
    window: dict | None = None, ssp_rcp_codes: dict | None = None,
) -> None:
    """Plot no-adaptation ember panels for chosen countries with post-2100 pathways.

    Pathways: median SLR vs the country's SSP population change, solid to 2100,
    then one branch per `variants` spec (see extend_growth_factors) to `to_year`.
    Shared linear scale 0..`vmax` people; y-axis sized to each panel's pathways.
    """
    eai_grid, slr_mm, growth_fracs = load_growth_matrix_csv(growth_matrix_path, window=window)
    countries = [i for i in countries if i in eai_grid.index]
    colors = viz.get("ssp_colors") or {}
    years = np.arange(2020, to_year + 1)
    i2100 = int(2100 - years[0])
    slr_paths = {s: slr_pathway(slr_traj_full, f"{s}_p50", years) / 1000.0 for s in ssps
                 if f"{s}_p50" in slr_traj_full.columns}
    ext = {v: extend_growth_factors(growth_df, spec, to_year=to_year) for v, spec in variants.items()}
    styles = dict(zip(variants, ["--", ":", "-."]))
    halo = [pe.Stroke(linewidth=4.0, foreground="w"), pe.Normal()]

    nrows = int(np.ceil(len(countries) / ncols))
    fs = 14
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols + 1.4, 4.5 * nrows + 0.9), sharex=True,
                             squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.115, top=0.96, wspace=0.2, hspace=0.25)
    im = None
    for ax, iso in zip(axes.flat, countries):
        m = build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, growth_fracs)
        im = draw_ember_panel(ax, m, slr_mm, np.array(growth_fracs), cmap=viz.get("cmap_ember", "YlOrRd"),
                              vmax=vmax, n_contours=10, contour_labels=False, fontsize=fs)
        y_lo, y_hi = min(growth_fracs) * 100.0, max(growth_fracs) * 100.0
        for ssp, x in slr_paths.items():
            c = colors.get(ssp, "k")
            for v in variants:
                y = (np.array([interpolate_growth_factor(ext[v], ssp, iso, int(t), 1.0) for t in years]) - 1.0) * 100.0
                y_lo, y_hi = min(y_lo, y.min()), max(y_hi, y.max())
                ax.plot(x[i2100:], y[i2100:], linestyle=styles[v], color=c, lw=2.1, zorder=5, path_effects=halo)
                ax.plot(x[-1], y[-1], "s", ms=5, color=c, mec="w", mew=0.8, zorder=6)
            ax.plot(x[:i2100 + 1], y[:i2100 + 1], "-", color=c, lw=2.6, zorder=5, path_effects=halo)
            for yr in (2050, 2100):
                i = int(yr - years[0])
                ax.plot(x[i], y[i], "o", ms=6, mfc="w", mec=c, mew=1.8, zorder=7)
        ax.set_ylim(y_lo - 5, y_hi + 8)
        ax.set_xlim(0, max(max(slr_mm) / 1000.0, max(p.max() for p in slr_paths.values())))
        ax.set_title(iso_to_name.get(iso, iso), fontsize=fs + 2)
    for ax in axes.flat[len(countries):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Global mean SLR (m)", fontsize=fs)
        ax.set_xticks([0, 0.5, 1.0, 1.5])
    for ax in axes[:, 0]:
        ax.set_ylabel("Population change\nrelative to 2020 (%)", fontsize=fs)
    handles = [Line2D([], [], color=colors.get(s, "k"), lw=2.3, label=f"{ssp_rcp_label(s, ssp_rcp_codes or {})} to 2100")
               for s in slr_paths]
    handles += [Line2D([], [], color="0.3", lw=1.9, ls=styles[v], label=f"after 2100: {spec.get('label', v)}")
                for v, spec in variants.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=fs, frameon=False,
               bbox_to_anchor=(0.48, 0.055))
    cax = fig.add_axes((0.90, 0.115, 0.015, 0.845))
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.ax.tick_params(labelsize=fs)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v / 1e6:g} M" if v >= 1e6 else (f"{v / 1e3:g} k" if v >= 1e3 else f"{v:g}")))
    cb.set_label("Expected annual impact, no adaptation (people)", fontsize=fs + 1)
    fig.savefig(out_path, dpi=int(viz.get("dpi", 200)), bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out_path.name}")


def plot_country_selection_ember(
    growth_matrix_path: Path, iso_to_name: dict[str, str], viz: dict, out_path: Path,
    countries: list[str], ncols: int = 4, n_contours: int = 5,
) -> None:
    """Plot no-adaptation ember panels for chosen countries on one shared scale.

    vmax is the largest EAI among them, rounded up to a clean multiple of `n_contours`.
    """
    eai_grid, slr_mm, growth_fracs = load_growth_matrix_csv(growth_matrix_path, window=viz.get("ember_display"))
    countries = [i for i in countries if i in eai_grid.index]
    matrices = {iso: build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, growth_fracs) for iso in countries}
    top = max(float(m.max()) for m in matrices.values()) or 1.0
    step = 10.0 ** np.floor(np.log10(top / n_contours))
    step = min((k * step for k in (1, 2, 2.5, 5, 10) if k * step * n_contours >= top), default=step * 10)
    vmax = step * n_contours
    nrows = int(np.ceil(len(countries) / ncols))
    fs = 12
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols + 1.2, 3.9 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.08, top=0.95, wspace=0.12, hspace=0.25)
    im = None
    for ax, iso in zip(axes.flat, countries):
        im = draw_ember_panel(ax, matrices[iso], slr_mm, np.array(growth_fracs), cmap=viz.get("cmap_ember", "YlOrRd"),
                              vmax=vmax, n_contours=n_contours, contour_labels=True, fontsize=fs, label_fontsize=11)
        ax.set_title(iso_to_name.get(iso, iso), fontsize=fs + 1)
    for ax in axes.flat[len(countries):]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Global mean SLR (m)", fontsize=fs)
        ax.set_xticks([0.0, 0.5, 1.0, 1.5])
    for ax in axes[:, 0]:
        ax.set_ylabel("Population change\nrelative to 2020 (%)", fontsize=fs)
    cax = fig.add_axes((0.90, 0.08, 0.015, 0.87))
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=fs - 1)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v / 1e6:g} M" if v >= 1e6 else (f"{v / 1e3:g} k" if v >= 1e3 else f"{v:g}")))
    cb.set_label("Expected annual impact (people)", fontsize=fs + 1)
    fig.savefig(out_path, dpi=int(viz.get("dpi", 200)), bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out_path.name}")


def _make_ember(
    matrix, slr_mm, growth_rates, title, slr_traj, ssp_growth_series,
    slr_uncertainty, highlight_years, viz, dpi, vmax, outpath,
):
    fig = plot_burning_ember(
        matrix, slr_mm, growth_rates, title=title,
        slr_trajectories=slr_traj, ssp_growth=ssp_growth_series,
        slr_uncertainty=slr_uncertainty,
        cmap=viz.get("cmap_ember", "YlOrRd"), vmax=vmax, dpi=dpi,
        figsize=tuple(viz.get("ember_figsize", (8, 6))),
        ssp_colors=viz.get("ssp_colors"), highlight_years=highlight_years,
        n_contours=int(viz.get("n_ember_contours", 6)),
    )
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Write the figures enabled in visualization.burning_ember_figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True, help="directory with exposure_*_growth_matrix.csv files")
    parser.add_argument("--out-dir", "--outdir", dest="outdir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    viz = cfg.get("visualization", {})
    adapt_cfg = cfg.get("adaptation", {})
    pg_cfg = cfg.get("population_growth", {})

    slr_intensities = adapt_cfg.get("slr_intensities", [])
    ssps = pg_cfg.get("ssps", ["SSP1", "SSP2", "SSP3", "SSP5"])
    highlight_years = viz.get("highlight_years", [2050, 2100])

    exp_dir = Path(args.expdir)
    out_dir = Path(args.outdir or viz.get("output_dir", f"{cfg['paths']['root']}/figures"))

    # p50 drives the overlay line; p17/p50/p83 the uncertainty subplot.
    traj_path = viz.get("slr_trajectories_csv", "")
    slr_traj = None
    slr_uncertainty: dict[str, pd.DataFrame] = {}
    if traj_path and Path(traj_path).exists():
        traj_full = load_slr_trajectories(traj_path)
        p50_cols = {c: c.replace("_p50", "") for c in traj_full.columns if c.endswith("_p50")}
        slr_traj = traj_full[list(p50_cols)].rename(columns=p50_cols)
        for ssp in ssps:
            cols = {f"{ssp}_p17": "p17", f"{ssp}_p50": "p50", f"{ssp}_p83": "p83"}
            present = {c: n for c, n in cols.items() if c in traj_full.columns}
            if len(present) == 3:
                slr_uncertainty[ssp] = traj_full[list(present)].rename(columns=present)

    # Overlay only; the heatmap background comes from the growth_matrix CSV.
    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    xlsx_path = Path(catalog.get_source("ssp_population_growth_factors").path)
    growth_df = load_ssp_growth_factors(xlsx_path) if xlsx_path.exists() else None

    dpi = int(viz.get("dpi", 200))

    iso_to_name = {v: k for k, v in _build_name_to_iso().items()}

    scenario_files: dict[str, Path] = {
        "baseline": exp_dir / "exposure_baseline_growth_matrix.csv"
    }
    for slr_int in slr_intensities:
        for meas in ["protect", "retreat", "avoid", "protect_avoid"]:
            p = exp_dir / f"exposure_{meas}_{slr_int}_growth_matrix.csv"
            if p.exists():
                scenario_files[f"{meas}_{slr_int}"] = p

    country_vmax = float(viz.get("country_eai_vmax", 2_000_000.0))

    figs = viz.get("burning_ember_figures", {})
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    top_cfg = viz.get("top_countries_ember", {})
    if scenario_files["baseline"].exists() and figs.get("top_countries", False):
        n_top = int(top_cfg.get("n", 12))
        plot_top_countries_ember(
            scenario_files["baseline"], iso_to_name, viz, out_dir / f"burning_ember_top{n_top}_baseline.png",
            n=n_top, ncols=int(top_cfg.get("ncols", 4)), vmax=float(top_cfg.get("vmax", 1e7)),
            rank_slr_mm=float(top_cfg.get("rank_slr_mm", 1400)),
            growth_df=growth_df if top_cfg.get("ssp_pathways", True) else None,
            slr_traj_full=traj_full if traj_path and Path(traj_path).exists() else None,
            ssps=ssps, ssp_rcp_codes=viz.get("ssp_rcp_codes", {}),
        )
    if scenario_files["baseline"].exists():
        sel_cfg = viz.get("ember_country_selection", {})
        if sel_cfg.get("countries") and figs.get("selection", False):
            plot_country_selection_ember(scenario_files["baseline"], iso_to_name, viz,
                                         out_dir / "burning_ember_selection_baseline.png", sel_cfg["countries"],
                                         ncols=int(sel_cfg.get("ncols", 4)),
                                         n_contours=int(sel_cfg.get("n_contours", 5)))
        set_cfg = viz.get("ember_country_set", {})
        ext_cfg = viz.get("extended_pathways", {})
        if (set_cfg.get("countries") and figs.get("country_set", True) and growth_df is not None
                and traj_path and Path(traj_path).exists()):
            plot_country_set_ember(
                scenario_files["baseline"], iso_to_name, viz, out_dir / "burning_ember_country_set_baseline.png",
                set_cfg["countries"], growth_df, traj_full, list(ext_cfg.get("ssps", ["SSP2", "SSP5"])),
                ext_cfg.get("variants") or {"stabilise": {"mode": "constant"}},
                to_year=int(ext_cfg.get("to_year", 2150)), ncols=int(set_cfg.get("ncols", 3)),
                vmax=float(set_cfg.get("vmax", 1e7)), window=set_cfg.get("display"),
                ssp_rcp_codes=viz.get("ssp_rcp_codes", {}))

    for scenario_label, growth_matrix_path in (scenario_files.items() if figs.get("per_scenario", False) else []):
        if not growth_matrix_path.exists():
            print(f"  Skipping {scenario_label}: missing {growth_matrix_path.name}.")
            continue

        eai_grid, slr_mm_grid, growth_fracs = load_growth_matrix_csv(growth_matrix_path, window=viz.get("ember_display"))
        growth_rates = np.array(growth_fracs)

        retry_transient_io((out_dir / "global").mkdir, parents=True, exist_ok=True)
        all_isos = eai_grid.index.tolist()
        ssp_growth_global = _build_ssp_growth(growth_df, slr_traj, ssps, all_isos, mean=True)

        matrix = build_growth_matrix_from_grid(eai_grid, slr_mm_grid, growth_fracs)
        _make_ember(matrix, slr_mm_grid, growth_rates, f"Global — {scenario_label}",
                    slr_traj, ssp_growth_global, slr_uncertainty, highlight_years, viz, dpi,
                    float(matrix.max()) or 1.0,
                    out_dir / "global" / f"burning_ember_global_{scenario_label}.png")

        country_dir = out_dir / "country"
        retry_transient_io(country_dir.mkdir, exist_ok=True)
        for iso in eai_grid.index:
            row_grid = eai_grid.loc[[iso]]
            matrix_c = build_growth_matrix_from_grid(row_grid, slr_mm_grid, growth_fracs)
            if matrix_c.max() == 0:
                continue
            ssp_growth_c = _build_ssp_growth(growth_df, slr_traj, ssps, [iso], mean=False)
            title = f"{iso_to_name.get(iso, iso)} — {scenario_label}"
            _make_ember(matrix_c, slr_mm_grid, growth_rates, title,
                        slr_traj, ssp_growth_c, slr_uncertainty, highlight_years, viz, dpi,
                        country_vmax,
                        country_dir / f"burning_ember_{iso}_{scenario_label}.png")

        print(f"  {scenario_label}: done")

    print(f"\nFigures written to: {out_dir}")


if __name__ == "__main__":
    main()
