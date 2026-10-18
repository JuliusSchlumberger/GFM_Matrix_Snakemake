"""Plot EAI time series along SSP trajectories (global, per country, per continent).

Reads exposure_*_ssp.csv (File 3), already resolved to each SSP's SLR and growth.

Usage:
    python analysis/plot_timeseries.py \\
        --config  snakemake_workflow/config/config.yml \\
        --expdir  D:/GFM/merged_results/exposure \\
        [--outdir D:/GFM/figures/timeseries]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402
from analysis_figures import CONTINENTS, iso_to_continent, ssp_rcp_label  # noqa: E402


def _draw_panel(
    ax: plt.Axes,
    eai_by_label: dict[str, pd.DataFrame],
    ssp: str,
    years: list[int],
    entity_isos: list[str],
    measure_colors: dict,
    slr_intensities: list[str],
    level_linestyles=("--", ":", "-."),
    baseline_linestyle: str = "-",
) -> float:
    level_rank = {level: i for i, level in enumerate(slr_intensities)}
    data_max = 0.0
    series_by_label: dict[str, pd.Series] = {}
    for label, df in eai_by_label.items():
        sub = df.loc[df.index.isin(entity_isos)]
        if sub.empty:
            continue
        ts_vals = [
            (yr, float(sub[f"EAI_{ssp}_{yr}"].sum()))
            for yr in years if f"EAI_{ssp}_{yr}" in sub.columns
        ]
        if ts_vals:
            yrs, vals = zip(*ts_vals)
            series_by_label[label] = pd.Series(vals, index=yrs)
            data_max = max(data_max, max(vals))

    # Partition on "_SLR_", not split("_"), so protect_avoid is not filed under protect.
    labels_by_measure: dict[str, list[str]] = {}
    for label in series_by_label:
        labels_by_measure.setdefault(label.partition("_SLR_")[0], []).append(label)

    for measure, labels in labels_by_measure.items():
        mc = measure_colors.get(measure, "#999999")
        if len(labels) > 1:
            stacked = pd.concat([series_by_label[l] for l in labels], axis=1)
            ax.fill_between(stacked.index, stacked.min(axis=1), stacked.max(axis=1),
                            color=mc, alpha=0.15, zorder=1, linewidth=0)
        for label in labels:
            intensity = label[len(measure) + 1:]  # "" for baseline (no suffix)
            rank = level_rank.get(intensity)
            ls = level_linestyles[rank % len(level_linestyles)] if rank is not None else baseline_linestyle
            s = series_by_label[label]
            # Markers show the computed output_years, which need not be evenly spaced.
            ax.plot(s.index, s.to_numpy(), linestyle=ls, linewidth=2, color=mc,
                    marker="o", markersize=4, markeredgewidth=0,
                    alpha=0.85, label=label, zorder=2)
    return data_max


def _timeseries_figure(
    eai_by_label: dict[str, pd.DataFrame],
    ssps: list[str],
    years: list[int],
    entity_isos: list[str],
    entity_label: str,
    measure_colors: dict,
    ssp_colors: dict,
    slr_intensities: list[str],
    dpi: int,
    ssp_rcp_codes: dict[str, int] | None = None,
    y_max: float | None = 2_000_000.0,
    level_linestyles: list[str] = ("--", ":", "-."),
    baseline_linestyle: str = "-",
) -> plt.Figure:
    # Colour = measure; linestyle = rank of the intensity in slr_intensities (cycling
    # level_linestyles); shaded band = min-max over a measure's intensities.
    any_df = next(iter(eai_by_label.values()))
    ssps_avail = [s for s in ssps if any(c.startswith(f"EAI_{s}_") for c in any_df.columns)]
    fig, axes = plt.subplots(1, len(ssps_avail), figsize=(4 * len(ssps_avail) + 2, 5),
                              sharey=True, dpi=dpi)
    data_max = 0.0
    if len(ssps_avail) == 1:
        axes = [axes]

    for ax, ssp in zip(axes, ssps_avail):
        col = ssp_colors.get(ssp, "#444444")
        data_max = max(data_max, _draw_panel(ax, eai_by_label, ssp, years, entity_isos, measure_colors,
                                             slr_intensities, level_linestyles, baseline_linestyle))
        ax.set_title(f"Dynamic flood risk under {ssp_rcp_label(ssp, ssp_rcp_codes)}",
                     color=col, fontsize=10)
        ax.set_xlabel("Year", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes:
        ax.set_ylim(0, y_max if y_max is not None else 1.05 * data_max or 1.0)

    axes[0].set_ylabel("EAI (people)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", fontsize=8,
                   bbox_to_anchor=(0.5, 0.02), ncol=5)
    fig.suptitle(entity_label, fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def _continent_overview_figure(
    eai_by_label: dict[str, pd.DataFrame],
    continent_isos: dict[str, list[str]],
    ssps: list[str],
    years: list[int],
    measure_colors: dict,
    slr_intensities: list[str],
    dpi: int,
    ssp_rcp_codes: dict[str, int] | None = None,
    level_linestyles=("--", ":", "-."),
    baseline_linestyle: str = "-",
) -> plt.Figure:
    # Rows sorted by no-adaptation EAI in the last year (max over ssps), descending.
    last = max(years)

    def _end_baseline(isos: list[str]) -> float:
        df = eai_by_label.get("baseline")
        if df is None:
            return 0.0
        sub = df.loc[df.index.isin(isos)]
        return max((float(sub[f"EAI_{s}_{last}"].sum()) for s in ssps if f"EAI_{s}_{last}" in sub.columns),
                   default=0.0)

    conts = sorted((c for c in continent_isos if continent_isos[c]),
                   key=lambda c: _end_baseline(continent_isos[c]), reverse=True)
    fig, axes = plt.subplots(len(conts), len(ssps), figsize=(4.6 * len(ssps) + 3.2, 2.7 * len(conts) + 0.6),
                             sharex=True, sharey="row", squeeze=False, dpi=dpi)
    fmt = plt.FuncFormatter(lambda v, _: f"{v / 1e6:g} M" if v >= 1e6 else (f"{v / 1e3:g} k" if v >= 1e3 else f"{v:g}"))
    for row, cont in zip(axes, conts):
        data_max = 0.0
        for ax, ssp in zip(row, ssps):
            data_max = max(data_max, _draw_panel(ax, eai_by_label, ssp, years, continent_isos[cont], measure_colors,
                                                 slr_intensities, level_linestyles, baseline_linestyle))
            ax.grid(alpha=0.3)
            ax.set_xticks([2025, 2050, 2075, 2100])
            ax.yaxis.set_major_formatter(fmt)
        row[0].set_ylim(0, 1.05 * data_max or 1.0)
        row[0].set_ylabel("EAI (people)", fontsize=10)
        row[0].annotate(cont, xy=(0, 0.5), xycoords="axes fraction", xytext=(-62, 0), textcoords="offset points",
                        rotation=90, ha="center", va="center", fontsize=13, fontweight="bold")
    for ax, ssp in zip(axes[0], ssps):
        ax.set_title(ssp_rcp_label(ssp, ssp_rcp_codes), fontsize=12)
    for ax in axes[-1]:
        ax.set_xlabel("Year", fontsize=10)
    fig.tight_layout(rect=(0.03, 0, 1, 1))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        names = {"baseline": "No adaptation", "protect": "Protect", "avoid": "Avoid", "retreat": "Retreat",
                 "protect_avoid": "Protect + Avoid"}

        def _pretty(label: str) -> str:
            measure, _, level = label.partition("_SLR_")
            return names.get(measure, measure) + (f" ({int(level) / 1000:g} m)" if level else "")

        order = sorted(range(len(labels)), key=lambda i: (labels[i] != "baseline", labels[i]))
        fig.legend([handles[i] for i in order], [_pretty(labels[i]) for i in order], loc="center left",
                   fontsize=10, bbox_to_anchor=(1.0, 0.5), frameon=False,
                   title="Strategy (design SLR)", title_fontsize=10)
    return fig


def main() -> None:
    """Write the global, per-country, per-continent and overview time-series figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    viz = cfg.get("visualization", {})
    pg_cfg = cfg.get("population_growth", {})
    adapt_cfg = cfg.get("adaptation", {})

    ssps = pg_cfg.get("ssps", ["SSP1", "SSP2", "SSP3", "SSP5"])
    years = [int(y) for y in pg_cfg.get("output_years", list(range(2025, 2105, 5)))]
    slr_intensities = adapt_cfg.get("slr_intensities", [])

    exp_dir = Path(args.expdir)
    out_dir = Path(args.outdir or str(
        Path(viz.get("output_dir", f"{cfg['paths']['root']}/figures")) / "timeseries"
    ))
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    dpi = int(viz.get("dpi", 200))
    m_colors = viz.get("measure_colors", {})
    s_colors = viz.get("ssp_colors", {})
    ssp_rcp_codes = viz.get("ssp_rcp_codes", {})
    y_max = float(viz.get("country_eai_vmax", 2_000_000.0))
    level_linestyles = viz.get("level_linestyles", ["--", ":", "-."])
    baseline_linestyle = viz.get("baseline_linestyle", "-")

    eai_by_label: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(exp_dir.glob("exposure_*_ssp.csv")):
        label = csv_path.stem.replace("exposure_", "").replace("_ssp", "")
        eai_by_label[label] = pd.read_csv(csv_path, index_col=0)

    if not eai_by_label:
        print(f"ERROR: no exposure_*_ssp.csv files found in {exp_dir}")
        sys.exit(1)

    all_isos = next(iter(eai_by_label.values())).index.tolist()

    def _save(isos, label, fname, y_max=y_max):
        fig = _timeseries_figure(eai_by_label, ssps, years, isos, label,
                                  m_colors, s_colors, slr_intensities, dpi,
                                  ssp_rcp_codes=ssp_rcp_codes, y_max=y_max,
                                  level_linestyles=level_linestyles,
                                  baseline_linestyle=baseline_linestyle)
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    # Fixed cap keeps country figures comparable; global/continent axes scale to their data.
    _save(all_isos, "Global", out_dir / "timeseries_global.png", y_max=None)
    print("Global done.")

    country_dir = out_dir / "country"
    retry_transient_io(country_dir.mkdir, exist_ok=True)
    for iso in sorted(all_isos):
        _save([iso], iso, country_dir / f"timeseries_{iso}.png")
    print(f"Per-country done → {country_dir}")

    continent_of = iso_to_continent(all_isos)
    cont_dir = out_dir / "continent"
    retry_transient_io(cont_dir.mkdir, exist_ok=True)
    for continent in CONTINENTS:
        cont_isos = [iso for iso, c in continent_of.items() if c == continent]
        if cont_isos:
            _save(cont_isos, continent, cont_dir / f"timeseries_{continent.replace(' ', '_')}.png", y_max=None)
    print(f"Per-continent done → {cont_dir}")

    ov_cfg = viz.get("timeseries_overview", {})
    ov_ssps = [s for s in ov_cfg.get("ssps", ["SSP2", "SSP5"]) if s in ssps]
    if ov_ssps:
        fig = _continent_overview_figure(
            eai_by_label, {c: [i for i, k in continent_of.items() if k == c] for c in CONTINENTS},
            ov_ssps, years, m_colors, slr_intensities, dpi, ssp_rcp_codes=ssp_rcp_codes,
            level_linestyles=level_linestyles, baseline_linestyle=baseline_linestyle)
        fig.savefig(cont_dir / "timeseries_continents_overview.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Continent overview ({', '.join(ov_ssps)}) → {cont_dir / 'timeseries_continents_overview.png'}")

    print(f"\nAll time series written to: {out_dir}")


if __name__ == "__main__":
    main()
