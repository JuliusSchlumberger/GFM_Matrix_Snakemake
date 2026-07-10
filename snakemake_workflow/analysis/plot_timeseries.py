"""Time-series of EAI along SSP sea-level-rise trajectories.

Reads the pre-resolved per-scenario "ssp" files (File 3) from
compute_exposure_analysis.py: exposure_{label}_ssp.csv, one column per
(SSP, year) already resolved to that SSP's real projected SLR and
population growth factor at file-creation time — no further SLR/growth
interpolation needed here.

Usage:
    python snakemake_workflow/analysis/plot_timeseries.py \\
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
from config_utils import load_config  # noqa: E402
from visualization import ssp_rcp_label  # noqa: E402


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
    y_max: float = 2_000_000.0,
    level_linestyles: list[str] = ("--", ":", "-."),
    baseline_linestyle: str = "-",
) -> plt.Figure:
    """One figure: one sub-panel per SSP, one line + intensity envelope per measure.

    Color encodes the adaptation measure (baseline/protect/retreat/avoid,
    via measure_colors) - one fixed color per measure. Linestyle encodes
    protection level: each measure's intensity variants (e.g.
    protect_SLR_250/500/1000) are ranked by their position in
    `slr_intensities` (never by the literal SLR value string, so which
    numeric intensity counts as "protection level 1/2/3" can change in
    config without touching this code) and assigned one of
    `level_linestyles` in that order, cycling if more levels are configured
    than styles listed (visualization.level_linestyles). Besides the
    individual lines, a shaded min-to-max envelope spans each measure's
    variants at each year. baseline has no intensity variant, so it's a
    single solid line (visualization.baseline_linestyle).
    """
    level_rank = {level: i for i, level in enumerate(slr_intensities)}

    any_df = next(iter(eai_by_label.values()))
    ssps_avail = [s for s in ssps if any(c.startswith(f"EAI_{s}_") for c in any_df.columns)]
    fig, axes = plt.subplots(1, len(ssps_avail), figsize=(4 * len(ssps_avail) + 2, 5),
                              sharey=True, dpi=dpi)
    if len(ssps_avail) == 1:
        axes = [axes]

    for ax, ssp in zip(axes, ssps_avail):
        col = ssp_colors.get(ssp, "#444444")
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

        # Group labels by measure (baseline / protect / retreat / avoid) so
        # multi-intensity measures get a min-max envelope across their
        # SLR_250/500/1000 variants, in addition to each individual line.
        labels_by_measure: dict[str, list[str]] = {}
        for label in series_by_label:
            labels_by_measure.setdefault(label.split("_")[0], []).append(label)

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
                # Marker at every actual data point - population_growth.output_years
                # is not evenly spaced (5-year steps until 2050, then 2060/2075/2100),
                # so a plain line would make the long unmarked stretches (e.g.
                # 2050-2060) look like real interpolated data rather than a
                # straight connector between the two nearest computed points.
                ax.plot(s.index, s.to_numpy(), linestyle=ls, linewidth=2, color=mc,
                         marker="o", markersize=4, markeredgewidth=0,
                         alpha=0.85, label=label, zorder=2)

        ax.set_title(f"Dynamic flood risk under {ssp_rcp_label(ssp, ssp_rcp_codes)}",
                     color=col, fontsize=10)
        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylim(0, y_max)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("EAI (people)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        # Below the panels, not the top-left corner: with a fixed y_max the
        # real data often only fills the bottom fraction of each panel, so a
        # top-left legend would overlap the y-axis tick labels and the first
        # panel's title.
        fig.legend(handles, labels, loc="upper center", fontsize=8,
                   bbox_to_anchor=(0.5, 0.02), ncol=5)
    fig.suptitle(entity_label, fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def main() -> None:
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
    out_dir.mkdir(parents=True, exist_ok=True)

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

    def _save(isos, label, fname):
        fig = _timeseries_figure(eai_by_label, ssps, years, isos, label,
                                  m_colors, s_colors, slr_intensities, dpi,
                                  ssp_rcp_codes=ssp_rcp_codes, y_max=y_max,
                                  level_linestyles=level_linestyles,
                                  baseline_linestyle=baseline_linestyle)
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    _save(all_isos, "Global", out_dir / "timeseries_global.png")
    print("Global done.")

    country_dir = out_dir / "country"
    country_dir.mkdir(exist_ok=True)
    for iso in sorted(all_isos):
        _save([iso], iso, country_dir / f"timeseries_{iso}.png")
    print(f"Per-country done → {country_dir}")

    first_df = next(iter(eai_by_label.values()))
    if "Region" in first_df.columns:
        cont_dir = out_dir / "continent"
        cont_dir.mkdir(exist_ok=True)
        for reg in first_df["Region"].dropna().unique():
            reg_isos = first_df.index[first_df["Region"] == reg].tolist()
            safe = reg.replace(" ", "_").replace("/", "-")
            _save(reg_isos, reg, cont_dir / f"timeseries_{safe}.png")
        print(f"Per-continent done → {cont_dir}")

    print(f"\nAll time series written to: {out_dir}")


if __name__ == "__main__":
    main()
