"""Relative importance of sea-level rise and population change for the no-adaptation EAI.

Without adaptation, EAI = E(SLR) * P: E is the EAI at today's population (the growth
matrix's 0 % row, src/pathways.py - linear extrapolation beyond the matrix) and P the
population factor relative to 2020 (IIASA SSP projections to 2100; after 2100 the
visualization.extended_pathways.variants continuations). The change since 2020 splits exactly:
    SLR alone          (E1 - E0) * P0
    population alone   (P1 - P0) * E0
    interaction        (E1 - E0) * (P1 - P0)   new people in areas that only flood with SLR
Sea level follows the median AR6 trajectory of the SSP-RCP.

Writes to --outdir (config visualization.slr_vs_population):
  slr_vs_population_waterfall_{SSP}_2100.png       per country (top n by EAI in 2100): EAI 2020
                                                   -> + SLR alone -> + population alone ->
                                                   + interaction, whisker = SLR p17-p83
  slr_vs_population_waterfall_{SSP}_{to_year}.png  same countries/order/x-axis, reference
                                                   population variant, band = range across the
                                                   population variants
  slr_vs_population_scatter_{SSP}_2100.png         countries whose EAI changes by at least
                                                   scatter_min_change: SLR part (x, log) vs
                                                   population part (y, symmetric log, signed),
                                                   interaction split equally; marker size = EAI
                                                   2100, colour = continent
  slr_vs_population_{SSP}.csv                      every country, 2100 and each to_year variant

Usage:
    python analysis/plot_slr_vs_population.py --config snakemake_workflow/config/config.yml \\
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pathways import eai_along_pathway  # noqa: E402
from population_extension import extend_growth_factors  # noqa: E402
from population_growth import _build_name_to_iso, interpolate_growth_factor, load_ssp_growth_factors  # noqa: E402
from visualization import (  # noqa: E402
    CONTINENTS, build_growth_matrix_from_grid, iso_to_continent, load_growth_matrix_csv, load_slr_trajectories,
    ssp_rcp_label,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
C_SLR, C_POP, C_INT = "#1f77b4", "#ff7f0e", "#9467bd"
CONTINENT_COLORS = dict(zip(CONTINENTS, plt.cm.Set2.colors))


def _people(v: float, _=None) -> str:
    a = abs(v)
    s = f"{a / 1e6:g} M" if a >= 1e6 else (f"{a / 1e3:g} k" if a >= 1e3 else f"{a:g}")
    return ("-" if v < 0 else "") + s


def _decompose(mats, slr_mm, gfr, traj, ext, ssp, iso, year, variant) -> dict:
    t = traj.index.to_numpy(float)
    s = {p: float(np.interp(year, t, traj[f"{ssp}_{p}"].to_numpy(float))) for p in ("p17", "p50", "p83")}

    def E(s_mm):  # EAI at today's population for global SLR s_mm
        return float(eai_along_pathway(mats[iso], slr_mm, gfr, [s_mm], [0.0])[0])

    p1 = interpolate_growth_factor(ext[variant], ssp, iso, year, 1.0)
    e0, e1 = E(0.0), E(s["p50"])
    return {"slr_m": s["p50"] / 1000, "pop_factor": p1, "EAI_2020": e0, "EAI_end": e1 * p1,
            "EAI_end_p17": E(s["p17"]) * p1, "EAI_end_p83": E(s["p83"]) * p1,
            "slr_alone": e1 - e0, "pop_alone": (p1 - 1) * e0, "interaction": (e1 - e0) * (p1 - 1)}


def _waterfall(ref, central, band, top, names, ssp, rcp, year, to_year, variants, v0, xmax, out_path, dpi):
    rows_df = ref.loc[top] if year == 2100 else central.loc[top]
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(top) + 2.4))
    y = -np.arange(len(top), dtype=float)
    for yi, iso in zip(y, top):
        r = rows_df.loc[iso]
        x = r.EAI_2020 / 1e6
        for val, col in ((r.slr_alone, C_SLR), (r.pop_alone, C_POP), (r.interaction, C_INT)):
            w = val / 1e6
            ax.barh(yi, w, left=x, height=0.66, color=col, alpha=1.0 if w >= 0 else 0.45,
                    hatch=None if w >= 0 else "////", edgecolor="white" if w < 0 else None, lw=0)
            x += w
        if year == 2100:
            ax.plot([r.EAI_end_p17 / 1e6, r.EAI_end_p83 / 1e6], [yi, yi], color="k", lw=1.0, zorder=4)
            ax.plot(r.EAI_end / 1e6, yi, "D", ms=6, color="w", mec="k", mew=1.0, zorder=5)
        else:
            lo, hi = band.loc[iso, "min"] / 1e6, band.loc[iso, "max"] / 1e6
            ax.fill_betweenx([yi - 0.45, yi + 0.45], lo, hi, color="0.15", alpha=0.25, lw=0, zorder=4)
            ax.plot([lo, lo], [yi - 0.45, yi + 0.45], color="0.15", lw=1.0, zorder=4)
            ax.plot([hi, hi], [yi - 0.45, yi + 0.45], color="0.15", lw=1.0, zorder=4)
            ax.plot(r.EAI_end / 1e6, yi, "D", ms=6, color="k", mec="k", zorder=5)
        ax.plot(r.EAI_2020 / 1e6, yi, "|", ms=10, color="k", mew=1.5, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([names.get(i, i) for i in top], fontsize=11)
    ax.set_ylim(y[-1] - 0.7, 0.7)
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Expected annual impact, no adaptation (million people)", fontsize=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [Line2D([], [], ls="none", marker="|", ms=10, mew=1.5, color="k", label="EAI 2020"),
               Patch(color=C_SLR, label="+ SLR alone (today's population)"),
               Patch(color=C_POP, label="+ population change alone (today's sea level)"),
               Patch(color=C_INT, label="+ interaction (new people x newly flooded areas)"),
               Patch(facecolor="0.6", hatch="////", edgecolor="white", label="negative (population decline)")]
    if year == 2100:
        handles += [Line2D([], [], ls="none", marker="D", ms=6, color="w", mec="k", label="EAI 2100"),
                    Line2D([], [], color="k", lw=1.0, label="EAI 2100, SLR p17-p83")]
        title = f"{ssp_rcp_label(ssp, rcp)}: change in EAI 2020-2100"
    else:
        handles += [Line2D([], [], ls="none", marker="D", ms=6, color="k",
                           label=f"EAI {to_year}, population {variants[v0].get('label', v0)} after 2100"),
                    Patch(facecolor="0.15", alpha=0.25, edgecolor="0.15",
                          label=f"EAI {to_year}, range across population variants")]
        title = f"{ssp_rcp_label(ssp, rcp)} continued: change in EAI 2020-{to_year} (median SLR)"
    ax.set_title(title, fontsize=13, loc="left")
    ax.legend(handles=handles, loc="lower right", fontsize=9.5, frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


def _scatter(ref, names, continent, ssp, rcp, min_change, n_labels, out_path, dpi):
    d = ref.copy()
    # two-part split for two axes: the interaction shared equally between the drivers
    d["slr_part"] = d.slr_alone + d.interaction / 2
    d["pop_part"] = d.pop_alone + d.interaction / 2
    d = d[((d.EAI_end - d.EAI_2020).abs() >= min_change) & (d.slr_part > 0)]
    if d.empty:
        print(f"  No country with an EAI change >= {min_change:,.0f} - scatter skipped")
        return
    d["continent"] = [continent.get(i, "Other") for i in d.index]
    size = 30 + 420 * np.sqrt(d.EAI_end / d.EAI_end.max())
    fig, ax = plt.subplots(figsize=(13, 8.5))
    lo, hi = d.slr_part.min() / 1.6, d.slr_part.max() * 1.6
    linthresh = 1e4
    xx = np.geomspace(lo, hi, 50)
    for k, ls in ((1.0, "-"), (0.1, "--")):
        xs = xx[k * xx >= linthresh]  # only where the y-axis is logarithmic (straight lines there)
        for sign in (1, -1):
            ax.plot(xs, sign * k * xs, color="0.6", lw=0.9, ls=ls, zorder=1)
        ax.annotate("population = SLR" if k == 1 else f"population = {k:g} x SLR", (xs[len(xs) // 3], k * xs[len(xs) // 3]),
                    xytext=(0, 4), textcoords="offset points", fontsize=9, color="0.45", ha="center", va="bottom")
    ax.axhline(0, color="k", lw=0.8, zorder=1)
    for c in [c for c in CONTINENTS if c in set(d.continent)] + (["Other"] if "Other" in set(d.continent) else []):
        q = d[d.continent == c]
        ax.scatter(q.slr_part, q.pop_part, s=size[q.index], color=CONTINENT_COLORS.get(c, "0.6"),
                   edgecolor="k", lw=0.5, alpha=0.85, zorder=3)
    for iso, r in d.sort_values("EAI_end", ascending=False).head(n_labels).iterrows():
        ax.annotate(names.get(iso, iso), (r.slr_part, r.pop_part), xytext=(6, 5 if r.pop_part >= 0 else -11),
                    textcoords="offset points", fontsize=9.5, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlim(lo, hi)
    ymax = max(d.pop_part.abs().max(), d.slr_part.max()) * 1.6
    ax.set_ylim(-ymax, ymax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.text(0.01, 0.99, "population growth adds", transform=ax.transAxes, ha="left", va="top", fontsize=10,
            color="0.35", style="italic")
    ax.text(0.01, 0.01, "population decline removes", transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            color="0.35", style="italic")
    ax.set_xlabel("Change in EAI due to sea-level rise, 2020-2100 (people)", fontsize=12)
    ax.set_ylabel("Change in EAI due to population change, 2020-2100 (people)", fontsize=12)
    ax.grid(alpha=0.25, which="major")
    conts = [c for c in CONTINENTS if c in set(d.continent)]
    h_cont = [Line2D([], [], ls="none", marker="o", color=CONTINENT_COLORS[c], mec="k", ms=9, label=c) for c in conts]
    ref_sizes = [v for v in (1e5, 1e6, 5e6) if v <= d.EAI_end.max()]
    h_size = [Line2D([], [], ls="none", marker="o", color="0.75", mec="k",
                     ms=np.sqrt(30 + 420 * np.sqrt(v / d.EAI_end.max())), label=f"EAI 2100 = {_people(v)}")
              for v in ref_sizes]
    ax.legend(handles=h_cont + h_size, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10, frameon=False,
              title=f"{ssp_rcp_label(ssp, rcp)}\n{len(d)} countries with\n|change in EAI| >= {_people(min_change)}",
              title_fontsize=10, alignment="left", labelspacing=1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Written: {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True, help="directory with exposure_baseline_growth_matrix.csv")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz = cfg.get("visualization", {})
    sv = viz.get("slr_vs_population", {})
    ep = viz.get("extended_pathways", {})
    ssps = list(sv.get("ssps", ["SSP2", "SSP5"]))
    n_top = int(sv.get("waterfall_n", 20))
    min_change = float(sv.get("scatter_min_change", 1e5))
    n_labels = int(sv.get("scatter_labels", 25))
    to_year = int(ep.get("to_year", 2150))
    variants: dict[str, dict] = ep.get("variants") or {"stabilise": {"mode": "constant"}}
    v0 = next(iter(variants))
    rcp = viz.get("ssp_rcp_codes", {})
    dpi = int(viz.get("dpi", 200))
    out_dir = Path(args.outdir or viz.get("output_dir", f"{cfg['paths']['root']}/figures"))
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    growth_df = load_ssp_growth_factors(Path(catalog.get_source("ssp_population_growth_factors").path))
    ext = {v: extend_growth_factors(growth_df, spec, to_year=to_year) for v, spec in variants.items()}
    traj = load_slr_trajectories(viz["slr_trajectories_csv"])
    names = {v: k for k, v in _build_name_to_iso().items()}
    grid, slr_mm, gfr = load_growth_matrix_csv(Path(args.expdir) / "exposure_baseline_growth_matrix.csv")
    mats = {iso: build_growth_matrix_from_grid(grid.loc[[iso]], slr_mm, gfr) for iso in grid.index}
    continent = iso_to_continent(grid.index.tolist())
    cases = [(2100, v0, "2100")] + [(to_year, v, f"{to_year} {v}") for v in variants]

    for ssp in ssps:
        d = pd.DataFrame([{"ssp": ssp, "ISO": iso, "name": names.get(iso, iso), "case": lab, "year": yr,
                           "population_after_2100": v if yr > 2100 else "",
                           **_decompose(mats, slr_mm, gfr, traj, ext, ssp, iso, yr, v)}
                          for iso in grid.index for yr, v, lab in cases])
        d.to_csv(out_dir / f"slr_vs_population_{ssp}.csv", index=False)

        ref = d[d.case == "2100"].set_index("ISO")
        top = ref.EAI_end.sort_values(ascending=False).index[:n_top].tolist()
        later = d[(d.year == to_year) & d.ISO.isin(top)]
        band = later.groupby("ISO").EAI_end.agg(["min", "max"])
        central = later[later.population_after_2100 == v0].set_index("ISO")
        both = pd.concat([ref.loc[top], central.loc[top]])
        xmax = max(ref.loc[top].EAI_end_p83.max(), band["max"].max(),
                   (both.EAI_2020 + both.slr_alone + both.pop_alone.clip(lower=0)
                    + both.interaction.clip(lower=0)).max()) / 1e6 * 1.03
        for yr in (2100, to_year):
            _waterfall(ref, central, band, top, names, ssp, rcp, yr, to_year, variants, v0, xmax,
                       out_dir / f"slr_vs_population_waterfall_{ssp}_{yr}.png", dpi)
        _scatter(ref, names, continent, ssp, rcp, min_change, n_labels,
                 out_dir / f"slr_vs_population_scatter_{ssp}_2100.png", dpi)


if __name__ == "__main__":
    main()
