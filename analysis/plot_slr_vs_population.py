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
  slr_vs_population_scatter_{SSPs}_2100.png        one figure, one subplot per SSP (panel
                                                   labels (a)/(b)): countries whose EAI changes
                                                   by at least scatter_min_change: SLR part
                                                   (x, log) vs population part (y, symmetric
                                                   log, signed), interaction split equally;
                                                   marker size = EAI 2100, colour = continent
  slr_vs_population_scatter_envelope_2100.png      single panel, the scatter_labels highest-EAI
                                                   countries only: dot position = SSP2 median;
                                                   dot size = EAI(2100) range across every SSP
                                                   (all configured, not just the 2 above) x SLR
                                                   p17/p83 combination; shaded convex hull around
                                                   each dot = that same combination's spread in
                                                   (SLR part, population part) space
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
from scipy.spatial import ConvexHull  # noqa: E402

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


# _scatter_panel/_scatter_envelope's figures grew (wider AND taller, same box_aspect=2
# proportions) so every one of their own fontsize= literals is scaled up by this factor
# too - otherwise the fonts shrink relative to the now-bigger plot box, which is exactly
# what happened when only figsize grew the first time this was tried.
FONT_SCALE = 1.5


def _ratio_lines(ax, lo, hi, linthresh=1e4):
    xx = np.geomspace(lo, hi, 50)
    for k, ls in ((1.0, "-"), (0.1, "--")):
        xs = xx[k * xx >= linthresh]  # only where the y-axis is logarithmic (straight lines there)
        for sign in (1, -1):
            ax.plot(xs, sign * k * xs, color="0.6", lw=0.9, ls=ls, zorder=1)
        # "population = SLR" misnamed the axes (both are EAI-change quantities, not
        # population/SLR themselves) - "r = <slope>" is just the ratio these lines plot.
        ax.annotate(f"r = {k:g}", (xs[len(xs) // 3], k * xs[len(xs) // 3]),
                    xytext=(0, 4), textcoords="offset points", fontsize=9 * FONT_SCALE, color="0.45",
                    ha="center", va="bottom")


def _filter_scatter(ref, continent, min_change):
    d = ref.copy()
    # two-part split for two axes: the interaction shared equally between the drivers
    d["slr_part"] = d.slr_alone + d.interaction / 2
    d["pop_part"] = d.pop_alone + d.interaction / 2
    d = d[((d.EAI_end - d.EAI_2020).abs() >= min_change) & (d.slr_part > 0)]
    d["continent"] = [continent.get(i, "Other") for i in d.index]
    return d


def _scatter_panel(ax, d, names, top_labels, size_max, panel_label, show_ylabel, min_change, M, linthresh):
    if d.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No country with an EAI change >= {min_change:,.0f}",
                ha="center", va="center", fontsize=11 * FONT_SCALE, color="0.4")
        return
    size = 30 + 420 * np.sqrt(d.EAI_end / size_max)
    _ratio_lines(ax, linthresh / 5, M, linthresh)
    ax.axhline(0, color="k", lw=0.8, zorder=1)
    for c in [c for c in CONTINENTS if c in set(d.continent)] + (["Other"] if "Other" in set(d.continent) else []):
        q = d[d.continent == c]
        ax.scatter(q.slr_part, q.pop_part, s=size[q.index], color=CONTINENT_COLORS.get(c, "0.6"),
                   edgecolor="k", lw=0.5, alpha=0.85, zorder=3)
    for iso in top_labels:
        if iso not in d.index:
            continue
        r = d.loc[iso]
        ax.annotate(names.get(iso, iso), (r.slr_part, r.pop_part), xytext=(6, 5 if r.pop_part >= 0 else -11),
                    textcoords="offset points", fontsize=9.5 * FONT_SCALE, zorder=4)
    # same transform (symlog, same linthresh) on both axes, x starting at 0 (not mirrored
    # negative - this data's slr_part is always positive) and y spanning -M to M. A
    # square box (x length == y length) is requested over a 2x-taller one (which would
    # match pixels-per-unit exactly, since y's -M..M span is twice x's 0..M span) - x is
    # not allowed to be the shorter side.
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlim(0, M)
    ax.set_ylim(-M, M)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.tick_params(axis="both", labelsize=10 * FONT_SCALE)
    # panel letter sits outside the axes box (upper-left), not over the data
    ax.text(-0.09, 1.03, panel_label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=16 * FONT_SCALE, fontweight="bold", clip_on=False)
    ax.set_xlabel("Change in EAI due to sea-level rise, 2020-2100 (people)", fontsize=12 * FONT_SCALE)
    if show_ylabel:
        ax.set_ylabel("Change in EAI due to population change, 2020-2100 (people)", fontsize=12 * FONT_SCALE)
    ax.grid(alpha=0.25, which="major")


def _scatter_figure(refs, names, continent, ssps, min_change, n_labels, out_path, dpi):
    # a single shared country SET for both panels - the union of whichever countries
    # qualify (|change in EAI| >= min_change) under EITHER scenario - not two
    # independently-thresholded, slightly different sets (previously 40 vs 46).
    qualifying = {ssp: _filter_scatter(refs[ssp], continent, min_change) for ssp in ssps}
    non_empty = [d for d in qualifying.values() if not d.empty]
    if not non_empty:
        print(f"  No country with an EAI change >= {min_change:,.0f} - scatter skipped")
        return
    country_union = sorted(set().union(*(set(d.index) for d in non_empty)))
    panels = {}
    for ssp in ssps:
        dd = refs[ssp].loc[country_union].copy()
        dd["slr_part"] = dd.slr_alone + dd.interaction / 2
        dd["pop_part"] = dd.pop_alone + dd.interaction / 2
        dd["continent"] = [continent.get(i, "Other") for i in dd.index]
        panels[ssp] = dd
    size_max = max(d.EAI_end.max() for d in panels.values())
    # labelled countries are also a single shared set (the union of each scenario's own
    # top n_labels by EAI_end) so both panels' labels are directly comparable.
    top_labels = sorted(set().union(
        *(set(d.EAI_end.sort_values(ascending=False).index[:n_labels]) for d in panels.values())))
    linthresh = 1e4
    M = max(max(d.slr_part.max(), d.pop_part.abs().max()) for d in panels.values()) * 1.3

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    # box_aspect=2 ties the drawn box's height to 2x its own width - widening the figure
    # without also growing its height just adds blank margin, the box itself doesn't
    # grow (that was the previous, wrong attempt here). Scaling BOTH dimensions by
    # FONT_SCALE grows the box for real, and every fontsize= in _scatter_panel/
    # _ratio_lines is scaled by that same factor so text doesn't shrink relative to it.
    fig, axes = plt.subplots(1, len(ssps), figsize=(9 * len(ssps) * FONT_SCALE, 9.5 * FONT_SCALE),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    conts_all = set()
    for i, (ax, ssp, lbl) in enumerate(zip(axes, ssps, panel_labels)):
        d = panels[ssp]
        _scatter_panel(ax, d, names, top_labels, size_max, lbl, i == 0, min_change, M, linthresh)
        conts_all |= set(d.continent)

    conts = [c for c in CONTINENTS if c in conts_all]
    h_cont = [Line2D([], [], ls="none", marker="o", color=CONTINENT_COLORS[c], mec="k", ms=9, label=c) for c in conts]
    ref_sizes = [v for v in (1e5, 1e6, 5e6) if v <= size_max]
    h_size = [Line2D([], [], ls="none", marker="o", color="0.75", mec="k",
                     ms=np.sqrt(30 + 420 * np.sqrt(v / size_max)), label=f"EAI 2100 = {_people(v)}")
              for v in ref_sizes]
    # anchored to the right panel's own axes (not a fig-level location) so it sits right
    # next to it instead of leaving a large gap to the figure's outer edge.
    axes[-1].legend(handles=h_cont + h_size, loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=10 * FONT_SCALE, frameon=False, labelspacing=1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out_path.name}")


def _envelope_points(mats, slr_mm, gfr, traj, ext, iso, year, variant, ssps) -> list[tuple[float, float, float]]:
    """(slr_part, pop_part, EAI_end) for every (ssp, SLR percentile) combination -
    the raw point cloud a country's uncertainty envelope is built from."""
    t = traj.index.to_numpy(float)

    def E(s_mm):
        return float(eai_along_pathway(mats[iso], slr_mm, gfr, [s_mm], [0.0])[0])

    e0 = E(0.0)
    pts = []
    for ssp in ssps:
        s = {p: float(np.interp(year, t, traj[f"{ssp}_{p}"].to_numpy(float))) for p in ("p17", "p50", "p83")}
        p1 = interpolate_growth_factor(ext[variant], ssp, iso, year, 1.0)
        for p in ("p17", "p50", "p83"):
            e1 = E(s[p])
            slr_alone, pop_alone = e1 - e0, (p1 - 1) * e0
            interaction = (e1 - e0) * (p1 - 1)
            pts.append((slr_alone + interaction / 2, pop_alone + interaction / 2, e1 * p1))
    return pts


def _scatter_envelope(ref_by_ssp, mats, slr_mm, gfr, traj, ext, variant, names, continent,
                      ref_ssp, env_ssps, rcp, n_labels, out_path, dpi):
    """Separate, single-panel figure for the n_labels highest-EAI countries (ranked by
    ref_ssp's own EAI 2100). Shape = convex hull over every (env_ssps x SLR p17/p50/p83)
    combination's own (slr_part, pop_part) - each env_ssps entry is a full SSP-RCP pairing
    (e.g. SSP2-RCP4.5) with its OWN SLR p17-p83 range, not one SLR band shared across
    scenarios. Two dots per country (a "dumbbell"), not one: the lower- and upper-bound
    EAI(2100) points within that same combination set, each sized like the main scatter's
    own EAI-2100 size encoding (so the two dots' size difference IS the uncertainty, not
    a separately-computed range number on a single dot)."""
    ref = ref_by_ssp[ref_ssp].copy()
    ref["slr_part"] = ref.slr_alone + ref.interaction / 2
    ref["pop_part"] = ref.pop_alone + ref.interaction / 2
    top = ref.EAI_end.sort_values(ascending=False).index[:n_labels]
    d = ref.loc[top].copy()
    d["continent"] = [continent.get(i, "Other") for i in d.index]

    x_floor = 10.0  # people - keeps a handful of near-zero/negative envelope points log-plottable
    linthresh = 1e4
    envelopes, hull_polys, lower_pt, upper_pt = {}, {}, {}, {}
    for iso in d.index:
        pts = _envelope_points(mats, slr_mm, gfr, traj, ext, iso, 2100, variant, env_ssps)
        xy = np.array([(max(x, x_floor), y) for x, y, _ in pts])
        envelopes[iso] = xy
        hull_polys[iso] = xy[ConvexHull(xy).vertices] if len(xy) >= 3 and not np.allclose(xy, xy[0]) else xy
        i_lo = min(range(len(pts)), key=lambda k: pts[k][2])
        i_hi = max(range(len(pts)), key=lambda k: pts[k][2])
        lower_pt[iso] = (max(pts[i_lo][0], x_floor), pts[i_lo][1], pts[i_lo][2])
        upper_pt[iso] = (max(pts[i_hi][0], x_floor), pts[i_hi][1], pts[i_hi][2])

    eai_max = max(upper_pt[i][2] for i in d.index)

    def _msize(eai):
        return 30 + 420 * np.sqrt(max(eai, 0.0) / eai_max)

    # both dimensions scaled by FONT_SCALE (same 2:3 ratio as the original 10x15) - see
    # _scatter_figure for why box_aspect requires growing both, not just width.
    fig, ax = plt.subplots(figsize=(12 * FONT_SCALE, 12.5 * FONT_SCALE))
    M = max(max(xy[:, 0].max(), np.abs(xy[:, 1]).max()) for xy in envelopes.values()) * 1.3
    _ratio_lines(ax, linthresh / 5, M, linthresh)
    ax.axhline(0, color="k", lw=0.8, zorder=1)

    for iso in d.index:
        poly = hull_polys[iso]
        color = CONTINENT_COLORS.get(d.loc[iso, "continent"], "0.6")
        if len(poly) >= 3:
            ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.18, zorder=2, lw=0)
            ax.plot(*np.vstack([poly, poly[:1]]).T, color=color, lw=1.0, alpha=0.6, zorder=2)

    for c in [c for c in CONTINENTS if c in set(d.continent)] + (["Other"] if "Other" in set(d.continent) else []):
        for iso in d[d.continent == c].index:
            color = CONTINENT_COLORS.get(c, "0.6")
            lx, ly, le = lower_pt[iso]
            ux, uy, ue = upper_pt[iso]
            ax.plot([lx, ux], [ly, uy], color=color, lw=1.1, alpha=0.7, zorder=4)
            ax.scatter([lx], [ly], s=_msize(le), color=color, edgecolor="k", lw=0.5, alpha=0.85, zorder=5)
            ax.scatter([ux], [uy], s=_msize(ue), color=color, edgecolor="k", lw=0.5, alpha=0.95, zorder=5)
    for iso in d.index:
        ux, uy, _ = upper_pt[iso]  # label anchored at the larger (upper-bound) dot
        ax.annotate(names.get(iso, iso), (ux, uy), xytext=(6, 5 if uy >= 0 else -11),
                    textcoords="offset points", fontsize=9.5 * FONT_SCALE, zorder=6)

    # same transform on both axes, x starting at 0, y spanning -M to M, square box - see
    # _scatter_panel for the full rationale.
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlim(0, M)
    ax.set_ylim(-M, M)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.tick_params(axis="both", labelsize=10 * FONT_SCALE)
    ax.set_xlabel("Change in EAI due to sea-level rise, 2020-2100 (people)", fontsize=12 * FONT_SCALE)
    ax.set_ylabel("Change in EAI due to population change, 2020-2100 (people)", fontsize=12 * FONT_SCALE)
    ax.grid(alpha=0.25, which="major")

    conts = [c for c in CONTINENTS if c in set(d.continent)]
    h_cont = [Line2D([], [], ls="none", marker="o", color=CONTINENT_COLORS[c], mec="k", ms=9, label=c) for c in conts]
    ref_sizes = [v for v in (1e5, 1e6, 5e6) if v <= eai_max]
    h_size = [Line2D([], [], ls="none", marker="o", color="0.75", mec="k",
                     ms=np.sqrt(_msize(v)), label=f"EAI 2100 = {_people(v)}")
              for v in ref_sizes]
    h_dumbbell = [Line2D([], [], color="0.4", lw=1.1, marker="o", mfc="0.75", mec="k", ms=7,
                        label="lower-bound to upper-bound EAI 2100\n(across env., dot size = that point's own EAI)")]
    # each env_ssps entry is already a full SSP-RCP pairing with its OWN SLR p17-p83
    # range (SSP2-RCP4.5's band differs from SSP5-RCP8.5's) - not one shared SLR band
    # applied uniformly across scenario choices.
    scen_label = " & ".join(ssp_rcp_label(s, rcp) for s in env_ssps)
    h_shape = [Patch(facecolor="0.6", alpha=0.3, edgecolor="0.6",
                     label=f"envelope: {scen_label}\n(each with its own SLR p17-p83 range)")]
    ax.legend(handles=h_cont + h_size + h_dumbbell + h_shape, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=10 * FONT_SCALE, frameon=False, alignment="left", labelspacing=1.1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out_path.name}")


def main() -> None:
    """Parse CLI arguments and plot the SLR-vs-population decomposition of no-adaptation EAI.

    For each configured SSP, decomposes the change in EAI 2020-2100 (and extended years)
    into SLR-alone, population-alone and interaction terms, then writes per-country
    waterfall plots, cross-country scatter plots and the top-N envelope scatter, plus one
    CSV per SSP, to --outdir.
    """
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

    refs_by_ssp = {}
    for ssp in ssps:
        d = pd.DataFrame([{"ssp": ssp, "ISO": iso, "name": names.get(iso, iso), "case": lab, "year": yr,
                           "population_after_2100": v if yr > 2100 else "",
                           **_decompose(mats, slr_mm, gfr, traj, ext, ssp, iso, yr, v)}
                          for iso in grid.index for yr, v, lab in cases])
        d.to_csv(out_dir / f"slr_vs_population_{ssp}.csv", index=False)

        ref = d[d.case == "2100"].set_index("ISO")
        refs_by_ssp[ssp] = ref
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

    # 19, not scatter_labels (18): SSP2's and SSP5's own top-18-by-EAI lists turned out
    # nearly identical (confirmed 2026-10) - one extra slot keeps the shared label set
    # from dropping a country that only narrowly missed either scenario's own top 18.
    _scatter_figure(refs_by_ssp, names, continent, ssps, min_change, n_labels + 1,
                    out_dir / f"slr_vs_population_scatter_{'_'.join(ssps)}_2100.png", dpi)

    # just the 2 configured SSPs (not SSP1 too) - each SSP here already carries its own
    # RCP (SSP2-RCP4.5, SSP5-RCP8.5), so this is 2 full SSP-RCP combinations, not 3.
    env_ssps = ssps
    ref_ssp = "SSP2" if "SSP2" in refs_by_ssp else ssps[0]
    _scatter_envelope(refs_by_ssp, mats, slr_mm, gfr, traj, ext, v0, names, continent,
                      ref_ssp, env_ssps, rcp, n_labels,
                      out_dir / "slr_vs_population_scatter_envelope_2100.png", dpi)


if __name__ == "__main__":
    main()
