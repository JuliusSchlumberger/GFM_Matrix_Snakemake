"""Split the change in no-adaptation EAI into SLR, population and interaction terms.

Writes waterfall and scatter figures plus one CSV per SSP (visualization.slr_vs_population).

Usage:
    python analysis/plot_slr_vs_population.py --config snakemake_workflow/config/config.yml \\
        --expdir <merged_results>/exposure [--out-dir <figures>]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import _people  # noqa: E402
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pathways import eai_along_pathway  # noqa: E402
from population_extension import extend_growth_factors  # noqa: E402
from population_growth import _build_name_to_iso, interpolate_growth_factor, load_ssp_growth_factors  # noqa: E402
from analysis_figures import (  # noqa: E402
    CONTINENTS, build_growth_matrix_from_grid, iso_to_continent, load_growth_matrix_csv, load_slr_trajectories,
    ssp_rcp_label,
)
from _figure_style import (  # noqa: E402
    CONTINENT_COLORS, DOUBLE_COLUMN_MM, MM_PER_INCH, SLR_POP_DECOMPOSITION_COLORS,
    apply_style, panel_letters, save_figure, write_caption,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
C_SLR = SLR_POP_DECOMPOSITION_COLORS["slr"]
C_POP = SLR_POP_DECOMPOSITION_COLORS["pop"]
C_INT = SLR_POP_DECOMPOSITION_COLORS["interaction"]
FIG_WIDTH_IN = DOUBLE_COLUMN_MM / MM_PER_INCH
# Cosmetic cap on text-annotated countries in the dense scatter/envelope figures (section 9's
# 180 mm design width leaves no room for every qualifying country's name); all qualifying
# countries still get a marker/hull - this only trims which ones are name-labelled.
MAX_SCATTER_LABELS = 5


def _decompose(mats, slr_mm, gfr, traj, ext, ssp, iso, year, variant) -> dict:
    # EAI = E(SLR) * P, E = EAI at today's population (growth-matrix 0 % row, linearly
    # extrapolated beyond it), P = population factor vs 2020 (IIASA SSP to 2100, then the
    # extended_pathways variants), SLR = median AR6 trajectory. Exact split since 2020:
    # SLR (E1-E0)*P0, population (P1-P0)*E0, interaction (E1-E0)*(P1-P0).
    t = traj.index.to_numpy(float)
    s = {p: float(np.interp(year, t, traj[f"{ssp}_{p}"].to_numpy(float))) for p in ("p17", "p50", "p83")}

    def E(s_mm):
        return float(eai_along_pathway(mats[iso], slr_mm, gfr, [s_mm], [0.0])[0])

    p1 = interpolate_growth_factor(ext[variant], ssp, iso, year, 1.0)
    e0, e1 = E(0.0), E(s["p50"])
    return {"slr_m": s["p50"] / 1000, "pop_factor": p1, "EAI_2020": e0, "EAI_end": e1 * p1,
            "EAI_end_p17": E(s["p17"]) * p1, "EAI_end_p83": E(s["p83"]) * p1,
            "slr_alone": e1 - e0, "pop_alone": (p1 - 1) * e0, "interaction": (e1 - e0) * (p1 - 1)}


def _waterfall(ref, central, band, top, names, ssp, rcp, year, to_year, variants, v0, xmax, out_path):
    rows_df = ref.loc[top] if year == 2100 else central.loc[top]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 0.3 * len(top) + 1.6))
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
    ax.set_yticklabels([names.get(i, i) for i in top], fontsize=7)
    ax.set_ylim(y[-1] - 0.7, 0.7)
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Expected annual impact, no adaptation (million people)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
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
        message = f"{ssp_rcp_label(ssp, rcp)}: change in EAI 2020-2100"
    else:
        handles += [Line2D([], [], ls="none", marker="D", ms=6, color="k",
                           label=f"EAI {to_year}, population {variants[v0].get('label', v0)} after 2100"),
                    Patch(facecolor="0.15", alpha=0.25, edgecolor="0.15",
                          label=f"EAI {to_year}, range across population variants")]
        message = f"{ssp_rcp_label(ssp, rcp)} continued: change in EAI 2020-{to_year} (median SLR)"
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        message,
        "Horizontal bars: EAI 2020 (tick mark) extended by the SLR-alone, population-alone and "
        "interaction terms of the decomposition (colours below), to EAI at the end year (diamond).",
        f"Colours: SLR alone {C_SLR}, population alone {C_POP}, interaction {C_INT}; hatched "
        "bars mark a negative (population-decline) contribution.",
        "EAI 2100, SLR p17-p83: black line through the 2100 diamond, the AR6 median-trajectory "
        "17th-83rd percentile range." if year == 2100 else
        "Grey band: range of EAI across the configured post-2100 population variants, at median SLR.",
        f"Countries: the {len(top)} with the highest EAI in 2100, no adaptation.",
    ])
    print(f"  Written: {out_path.name}")


def _ratio_lines(ax, lo, hi, linthresh=1e4):
    xx = np.geomspace(lo, hi, 50)
    for k, ls in ((1.0, "-"), (0.1, "--")):
        xs = xx[k * xx >= linthresh]  # only where the y-axis is logarithmic (straight lines there)
        for sign in (1, -1):
            ax.plot(xs, sign * k * xs, color="0.6", lw=0.9, ls=ls, zorder=1)
        ax.annotate(f"r = {k:g}", (xs[len(xs) // 3], k * xs[len(xs) // 3]),
                    xytext=(0, 4), textcoords="offset points", fontsize=7, color="0.45",
                    ha="center", va="bottom")


def _filter_scatter(ref, continent, min_change):
    d = ref.copy()
    # interaction shared equally between the two drivers
    d["slr_part"] = d.slr_alone + d.interaction / 2
    d["pop_part"] = d.pop_alone + d.interaction / 2
    d = d[((d.EAI_end - d.EAI_2020).abs() >= min_change) & (d.slr_part > 0)]
    d["continent"] = [continent.get(i, "Other") for i in d.index]
    return d


def _scatter_panel(ax, d, names, label_isos, size_max, letter, ssp_label, min_change, M, linthresh):
    if d.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"No country with an EAI change >= {min_change:,.0f}",
                ha="center", va="center", fontsize=7, color="0.4")
        return
    size = 6 + 90 * np.sqrt(d.EAI_end / size_max)
    _ratio_lines(ax, linthresh / 5, M, linthresh)
    ax.axhline(0, color="k", lw=0.8, zorder=1)
    for c in [c for c in CONTINENTS if c in set(d.continent)] + (["Other"] if "Other" in set(d.continent) else []):
        q = d[d.continent == c]
        ax.scatter(q.slr_part, q.pop_part, s=size[q.index], color=CONTINENT_COLORS.get(c, "0.6"),
                   edgecolor="k", lw=0.4, alpha=0.85, zorder=3)
    # Labelled points, sorted so a fixed vertical stagger (below) separates ones that cluster
    # close together rather than relying on an automatic label-placement dependency.
    below = sorted((i for i in label_isos if i in d.index and d.loc[i, "pop_part"] < 0),
                   key=lambda i: d.loc[i, "pop_part"])
    above = sorted((i for i in label_isos if i in d.index and d.loc[i, "pop_part"] >= 0),
                   key=lambda i: -d.loc[i, "pop_part"])
    for group, sign in ((below, -1), (above, 1)):
        for k, iso in enumerate(group):
            r = d.loc[iso]
            near_right_edge = r.slr_part > 0.5 * M
            dy = sign * (4 + 13 * k)
            ax.annotate(names.get(iso, iso), (r.slr_part, r.pop_part),
                        xytext=(-5 if near_right_edge else 5, dy),
                        ha="right" if near_right_edge else "left",
                        textcoords="offset points", fontsize=7, zorder=4)
    # slr_part is always > 0, so x spans 0..M and y -M..M; square box rather than equal units.
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlim(0, M)
    ax.set_ylim(-M, M)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.tick_params(axis="both", labelsize=7)
    panel_letters(ax, letter, fontsize=8, label=ssp_label)
    ax.set_xlabel("Change in EAI due to sea-level rise, 2020-2100 (people)", fontsize=7)
    ax.grid(alpha=0.25, which="major")


def _scatter_figure(refs, names, continent, ssps, rcp, min_change, n_labels, out_path):
    # One shared country set: those qualifying under ANY scenario.
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
    top_labels = sorted(set().union(
        *(set(d.EAI_end.sort_values(ascending=False).index[:n_labels]) for d in panels.values())))
    # Name only the MAX_SCATTER_LABELS highest-EAI countries - every qualifying country still
    # gets a marker; this only trims text labels to fit the 180 mm panel width.
    label_isos = sorted(top_labels, key=lambda i: -max(panels[s].EAI_end.get(i, 0.0) for s in ssps))
    label_isos = label_isos[:MAX_SCATTER_LABELS]
    linthresh = 1e4
    M = max(max(d.slr_part.max(), d.pop_part.abs().max()) for d in panels.values()) * 1.3

    fig, axes = plt.subplots(1, len(ssps), figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN / len(ssps) + 1.05),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    conts_all = set()
    for i, (ax, ssp) in enumerate(zip(axes, ssps)):
        d = panels[ssp]
        _scatter_panel(ax, d, names, label_isos, size_max, chr(ord("a") + i), ssp_rcp_label(ssp, rcp),
                       min_change, M, linthresh)
        conts_all |= set(d.continent)

    conts = [c for c in CONTINENTS if c in conts_all]
    h_cont = [Line2D([], [], ls="none", marker="o", color=CONTINENT_COLORS[c], mec="k", ms=5, label=c) for c in conts]
    ref_sizes = [v for v in (1e5, 1e6, 5e6) if v <= size_max]
    h_size = [Line2D([], [], ls="none", marker="o", color="0.75", mec="k",
                     ms=np.sqrt(6 + 90 * np.sqrt(v / size_max)), label=f"EAI 2100 = {_people(v)}")
              for v in ref_sizes]
    fig.subplots_adjust(left=0.1, right=0.78, bottom=0.17, top=0.84, wspace=0.32)
    fig.supylabel("Change in EAI due to population change, 2020-2100 (people)", fontsize=7, x=0.01)
    axes[-1].legend(handles=h_cont + h_size, loc="center left", bbox_to_anchor=(1.04, 0.5),
                    fontsize=7, frameon=False, labelspacing=1.0)
    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        f"{' & '.join(ssp_rcp_label(s, rcp) for s in ssps)}: countries where the no-adaptation EAI "
        f"changes by at least {min_change:,.0f} people/yr between 2020 and 2100, split into the "
        "part driven by sea-level rise alone and the part driven by population change alone "
        "(interaction term shared equally between the two).",
        "Dashed/solid grey lines: constant ratio (0.1x, 1x) between the two parts.",
        "Marker colour: continent; marker size: EAI in 2100. Only the "
        f"{MAX_SCATTER_LABELS} highest-EAI countries are name-labelled; every qualifying country "
        "is plotted.",
    ])
    print(f"  Written: {out_path.name}")


def _envelope_points(mats, slr_mm, gfr, traj, ext, iso, year, variant, ssps) -> list[tuple[float, float, float]]:
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
                      ref_ssp, env_ssps, rcp, n_labels, out_path):
    # Top n_labels by ref_ssp EAI 2100. Hull over every (env_ssps x SLR p17/p50/p83) point, each
    # SSP with its own SLR band; dumbbell = lowest- and highest-EAI(2100) points, sized by EAI.
    ref = ref_by_ssp[ref_ssp].copy()
    ref["slr_part"] = ref.slr_alone + ref.interaction / 2
    ref["pop_part"] = ref.pop_alone + ref.interaction / 2
    top = ref.EAI_end.sort_values(ascending=False).index[:n_labels]
    d = ref.loc[top].copy()
    d["continent"] = [continent.get(i, "Other") for i in d.index]
    # Every top-n_labels country keeps its hull/dumbbell; only text labels are trimmed further
    # to fit the 180 mm panel (section 9).
    label_isos = set(d.EAI_end.sort_values(ascending=False).index[:MAX_SCATTER_LABELS])

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
        return 6 + 90 * np.sqrt(max(eai, 0.0) / eai_max)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.95))
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
    below = sorted((i for i in label_isos if upper_pt[i][1] < 0), key=lambda i: upper_pt[i][1])
    above = sorted((i for i in label_isos if upper_pt[i][1] >= 0), key=lambda i: -upper_pt[i][1])
    for group, sign in ((below, -1), (above, 1)):
        for k, iso in enumerate(group):
            ux, uy, _ = upper_pt[iso]
            near_right_edge = ux > 0.5 * M
            dy = sign * (4 + 16 * k)
            ax.annotate(names.get(iso, iso), (ux, uy), xytext=(-5 if near_right_edge else 5, dy),
                        ha="right" if near_right_edge else "left",
                        textcoords="offset points", fontsize=7, zorder=6)

    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlim(0, M)
    ax.set_ylim(-M, M)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_people))
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlabel("Change in EAI due to sea-level rise, 2020-2100 (people)", fontsize=7)
    ax.set_ylabel("Change in EAI due to population change, 2020-2100 (people)", fontsize=7)
    ax.grid(alpha=0.25, which="major")

    conts = [c for c in CONTINENTS if c in set(d.continent)]
    h_cont = [Line2D([], [], ls="none", marker="o", color=CONTINENT_COLORS[c], mec="k", ms=5, label=c) for c in conts]
    ref_sizes = [v for v in (1e5, 1e6, 5e6) if v <= eai_max]
    h_size = [Line2D([], [], ls="none", marker="o", color="0.75", mec="k",
                     ms=np.sqrt(_msize(v)), label=f"EAI 2100 = {_people(v)}")
              for v in ref_sizes]
    h_dumbbell = [Line2D([], [], color="0.4", lw=1.1, marker="o", mfc="0.75", mec="k", ms=5,
                        label="lower- to upper-bound EAI 2100\n(across env., size = that point's EAI)")]
    scen_label = " & ".join(ssp_rcp_label(s, rcp) for s in env_ssps)
    h_shape = [Patch(facecolor="0.6", alpha=0.3, edgecolor="0.6",
                     label=f"envelope: {scen_label}\n(each with its own SLR p17-p83 range)")]
    fig.subplots_adjust(left=0.1, right=0.7, bottom=0.2, top=0.97)
    ax.legend(handles=h_cont + h_size + h_dumbbell + h_shape, loc="upper left", bbox_to_anchor=(1.03, 1.0),
              fontsize=7, frameon=False, alignment="left", labelspacing=0.9)
    save_figure(fig, out_path)
    plt.close(fig)
    write_caption(out_path, [
        f"The {len(d)} highest-EAI-2100 countries under {ssp_rcp_label(ref_ssp, rcp)}, decomposed "
        "as in the scatter figure, with an envelope over every combination of "
        f"{' & '.join(ssp_rcp_label(s, rcp) for s in env_ssps)} and the AR6 SLR p17-p83 range.",
        "Shaded hull: the full envelope per country; dumbbell: the lowest- and highest-EAI(2100) "
        "point in that envelope, dot size proportional to that point's own EAI.",
        "Marker/hull colour: continent; dashed/solid grey lines: constant ratio (0.1x, 1x) "
        "between the two decomposition terms.",
        f"Only the {MAX_SCATTER_LABELS} highest-EAI countries are name-labelled; all "
        f"{len(d)} are plotted.",
    ])
    print(f"  Written: {out_path.name}")


def main() -> None:
    """Write the decomposition CSVs, waterfalls and scatter figures to --out-dir."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True, help="directory with exposure_baseline_growth_matrix.csv")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz = cfg["visualization"]
    sv = viz["slr_vs_population"]
    ep = viz["extended_pathways"]
    ssps = list(sv["ssps"])
    n_top = int(sv["waterfall_n"])
    min_change = float(sv["scatter_min_change"])
    n_labels = int(sv["scatter_labels"])
    to_year = int(ep["to_year"])
    variants: dict[str, dict] = ep.get("variants") or {"stabilise": {"mode": "constant"}}
    v0 = next(iter(variants))
    rcp = viz["ssp_rcp_codes"]
    out_dir = Path(args.out_dir or viz["output_dir"])
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    apply_style()

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
                       out_dir / f"slr_vs_population_waterfall_{ssp}_{yr}.png")

    # +1: keeps a country that narrowly missed one scenario's top list in the shared label set.
    _scatter_figure(refs_by_ssp, names, continent, ssps, rcp, min_change, n_labels + 1,
                    out_dir / f"slr_vs_population_scatter_{'_'.join(ssps)}_2100.png")

    env_ssps = ssps
    ref_ssp = "SSP2" if "SSP2" in refs_by_ssp else ssps[0]
    _scatter_envelope(refs_by_ssp, mats, slr_mm, gfr, traj, ext, v0, names, continent,
                      ref_ssp, env_ssps, rcp, n_labels,
                      out_dir / "slr_vs_population_scatter_envelope_2100.png")


if __name__ == "__main__":
    main()
