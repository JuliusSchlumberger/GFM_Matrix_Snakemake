"""Draw the gauge-validation figures of INPUT_ANALYSIS.md §8.7 from the exported tables.

Reads boundary_test/input_analysis_data/gauge_validation/ (`export_gauge_validation.py`) and writes
gauge_validation_{regions,tc_raise,tide_neighbours}.{png,pdf,txt} into docs/.

Usage:
    python docs/figure_scripts/plot_gauge_validation.py [--tables <dir>] [--out-dir <dir>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis"))
import _figure_style as fs  # noqa: E402

TABLES = REPO / "boundary_test" / "input_analysis_data" / "gauge_validation"
OUT_DIR = REPO / "docs"
WIDTH_IN = fs.DOUBLE_COLUMN_MM / fs.MM_PER_INCH
MM = 1 / fs.MM_PER_INCH
PRODUCT, COAST_RP = "new, r left out", "COAST-RP"
LETTER_PT = 9
# tide factor through water / within 25 km: classes symmetric in log, round ratios
FACTOR_BREAKS = (0.67, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5)
TIDE_RULES = {"25 km (production)": "within 25 km", "along water (GEBCO)": "through GEBCO water",
              "along water (OSM)": "through OSM water", "FES amplitude within x1.2": "similar FES amplitude"}
TIDE_GROUPS = (("TC group", "all", "all gauges"), ("region", "N.W.North-America (NWN)", "N.W. North America"),
               ("region", "W.North-America (WNA)", "W. North America"),
               ("region", "West&Central-Europe (WCE)", "W. & C. Europe"), ("region", "Mediterranean (MED)", "Mediterranean"),
               ("region", "S.Australia (SAU)", "S. Australia"), ("spring tidal range (m)", "(4, 20]", "spring range > 4 m"))
TC_VARIANTS = (  # (score level, form, row label, in sample)
    ("no raise", "none", "no raise", False),
    ("shift, basin left out", "shift", "shift, other basins' fit", False),
    ("shift, own basin, in sample", "shift", "shift, own-basin fit (in sample)", True),
    ("shift, own basin, gauges left out", "shift", "shift, own-basin fit", False),
    ("shift, all basins, gauges left out", "shift", "shift, all-basin fit (production)", False),
    ("surge scale, basin left out", "factor", "factor, other basins' fit", False),
    ("surge scale, own basin, gauges left out", "factor", "factor, own-basin fit", False),
    ("surge scale, all basins, gauges left out", "factor", "factor, all-basin fit", False),
)
BASIN_NAMES = {"WP": "W. Pacific", "NA": "N. Atlantic", "SP": "S. Pacific", "SI": "S. Indian", "NI": "N. Indian"}
REGION_NAMES = {"E.Asia (EAS)": "E. Asia", "E.Australia (EAU)": "E. Australia",
                "E.North-America (ENA)": "E. North America", "Equatorial.Pacific-Ocean (EPO)": "Equatorial Pacific",
                "Mediterranean (MED)": "Mediterranean", "N.Europe (NEU)": "N. Europe",
                "N.Pacific-Ocean (NPO)": "N. Pacific", "N.W.North-America (NWN)": "N.W. North America",
                "New-Zealand (NZ)": "New Zealand", "S.Australia (SAU)": "S. Australia", "S.E.Asia (SEA)": "S.E. Asia",
                "S.Pacific-Ocean (SPO)": "S. Pacific", "W.North-America (WNA)": "W. North America",
                "West&Central-Europe (WCE)": "W. & C. Europe"}


def _csv(tables: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(tables / name, keep_default_na=False)


def _dot_axes(ax, n_rows: int) -> None:
    fs.style_axes_muted(ax)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(n_rows - 0.5, -0.5)


def _caption(path: Path, lines: list[str]) -> None:
    fs.write_caption(path, [" ".join(" ".join(line.split()) for line in lines)])


def regions(tables: Path, out_dir: Path) -> None:
    summary, breakdown = _csv(tables, "levels_summary.csv"), _csv(tables, "levels_breakdown.csv")
    rows = [summary[(summary.gauges == "all") & (summary.rp == 100)].assign(label="All gauges")]
    reg = breakdown[(breakdown.split == "region") & (breakdown.rp == 100) & (breakdown.gauges != "other")]
    order = reg[reg.level == PRODUCT].sort_values("mae").gauges.tolist()
    for name in order:
        rows.append(reg[reg.gauges == name].assign(label=REGION_NAMES[name]))
    data = pd.concat(rows)
    labels = [f"{lab} ({int(n)})" for lab, n in data[data.level == PRODUCT][["label", "n"]].to_numpy()]
    y = np.arange(len(labels))
    fig, (ax_mae, ax_bias) = plt.subplots(1, 2, figsize=(WIDTH_IN, 105 * MM), sharey=True,
                                          gridspec_kw={"width_ratios": (1, 1.3)})
    for k, (level, source) in enumerate(((COAST_RP, "COAST-RP"), (PRODUCT, "GFM product"))):
        color, marker = fs.STORM_TIDE_SOURCE_STYLES[source]
        d = data[data.level == level]
        ax_mae.scatter(d.mae, y, color=color, marker=marker, s=16, zorder=3, label=source)
        off = y + (0.17 if k else -0.17)
        ax_bias.hlines(off, d.p10, d.p90, color=color, linewidth=1.0)
        ax_bias.scatter(d["median"], off, color=color, marker=marker, s=16, zorder=3)
    both = data.pivot_table(index="label", columns="level", values="mae", sort=False)
    ax_mae.hlines(y, both[PRODUCT], both[COAST_RP], color=fs.GRID, linewidth=1.5, zorder=1)
    ax_mae.set(xlim=(0, 0.6), xticks=np.arange(0, 0.61, 0.1), yticks=y, yticklabels=labels,
               xlabel="RP100 mean absolute error (m)")
    ax_bias.axvline(0, color=fs.MUTED, linewidth=0.6, zorder=1)
    ax_bias.set(xlim=(-1.0, 0.8), xticks=np.arange(-1.0, 0.81, 0.2), xlabel="RP100 station - gauge (m)")
    for ax, letter in ((ax_mae, "a"), (ax_bias, "b")):
        _dot_axes(ax, len(y))
        ax.axhline(0.5, color=fs.MUTED, linewidth=0.5)
        fs.panel_letters(ax, letter, fontsize=LETTER_PT)
    fig.legend(*ax_mae.get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2, handletextpad=0.2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out_dir / "gauge_validation_regions.png"
    fs.save_figure(fig, path)
    plt.close(fig)
    _caption(path, [
        "The GFM storm-tide product's RP100 level is closer to the tide gauges than COAST-RP's in every region.",
        f"Scored at the {int(data.n.iloc[0])} GESLA gauges that have a product station within 25 km, against the",
        "gauge's own skew-surge convolution (gauge surge on the gauge tide); each gauge's surge ratio r and tide",
        "factor are computed without the gauge. (a) Mean absolute error of station - gauge at RP100; blue circles",
        "the GFM product, orange squares COAST-RP; regions (IPCC AR6, number of gauges in brackets) sorted by the",
        "product's error, regions with fewer than 15 gauges left out. (b) Median of station - gauge at RP100",
        "(marker) and its 10-90% range across the region's gauges (line).",
    ])


def tc_raise(tables: Path, out_dir: Path) -> None:
    scores = _csv(tables, "tc_raise_cv_scores.csv")
    tcd = scores[(scores.split == "TC group") & (scores.gauges == "TC-dominated gauges")]
    boot = _csv(tables, "tc_shift_bootstrap.csv")
    shift_all = float(_csv(tables, "tc_shifts_production.csv").set_index("basin").shift_m["all"])
    y = np.arange(len(TC_VARIANTS))
    fig = plt.figure(figsize=(WIDTH_IN, 150 * MM))
    grid = fig.add_gridspec(2, 3, height_ratios=(len(TC_VARIANTS), 5.5), hspace=0.45)
    ax_mae, ax_bias, ax_rp2 = (fig.add_subplot(grid[0, k]) for k in range(3))
    ax_basin = fig.add_subplot(grid[1, 1:])
    for j, (level, form, _, in_sample) in enumerate(TC_VARIANTS):
        color, marker = fs.TC_RAISE_STYLES[form]
        face = "white" if in_sample else color
        at = {rp: tcd[(tcd.level == level) & (tcd.rp == rp)].iloc[0] for rp in (2, 100)}
        ax_mae.scatter(at[100].mae, j, color=face, edgecolors=color, marker=marker, s=20, zorder=3)
        ax_bias.hlines(j, at[100].p10, at[100].p90, color=color, linewidth=1.0)
        ax_bias.scatter(at[100]["median"], j, color=face, edgecolors=color, marker=marker, s=20, zorder=3)
        ax_rp2.scatter(at[2].mae, j, color=face, edgecolors=color, marker=marker, s=20, zorder=3)
    ax_mae.set(xlim=(0.3, 0.5), xticks=(0.3, 0.4, 0.5), xlabel="RP100 mean abs. error (m)")
    ax_bias.axvline(0, color=fs.MUTED, linewidth=0.6, zorder=1)
    ax_bias.set(xlim=(-1.0, 1.5), xticks=(-1.0, 0.0, 1.0), xlabel="RP100 station - gauge (m)")
    ax_rp2.set(xlim=(0.06, 0.11), xticks=(0.06, 0.08, 0.10), xlabel="RP2 mean abs. error (m)")
    for ax, letter in ((ax_mae, "a"), (ax_bias, "b"), (ax_rp2, "c")):
        _dot_axes(ax, len(y))
        ax.set_yticks(y, [v[2] for v in TC_VARIANTS] if ax is ax_mae else [])
        fs.panel_letters(ax, letter, fontsize=LETTER_PT)
    basins = [b for b in BASIN_NAMES if b in set(boot.basin)]
    yb = np.arange(len(basins))
    own = boot[boot.fit == "own basin"].set_index("basin")
    other = boot[boot.fit == "basin left out"].set_index("basin")
    color, marker = fs.TC_RAISE_STYLES["shift"]
    ax_basin.hlines(yb, own.p05[basins], own.p95[basins], color=color, linewidth=1.0)
    ax_basin.scatter(own.value[basins], yb, color=color, marker=marker, s=20, zorder=3,
                     label="fitted on the basin's own gauges (5-95% bootstrap)")
    ax_basin.scatter(other.value[basins], yb, color="white", edgecolors=color, marker="v", s=24, zorder=3,
                     label="fitted on the other basins' gauges")
    ax_basin.axvline(shift_all, color=fs.INK, linewidth=0.8, linestyle="--",
                     label=f"one value for all basins (production, {shift_all:.2f} m)")
    ax_basin.set(xlim=(0, 0.4), xticks=np.arange(0, 0.41, 0.1), xlabel="COAST-RP_TC shift (m)")
    ax_basin.set_yticks(yb, [f"{BASIN_NAMES[b]} ({int(own.gauges[b])})" for b in basins])
    _dot_axes(ax_basin, len(basins))
    fs.panel_letters(ax_basin, "d", fontsize=LETTER_PT)
    ax_basin.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.35, -0.28), handletextpad=0.3)
    fig.subplots_adjust(left=0.3, right=0.97, top=0.95, bottom=0.24, wspace=0.18)
    path = out_dir / "gauge_validation_tc_raise.png"
    fs.save_figure(fig, path)
    plt.close(fig)
    _caption(path, [
        "Fitted per cyclone basin, the COAST-RP_TC raise scores no better out of sample than one shift for all",
        "basins, which production uses.",
        f"(a-c) Scores at the {int(tcd.n.iloc[0])} TC-dominated gauges against the gauge's own skew-surge",
        "convolution: RP100 mean absolute error, RP100 median of station - gauge with its 10-90% range, and RP2",
        "mean absolute error. Each raise is fitted to zero the median RP100 error of the TC-dominated gauges; except",
        "where marked in sample (hollow marker), every TC-dominated gauge within 300 km of the scored gauge is left",
        "out of the fit. Black cross: no raise; blue circles: a shift added to COAST-RP_TC; green diamonds: a factor",
        "on COAST-RP_TC above its own RP2 level (its tide floor). (d) The shift per basin (number of TC-dominated",
        "gauges in brackets): fitted on the basin's own gauges (filled circle, 5-95% range of 2,000 bootstrap",
        "resamples of the gauges), fitted on the other basins' gauges (hollow triangle), and the one value fitted",
        "on all basins that production applies (dashed line). Shifts are tried from 0 to 0.4 m, so the bootstrap",
        "ranges stop at those bounds.",
    ])


def tide_neighbours(tables: Path, out_dir: Path) -> None:
    st = _csv(tables, "tide_factor_stations.csv")
    ratio = st.tide_factor_through_water / st.tide_factor_25km
    changed = ~np.isclose(ratio, 1.0)
    order = np.argsort(np.abs(np.log(ratio[changed].to_numpy())))
    levels = _csv(tables, "tide_neighbours_level_scores.csv")
    amp = _csv(tables, "tide_neighbours_amplitude_scores.csv")
    colors = (list(plt.get_cmap("Blues_r")(np.linspace(0.05, 0.75, 4))) + ["#f4f4f4"]
              + list(plt.get_cmap("Oranges")(np.linspace(0.25, 0.95, 4))))
    cmap = ListedColormap(colors[1:-1])
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    norm = BoundaryNorm(FACTOR_BREAKS, cmap.N)
    fig = plt.figure(figsize=(WIDTH_IN, 150 * MM))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.55, 1), width_ratios=(2.2, 1))
    ax_map = fig.add_subplot(grid[0, :], projection=ccrs.EqualEarth())
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor=fs.LAND_COLOR, edgecolor="none", zorder=0)
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor=fs.MUTED, zorder=1)
    ax_map.spines["geo"].set_linewidth(0.4)
    pts = st[changed].iloc[order]
    sc = ax_map.scatter(pts.lon, pts.lat, c=ratio[changed].iloc[order], cmap=cmap, norm=norm, s=4, linewidths=0,
                        transform=ccrs.PlateCarree(), zorder=3, rasterized=True)
    bar = fig.colorbar(sc, ax=ax_map, orientation="vertical", extend="both", shrink=0.8, pad=0.02,
                       ticks=FACTOR_BREAKS)
    bar.set_label("Tide factor, through water / within 25 km")
    bar.ax.set_yticklabels([f"{b:g}" for b in FACTOR_BREAKS])
    fs.panel_letters(ax_map, "a", fontsize=LETTER_PT)
    ax_lev, ax_amp = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    y = np.arange(len(TIDE_GROUPS))
    for k, (rule, name) in enumerate(TIDE_RULES.items()):
        color, marker = fs.TIDE_NEIGHBOUR_STYLES[name]
        vals = [levels[(levels.split == s) & (levels.gauges == g) & (levels.rp == 100) & (levels.level == rule)]
                .mae.iloc[0] for s, g, _ in TIDE_GROUPS]
        ax_lev.scatter(vals, y + (k - 1.5) * 0.14, color=color, marker=marker, s=14, zorder=3, label=name)
        a = amp[(amp.tide == rule)].set_index("gauges").mae_rel
        ax_amp.scatter(100 * a[["all", "gauge amplitude > 2 m"]], [0 + (k - 1.5) * 0.14, 1 + (k - 1.5) * 0.14],
                       color=color, marker=marker, s=14, zorder=3)
    n_lev = levels[(levels.split == "TC group") & (levels.gauges == "all")].n.iloc[0]
    ax_lev.set(xlim=(0, 0.4), xticks=np.arange(0, 0.41, 0.1), xlabel="RP100 mean absolute error (m)")
    ax_lev.set_yticks(y, [lab for *_, lab in TIDE_GROUPS])
    _dot_axes(ax_lev, len(y))
    n_amp = amp.set_index("gauges").n
    ax_amp.set(xlim=(0, 20), xticks=np.arange(0, 21, 5), xlabel="Tide amplitude error (%)")
    ax_amp.set_yticks([0, 1], [f"all ({int(n_amp['all'].iloc[0])})",
                               f"amplitude > 2 m ({int(n_amp['gauge amplitude > 2 m'].iloc[0])})"])
    _dot_axes(ax_amp, 2)
    for ax, letter in ((ax_lev, "b"), (ax_amp, "c")):
        fs.panel_letters(ax, letter, fontsize=LETTER_PT)
    fig.legend(*ax_lev.get_legend_handles_labels(), frameon=False, loc="lower center", ncol=4,
               handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out_dir / "gauge_validation_tide_neighbours.png"
    fs.save_figure(fig, path)
    plt.close(fig)
    _caption(path, [
        "Correcting the FES tide only with tide gauges reached through water (production) improves the fjord coasts",
        "and makes no region worse; through GEBCO water the gain comes with failures where its cells close inlets.",
        "(a) Ratio of each product station's tide factor with gauges reached through water outside the OSM land",
        f"polygons (production) to the factor with all gauges within 25 km; the {int(changed.sum())} of {len(st)}",
        "stations where it changes, larger changes drawn on top (blue: lowered, orange: raised); Equal Earth",
        f"projection. (b) RP100 mean absolute error at the {int(n_lev)} ratio gauges (gauge site left out of the",
        "tide factor), for all gauges, the regions where the rules differ and gauges with a spring tidal range",
        "above 4 m. (c) Mean absolute relative error of the corrected FES tide amplitude (sum of M2, S2, N2, K1",
        "and O1) at the usable GESLA tide gauges, each with its own site left out. Neighbour rules: all gauges",
        "within 25 km (orange square), through GEBCO 15-arcsecond water below 0 m (vermilion triangle), through",
        "water outside the OSM land polygons on 100 m cells (blue circle, production), and gauges whose FES",
        "amplitude is within a factor 1.2 of the point's (pink diamond).",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", default=str(TABLES))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    tables, out_dir = Path(args.tables), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fs.apply_style()
    for draw in (regions, tc_raise, tide_neighbours):
        draw(tables, out_dir)


if __name__ == "__main__":
    main()
