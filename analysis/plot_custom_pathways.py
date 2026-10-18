"""Plot DAPP metro-map pathways from compute_custom_pathways.py CSVs (one track per measure).

Style after https://github.com/JuliusSchlumberger/Pathways_Analysis_Dashboard
(scripts/PathwaysMaps). Writes a combined countries figure and one continents figure per SSP.

Usage:
    python analysis/plot_custom_pathways.py --csv-ssp2 <figures>/custom_pathways_SSP2.csv \\
        --csv-ssp5 <figures>/custom_pathways_SSP5.csv [--to-year 2150] [--out-dir <figures>]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from population_growth import _build_name_to_iso  # noqa: E402

SSP_DISPLAY = {"SSP2": "SSP2-4.5", "SSP5": "SSP5-8.5"}

MEASURE_COLORS = {"baseline": "0.55"}
MEASURE_LABEL = {"avoid": "Avoid", "protect": "Protect", "retreat": "Retreat", "protect_avoid": "Protect + Avoid"}
# ColorBrewer "Paired": small/large share a hue.
MEASURE_PAIR_COLORS = {
    "avoid": {"small": "#A6CEE3", "large": "#1F78B4"},
    "protect": {"small": "#FDBF6F", "large": "#FF7F00"},
    "protect_avoid": {"small": "#CAB2D6", "large": "#6A3D9A"},
    "retreat": {"small": "#B2DF8A", "large": "#33A02C"},
}
# Top to bottom; intensity is the CSV's "small"/"large" label, not the SLR_* config value.
TRACKS = [("baseline", None), ("avoid", "small"), ("avoid", "large"), ("protect", "small"),
         ("protect", "large"), ("protect_avoid", "small"), ("protect_avoid", "large"),
         ("retreat", "small"), ("retreat", "large")]
TRACK_Y = {b: len(TRACKS) - i for i, b in enumerate(TRACKS)}
TRACK_LABEL = {("baseline", None): "Baseline"}
for _m in ("avoid", "protect", "retreat"):
    TRACK_LABEL[(_m, "small")] = f"{MEASURE_LABEL[_m]} (small)"
    TRACK_LABEL[(_m, "large")] = f"{MEASURE_LABEL[_m]} (large)"
TRACK_LABEL[("protect_avoid", "small")] = f"{MEASURE_LABEL['protect_avoid']}\n(small)"
TRACK_LABEL[("protect_avoid", "large")] = f"{MEASURE_LABEL['protect_avoid']}\n(large)"

TRACK_COLOR: dict[tuple[str, str | None], tuple[float, float, float] | str] = {
    ("baseline", None): MEASURE_COLORS["baseline"],
}
for _m, _pair in MEASURE_PAIR_COLORS.items():
    TRACK_COLOR[(_m, "small")] = _pair["small"]
    TRACK_COLOR[(_m, "large")] = _pair["large"]

# Shared segments overlap into one line. A tipping point runs TIP_OVERSHOOT_YEARS past its
# end year, capped by a black bar; the connector still leaves at the real switch year.
LINE_LW = 4.0
TIP_OVERSHOOT_YEARS = 2.5
TICK_HALF_HEIGHT = 0.32
ARROW_MS = 16
START_MS = 13


def _block_parts(block: str) -> tuple[str, str]:
    measure, _, intensity = block.rpartition("_")
    return measure, intensity


def chain_segments(row: pd.Series, to_year: int) -> tuple[list[tuple[str, float, float]], bool]:
    """Return (block, start_year, end_year) segments of one sequence and whether it succeeds."""
    blocks = [b.strip() for b in row["chain"].split(">")]
    raw = row["stage_years"]
    switch_years = [float(y) for y in str(raw).split(";") if y.strip()] if pd.notna(raw) and str(raw).strip() else []
    success = str(row["outcome"]).startswith("success")
    boundaries = [float(row["baseline_tip_year"])] + switch_years + ([float(to_year)] if success else [])
    return list(zip(blocks, boundaries[:-1], boundaries[1:])), success


YTICK_FONTSIZE = 15
AXIS_FONTSIZE = 17
ANNOT_FONTSIZE = 15
TITLE_FONTSIZE = 19


def _style_axes(ax: plt.Axes, to_year: int, title: str | None, show_yticklabels: bool = True) -> None:
    ax.set_xlim(2020, to_year + 1)
    ax.set_yticks([TRACK_Y[t] for t in TRACKS])
    ax.tick_params(axis="y", length=0)
    if show_yticklabels:
        # Re-setting labels would undo sharey's auto-hide on other columns.
        labels = ax.set_yticklabels([TRACK_LABEL[t] for t in TRACKS], fontsize=YTICK_FONTSIZE)
        for lbl, t in zip(labels, TRACKS):
            lbl.set_color(TRACK_COLOR[t])  # the legend does not name measure colours
    ax.set_ylim(0.3, len(TRACKS) + 0.7)
    ax.tick_params(axis="x", labelsize=AXIS_FONTSIZE - 1)
    ax.grid(axis="x", alpha=0.25)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.plot(1, 0, ">", color="black", ms=9, transform=ax.transAxes, clip_on=False, zorder=10)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")


def plot_entity_panel(ax: plt.Axes, title: str | None, g: pd.DataFrame, to_year: int,
                      show_yticklabels: bool = True) -> None:
    """Draw one entity's sequences; a never-tipping baseline gets a bare trunk panel."""
    base_track = TRACK_Y[("baseline", None)]

    if g.empty or (g["outcome"] == "no_adaptation_needed").all():
        ax.plot([2020, to_year], [base_track, base_track], color=MEASURE_COLORS["baseline"],
                lw=LINE_LW, zorder=2, solid_capstyle="butt")
        ax.plot(to_year, base_track, ">", ms=ARROW_MS, color=MEASURE_COLORS["baseline"], zorder=5)
        ax.text(0.5, 0.5, f"No adaptation needed by {to_year}", transform=ax.transAxes,
                ha="center", va="center", fontsize=ANNOT_FONTSIZE + 2, color="0.4", style="italic")
        _style_axes(ax, to_year, title, show_yticklabels)
        return

    baseline_tip = float(g["baseline_tip_year"].iloc[0])
    # One shared baseline trunk for all sequences, with its own tipping bar.
    base_x_end = baseline_tip + TIP_OVERSHOOT_YEARS
    ax.plot([2020, base_x_end], [base_track, base_track], color=MEASURE_COLORS["baseline"],
            lw=LINE_LW, zorder=2, solid_capstyle="butt")
    ax.plot([base_x_end, base_x_end], [base_track - TICK_HALF_HEIGHT, base_track + TICK_HALF_HEIGHT],
            color="black", lw=LINE_LW, zorder=6, solid_capstyle="butt")

    for _, row in g.sort_values("sequence_id").iterrows():
        segments, success = chain_segments(row, to_year)
        prev_track = base_track
        for i, (block, y0, y1) in enumerate(segments):
            measure, intensity = _block_parts(block)
            track = TRACK_Y[(measure, intensity)]
            is_last = i == len(segments) - 1
            c = TRACK_COLOR[(measure, intensity)]
            if track != prev_track:
                # connectors are grey: they mark a transition, not a measure
                ax.plot([y0, y0], [prev_track, track], color=MEASURE_COLORS["baseline"],
                        lw=LINE_LW, zorder=2)
            ax.plot(y0, track, "o", ms=START_MS, mfc=c, mec=c, zorder=5)
            if is_last and success:
                ax.plot([y0, y1], [track, track], color=c, lw=LINE_LW,
                        zorder=3, solid_capstyle="butt")
                ax.plot(y1, track, ">", ms=ARROW_MS, color=c, zorder=5)
            else:
                x_end = y1 + TIP_OVERSHOOT_YEARS
                ax.plot([y0, x_end], [track, track], color=c, lw=LINE_LW,
                        zorder=3, solid_capstyle="butt")
                ax.plot([x_end, x_end], [track - TICK_HALF_HEIGHT, track + TICK_HALF_HEIGHT],
                        color="black", lw=LINE_LW, zorder=6, solid_capstyle="butt")
            prev_track = track

    ax.axvline(baseline_tip, color=MEASURE_COLORS["baseline"], lw=0.8, ls=":", zorder=1)
    _style_axes(ax, to_year, title, show_yticklabels)


def _add_legend(fig: plt.Figure, to_year: int, bbox_y: float = 0.02) -> None:
    handles = [Line2D([], [], ls="none", marker="o", mfc="0.3", mec="0.3", ms=START_MS, label="new strategy implemented"),
              Line2D([], [], ls="none", marker="|", color="black", mew=LINE_LW, ms=14, label="tipping point"),
              Line2D([], [], ls="none", marker=">", color="0.3", ms=ARROW_MS, label=f"holds beyond {to_year}")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=AXIS_FONTSIZE, frameon=False,
              bbox_to_anchor=(0.5, bbox_y))


def make_figure(sub: pd.DataFrame, entities: list[str], titles: dict[str, str], to_year: int, out_path: Path, dpi: int) -> None:
    """Plot one panel per entity in a grid of up to 3 columns."""
    ncols = min(3, len(entities))
    nrows = int(np.ceil(len(entities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 5.6 * nrows), squeeze=False,
                             sharex=True, sharey=True)
    # last filled row per column (ragged last row)
    col_last_row = {}
    for idx in range(len(entities)):
        row, col = divmod(idx, ncols)
        col_last_row[col] = row
    for idx, (ax, entity) in enumerate(zip(axes.flat, entities)):
        row, col = divmod(idx, ncols)
        plot_entity_panel(ax, titles.get(entity, entity), sub[sub["entity"] == entity], to_year,
                          show_yticklabels=(col == 0))
        if row == col_last_row[col]:
            ax.set_xlabel("Year", fontsize=AXIS_FONTSIZE)
    for ax in axes.flat[len(entities):]:
        ax.set_visible(False)

    _add_legend(fig, to_year)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_countries_combined_figure(
    sub_by_ssp: dict[str, pd.DataFrame], countries: list[str], titles: dict[str, str],
    to_year: int, out_path: Path, dpi: int,
) -> None:
    """Plot countries as columns and SSPs as rows in one figure."""
    ssps = list(sub_by_ssp)
    fig, axes = plt.subplots(len(ssps), len(countries), figsize=(6.8 * len(countries), 5.6 * len(ssps)), squeeze=False,
                             sharex=True, sharey=True)
    for r, ssp in enumerate(ssps):
        sub = sub_by_ssp[ssp]
        for c, iso in enumerate(countries):
            ax = axes[r, c]
            title = titles.get(iso, iso) if r == 0 else None
            plot_entity_panel(ax, title, sub[sub["entity"] == iso], to_year, show_yticklabels=(c == 0))
            if r == len(ssps) - 1:
                ax.set_xlabel("Year", fontsize=AXIS_FONTSIZE)
        # -150 pt clears the first column's wide y-tick labels
        axes[r, 0].annotate(SSP_DISPLAY.get(ssp, ssp), xy=(0, 0.5), xycoords="axes fraction", xytext=(-150, 0),
                            textcoords="offset points", rotation=90, ha="center", va="center",
                            fontsize=TITLE_FONTSIZE, fontweight="bold")

    _add_legend(fig, to_year)
    fig.tight_layout(rect=(0.07, 0.07, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Write the countries and per-SSP continents pathway maps to --out-dir."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-ssp2", required=True)
    parser.add_argument("--csv-ssp5", required=True)
    parser.add_argument("--to-year", type=int, default=2150)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    dfs = {"SSP2": pd.read_csv(args.csv_ssp2), "SSP5": pd.read_csv(args.csv_ssp5)}
    sub_by_ssp = {ssp: df[df["ssp"] == ssp] for ssp, df in dfs.items()}
    for ssp, sub in sub_by_ssp.items():
        if sub.empty:
            raise SystemExit(f"No rows for ssp={ssp}")

    iso_to_name = {v: k for k, v in _build_name_to_iso().items()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    countries = list(dict.fromkeys(sub_by_ssp["SSP2"].loc[sub_by_ssp["SSP2"]["entity_type"] == "country", "entity"]))
    if countries:
        titles = {iso: iso_to_name.get(iso, iso) for iso in countries}
        out_path = out_dir / "custom_pathways_map_countries_combined.png"
        make_countries_combined_figure(sub_by_ssp, countries, titles, args.to_year, out_path, args.dpi)
        print(f"Written: {out_path}")

    for ssp, sub in sub_by_ssp.items():
        continents = list(dict.fromkeys(sub.loc[sub["entity_type"] == "continent", "entity"]))
        if continents:
            titles = {c: c for c in continents}
            out_path = out_dir / f"custom_pathways_map_continents_{ssp}.png"
            make_figure(sub, continents, titles, args.to_year, out_path, args.dpi)
            print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
