"""Adaptation pathways map for a small, hand-picked set of sequences
(compute_custom_pathways.py's own CUSTOM_SEQUENCES), drawn in the style of
https://github.com/JuliusSchlumberger/Pathways_Analysis_Dashboard/tree/main/scripts/PathwaysMaps:
ONE FIXED HORIZONTAL TRACK PER MEASURE/DESIGN-INTENSITY BLOCK (not one lane
per sequence, which was this repo's own earlier, simpler attempt - see
plot_adaptation_pathways.py). Each sequence is a route that travels along a
track for as long as that stage holds, then jumps (a vertical connector) to
its next measure's track at the switch year - the standard DAPP "metro map"
convention. Multiple sequences sharing a track at an overlapping time are
kept visually distinguishable the same way the reference repo's own
y-offset/base_y mechanism does, simplified here (only 4 fixed sequences, no
need for that repo's own combinatorial offset optimizer): a small fixed
per-sequence y-offset within every track it visits, plus its own linestyle.

Reads compute_custom_pathways.py's own two CSVs (one per SSP) - no
recomputation. Writes THREE figures: one combined countries figure (SSP2 and
SSP5 as rows, one column per country, row labelled by a rotated "SSP2"/"SSP5"
annotation on the left) and one continents figure per SSP (6 panels each -
entity_type straight from the CSVs). Country panels are subtitled with the
real country name (Natural Earth via population_growth._build_name_to_iso,
top row only); continent panels with the entity string itself (already the
name). A continent whose baseline never tips ("no_adaptation_needed") still
gets a panel - just the baseline trunk running the whole horizon with no
tipping point, not omitted.

Usage:
    python analysis/plot_custom_pathways.py --csv-ssp2 <figures>/custom_pathways_SSP2.csv \\
        --csv-ssp5 <figures>/custom_pathways_SSP5.csv [--to-year 2150] [--outdir <figures>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from population_growth import _build_name_to_iso  # noqa: E402

SSP_DISPLAY = {"SSP2": "SSP2-4.5", "SSP5": "SSP5-8.5"}

MEASURE_COLORS = {"baseline": "0.55", "avoid": "#4C72B0", "protect": "#DD8452", "retreat": "#55A868",
                  "protect_avoid": "#8172B2"}
MEASURE_LABEL = {"avoid": "Avoid", "protect": "Protect", "retreat": "Retreat", "protect_avoid": "Protect + Avoid"}
# fixed track order, top to bottom - keyed by (measure, intensity) where
# intensity is "small"/"large" (DESIGN_LABEL's own vocabulary, which is what
# _block_parts gets back from the CSV's "measure_small"/"measure_large"
# chain-string blocks - NOT the raw "SLR_500"/"SLR_1000" config values).
TRACKS = [("baseline", None), ("avoid", "small"), ("avoid", "large"), ("protect", "small"),
         ("protect", "large"), ("protect_avoid", "small"), ("protect_avoid", "large"),
         ("retreat", "small"), ("retreat", "large")]
TRACK_Y = {b: len(TRACKS) - i for i, b in enumerate(TRACKS)}  # baseline at top
TRACK_LABEL = {("baseline", None): "Baseline (no adaptation)"}
for _m in ("avoid", "protect", "protect_avoid", "retreat"):
    TRACK_LABEL[(_m, "small")] = f"{MEASURE_LABEL[_m]} (small)"
    TRACK_LABEL[(_m, "large")] = f"{MEASURE_LABEL[_m]} (large)"


def _lighten(hex_color: str, amount: float) -> tuple[float, float, float]:
    """Blend `hex_color` toward white by `amount` (0=unchanged, 1=white)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# small/large get distinct shades of the same measure colour (small = lighter
# tint, large = the full-saturation base colour) so intensity reads from
# colour too, not just track position.
TRACK_COLOR: dict[tuple[str, str | None], tuple[float, float, float] | str] = {
    ("baseline", None): MEASURE_COLORS["baseline"],
}
for _m, _c in MEASURE_COLORS.items():
    if _m == "baseline":
        continue
    TRACK_COLOR[(_m, "small")] = _lighten(_c, 0.55)
    TRACK_COLOR[(_m, "large")] = _c

# One uniform line style/thickness throughout (sequences are no longer told
# apart by offset or dash pattern - identical segments shared by more than
# one sequence simply overlap into a single line, exactly like a real metro
# map, and sequences only read as visually distinct where they actually
# diverge onto a different track or a different time). A tipping point -
# every intermediate switch AND a final failure - is marked by letting the
# horizontal line run TIP_OVERSHOOT_YEARS past its real end year and capping
# that overshoot with a short, thick black vertical bar (the "black l"); the
# vertical connector to the next track (where there is one) still departs
# from the REAL switch year, so the coloured overshoot reads as a trailing
# flag on the track that just tipped, not a delay in the handover itself.
# Success (holds beyond --to-year) keeps its arrow marker - only the FAIL
# marker changes (black bar instead of an X).
LINE_LW = 4.0
TIP_OVERSHOOT_YEARS = 2.5
TICK_HALF_HEIGHT = 0.32
ARROW_MS = 16
START_MS = 13  # filled circle marking where a sequence starts a new measure


def _block_parts(block: str) -> tuple[str, str]:
    measure, _, intensity = block.rpartition("_")
    return measure, intensity


def chain_segments(row: pd.Series, to_year: int) -> tuple[list[tuple[str, float, float]], bool]:
    blocks = [b.strip() for b in row["chain"].split(">")]
    raw = row["stage_years"]
    switch_years = [float(y) for y in str(raw).split(";") if y.strip()] if pd.notna(raw) and str(raw).strip() else []
    success = str(row["outcome"]).startswith("success")
    boundaries = [float(row["baseline_tip_year"])] + switch_years + ([float(to_year)] if success else [])
    return list(zip(blocks, boundaries[:-1], boundaries[1:])), success


def _style_axes(ax: plt.Axes, to_year: int, title: str | None) -> None:
    ax.set_xlim(2020, to_year + 1)
    ax.set_yticks([TRACK_Y[t] for t in TRACKS])
    labels = ax.set_yticklabels([TRACK_LABEL[t] for t in TRACKS], fontsize=8.5)
    for lbl, t in zip(labels, TRACKS):
        lbl.set_color(TRACK_COLOR[t])  # compensates for the legend no longer naming measure colours
    ax.set_ylim(0.3, len(TRACKS) + 0.7)
    ax.grid(axis="x", alpha=0.25)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")


def plot_entity_panel(ax: plt.Axes, title: str | None, g: pd.DataFrame, to_year: int) -> None:
    base_track = TRACK_Y[("baseline", None)]
    thr = float(g["threshold"].iloc[0]) if not g.empty else None

    if g.empty or (g["outcome"] == "no_adaptation_needed").all():
        ax.plot([2020, to_year], [base_track, base_track], color=MEASURE_COLORS["baseline"],
                lw=LINE_LW, zorder=2, solid_capstyle="butt")
        ax.plot(to_year, base_track, ">", ms=ARROW_MS, color=MEASURE_COLORS["baseline"], zorder=5)
        ax.text(0.5, 0.5, f"No adaptation needed by {to_year}", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="0.4", style="italic")
        _style_axes(ax, to_year, title)
        if thr is not None:
            # bottom-right here (not top-right like the main branch below) -
            # baseline's own arrow occupies the full top row in this trivial
            # case, so bottom-right is the clear corner instead.
            ax.text(0.98, 0.04, f"tipping threshold: {thr:,.0f} EAI/yr", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5, color="0.35")
        return

    baseline_tip = float(g["baseline_tip_year"].iloc[0])
    # a single shared trunk, not one per sequence - every sequence starts here
    ax.plot([2020, baseline_tip], [base_track, base_track], color=MEASURE_COLORS["baseline"],
            lw=LINE_LW, zorder=2, solid_capstyle="butt")

    for _, row in g.sort_values("sequence_id").iterrows():
        segments, success = chain_segments(row, to_year)
        prev_track = base_track
        for i, (block, y0, y1) in enumerate(segments):
            measure, intensity = _block_parts(block)
            track = TRACK_Y[(measure, intensity)]
            is_last = i == len(segments) - 1
            c = TRACK_COLOR[(measure, intensity)]
            if track != prev_track:
                ax.plot([y0, y0], [prev_track, track], color=c, lw=LINE_LW, zorder=2)
            # filled circle where this sequence starts using this measure
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

    ax.axvline(baseline_tip, color="0.5", lw=0.8, ls=":", zorder=1)
    _style_axes(ax, to_year, title)
    ax.text(0.98, 0.97, f"tipping threshold: {thr:,.0f} EAI/yr", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="0.35")


TIPPING_DEF = ("A tipping point is reached when a measure's modelled Expected Annual Impact (EAI, people "
              "affected per year) first reaches the panel's own threshold (annotated in each panel).")


def _add_legend_and_caption(fig: plt.Figure, to_year: int, bbox_y: float = 0.045) -> None:
    # Only the three requested markers - measure identity now reads off the
    # colour-matched y-tick labels instead of a colour legend.
    handles = [Line2D([], [], ls="none", marker="o", mfc="0.3", mec="0.3", ms=START_MS, label="measure starts here"),
              Line2D([], [], ls="none", marker=">", color="0.3", ms=ARROW_MS, label=f"holds beyond {to_year}"),
              Line2D([], [], ls="none", marker="|", color="black", mew=LINE_LW, ms=14, label="tipping point reached")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, bbox_y))
    fig.text(0.5, 0.01, TIPPING_DEF, ha="center", va="bottom", fontsize=8, color="0.25", wrap=True)


def make_figure(sub: pd.DataFrame, entities: list[str], titles: dict[str, str], to_year: int, out_path: Path, dpi: int) -> None:
    ncols = min(3, len(entities))
    nrows = int(np.ceil(len(entities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 5.6 * nrows), squeeze=False)
    for ax, entity in zip(axes.flat, entities):
        plot_entity_panel(ax, titles.get(entity, entity), sub[sub["entity"] == entity], to_year)
        ax.set_xlabel("Year", fontsize=11)
    for ax in axes.flat[len(entities):]:
        ax.set_visible(False)

    _add_legend_and_caption(fig, to_year)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_countries_combined_figure(
    sub_by_ssp: dict[str, pd.DataFrame], countries: list[str], titles: dict[str, str],
    to_year: int, out_path: Path, dpi: int,
) -> None:
    ssps = list(sub_by_ssp)
    fig, axes = plt.subplots(len(ssps), len(countries), figsize=(6.8 * len(countries), 5.6 * len(ssps)), squeeze=False)
    for r, ssp in enumerate(ssps):
        sub = sub_by_ssp[ssp]
        for c, iso in enumerate(countries):
            ax = axes[r, c]
            title = titles.get(iso, iso) if r == 0 else None  # country name only on the top row
            plot_entity_panel(ax, title, sub[sub["entity"] == iso], to_year)
            if r == len(ssps) - 1:
                ax.set_xlabel("Year", fontsize=11)
        axes[r, 0].annotate(SSP_DISPLAY.get(ssp, ssp), xy=(0, 0.5), xycoords="axes fraction", xytext=(-100, 0),
                            textcoords="offset points", rotation=90, ha="center", va="center",
                            fontsize=14, fontweight="bold")

    _add_legend_and_caption(fig, to_year)
    fig.tight_layout(rect=(0.05, 0.1, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-ssp2", required=True)
    parser.add_argument("--csv-ssp5", required=True)
    parser.add_argument("--to-year", type=int, default=2150)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    dfs = {"SSP2": pd.read_csv(args.csv_ssp2), "SSP5": pd.read_csv(args.csv_ssp5)}
    sub_by_ssp = {ssp: df[df["ssp"] == ssp] for ssp, df in dfs.items()}
    for ssp, sub in sub_by_ssp.items():
        if sub.empty:
            raise SystemExit(f"No rows for ssp={ssp}")

    iso_to_name = {v: k for k, v in _build_name_to_iso().items()}
    out_dir = Path(args.outdir)
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
