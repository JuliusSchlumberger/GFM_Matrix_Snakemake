"""Shared colours, fonts, sizes, panel lettering and captions for figures
(CLAUDE.md "Figures" / the general CLAUDE.md's section 9).

AGREE = both sides wet. A_ONLY/B_ONLY are direction-free; keep one A/B convention
per comparison type (e.g. A = the physics reference) so colours mean the same.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt

# Background colours distinct from the category palette.
LAND_COLOR = "#cfcfcf"
WATER_COLOR = "#aec6d8"

# SFINCS boundary/forcing diagnostics (sfincs_tiles/boundary_lines_plot.py,
# plot_sfincs_tile_diagnostics.py): a waterlevel-boundary cell/conflict marker (red) and
# a SFINCS-sourced series/point marker (blue), each one colour shared across both scripts.
BOUNDARY_CELL_COLOR = "#d62728"
SFINCS_COLOR = "#1f4e8c"

# ColorBrewer Set1 (colour-vision-deficiency safe); mirrored by config.yml
# validation.plots.agreement_colors.
AGREE_COLOR = "#4daf4a"
A_ONLY_COLOR = "#377eb8"
B_ONLY_COLOR = "#e41a1c"

LAND_LABEL = "Land"
WATER_LABEL = "Permanent water"

# Muted ink/grid theme (input_analysis/make_boundary_forcing_figures.py): near-black text,
# grey de-emphasised series, light grid - an alternative to apply_style()'s defaults for
# figures with many overlaid series where full-black ink would be too heavy.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Colour-blind-safe sequential maps (section 9 "Colour") - matplotlib built-ins, so
# picking them needs no new dependency (cmocean/Crameri are not installed in this
# project's envs). Single-hue ColorBrewer ramps for the common geophysical quantities
# that recur across this project's figures, so each has one fixed identity throughout;
# SEQUENTIAL_CMAP is the fallback for anything else.
SEQUENTIAL_CMAP = "viridis"
SEQUENTIAL_CMAP_ELEVATION = "Greens"
SEQUENTIAL_CMAP_DEPTH = "Blues"
SEQUENTIAL_CMAP_FRICTION = "Oranges"

SINGLE_COLUMN_MM = 90
DOUBLE_COLUMN_MM = 180
MM_PER_INCH = 25.4

# SSP identity across the burning-ember / pathways / tipping-points figures. Not
# red/green (section 9: never that pair as the only distinction) and chosen to read
# against the burning-ember YlOrRd background too, since warm hues (red/orange/yellow)
# blend into it. Line style and marker vary per SSP as well, so colour is never the
# only cue (section 9 "Categories are double-encoded") wherever shape/style isn't
# already claimed by something else (e.g. strategy, in the tipping-points figure).
SSP_COLORS = {
    "SSP1": "#0072B2",  # blue
    "SSP2": "#CC79A7",  # reddish purple
    "SSP3": "#56B4E9",  # sky blue
    "SSP5": "#3B3B3B",  # charcoal grey
}
SSP_LINESTYLES = {"SSP1": "-.", "SSP2": "-", "SSP3": ":", "SSP5": "--"}
SSP_MARKERS = {"SSP1": "s", "SSP2": "o", "SSP3": "^", "SSP5": "D"}

# Adaptation-measure identity. Single colour per measure (plot_timeseries.py,
# plot_extended_pathways.py) - not yet reconciled with MEASURE_PAIR_COLORS below
# (plot_custom_pathways.py), which gives the same measure names different hues
# (e.g. "avoid" is green here, blue there) to free up small/large as a second colour
# dimension; section 9 wants one colour per variable across every figure, so this is
# a known, open inconsistency (TODO.md), not a decision made here.
MEASURE_COLORS = {
    "baseline": "#7f7f7f",
    "protect": "#1f77b4",
    "retreat": "#9467bd",
    "avoid": "#2ca02c",
    "protect_avoid": "#17becf",
}

# ColorBrewer "Paired": small/large intensity shares a hue, so colour still identifies
# the measure on its own; baseline is a neutral grey, reserved for context (section 9).
BASELINE_TRACK_COLOR = "0.55"
MEASURE_PAIR_COLORS = {
    "avoid": {"small": "#A6CEE3", "large": "#1F78B4"},
    "protect": {"small": "#FDBF6F", "large": "#FF7F00"},
    "protect_avoid": {"small": "#CAB2D6", "large": "#6A3D9A"},
    "retreat": {"small": "#B2DF8A", "large": "#33A02C"},
}

# EAI-change decomposition (plot_slr_vs_population.py waterfalls/scatter): SLR alone,
# population alone, and their interaction term.
SLR_POP_DECOMPOSITION_COLORS = {"slr": "#1f77b4", "pop": "#ff7f0e", "interaction": "#9467bd"}

# Continent identity (plot_slr_vs_population.py scatter/envelope) - ColorBrewer Set2; the
# only continent-colour mapping in the project. Keys must match src/analysis_figures.py's
# CONTINENTS list; a continent missing from this dict (e.g. "Other") falls back to context
# grey at the call site, per section 9.
CONTINENT_COLORS = {
    "Africa": "#66c2a5",
    "Asia": "#fc8d62",
    "Europe": "#8da0cb",
    "North America": "#e78ac3",
    "South America": "#a6d854",
    "Oceania": "#ffd92f",
}

# Storm-tide level sources and rule variants (docs/figure_scripts/plot_gauge_validation.py),
# Okabe-Ito, each with its own marker. The production variant is blue in every panel.
STORM_TIDE_SOURCE_STYLES = {
    "GFM product": ("#0072B2", "o"),
    "COAST-RP": ("#E69F00", "s"),
}
TC_RAISE_STYLES = {
    "none": ("#000000", "X"),
    "shift": ("#0072B2", "o"),
    "factor": ("#009E73", "D"),
}
TIDE_NEIGHBOUR_STYLES = {
    "within 25 km": ("#E69F00", "s"),
    "through GEBCO water": ("#D55E00", "^"),
    "through OSM water": ("#0072B2", "o"),
    "similar FES amplitude": ("#CC79A7", "D"),
}


def apply_style() -> None:
    """Set this project's figure rcParams once (section 9 "Output and checks").

    Called explicitly by each figure script rather than on import, so importing this
    module stays side-effect-free (CLAUDE.md section 6).
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })


def _savefig_retry(fig: plt.Figure, path: Path, **kwargs) -> None:
    # Windows intermittently refuses to truncate a file another process (viewer, indexer) holds.
    for attempt in range(5):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save `fig` as PNG (300 dpi) and PDF (fonts embedded) under `path`'s stem."""
    path = Path(path)
    _savefig_retry(fig, path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    _savefig_retry(fig, path.with_suffix(".pdf"), bbox_inches="tight")


def style_axes_muted(ax: plt.Axes) -> None:
    """Apply the muted ink/grid theme (INK/MUTED/GRID) to one axes."""
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def orient_north_up(arr, transform):
    """Flip `arr` so row 0 is north and column 0 is west, for imshow origin="upper".

    SFINCS subgrid rasters are stored south-up (positive y step). Rotated grids
    keep their rotation; use pcolormesh where exact geometry matters.
    """
    import numpy as np

    row_step_north = transform.e
    col_step_east = transform.a
    if row_step_north > 0:
        arr = np.flip(arr, axis=0)
    if col_step_east < 0:
        arr = np.flip(arr, axis=1)
    return arr


def panel_letters(ax: plt.Axes, letter: str | None, fontsize: float = 11, label: str | None = None) -> None:
    """Draw '(letter)', or '(letter) label' if `label` is given, just above the axes'
    top-left corner; no-op if `letter` is falsy.

    Maps never use titles; captions go in `draw_caption_box`. For a small-multiple
    panel's short header (section 9's exception to "no titles"), pass it as `label`
    rather than a separate `ax.set_title` - one left-aligned string, not a centered
    title floating independently of the letter it has to line up with. `label` must be
    single-line: a second line would sit under the letter, not after it, breaking that
    same alignment. `fontsize` defaults to the original, larger-figure value; pass a
    smaller one (still >= 7, section 9's floor) for a figure with many small panels,
    where 11 pt reads oversized relative to the panels.
    """
    if not letter:
        return
    if label:
        assert "\n" not in label, "panel_letters label must be single-line - no separate title"
    text = f"({letter}) {label}" if label else f"({letter})"
    ax.text(
        0.0, 1.02, text, transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
        ha="left", va="bottom",
    )


def draw_caption_box(ax: plt.Axes, lines: list[str] | str, loc: str = "upper left", fontsize: float = 8) -> None:
    """Draw an in-axes caption box (maps never use titles); no-op if `lines` is falsy.

    `loc`: "upper left", "upper right", "lower left" or "lower right".
    """
    if not lines:
        return
    if isinstance(lines, str):
        lines = [lines]
    coords = {
        "upper left": (0.03, 0.97, "top", "left"),
        "upper right": (0.97, 0.97, "top", "right"),
        "lower left": (0.03, 0.03, "bottom", "left"),
        "lower right": (0.97, 0.03, "bottom", "right"),
    }
    x, y, va, ha = coords[loc]
    ax.text(
        x, y, "\n".join(lines), transform=ax.transAxes, fontsize=fontsize, va=va, ha=ha,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5, pad=4.0),
    )


def write_caption(figure_path: Path, lines: list[str]) -> None:
    """Write `lines` to a `.txt` file next to `figure_path` (CLAUDE.md "Figures")."""
    Path(figure_path).with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


