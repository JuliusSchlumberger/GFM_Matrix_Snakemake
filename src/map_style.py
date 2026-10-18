"""Shared colours, labels and panel lettering for flood-extent agreement maps.

AGREE = both sides wet. A_ONLY/B_ONLY are direction-free; keep one A/B convention
per comparison type (e.g. A = the physics reference) so colours mean the same.
"""

import matplotlib.pyplot as plt

# Background colours distinct from the category palette.
LAND_COLOR = "#cfcfcf"
WATER_COLOR = "#aec6d8"

# ColorBrewer Set1 (colour-vision-deficiency safe); mirrored by config.yml
# validation.plots.agreement_colors.
AGREE_COLOR = "#4daf4a"
A_ONLY_COLOR = "#377eb8"
B_ONLY_COLOR = "#e41a1c"

LAND_LABEL = "Land"
WATER_LABEL = "Permanent water"


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


def draw_panel_letter(ax: plt.Axes, letter: str | None) -> None:
    """Draw '(letter)' just above the axes' top-left corner; no-op if falsy.

    Maps never use titles; captions go in `draw_caption_box`.
    """
    if not letter:
        return
    ax.text(
        0.0, 1.02, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold",
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


