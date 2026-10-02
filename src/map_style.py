"""Shared visual style for every flood-extent agreement / inundation map in
this codebase - `validation/plot_agreement_map.py` (model-vs-benchmark,
`src/validation.py`'s own confusion_counts_soft) and `sfincs_tiles/*.py`
(model-vs-model: eikonal-vs-SFINCS). Land/water background colors, the
agree/first-only/second-only category palette, and the panel-lettering
convention are centralized here so every spatial map in the repo looks the
same regardless of which comparison produced it or which rendering path
(a DeltaDTM permanent-water raster mask, a per-tile `mog` land/ocean/
waterbody mask, cartopy's own Natural Earth land feature) built its
land/water background - this module does not try to unify the background
DATA sources, which genuinely differ per caller's own available inputs,
only the COLORS, LABELS, and lettering every caller draws with.

Category semantics: `AGREE_COLOR` is always "both sides agree this cell is
wet". `A_ONLY_COLOR`/`B_ONLY_COLOR` are a fixed, direction-free pair - which
real-world meaning they take (benchmark validation's "model under-predicts"/
"model over-predicts", or an eikonal-vs-SFINCS comparison's "SFINCS only"/
"eikonal only") is the caller's own choice of label text, not a different
color. Pick a consistent A/B convention per comparison type and keep it
(e.g. always A=the physics reference when comparing two models) so the same
color means the same thing across every plot of that comparison type.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Land grey / permanent-water blue-grey - distinct from AGREE/A_ONLY/B_ONLY
# below so the background is never mistaken for a 4th category.
LAND_COLOR = "#cfcfcf"
WATER_COLOR = "#aec6d8"

AGREE_COLOR = "#4daf4a"    # green - ColorBrewer Set1, distinguishable under
A_ONLY_COLOR = "#377eb8"   # common colour-vision deficiencies - the same
B_ONLY_COLOR = "#e41a1c"   # 3 hues validation/plot_agreement_map.py already
# used before this module existed (config.yml's own validation.plots.agreement_colors
# mirrors these exact values) - kept as the one fixed palette every category
# map in the repo now draws from, rather than each module picking its own.

LAND_LABEL = "Land"
WATER_LABEL = "Permanent water"


def draw_panel_letter(ax: plt.Axes, letter: str | None) -> None:
    """'(a)'/'(b)'/... drawn just OUTSIDE the axes (above its top-left
    corner, axes-fraction y > 1) - a bare letter, never inside the data
    area and NEVER baked into a title. No map in this codebase has a
    title or suptitle - a panel that needs a real caption uses
    `draw_caption_box` (an in-axes text box) instead, never
    `ax.set_title`/`fig.suptitle`. No-op if `letter` is falsy, so callers
    can pass `None` unconditionally for a single-panel figure that needs
    no letter at all.
    """
    if not letter:
        return
    ax.text(
        0.0, 1.02, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold",
        ha="left", va="bottom",
    )


def draw_caption_box(ax: plt.Axes, lines: list[str] | str, loc: str = "upper left") -> None:
    """In-axes text box - the one way any map in this codebase conveys
    context (tile ID, CSI, cell counts, what a panel shows, ...) - NEVER an
    `ax.set_title()`/`fig.suptitle()`. No-op on an empty/falsy `lines`, so
    callers can pass "nothing to show" unconditionally.

    `loc` is one of "upper left" (default, pairs naturally with
    `draw_panel_letter`'s own top-left corner), "upper right", "lower left",
    "lower right".
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
        x, y, "\n".join(lines), transform=ax.transAxes, fontsize=8, va=va, ha=ha,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5, pad=4.0),
    )


def land_water_legend_handles(
    agree_label: str, a_only_label: str, b_only_label: str,
    extra: list[Patch] | None = None,
) -> list[Patch]:
    """The standard 5-entry legend every agreement/inundation map in this
    repo should show: the 3 category colors (caller-supplied labels - see
    this module's own docstring on the A/B convention) plus the 2 shared
    background colors (fixed "Land"/"Permanent water" labels - a reader
    comparing two different maps should see the same background legend
    text on both). `extra` appends any genuinely map-specific entries
    (e.g. a tile-footprint outline) after the standard 5.
    """
    handles = [
        Patch(facecolor=AGREE_COLOR, label=agree_label),
        Patch(facecolor=A_ONLY_COLOR, label=a_only_label),
        Patch(facecolor=B_ONLY_COLOR, label=b_only_label),
        Patch(facecolor=LAND_COLOR, edgecolor="black", linewidth=0.3, label=LAND_LABEL),
        Patch(facecolor=WATER_COLOR, edgecolor="black", linewidth=0.3, label=WATER_LABEL),
    ]
    if extra:
        handles.extend(extra)
    return handles
