"""Render the flood-variant comparison figure for docs/methods_02_flood_depth.md.

Runs all four FLOOD_MODELS (src/flood_model.py) on one synthetic scenario built to
isolate what each added mechanism changes: a coast-connectivity check (the walled-off
pond), path-following vs straight-line attenuation (the sill detour), and a duration-based
friction correction. The duration correction here is NOT production's actual
event_volume.DurationCorrection - see elevation_duration_multiplier's docstring and
TODO.md's "Open methods questions" entry. Panel letters, caption file, colours, sizing
and save settings all follow analysis/_figure_style.py, per the general CLAUDE.md's
section 9 - no in-image titles or descriptive text (that lives in the companion .txt
file instead), single-hue sequential colour maps per variable, 180 mm final width,
PNG+PDF output. SHOW_COLORBARS is an explicit, documented override of that section's "one
shared colour bar per group of panels" rule - see its own comment for why.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
import _figure_style  # noqa: E402
from flood_model import flood_depth_dense  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = OUT_DIR / "flood_variant_comparison.png"

POINT_COLOR = "#8a3ffc"

# style-override: one shared colour bar per group of panels | the discrete 0.20 m depth
# bands plus the three annotated sample-point values already give exact numbers, which a
# colour bar on top of eight small panels would mostly just crowd. Flip to True to render
# the default shared colour bars instead (plumbing below is in place but not visually tuned,
# since it's off by default).
SHOW_COLORBARS = False

SEED_LEVEL_M = 2.5
RATE_M_PER_CELL = 0.01  # straight_line's attenuation_m_per_m and eikonal's friction
DEPTH_BIN_M = 0.20  # isoline-style colour banding step
ELEVATION_VMAX_M = 3.0  # land colour ramp runs 0..this; everything below 0 is one flat colour

N_ROWS, N_COLS = 60, 160
MIN_ELEVATION_M = 0.0  # background at the coast - sea level; only the pond's interior and the
# berm/sill (below/above this) leave it, so the below-0 colour stays confined to the pond
MAX_ELEVATION_M = 2.3  # background at the domain's far inland edge - below SEED_LEVEL_M, so every
# background cell is exceeded by the surge at some point (briefly, near the inland edge) - this
# is what gives elevation_duration_multiplier a real gradient of exceedance durations to work
# with; see that function's docstring for why BERM/SILL below are deliberately NOT part of that.
PIT_ROWS, PIT_COLS = (0, 25), (15, 35)  # interior - row 0 = the domain's own upper edge
BERM_ROWS, BERM_COLS = (0, 27), (13, 37)  # enclosing wall; open at the top, where the domain's own
# edge already closes it (no separate top wall segment needed) - the pond stretches to row 0
BERM_ELEVATION_M = 3.0  # above the boundary level everywhere - never floods, no solve needed to know that
PIT_FLOOR_M = -1.0
RIDGE_ROWS, RIDGE_COLS = (9, N_ROWS), (70, 77)  # the gap is rows 0:9
SILL_ELEVATION_M = 3.0  # same margin above the boundary level as the pond's berm - see build_variant_scenario

SAMPLE_POINTS = {
    "pit": ((PIT_ROWS[0] + PIT_ROWS[1]) // 2, (PIT_COLS[0] + PIT_COLS[1]) // 2),
    "near": (50, 80),  # just past the sill's gap
    "far": (50, 130),  # well apart from "near" - avoids label overlap at subplot scale
}
# (dx, dy, va) per point label, close to the marker but offset in opposite directions so
# "near"/"far" labels can't collide despite sharing a row
LABEL_OFFSET = {"pit": (0, -5, "bottom"), "near": (0, -5, "bottom"), "far": (0, 6, "top")}

# Sized for this figure's actual 180 mm print width (eight small panels), not the ~14 inch
# draft size these were first tuned at - at that size the same point values read oversized
# relative to each panel and ran the panel letters into the grey group boxes above them.
# LABEL_FONTSIZE sits at section 9's 7 pt floor; LETTER_FONTSIZE one step above it so panel
# letters still read as a level above point labels.
LABEL_FONTSIZE = 7
LETTER_FONTSIZE = 8
LINEWIDTH = 1.0
MARKER_SIZE = 12

assert BERM_ELEVATION_M > SEED_LEVEL_M and SILL_ELEVATION_M > SEED_LEVEL_M, (
    "both barriers must stay above the boundary level - see build_variant_scenario's docstring "
    "for why a merely-overtoppable sill doesn't reliably force the intended detour here"
)


def build_variant_scenario(n_rows: int = N_ROWS, n_cols: int = N_COLS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ocean, a floodplain sloping from MIN_ELEVATION_M to MAX_ELEVATION_M, a walled-off
    pond open at the domain's upper edge, and a sill with a gap.

    A sill at a more "realistic", merely-overtoppable height (e.g. ~1.2 m above background,
    nowhere near the boundary level) was tried first and rejected: on a uniform-friction plain,
    eikonal's cost-distance treats elevation as a gate, not a cost, so a cell only ever becomes
    genuinely impassable if its *own first-pass level* comes out at or below its elevation: it
    is the margin between the two that must hold, not the sill's absolute height. Here that
    margin flipped the wrong way (the first pass reached it at a higher level than intended,
    everywhere along the sill, regardless of row - a uniform strip never isolates a path the way
    a single pinch point would), so it never blocked - straight_line and eikonal came out
    (numerically) identical, defeating the point of this panel. SILL_ELEVATION_M is kept at the
    same safely-above-MAX_ELEVATION_M margin as the pond's own wall instead, so blocking is exact
    and immediate rather than relying on a fragile numerical margin.
    """
    rise = np.arange(n_cols, dtype=np.float32) / (n_cols - 1)
    background = (MIN_ELEVATION_M + (MAX_ELEVATION_M - MIN_ELEVATION_M) * rise)[np.newaxis, :]
    dem = np.broadcast_to(background, (n_rows, n_cols)).copy()
    mask = np.zeros((n_rows, n_cols), dtype=np.int8)
    friction = np.full((n_rows, n_cols), RATE_M_PER_CELL, dtype=np.float32)

    ocean_cols = 6
    mask[:, :ocean_cols] = 1
    dem[:, :ocean_cols] = 0.0

    dem[BERM_ROWS[0]:BERM_ROWS[1], BERM_COLS[0]:BERM_COLS[1]] = BERM_ELEVATION_M
    dem[PIT_ROWS[0]:PIT_ROWS[1], PIT_COLS[0]:PIT_COLS[1]] = PIT_FLOOR_M

    dem[RIDGE_ROWS[0]:RIDGE_ROWS[1], RIDGE_COLS[0]:RIDGE_COLS[1]] = SILL_ELEVATION_M

    return dem, mask, friction


OFFSHORE_BOUNDARY_COL = 5  # column the surge level is imposed at - within the ocean mask, at
# elevation 0; marked by a plain red line on the elevation panel (no text - see _figure_style)


def _seeds(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(n_rows, dtype=np.int64)
    cols = np.full(n_rows, OFFSHORE_BOUNDARY_COL, dtype=np.int64)
    values = np.full(n_rows, SEED_LEVEL_M)
    return rows, cols, values


def run_base_variants(dem: np.ndarray, mask: np.ndarray, friction: np.ndarray) -> dict[str, tuple[np.ndarray, dict]]:
    """Run the four FLOOD_MODELS, no duration correction."""
    transform = Affine.identity()
    seed_rows, seed_cols, seed_values = _seeds(dem.shape[0])
    common = dict(dem=dem, mask=mask, friction=friction, transform=transform,
                  seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
                  k=1, idw_power=1.0, cell_dx_m=1.0, cell_dy_m=1.0)
    results = {}
    for model in ("bathtub", "bathtub_connected"):
        results[model] = flood_depth_dense(**common, model=model)
    results["straight_line"] = flood_depth_dense(**common, model="straight_line", attenuation_m_per_m=RATE_M_PER_CELL)
    results["eikonal"] = flood_depth_dense(
        **common, model="eikonal", obstacle_coupling=True, max_rounds=60, max_outer_iterations=3,
        waterlevel_epsilon_m=0.03,
    )
    return results


DURATION_ALPHA = 1.0
DURATION_MAX_MULTIPLIER = 10.0
DURATION_DT_S = 1800.0
DURATION_N_STEPS = 48  # 24h at 30-min steps
DURATION_T_REF_S = DURATION_N_STEPS * DURATION_DT_S  # the longest duration any cell can see here
# (the full storm) - see elevation_duration_multiplier's docstring for why the reference must be
# the longest, not some arbitrary shorter value


def build_hydrograph() -> np.ndarray:
    """A synthetic single-peak 24h storm tide (m), peaking at SEED_LEVEL_M."""
    t = np.arange(DURATION_N_STEPS)
    peak = DURATION_N_STEPS // 2
    return SEED_LEVEL_M * 0.5 * (1.0 + np.cos(2 * np.pi * (t - peak) / DURATION_N_STEPS))


def elevation_duration_multiplier(dem: np.ndarray, hydrograph: np.ndarray) -> np.ndarray:
    """Friction multiplier (T_ref/T)^alpha, T = how long the boundary forcing alone exceeds a
    cell's own elevation - directly, with no first solve (cf. event_volume.DurationCorrection,
    production's actual mechanism, which instead needs a first eikonal pass's depth to define a
    head above each cell's spill level; see TODO.md "Open methods questions" for why this figure
    uses the direct comparison instead).

    T_ref = DURATION_T_REF_S = the longest duration any cell in the domain can see (the full
    storm) - not an arbitrary shorter reference. That pins the multiplier to exactly 1.0 at the
    longest-duration (always-wet, coastal) cells and >1.0 everywhere else, so the corrected
    friction is never below the uncorrected baseline anywhere. A shorter T_ref (an earlier version
    of this used a fixed 12h) gives multiplier<1 - i.e. LESS friction than baseline - at every
    cell exceeded for longer than T_ref, which is exactly backwards: duration correction is meant
    to account for a short exceedance holding less water behind a barrier, never to make a
    long-exceedance cell more permeable than the uncorrected case. That bug was visible as
    duration-corrected depths exceeding the uncorrected depths at some points, which cannot happen
    once multiplier >= 1 everywhere (both straight_line's attenuation_m_per_m and eikonal's cost
    only ever increase with friction, so flood depth can only shrink or stay the same).

    Cells the forcing never reaches at all (duration 0, floored to one timestep) get the highest
    multiplier (capped at DURATION_MAX_MULTIPLIER). BERM_ELEVATION_M/SILL_ELEVATION_M (3.0 m) are
    both in this never-exceeded bucket - the 2.5 m surge peak never reaches them, so they are hard
    obstacles for Box B's connectivity/detour test, not "briefly-overtopped sills" for this
    correction. The short-but-nonzero-duration story this function is meant to show instead comes
    from the open background slope (MIN_ELEVATION_M..MAX_ELEVATION_M spans up to just under the
    surge peak), which gives a real gradient of exceedance durations with no barrier needed.
    """
    exceeds = hydrograph[:, np.newaxis, np.newaxis] > dem[np.newaxis, :, :]
    duration_s = np.maximum(exceeds.sum(axis=0) * DURATION_DT_S, DURATION_DT_S)
    return np.minimum((DURATION_T_REF_S / duration_s) ** DURATION_ALPHA, DURATION_MAX_MULTIPLIER).astype(np.float32)


def run_duration_correction(dem: np.ndarray, mask: np.ndarray, friction: np.ndarray) -> dict:
    """The duration-modified friction field, and the eikonal depth it gives.

    The multiplier is elevation_duration_multiplier's direct, single-pass function of elevation
    alone, so its spatial pattern is exactly the elevation panel's own contours, rescaled - not an
    independently-shaped one, and not production's actual (first-pass/head-based) mechanism; see
    that function's docstring.

    `straight_line` is skipped here: it has no spatial friction field to correct, so a "duration
    correction" for it would only ever be a scalar rescaling of its one attenuation_m_per_m - not
    a meaningful comparison to show beside eikonal's actually spatial correction. (That scalar
    rescaling is still what flood_depth_dense itself falls back to when it reports duration
    correction's effect in aggregate, diagnostics["mean_duration_multiplier"] - just not
    reproduced in this figure.)
    """
    transform = Affine.identity()
    seed_rows, seed_cols, seed_values = _seeds(dem.shape[0])
    hydrograph = build_hydrograph()
    multiplier = elevation_duration_multiplier(dem, hydrograph)
    modified_friction = friction * multiplier

    eikonal_wd, eikonal_diag = flood_depth_dense(
        dem=dem, mask=mask, friction=modified_friction, transform=transform,
        seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values, k=1, idw_power=1.0,
        cell_dx_m=1.0, cell_dy_m=1.0, model="eikonal", obstacle_coupling=True, max_rounds=60, max_outer_iterations=3,
        waterlevel_epsilon_m=0.03,
    )
    return {"modified_friction": modified_friction, "eikonal": eikonal_wd, "eikonal_diag": eikonal_diag}


def _elevation_cmap_norm(vmax: float = ELEVATION_VMAX_M) -> tuple[ListedColormap, Normalize]:
    """A single-hue (Greens) land colour ramp from 0 to vmax; everything below 0 is one
    flat colour.

    Without a colourbar (SHOW_COLORBARS=False), below/above 0 is meant to read as a plain
    categorical water/land split (_figure_style.WATER_COLOR), not a quantitative scale either
    side.
    """
    cmap = ListedColormap(plt.get_cmap(_figure_style.SEQUENTIAL_CMAP_ELEVATION)(np.linspace(0.15, 0.95, 256)))
    cmap.set_under(_figure_style.WATER_COLOR)
    return cmap, Normalize(vmin=0.0, vmax=vmax, clip=False)


def _depth_cmap_norm(vmax: float) -> tuple[ListedColormap, BoundaryNorm]:
    n_bins = max(1, int(np.ceil(vmax / DEPTH_BIN_M)))
    bounds = np.arange(0, (n_bins + 1) * DEPTH_BIN_M, DEPTH_BIN_M)
    cmap = ListedColormap(plt.get_cmap(_figure_style.SEQUENTIAL_CMAP_DEPTH)(np.linspace(0.15, 0.95, n_bins)))
    return cmap, BoundaryNorm(bounds, cmap.N)


def _mark_offshore_boundary(ax) -> None:
    """Red line at the offshore boundary-forcing column - drawn on every panel."""
    ax.axvline(OFFSHORE_BOUNDARY_COL, color="red", linewidth=LINEWIDTH, zorder=6)


def _hide_axes_frame(ax) -> None:
    """No ticks and no spines - row/column index isn't a physical coordinate here, so a
    partial axis frame (section 9 hides only the top/right spines by default) would read
    as a broken x/y axis rather than a deliberately bare one.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_depth(ax, mask: np.ndarray, wd: np.ndarray, cmap, norm):
    water = np.where(mask == 1, 1, 0)
    ax.imshow(water, cmap=ListedColormap([_figure_style.LAND_COLOR, _figure_style.WATER_COLOR]), vmin=0, vmax=1)
    depth = np.where(mask == 1, np.nan, wd)
    depth = np.where(depth <= 0, np.nan, depth)
    im = ax.imshow(depth, cmap=cmap, norm=norm)
    _mark_offshore_boundary(ax)
    for name, (r, c) in SAMPLE_POINTS.items():
        ax.scatter([c], [r], marker="o", s=MARKER_SIZE, facecolor="white", edgecolor=POINT_COLOR,
                   linewidth=LINEWIDTH, zorder=5)
        dx, dy, va = LABEL_OFFSET[name]
        ax.annotate(
            f"{wd[r, c]:.2f}", xy=(c, r), xytext=(c + dx, r + dy), fontsize=LABEL_FONTSIZE, color="black",
            ha="center", va=va, weight="bold", zorder=6,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", linewidth=0.5, pad=1.0, boxstyle="round,pad=0.15"),
        )
    _hide_axes_frame(ax)
    return im


def _group_bbox(axes_list, pad: float = 0.012, pad_top: float = 0.0) -> tuple[float, float, float, float]:
    positions = [ax.get_position() for ax in axes_list]
    x0 = min(p.x0 for p in positions) - pad
    x1 = max(p.x1 for p in positions) + pad
    y0 = min(p.y0 for p in positions) - pad
    y1 = max(p.y1 for p in positions) + pad_top
    return x0, y0, x1 - x0, y1 - y0


def _draw_group_box(fig, axes_list, pad_top: float) -> None:
    """A filled light-grey panel behind a group of panels - no label, no edge.

    pad_top must clear the panel letters of the group's top row (_figure_style.panel_letters
    sits just above its axes), or the box's top edge cuts through them.
    """
    x0, y0, w, h = _group_bbox(axes_list, pad_top=pad_top)
    patch = FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        transform=fig.transFigure, figure=fig, facecolor="#e6e6e6", edgecolor="none", zorder=-1,
    )
    fig.add_artist(patch)


def _write_figure_caption() -> None:
    _figure_style.write_caption(OUT_PATH, [
        "The four FLOOD_MODELS variants (src/flood_model.py) disagree sharply once a scenario has",
        "a disconnected basin and a detour around an obstacle, which is the sanity check and the",
        "explanation this figure exists to give, panel by panel.",
        "",
        "Scenario: ocean at the left edge, with a surge boundary condition (red line, every panel)",
        "rising over 24h; a floodplain sloping from 0 m at the coast to 2.3 m at the domain's far",
        "inland edge; a pond below sea level (-1 m), walled off on three sides at 3 m and open at",
        "the domain's own upper edge; a 3 m sill blocking a direct route inland except around a",
        "gap at one end.",
        "",
        "Grey boxes: the left box (a-c) is the scenario's static inputs; the right box (d-h) is",
        "every flood-depth result on that scenario.",
        "",
        "Colours: single-hue, colour-blind-safe sequential ramps, one per variable - elevation",
        "(a) uses Greens, flood depth (d-h) uses Blues, friction (b, c) uses Oranges (light = low",
        "in all three). Depth is binned into discrete 0.20 m steps. Grey = dry land; light",
        "blue-grey = permanent water (including below-sea-level elevation in (a), shown as one",
        "flat colour, not part of the scale).",
        "",
        "No colour bars: deliberately overridden here (SHOW_COLORBARS=False in the script) in",
        "favour of the discrete depth bands plus the annotated sample-point values below, which",
        "already give exact numbers; set SHOW_COLORBARS=True to restore one shared colour bar per",
        "panel group instead.",
        "",
        "Markers: white circles with a purple edge = three fixed sample points (pond interior,",
        "near the sill's gap, far past it); the black-on-white number beside each is that point's",
        "flood depth (m) under the panel's model.",
        "",
        "This is a deterministic synthetic test scenario, not a result with uncertainty to report.",
        "No CRS: the domain is an unprojected synthetic grid, not a geographic map.",
        "",
        "Panels:",
        "(a) elevation (m)",
        "(b) friction - spatially uniform attenuation rate",
        "(c) friction x a direct duration multiplier (time the boundary forcing exceeds each",
        "    cell's own elevation) - NOT production's actual event_volume.DurationCorrection;",
        "    see TODO.md \"Open methods questions\" and elevation_duration_multiplier's docstring",
        "(d) bathtub - IDW-interpolated level, no coast-connectivity check",
        "(e) bathtub_connected - (d) + coast-connectivity filter",
        "(f) straight_line - uniform rate, attenuated by straight-line distance",
        "(g) eikonal - uniform rate, attenuated by the real, connectivity-respecting path",
        "(h) eikonal, run with the duration-corrected friction field from (c)",
    ])


def make_figure() -> None:
    _figure_style.apply_style()
    dem, mask, friction = build_variant_scenario()
    base = run_base_variants(dem, mask, friction)
    dur = run_duration_correction(dem, mask, friction)

    print("=== base variants (no duration correction) ===")
    for name, (wd, diag) in base.items():
        pts = {label: float(wd[r, c]) for label, (r, c) in SAMPLE_POINTS.items()}
        print(f"  {name:18s} points={pts}  diag={diag}")

    print("=== duration correction (eikonal only) ===")
    pts = {label: float(dur["eikonal"][r, c]) for label, (r, c) in SAMPLE_POINTS.items()}
    print(f"  updated eikonal points={pts}")
    print(f"  updated eikonal diag={dur['eikonal_diag']}")

    vmax_depth = max(
        max(float(np.nanmax(wd)) for wd, _ in base.values()),
        float(np.nanmax(dur["eikonal"])),
    )
    depth_cmap, depth_norm = _depth_cmap_norm(vmax_depth)
    vmax_friction = float(dur["modified_friction"].max())
    friction_cmap = plt.get_cmap(_figure_style.SEQUENTIAL_CMAP_FRICTION)
    friction_norm = Normalize(vmin=0, vmax=vmax_friction)

    # Axes are placed by hand (fig.add_axes), not GridSpec: GridSpec's wspace/hspace are a
    # single relative knob per figure, which can't give column 0 - column 1 a bigger gap than
    # column 1 - column 2, or rows enough gap to clear the panel letters, without also fighting
    # imshow's default aspect="equal" (which otherwise letterboxes every panel inside a
    # mismatched cell and opens up large, uncontrolled gaps of its own).
    margin_left, margin_right = 0.02, 0.02
    margin_top, margin_bottom = 0.07, 0.02  # top also clears row 0's panel letters
    col_gap_wide = 0.055  # column 0 (inputs) to column 1 - deliberately bigger
    col_gap_tight = 0.03  # column 1 to column 2
    row_gap = 0.055  # between every row - enough that a row's panel letters (just above its
    # own axes) never reach into the row above

    col_width = (1 - margin_left - margin_right - col_gap_wide - col_gap_tight) / 3
    row_height = (1 - margin_top - margin_bottom - 2 * row_gap) / 3
    col_x0 = [
        margin_left,
        margin_left + col_width + col_gap_wide,
        margin_left + 2 * col_width + col_gap_wide + col_gap_tight,
    ]
    row_y0 = [1 - margin_top - row_height, 1 - margin_top - 2 * row_height - row_gap,
              1 - margin_top - 3 * row_height - 2 * row_gap]

    # figsize's aspect ratio matches each panel's (N_ROWS/N_COLS) so imshow's equal-aspect
    # scaling doesn't shrink panels inside their allotted axes box. Width is this figure's
    # final, printed size (section 9 "design at final size") - never rescaled afterwards.
    fig_width = _figure_style.DOUBLE_COLUMN_MM / _figure_style.MM_PER_INCH
    fig_height = col_width * fig_width / ((N_COLS / N_ROWS) * row_height)
    fig = plt.figure(figsize=(fig_width, fig_height))

    def _axes(row: int, col: int):
        return fig.add_axes((col_x0[col], row_y0[row], col_width, row_height))

    ax_dem = _axes(0, 0)
    ax_bt = _axes(0, 1)
    ax_btc = _axes(0, 2)
    ax_fric = _axes(1, 0)
    ax_sl = _axes(1, 1)
    ax_eik = _axes(1, 2)
    ax_fric_mod = _axes(2, 0)
    ax_eik_dc = _axes(2, 1)  # shifted left from column 2 - no bare gap beside it

    elev_cmap, elev_norm = _elevation_cmap_norm()
    elev_cmap.set_bad(_figure_style.WATER_COLOR)  # ocean cells sit at dem==0, same as the pond's
    # shoreline - mask them out rather than let them read as the land ramp's start colour
    ax_dem.imshow(np.where(mask == 1, np.nan, dem), cmap=elev_cmap, norm=elev_norm)
    _mark_offshore_boundary(ax_dem)
    _hide_axes_frame(ax_dem)

    ax_fric.imshow(np.where(mask == 1, np.nan, friction), cmap=friction_cmap, norm=friction_norm)
    _mark_offshore_boundary(ax_fric)
    _hide_axes_frame(ax_fric)

    for ax, name in zip((ax_bt, ax_btc, ax_sl, ax_eik), ("bathtub", "bathtub_connected", "straight_line", "eikonal")):
        wd, _ = base[name]
        _draw_depth(ax, mask, wd, depth_cmap, depth_norm)

    ax_fric_mod.imshow(np.where(mask == 1, np.nan, dur["modified_friction"]), cmap=friction_cmap, norm=friction_norm)
    _mark_offshore_boundary(ax_fric_mod)
    _hide_axes_frame(ax_fric_mod)

    _draw_depth(ax_eik_dc, mask, dur["eikonal"], depth_cmap, depth_norm)

    if SHOW_COLORBARS:
        # section 9's default: one shared colour bar per group of panels on the same scale
        from matplotlib.cm import ScalarMappable

        fig.colorbar(ScalarMappable(norm=elev_norm, cmap=elev_cmap), ax=ax_dem, fraction=0.08,
                     pad=0.02, label="Elevation (m)")
        fig.colorbar(ScalarMappable(norm=friction_norm, cmap=friction_cmap), ax=[ax_fric, ax_fric_mod],
                     fraction=0.08, pad=0.02, label="Friction (m/cell)")
        fig.colorbar(ScalarMappable(norm=depth_norm, cmap=depth_cmap),
                     ax=[ax_bt, ax_btc, ax_sl, ax_eik, ax_eik_dc], fraction=0.04, pad=0.02,
                     label="Flood depth (m)", extend="max")

    # drawn before the panel letters, and padded enough to clear them, so the letters sit
    # inside the (filled) box rather than straddling its top edge. Computed from the
    # letter's own rendered height (fontsize in points -> figure-fraction via fig_height,
    # which is itself much smaller than this script's original ~14 inch draft size) rather
    # than a flat constant tuned at that other size - a fixed figure-fraction pad doesn't
    # track a final print size this much smaller.
    letter_height_fig_frac = (LETTER_FONTSIZE / 72.0) / fig_height
    letter_clearance = 1.02 * row_height - row_height + letter_height_fig_frac * 1.6
    _draw_group_box(fig, [ax_dem, ax_fric, ax_fric_mod], pad_top=letter_clearance)
    _draw_group_box(fig, [ax_bt, ax_btc, ax_sl, ax_eik, ax_eik_dc], pad_top=letter_clearance)

    # column 0 (the static inputs) gets (a)-(c) top to bottom; everything else is lettered
    # left to right, top to bottom, the usual reading order
    for ax, letter in zip(
        (ax_dem, ax_fric, ax_fric_mod, ax_bt, ax_btc, ax_sl, ax_eik, ax_eik_dc),
        "abcdefgh",
    ):
        _figure_style.panel_letters(ax, letter, fontsize=LETTER_FONTSIZE)

    _figure_style.save_figure(fig, OUT_PATH)
    plt.close(fig)
    _write_figure_caption()
    print(f"Wrote {OUT_PATH.with_suffix('.png')}, {OUT_PATH.with_suffix('.pdf')} "
          f"and {OUT_PATH.with_suffix('.txt')}")


if __name__ == "__main__":
    make_figure()
