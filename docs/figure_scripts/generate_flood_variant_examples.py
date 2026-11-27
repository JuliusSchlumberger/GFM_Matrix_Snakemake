"""Render the flood-variant comparison figure for docs/methods_02_flood_depth.md.

Runs the FLOOD_MODELS variants (src/flood_model.py) on one synthetic scenario built to
isolate what each added mechanism changes: a coast-connectivity check (the walled-off
pond), path-following vs straight-line attenuation (the sill detour), a land-cover-derived
(spatially varying) vs uniform attenuation rate, and the storm-volume cap (event_volume.VolumeCap)
on the land-cover-driven eikonal result. The land cover here is a synthetic four-class
mosaic (build_landcover), not real land-cover data, coloured with conventional land-use-map
colours - see that function's docstring and LANDCOVER_COLORS. Panel letters, caption file,
colours, sizing and save settings all follow analysis/_figure_style.py, per the general
CLAUDE.md's section 9 - no in-image titles or descriptive text (that lives in the companion
.txt file instead), single-hue sequential colour maps per variable (plus the land-cover
palette, and a rank-ordered Oranges recolouring of those same classes for friction -
_friction_class_cmap), 180 mm final width, PNG+PDF output.
SHOW_COLORBARS is an explicit, documented override of that section's "one shared colour
bar per group of panels" rule - see its own comment for why.
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
from event_volume import VolumeCap  # noqa: E402
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
# background cell is reachable by the surge at some level, leaving room for the land-cover and
# uniform friction rates (RATE_M_PER_CELL and LANDCOVER_*_FRICTION_M_PER_CELL) to actually
# differentiate how far inland the water gets, rather than being capped by dry ground everywhere.
PIT_ROWS, PIT_COLS = (0, 25), (15, 35)  # interior - row 0 = the domain's own upper edge
BERM_ROWS, BERM_COLS = (0, 27), (13, 37)  # enclosing wall; open at the top, where the domain's own
# edge already closes it (no separate top wall segment needed) - the pond stretches to row 0
BERM_ELEVATION_M = 3.0  # above the boundary level everywhere - never floods, no solve needed to know that
PIT_FLOOR_M = -1.0
RIDGE_ROWS, RIDGE_COLS = (9, N_ROWS), (70, 77)  # the gap is rows 0:9
SILL_ELEVATION_M = 3.0  # same margin above the boundary level as the pond's berm - see build_variant_scenario

# Four land-cover classes, loosely following a real coastal-to-inland progression (bare tidal
# fringe, open grass/cropland as the dominant background, patches of shrub and forest further
# inland) rather than real land-cover data - see build_landcover. Patches are placed clear of
# the pond (cols 15-37) and the sill (cols 70-77) so the land cover never changes the
# connectivity/detour story, only how far the water reaches once it does.
LANDCOVER_TIDAL_FLAT_COLS = (6, 16)  # immediately inland of the ocean (ocean_cols=6), all rows
LANDCOVER_SHRUBLAND_ROWS, LANDCOVER_SHRUBLAND_COLS = (30, 55), (45, 65)
LANDCOVER_FOREST_ROWS, LANDCOVER_FOREST_COLS = (10, 35), (95, 125)

# Friction (m/cell): tidal flat < grassland/cropland (the open background, close to the old
# uniform rate) < shrubland < forest, so the four classes actually differentiate inland reach.
LANDCOVER_TIDAL_FLAT_FRICTION_M_PER_CELL = RATE_M_PER_CELL * 0.3
LANDCOVER_GRASSLAND_FRICTION_M_PER_CELL = RATE_M_PER_CELL * 0.8
LANDCOVER_SHRUBLAND_FRICTION_M_PER_CELL = RATE_M_PER_CELL * 1.8
LANDCOVER_FOREST_FRICTION_M_PER_CELL = RATE_M_PER_CELL * 3.5

# Class order matches LANDCOVER_*_FRICTION_M_PER_CELL above (tidal flat, grassland/cropland,
# shrubland, forest). Conventional land-use-map colours (tan bare ground, khaki/olive
# grassland-cropland, ochre scrub, forest green), not this project's abstract qualitative
# palette, since land cover (b) is meant to read like an ordinary land-use map at a glance.
# Panel (c) (friction) does NOT reuse these - it is coloured by rank on the Oranges ramp
# instead (_friction_class_cmap), so the two panels encode different things (land-use identity
# vs attenuation order) and are free to use unrelated colours.
LANDCOVER_COLORS = ("#D9C9A3", "#CDCB6E", "#C8963E", "#2E7D32")

# event_volume.VolumeCap parameters. VOLUME_CAP_N_RINGS matches config.yml's
# simulation.flooding.volume_cap default; VOLUME_CAP_VELOCITY_M_S does NOT (that default,
# 1.0 m/s, swamps this tiny 1 m/cell domain's whole storage in a single entry cell's inflow -
# tried and found to cap nothing at all, at any n_rings). Tuned down here, for this domain only,
# for a visible but moderate cap - most of the flooded patch survives, only its outermost,
# highest-attenuated cells are trimmed - see build_volume_cap.
VOLUME_CAP_VELOCITY_M_S = 0.0007
VOLUME_CAP_N_RINGS = 6

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


def build_landcover(n_rows: int, n_cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic four-class land cover (class 0-3, friction m/cell): grassland/cropland
    background, a tidal-flat fringe at the coast, and a shrubland and a forest patch inland.

    Not real land-cover data: a simple geometric mosaic (LANDCOVER_*_COLS/ROWS) stands in for
    src/flood_model.py's friction_source="landcover" path to have a spatially varying field to
    run against, without pulling in the real land-cover raster or compute_friction.py's
    classification.
    """
    landcover_class = np.ones((n_rows, n_cols), dtype=np.int8)  # 1 = grassland/cropland, the background
    tidal = slice(*LANDCOVER_TIDAL_FLAT_COLS)
    landcover_class[:, tidal] = 0
    landcover_class[LANDCOVER_SHRUBLAND_ROWS[0]:LANDCOVER_SHRUBLAND_ROWS[1],
                    LANDCOVER_SHRUBLAND_COLS[0]:LANDCOVER_SHRUBLAND_COLS[1]] = 2
    landcover_class[LANDCOVER_FOREST_ROWS[0]:LANDCOVER_FOREST_ROWS[1],
                    LANDCOVER_FOREST_COLS[0]:LANDCOVER_FOREST_COLS[1]] = 3

    friction_by_class = np.array([
        LANDCOVER_TIDAL_FLAT_FRICTION_M_PER_CELL, LANDCOVER_GRASSLAND_FRICTION_M_PER_CELL,
        LANDCOVER_SHRUBLAND_FRICTION_M_PER_CELL, LANDCOVER_FOREST_FRICTION_M_PER_CELL,
    ], dtype=np.float32)
    friction_landcover = friction_by_class[landcover_class]
    return landcover_class, friction_landcover


OFFSHORE_BOUNDARY_COL = 5  # column the surge level is imposed at - within the ocean mask, at
# elevation 0; marked by a plain red line on the elevation panel (no text - see _figure_style)


def _seeds(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(n_rows, dtype=np.int64)
    cols = np.full(n_rows, OFFSHORE_BOUNDARY_COL, dtype=np.int64)
    values = np.full(n_rows, SEED_LEVEL_M)
    return rows, cols, values


HYDROGRAPH_DT_S = 1800.0
HYDROGRAPH_N_STEPS = 48  # 24h at 30-min steps


def build_hydrograph() -> np.ndarray:
    """A synthetic single-peak 24h storm tide (m), peaking at SEED_LEVEL_M.

    Feeds event_volume.VolumeCap's storm-volume budget (run_base_variants); only its shape
    relative to the peak matters there, as event_volume.DurationCorrection's hydrographs.
    """
    t = np.arange(HYDROGRAPH_N_STEPS)
    peak = HYDROGRAPH_N_STEPS // 2
    return SEED_LEVEL_M * 0.5 * (1.0 + np.cos(2 * np.pi * (t - peak) / HYDROGRAPH_N_STEPS))


def build_volume_cap() -> VolumeCap:
    """A VolumeCap for this synthetic, CRS-less domain: one arbitrary station.

    With a single station, VolumeCap.budgets' nearest-station lookup is trivial (always that
    one station) regardless of its lon/lat value, so an arbitrary placement is fine here - same
    reasoning as `_seeds` imposing one constant level with no real geography. VOLUME_CAP_VELOCITY_M_S
    is tuned for this domain's scale, not production's - see its own comment.
    """
    return VolumeCap(
        station_lonlat=np.array([[0.0, 0.0]]), series=build_hydrograph()[np.newaxis, :],
        dt_s=HYDROGRAPH_DT_S, velocity_m_s=VOLUME_CAP_VELOCITY_M_S, n_rings=VOLUME_CAP_N_RINGS, crs=None,
    )


def run_base_variants(
    dem: np.ndarray, mask: np.ndarray, friction: np.ndarray, friction_landcover: np.ndarray,
) -> dict[str, tuple[np.ndarray, dict]]:
    """Run bathtub, bathtub_connected, straight_line, and eikonal at the uniform rate, the
    land-cover-derived rate, and the land-cover-derived rate with the volume cap."""
    transform = Affine.identity()
    seed_rows, seed_cols, seed_values = _seeds(dem.shape[0])
    common = dict(dem=dem, mask=mask, transform=transform,
                  seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
                  k=1, idw_power=1.0, cell_dx_m=1.0, cell_dy_m=1.0)
    eikonal_kwargs = dict(obstacle_coupling=True, max_rounds=60, max_outer_iterations=3, waterlevel_epsilon_m=0.03)
    results = {}
    for model in ("bathtub", "bathtub_connected"):
        results[model] = flood_depth_dense(**common, friction=friction, model=model)
    results["straight_line"] = flood_depth_dense(
        **common, friction=friction, model="straight_line", attenuation_m_per_m=RATE_M_PER_CELL,
    )
    results["eikonal_uniform"] = flood_depth_dense(**common, friction=friction, model="eikonal", **eikonal_kwargs)
    results["eikonal_landcover"] = flood_depth_dense(
        **common, friction=friction_landcover, model="eikonal", **eikonal_kwargs,
    )
    results["eikonal_landcover_cap"] = flood_depth_dense(
        **common, friction=friction_landcover, model="eikonal", volume_cap=build_volume_cap(), **eikonal_kwargs,
    )
    return results


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


def _friction_class_cmap(n_classes: int) -> ListedColormap:
    """Oranges, sampled at `n_classes` evenly spaced points - lightest = lowest-friction class.

    Classes (LANDCOVER_*_FRICTION_M_PER_CELL) are coloured by rank, not by their actual
    friction value: with only 4 classes spanning an 11x range, sampling Oranges at the
    classes' own (very unevenly spaced) fractional positions would under-use the light end of
    the ramp and bunch the darker classes together. Evenly spacing the 4 samples instead gives
    an unambiguous, strictly increasing light-to-dark reading, at the cost of not being a
    literal magnitude scale (it is ordinal, like the land-cover classes it's derived from).
    """
    return ListedColormap(plt.get_cmap(_figure_style.SEQUENTIAL_CMAP_FRICTION)(np.linspace(0.2, 0.95, n_classes)))


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


def _draw_landcover(ax, mask: np.ndarray, landcover_class: np.ndarray) -> None:
    """Qualitative land-cover panel - LANDCOVER_COLORS, one per class, ocean masked out."""
    cmap = ListedColormap(LANDCOVER_COLORS)
    cmap.set_bad(_figure_style.WATER_COLOR)
    classes = np.where(mask == 1, np.nan, landcover_class)
    ax.imshow(classes, cmap=cmap, vmin=0, vmax=len(LANDCOVER_COLORS) - 1)
    _mark_offshore_boundary(ax)
    _hide_axes_frame(ax)


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
        "The FLOOD_MODELS variants (src/flood_model.py) disagree sharply once a scenario has a",
        "disconnected basin, a detour around an obstacle, and a land-cover-driven split in",
        "propagation cost, which is the sanity check and the explanation this figure exists to",
        "give, panel by panel.",
        "",
        "Scenario: ocean at the left edge, with a surge boundary condition (red line, every panel)",
        "rising over 24h; a floodplain sloping from 0 m at the coast to 2.3 m at the domain's far",
        "inland edge; a pond below sea level (-1 m), walled off on three sides at 3 m and open at",
        "the domain's own upper edge; a 3 m sill blocking a direct route inland except around a",
        "gap at one end. A synthetic four-class land cover (build_landcover, not real land-cover",
        "data) gives a tidal-flat fringe at the coast, grassland/cropland as the open background,",
        "and a shrubland and a forest patch further inland, each placed clear of the pond and",
        "the sill.",
        "",
        "Grey boxes: the left box (a-c) is the scenario's static inputs; the right box (d-i) is",
        "every flood-depth result on that scenario.",
        "",
        "Colours: elevation (a) uses Greens and flood depth (d-i) uses Blues - single-hue,",
        "colour-blind-safe sequential ramps (light = low). Land cover (b) uses conventional",
        "land-use-map colours instead (qualitative, not a scale): tan = tidal flat, khaki =",
        "grassland/cropland (the background), ochre = shrubland, forest green = forest. Friction",
        "(c) reuses those same four classes but recoloured on the Oranges ramp by rank, lightest",
        "to darkest in the same tidal-flat/grassland/shrubland/forest order as (b) - i.e. by",
        "increasing attenuation, not by (b)'s land-use identity. Depth is binned into discrete",
        "0.20 m steps. Grey = dry land; light blue-grey = permanent water (including below-sea-",
        "level elevation in (a), shown as one flat colour, not part of the scale).",
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
        "(b) land cover class (synthetic, four classes, build_landcover): a tidal-flat fringe at",
        "    the coast, grassland/cropland as the open background, and a shrubland and a forest",
        "    patch further inland - not real land-cover data",
        "(c) friction derived from (b)'s classes - spatially varying attenuation rate, recoloured",
        "    by increasing rank (tidal flat < grassland/cropland < shrubland < forest), not (b)'s",
        "    land-use colours",
        "(d) bathtub - IDW-interpolated level, no coast-connectivity check",
        "(e) bathtub_connected - (d) + coast-connectivity filter",
        "(f) straight_line - uniform rate (RATE_M_PER_CELL), attenuated by straight-line distance",
        "(g) eikonal - uniform rate, attenuated by the real, connectivity-respecting path",
        "(h) eikonal - land-cover-derived rate (c), attenuated by the real, connectivity-",
        "    respecting path",
        "(i) eikonal, land-cover-derived rate (c) + the storm-volume cap (event_volume.VolumeCap):",
        "    each coast-connected patch is capped at the volume that can cross into it during the",
        "    storm (build_hydrograph), so depth in (i) is never above (h)'s",
    ])


def make_figure() -> None:
    _figure_style.apply_style()
    dem, mask, friction = build_variant_scenario()
    landcover_class, friction_landcover = build_landcover(N_ROWS, N_COLS)
    base = run_base_variants(dem, mask, friction, friction_landcover)

    print("=== variants ===")
    for name, (wd, diag) in base.items():
        pts = {label: float(wd[r, c]) for label, (r, c) in SAMPLE_POINTS.items()}
        print(f"  {name:22s} points={pts}  diag={diag}")

    vmax_depth = max(float(np.nanmax(wd)) for wd, _ in base.values())
    depth_cmap, depth_norm = _depth_cmap_norm(vmax_depth)
    # friction (uniform) isn't plotted (only used to run straight_line/eikonal_uniform); (c)
    # colours the 4 land-cover classes by friction rank, not by their raw value - see
    # _friction_class_cmap.
    friction_class_cmap = _friction_class_cmap(len(LANDCOVER_COLORS))

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
    n_rows_grid = 3  # column 0 (elevation, land cover, land-cover friction) and the 2-column
    # results grid (6 results) both need exactly 3 rows - every cell is used, none left empty

    col_width = (1 - margin_left - margin_right - col_gap_wide - col_gap_tight) / 3
    row_height = (1 - margin_top - margin_bottom - (n_rows_grid - 1) * row_gap) / n_rows_grid
    col_x0 = [
        margin_left,
        margin_left + col_width + col_gap_wide,
        margin_left + 2 * col_width + col_gap_wide + col_gap_tight,
    ]
    row_y0 = [1 - margin_top - (row + 1) * row_height - row * row_gap for row in range(n_rows_grid)]

    # figsize's aspect ratio matches each panel's (N_ROWS/N_COLS) so imshow's equal-aspect
    # scaling doesn't shrink panels inside their allotted axes box. Width is this figure's
    # final, printed size (section 9 "design at final size") - never rescaled afterwards.
    fig_width = _figure_style.DOUBLE_COLUMN_MM / _figure_style.MM_PER_INCH
    fig_height = col_width * fig_width / ((N_COLS / N_ROWS) * row_height)
    fig = plt.figure(figsize=(fig_width, fig_height))

    def _axes(row: int, col: int):
        return fig.add_axes((col_x0[col], row_y0[row], col_width, row_height))

    ax_dem = _axes(0, 0)
    ax_landcover = _axes(1, 0)
    ax_landcover_fric = _axes(2, 0)
    ax_bt = _axes(0, 1)
    ax_btc = _axes(0, 2)
    ax_sl = _axes(1, 1)
    ax_eik = _axes(1, 2)
    ax_eik_lc = _axes(2, 1)
    ax_eik_lc_cap = _axes(2, 2)

    elev_cmap, elev_norm = _elevation_cmap_norm()
    elev_cmap.set_bad(_figure_style.WATER_COLOR)  # ocean cells sit at dem==0, same as the pond's
    # shoreline - mask them out rather than let them read as the land ramp's start colour
    ax_dem.imshow(np.where(mask == 1, np.nan, dem), cmap=elev_cmap, norm=elev_norm)
    _mark_offshore_boundary(ax_dem)
    _hide_axes_frame(ax_dem)

    _draw_landcover(ax_landcover, mask, landcover_class)

    friction_class_cmap.set_bad(_figure_style.WATER_COLOR)
    ax_landcover_fric.imshow(np.where(mask == 1, np.nan, landcover_class), cmap=friction_class_cmap,
                             vmin=0, vmax=len(LANDCOVER_COLORS) - 1)
    _mark_offshore_boundary(ax_landcover_fric)
    _hide_axes_frame(ax_landcover_fric)

    for ax, name in zip(
        (ax_bt, ax_btc, ax_sl, ax_eik, ax_eik_lc, ax_eik_lc_cap),
        ("bathtub", "bathtub_connected", "straight_line", "eikonal_uniform", "eikonal_landcover",
         "eikonal_landcover_cap"),
    ):
        wd, _ = base[name]
        _draw_depth(ax, mask, wd, depth_cmap, depth_norm)

    if SHOW_COLORBARS:
        # section 9's default: one shared colour bar per group of panels on the same scale.
        # (c) gets none even here: it is coloured by class rank (_friction_class_cmap), not a
        # literal magnitude, so a colour bar would misstate it as one - same reason (b) (also
        # qualitative) never gets one either.
        from matplotlib.cm import ScalarMappable

        fig.colorbar(ScalarMappable(norm=elev_norm, cmap=elev_cmap), ax=ax_dem, fraction=0.08,
                     pad=0.02, label="Elevation (m)")
        fig.colorbar(ScalarMappable(norm=depth_norm, cmap=depth_cmap),
                     ax=[ax_bt, ax_btc, ax_sl, ax_eik, ax_eik_lc, ax_eik_lc_cap], fraction=0.04, pad=0.02,
                     label="Flood depth (m)", extend="max")

    # drawn before the panel letters, and padded enough to clear them, so the letters sit
    # inside the (filled) box rather than straddling its top edge. Computed from the
    # letter's own rendered height (fontsize in points -> figure-fraction via fig_height,
    # which is itself much smaller than this script's original ~14 inch draft size) rather
    # than a flat constant tuned at that other size - a fixed figure-fraction pad doesn't
    # track a final print size this much smaller.
    letter_height_fig_frac = (LETTER_FONTSIZE / 72.0) / fig_height
    letter_clearance = 1.02 * row_height - row_height + letter_height_fig_frac * 1.6
    _draw_group_box(fig, [ax_dem, ax_landcover, ax_landcover_fric], pad_top=letter_clearance)
    _draw_group_box(fig, [ax_bt, ax_btc, ax_sl, ax_eik, ax_eik_lc, ax_eik_lc_cap], pad_top=letter_clearance)

    # column 0 (the static inputs) gets (a)-(c) top to bottom; everything else is lettered
    # left to right, top to bottom, the usual reading order
    for ax, letter in zip(
        (ax_dem, ax_landcover, ax_landcover_fric, ax_bt, ax_btc, ax_sl, ax_eik, ax_eik_lc, ax_eik_lc_cap),
        "abcdefghi",
    ):
        _figure_style.panel_letters(ax, letter, fontsize=LETTER_FONTSIZE)

    _figure_style.save_figure(fig, OUT_PATH)
    plt.close(fig)
    _write_figure_caption()
    print(f"Wrote {OUT_PATH.with_suffix('.png')}, {OUT_PATH.with_suffix('.pdf')} "
          f"and {OUT_PATH.with_suffix('.txt')}")


if __name__ == "__main__":
    make_figure()
