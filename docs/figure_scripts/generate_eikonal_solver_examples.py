"""Render the synthetic eikonal-solver figures for docs/methods_02_flood_depth.md.

Runs the production solver (src/eikonal.py, src/flood_model.py) on toy grids:
rounds/epsilon (spiral maze), block skipping, obstacle coupling (tall ridge). Styled
per the general CLAUDE.md's section 9 via analysis/_figure_style.py: no titles (short
row/column headers on small-multiple panels are that section's named exception), every
panel lettered, a caption file per figure, single-hue sequential colour maps, 180 mm
final width, PNG+PDF output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from eikonal import UNREACHED_T, _block_sweep, _dense_sweep, _ORTHANT_ORDER, solve_eikonal_dense  # noqa: E402
from flood_model import flood_depth_dense  # noqa: E402
import _figure_style  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1]

WATER_COLOR = "#2a78d6"
LAND_COLOR = "#d8d8d4"
WALL_COLOR = "#4a4a46"
BLOCK_COLOR = "#d6572a"
NOT_REACHED_COLOR = "#eeeeee"
SWEEP_FILL_COLOR = "#f4d35e"  # not blue: blue reads as "flooded"
ATTENUATION_CMAP = "Blues"  # light = low attenuation (near seed); section 9 "Colour"
DEPTH_BIN_M = 0.10  # isoline-style banding for figure 3 - a continuous scale hid the
# (small but real) depth difference obstacle coupling makes

# Sized for these figures' actual 180 mm print width, not a larger draft size - section
# 9's floor for the smaller one is 7 pt text / 0.5 pt lines.
FIG_WIDTH_IN = _figure_style.DOUBLE_COLUMN_MM / _figure_style.MM_PER_INCH
LETTER_FONTSIZE = 8  # panel letters, and short headers appended to them (see panel_letters)
LABEL_FONTSIZE = 7


def build_spiral_friction(size: int = 27, n_rings: int = 3, gap_width: int = 9) -> np.ndarray:
    """Build nested high-friction rings with alternating gaps (a spiral maze).

    The spiral path reverses direction repeatedly: a known slow case for Fast
    Sweeping, since each round relays information once per sweep direction.
    """
    friction = np.full((size, size), 1.0, dtype=np.float32)
    cy, cx = size // 2, size // 2
    wall_friction = 50.0
    radii = np.linspace(size // 2 - 2, 5, n_rings).astype(int)
    sides = ["S", "N", "E", "W"]
    for i, r in enumerate(radii):
        r0, r1 = cy - r, cy + r
        c0, c1 = cx - r, cx + r
        friction[r0:r0 + 2, c0:c1 + 1] = wall_friction
        friction[r1 - 1:r1 + 1, c0:c1 + 1] = wall_friction
        friction[r0:r1 + 1, c0:c0 + 2] = wall_friction
        friction[r0:r1 + 1, c1 - 1:c1 + 1] = wall_friction

        side = sides[i % len(sides)]
        gc = slice(cx - gap_width // 2, cx + gap_width // 2 + 1)
        gr = slice(cy - gap_width // 2, cy + gap_width // 2 + 1)
        if side == "S":
            friction[r1 - 1:r1 + 1, gc] = 1.0
        elif side == "N":
            friction[r0:r0 + 2, gc] = 1.0
        elif side == "E":
            friction[gr, c1 - 1:c1 + 1] = 1.0
        else:
            friction[gr, c0:c0 + 2] = 1.0
    return friction


# (orthant, label, example-cell offset, row-neighbour offset, column-neighbour
# offset) in round order; offsets match the quadrant each orthant reaches alone.
SWEEP_NEIGHBOR_INFO = [
    (1, "sweep 1", (2, 2), (1, 2), (2, 1)),
    (4, "sweep 2", (2, -2), (1, -2), (2, -1)),
    (3, "sweep 3", (-2, -2), (-1, -2), (-2, -1)),
    (2, "sweep 4", (-2, 2), (-1, 2), (-2, 1)),
]


def _draw_sweep_quadrant_diagram(
    ax, orthant: int, example_offset: tuple[int, int],
    neighbor_a_offset: tuple[int, int], neighbor_b_offset: tuple[int, int], size: int = 7,
) -> None:
    friction = np.ones((size, size), dtype=np.float32)
    cy, cx = size // 2, size // 2
    t = np.full((size + 1, size + 1), 99.0, dtype=np.float32)
    # Cell (r, c) reads vertex (r+1, c+1), so the displayed seed is t[cy+1, cx+1].
    t[cy + 1, cx + 1] = 0.0
    neg_two, eight, four = np.float32(-2.0), np.float32(8.0), np.float32(4.0)
    _dense_sweep(t, friction, orthant, neg_two, eight, four)
    reached = t[1:, 1:] < 99.0

    ax.imshow(np.ones((size, size)), cmap=ListedColormap([NOT_REACHED_COLOR]), vmin=0, vmax=1)
    ax.imshow(np.where(reached, 1, np.nan), cmap=ListedColormap([SWEEP_FILL_COLOR]), vmin=0, vmax=1)
    ax.scatter([cx], [cy], marker="*", s=170, color=BLOCK_COLOR, edgecolor="white", linewidth=0.8, zorder=5)

    ex_row, ex_col = cy + example_offset[0], cx + example_offset[1]
    ax.annotate(
        # shrinkA/B are points, not data units - at this panel's actual ~90 pt size
        # (section 9's 180 mm width over 4 small-multiple columns), the old 14/22 pt
        # shrink ate nearly the whole ~35 pt arrow; a few points is enough to clear
        # the centre marker and the target box.
        "", xy=(ex_col, ex_row), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color=BLOCK_COLOR, linewidth=1.4, mutation_scale=10,
                         shrinkA=3, shrinkB=5),
        zorder=6,
    )
    for row_off, col_off in (neighbor_a_offset, neighbor_b_offset):
        n_row, n_col = cy + row_off, cx + col_off
        ax.add_patch(plt.Rectangle((n_col - 0.5, n_row - 0.5), 1, 1, fill=False,
                                    edgecolor=BLOCK_COLOR, linewidth=1.5, linestyle=(0, (2, 1)), zorder=6))
    ax.add_patch(plt.Rectangle((ex_col - 0.5, ex_row - 0.5), 1, 1, fill=False, edgecolor=BLOCK_COLOR, linewidth=2.0, zorder=7))
    ax.text(ex_col, ex_row, "t", fontsize=LABEL_FONTSIZE, ha="center", va="center", color=BLOCK_COLOR,
            weight="bold", zorder=8)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure_1() -> None:
    """Write eikonal_example_rounds.png: sweep quadrants, rounds, max change vs epsilon."""
    _figure_style.apply_style()
    friction = build_spiral_friction()
    n_rows, n_cols = friction.shape
    seed_rows = np.array([0], dtype=np.int64)
    seed_cols = np.array([n_cols // 2], dtype=np.int64)
    seed_values = np.array([0.0])

    round_checkpoints = [1, 2, 5, 12]
    snapshots = {}
    for n in round_checkpoints:
        t = solve_eikonal_dense(friction, seed_rows, seed_cols, seed_values, epsilon=0.0, sweep_budget=4 * n)
        snapshots[n] = t[1:, 1:]

    epsilon = 0.03
    max_change_per_round = []
    prev = None
    for n in range(1, 16):
        t = solve_eikonal_dense(friction, seed_rows, seed_cols, seed_values, epsilon=0.0, sweep_budget=4 * n)
        cur = t[1:, 1:]
        if prev is not None:
            max_change_per_round.append(float(np.max(np.abs(cur - prev))))
        prev = cur

    # Rows 0 and 1 get equal height_ratios (not the sweep diagrams' and snapshots' own
    # grid resolutions, which are irrelevant to on-screen size): both hold square-aspect
    # panels over the same 4 equal-width columns, so equal row heights is what makes them
    # render at the same size with their columns lined up left-to-right between rows.
    # 5 columns, not 4: a dedicated (narrow) column for row 1's colour bar, placed with
    # its own `cax` rather than fig.colorbar(ax=panel_b_axes). The latter reflows the
    # axes it's given to make room for itself, which silently tightened row 1's column
    # spacing relative to row 0's untouched one - the two rows no longer matched despite
    # sharing one GridSpec. A dedicated cax touches only its own cell.
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.62))
    gs = fig.add_gridspec(
        3, 5, height_ratios=[1, 1, 0.55], width_ratios=[1, 1, 1, 1, 0.16],
        hspace=0.22, wspace=0.08, top=0.93, bottom=0.08, left=0.06, right=0.95,
    )

    panel_a_axes, panel_a_labels = [], []
    for i, (orthant, label, ex_off, na_off, nb_off) in enumerate(SWEEP_NEIGHBOR_INFO):
        ax = fig.add_subplot(gs[0, i])
        _draw_sweep_quadrant_diagram(ax, orthant, ex_off, na_off, nb_off)
        panel_a_axes.append(ax)
        panel_a_labels.append(label)

    seed_row, seed_col = 0, n_cols // 2
    wall_mask = friction > 10
    converged = snapshots[round_checkpoints[-1]]
    raw_vmax = float(np.max(converged[converged < 90]))
    # Maze friction units are arbitrary (1 open, 50 walls); rescale to 0-2 m for display.
    DISPLAY_VMAX_M = 2.0
    panel_b_axes, panel_b_labels = [], []
    for i, n in enumerate(round_checkpoints):
        ax = fig.add_subplot(gs[1, i])
        unreached_mask = wall_mask | (snapshots[n] >= 90)
        arr = np.where(unreached_mask, np.nan, snapshots[n] / raw_vmax * DISPLAY_VMAX_M)
        ax.imshow(np.ones_like(arr), cmap=ListedColormap([NOT_REACHED_COLOR]), origin="upper")
        im = ax.imshow(arr, cmap=ATTENUATION_CMAP, vmin=0, vmax=DISPLAY_VMAX_M, origin="upper")
        ax.imshow(np.where(wall_mask, 1, np.nan), cmap=ListedColormap([WALL_COLOR]), origin="upper")
        ax.scatter([seed_col], [seed_row], marker="*", s=80, color=BLOCK_COLOR,
                   edgecolor="white", linewidth=0.6, zorder=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_b_axes.append(ax)
        panel_b_labels.append(f"round {n}")

    cax = fig.add_subplot(gs[1, 4])
    fig.colorbar(im, cax=cax, label="Cumulative attenuation from seed (m)")

    ax2 = fig.add_subplot(gs[2, :4])
    rounds_x = np.arange(2, 2 + len(max_change_per_round))
    display_change = np.maximum(max_change_per_round, 1e-3)  # log-scale floor; 0 = fully converged
    ax2.plot(rounds_x, display_change, color="#2a78d6", linewidth=1.2, marker="o", markersize=2.5)
    ax2.axhline(epsilon, color=BLOCK_COLOR, linestyle="--", linewidth=1.0, label=f"epsilon = {epsilon}")
    ax2.set_yscale("log")
    ax2.set_xlabel("Round", fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel("Max change", fontsize=LABEL_FONTSIZE)
    ax2.tick_params(labelsize=LABEL_FONTSIZE)
    ax2.legend(fontsize=LABEL_FONTSIZE, loc="upper right", frameon=False)
    ax2.grid(True, alpha=0.3, linewidth=0.5)

    all_axes = (*panel_a_axes, *panel_b_axes, ax2)
    all_labels = (*panel_a_labels, *panel_b_labels, None)
    for ax, letter, label in zip(all_axes, "abcdefghi", all_labels):
        _figure_style.panel_letters(ax, letter, fontsize=LETTER_FONTSIZE, label=label)

    _figure_style.save_figure(fig, OUT_DIR / "eikonal_example_rounds.png")
    plt.close(fig)
    _figure_style.write_caption(OUT_DIR / "eikonal_example_rounds.png", [
        "The Fast Sweeping solver needs several rounds of four directional sweeps to",
        "converge, and stops once the largest change between rounds drops below epsilon.",
        "",
        "Scenario: a synthetic spiral maze (nested high-friction rings with alternating",
        "gaps) seeded along the top edge - a known slow case, since the only path reverses",
        "direction repeatedly and each round only relays information once per sweep",
        "direction.",
        "",
        "Colours: (a)-(d) - yellow = reached by that single sweep direction, light grey =",
        "not yet reached, dark grey = wall. (e)-(h) - Blues, light = low cumulative",
        "attenuation from the seed (m), dark = high; dark grey = wall; light grey = not",
        "yet reached at that round. No CRS or uncertainty: a synthetic unprojected grid,",
        "deterministic solve.",
        "",
        "Markers: orange star = the seed cell (a-h). In (a)-(d), the orange arrow points at",
        "one example cell 't' (bold outline) that direction's sweep updates, from the two",
        "neighbour cells (dashed outline) it reads to do so.",
        "",
        "Panels:",
        "(a)-(d) one sweep direction each, in round order - the region each one alone",
        "     reaches from the seed before the next direction starts",
        "(e)-(h) the solution after 1, 2, 5 and 12 rounds (4/8/20/48 sweeps)",
        "(i) the largest change between consecutive rounds, vs. epsilon (dashed line) -",
        "     the convergence check the solver itself uses",
    ])
    print(f"Wrote {OUT_DIR / 'eikonal_example_rounds.png'}")
    print(f"  max_change by round {rounds_x[-1]}: {max_change_per_round[-1]:.6g} (epsilon={epsilon})")


def build_ridge_scenario(n_rows: int = 70, n_cols: int = 110) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ocean, a tall low-friction ridge (rows 15-55) with open flanks, and a basin.

    The ridge is the cheapest raw route inland and inflates basin depth; the
    flanks keep the basin connected, so obstacle coupling changes its depth,
    not whether it floods. Friction is cost per grid cell (no distance term),
    sized relative to the 3.0 m boundary level, not a Manning's n scale.
    """
    dem = np.full((n_rows, n_cols), 1.0, dtype=np.float32)
    mask = np.zeros((n_rows, n_cols), dtype=np.int8)
    friction = np.full((n_rows, n_cols), 0.01, dtype=np.float32)

    ocean_cols = 6
    mask[:, :ocean_cols] = 1
    dem[:, :ocean_cols] = 0.0

    ridge_lo, ridge_hi = 35, 42
    ridge_row_lo, ridge_row_hi = 15, 55
    dem[ridge_row_lo:ridge_row_hi, ridge_lo:ridge_hi] = 8.0        # above the 3.0 m boundary level
    friction[ridge_row_lo:ridge_row_hi, ridge_lo:ridge_hi] = 0.001  # deliberately low friction

    return dem, mask, friction


def make_figure_3() -> None:
    """Write eikonal_example_obstacle_coupling.png: basin depth with/without coupling."""
    _figure_style.apply_style()
    dem, mask, friction = build_ridge_scenario()
    n_rows, n_cols = dem.shape
    transform = Affine.identity()

    ocean = mask == 1
    coastline_rows, coastline_cols = np.nonzero(ocean & np.roll(mask != 1, -1, axis=1))
    seed_values = np.full(len(coastline_rows), 3.0)

    results = {}
    for coupling in (False, True):
        wd, diag = flood_depth_dense(
            dem, mask, friction, transform,
            seed_rows=coastline_rows, seed_cols=coastline_cols, seed_values=seed_values,
            obstacle_coupling=coupling, max_rounds=40, max_outer_iterations=3,
            waterlevel_epsilon_m=0.03,
        )
        results[coupling] = (wd, diag)
        print(f"obstacle_coupling={coupling}: {diag}, "
              f"n_flooded_in_basin={int((wd[:, 42:] > 0).sum())}")

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * (5.2 / 15)))

    depth_vmax = max(float(np.nanmax(wd)) for wd, _ in results.values())
    n_bins = max(1, int(np.ceil(depth_vmax / DEPTH_BIN_M)))
    depth_bounds = np.arange(0, (n_bins + 1) * DEPTH_BIN_M, DEPTH_BIN_M)
    depth_cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.15, 0.95, n_bins)))
    depth_norm = BoundaryNorm(depth_bounds, depth_cmap.N)

    labels = ("without coupling", "with coupling")
    for ax, coupling in zip(axes[:2], (False, True)):
        wd, _ = results[coupling]
        # depth is 0 (not NaN) on the dry ridge, so the wall must be drawn last, on top
        # of the depth layer, or its grey gets painted over by the depth cmap's vmin.
        depth_display = np.where(mask == 1, np.nan, wd)
        ax.imshow(depth_display, cmap=depth_cmap, norm=depth_norm, origin="upper")
        ax.imshow(np.where(dem > 4, 1, np.nan), cmap=ListedColormap([WALL_COLOR]), origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    # style-override: one shared colour bar per group of panels | the discrete 0.10 m
    # bands plus (a)/(b)'s visible band-contour kink already make the comparison, and
    # (c)'s bands are a different, incommensurable quantity (depth over-estimation, not
    # depth) anyway - a colour bar for each would mostly just crowd three small panels.

    ax = axes[2]
    wd_off, _ = results[False]
    wd_on, _ = results[True]
    depth_reduction = np.where(mask == 1, np.nan, wd_off - wd_on)
    red_vmax = float(np.nanmax(depth_reduction))
    red_n_bins = max(1, int(np.ceil(red_vmax / DEPTH_BIN_M)))
    red_bounds = np.arange(0, (red_n_bins + 1) * DEPTH_BIN_M, DEPTH_BIN_M)
    red_cmap = ListedColormap(plt.get_cmap("Reds")(np.linspace(0.15, 0.95, red_n_bins)))
    red_norm = BoundaryNorm(red_bounds, red_cmap.N)
    ax.imshow(depth_reduction, cmap=red_cmap, norm=red_norm, origin="upper")
    ax.imshow(np.where(dem > 4, 1, np.nan), cmap=ListedColormap([WALL_COLOR]), origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for ax, letter, label in zip(axes, "abc", (*labels, "depth over-estimation")):
        _figure_style.panel_letters(ax, letter, fontsize=LETTER_FONTSIZE, label=label)

    _figure_style.save_figure(fig, OUT_DIR / "eikonal_example_obstacle_coupling.png")
    plt.close(fig)
    _figure_style.write_caption(OUT_DIR / "eikonal_example_obstacle_coupling.png", [
        "The outer obstacle-coupling loop matters: a friction-cheap crossing of a",
        "physically-impassable ridge inflates basin depth behind it unless the loop",
        "detects and blocks that crossing.",
        "",
        "Scenario: ocean at the left edge; a tall ridge (dark grey block), well above the",
        "3.0 m boundary level, with open flanks above and below it; a basin inland of the",
        "ridge. The ridge is the cheapest raw route inland, so an uncoupled solve lets",
        "water cross it anyway; the open flanks keep the basin connected regardless, so",
        "coupling changes its depth, not whether it floods. No CRS or uncertainty: a",
        "synthetic unprojected grid, deterministic solve.",
        "",
        "Colours: dark grey = the ridge (impassable). Blues (a, b) and Reds (c) - discrete",
        "0.10 m bands, light = low, dark = high; a continuous scale hid how small (but",
        "real) a difference obstacle coupling makes here, in both the depth itself (a vs",
        "b) and the over-estimation (c). No colour bars: (a)/(b)'s band-contour kink and",
        "(c)'s bands already make the comparison; (c) is a different quantity from (a)/(b)",
        "anyway, so a shared one would not help.",
        "",
        "Panels:",
        "(a) basin depth without obstacle coupling - inflated by the cheap ridge crossing",
        "(b) basin depth with obstacle coupling - the crossing is detected and blocked",
        "(c) (a) minus (b): how much (a) over-estimates depth by not blocking the cheap",
        "    crossing, concentrated in the interior rows behind the ridge; the open",
        "    flanks above/below it are ~zero",
    ])
    print(f"Wrote {OUT_DIR / 'eikonal_example_obstacle_coupling.png'}")


FIG2_BLOCK = 16  # production uses SWEEP_BLOCK_SIZE=64


def _traced_block_solve(friction, seed_rows, seed_cols, seed_values, max_rounds: int):
    m, n = friction.shape
    t = np.full((m + 1, n + 1), UNREACHED_T, dtype=friction.dtype)
    t[seed_rows, seed_cols] = seed_values
    neg_two, eight, four = np.float32(-2.0), np.float32(8.0), np.float32(4.0)
    nbr, nbc = -(-(m + 1) // FIG2_BLOCK), -(-(n + 1) // FIG2_BLOCK)
    changed_at = np.full((nbr, nbc), -1, dtype=np.int64)
    swept_at = np.full((4, nbr, nbc), -1, dtype=np.int64)
    sweep_idx = 0
    records = []
    for _ in range(max_rounds):
        first_sweep = sweep_idx + 1
        n_swept, max_change = 0, 0.0
        for orthant in _ORTHANT_ORDER:
            sweep_idx += 1
            change, swept = _block_sweep(t, friction, orthant, neg_two, eight, four,
                                         FIG2_BLOCK, changed_at, swept_at, sweep_idx)
            n_swept += swept
            max_change = max(max_change, change)
        records.append({
            "processed": (swept_at >= first_sweep).any(axis=0),
            "frac_swept": n_swept / (4 * changed_at.size),
        })
        if max_change <= 0.0:
            break
    return t, records


def build_coastal_maze_scene(n_rows: int = 175, n_cols: int = 255, maze_size: int = 97) -> np.ndarray:
    """Build open land with one spiral maze whose walls are impassable (friction 9999).

    The (n_rows+1, n_cols+1) vertex grid must be a multiple of FIG2_BLOCK, or a
    thin edge block reads as "skipped" without real skipping - asserted below rather
    than left to fail silently as a misleading edge block in the figure.
    """
    assert (n_rows + 1) % FIG2_BLOCK == 0 and (n_cols + 1) % FIG2_BLOCK == 0, (
        f"(n_rows+1, n_cols+1)=({n_rows + 1}, {n_cols + 1}) must be a multiple of FIG2_BLOCK={FIG2_BLOCK}"
    )
    friction = np.full((n_rows, n_cols), 0.01, dtype=np.float32)
    maze = build_spiral_friction(size=maze_size, n_rings=10, gap_width=7)
    r0, c0 = n_rows - maze_size - 6, n_cols - maze_size - 8
    friction[r0:r0 + maze_size, c0:c0 + maze_size] = np.where(maze > 10, np.float32(9999.0), np.float32(0.01))
    return friction


def make_figure_2() -> None:
    """Write eikonal_example_block_sweep.png: re-swept vs skipped blocks per round."""
    _figure_style.apply_style()
    friction = build_coastal_maze_scene()
    n_rows, n_cols = friction.shape
    seed_rows = np.zeros(n_cols, dtype=np.int64)
    seed_cols = np.arange(n_cols, dtype=np.int64)
    seed_values = np.zeros(n_cols, dtype=np.float32)

    t_blk, records = _traced_block_solve(friction, seed_rows, seed_cols, seed_values, max_rounds=200)
    # Block skipping must be bit-identical to the production solver and plain sweeps.
    t_prod = solve_eikonal_dense(friction, seed_rows, seed_cols, seed_values, epsilon=0.0, max_rounds=200)
    assert np.array_equal(t_blk, t_prod), "illustration block solve differs from production solve_eikonal_dense"
    t_dense = np.full_like(t_blk, UNREACHED_T)
    t_dense[seed_rows, seed_cols] = seed_values
    for _ in range(len(records)):
        for orthant in _ORTHANT_ORDER:
            _dense_sweep(t_dense, friction, orthant, np.float32(-2.0), np.float32(8.0), np.float32(4.0))
    assert np.array_equal(t_blk, t_dense), "block solve differs from plain dense sweeps"

    n_rounds = len(records)
    frac = np.array([r["frac_swept"] for r in records])
    print(f"Figure 2: {n_rounds} rounds to full convergence, "
          f"block visits processed overall {100 * frac.mean():.1f}%")

    checkpoints = [1, 3, 8, 16, 22]
    assert max(checkpoints) <= n_rounds, (
        f"checkpoints {checkpoints} exceed {n_rounds} rounds to convergence - pick earlier checkpoints"
    )
    skip_color, swept_color = "#e4e4e0", BLOCK_COLOR
    block_cmap = ListedColormap([skip_color, swept_color])

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * (3.1 / 14) * 1.05))
    gs = fig.add_gridspec(1, len(checkpoints), wspace=0.08, top=0.84, bottom=0.17, left=0.02, right=0.98)

    panel_axes, panel_labels = [], []
    for i, k in enumerate(checkpoints):
        rec = records[k - 1]
        ax = fig.add_subplot(gs[0, i])
        ax.pcolormesh(rec["processed"].astype(int), cmap=block_cmap, vmin=0, vmax=1,
                      edgecolors="white", linewidth=0.8)
        nbr, nbc = rec["processed"].shape
        ax.plot([0, nbc], [0, 0], color=WATER_COLOR, linewidth=3, solid_capstyle="butt", clip_on=False)
        ax.set_xlim(0, nbc)
        ax.set_ylim(nbr, 0)
        ax.set_aspect("equal")
        ax.set_anchor("N")  # pin to the cell's top, so freed vertical room collapses below it
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        panel_axes.append(ax)
        panel_labels.append(f"round {k}")

    # aspect="equal" axes only get their final (shrunk) position at draw time; anchoring
    # the legend to a guessed figure-fraction left a large, unexplained gap below the
    # panels (they're narrower than their nominal cell once square-constrained). Anchor
    # to the panels' own measured bottom edge instead, so the gap is set directly.
    fig.canvas.draw()
    panel_bottom = min(ax.get_position().y0 for ax in panel_axes)
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=swept_color),
                 plt.Rectangle((0, 0), 1, 1, color=skip_color),
                 plt.Line2D([0], [0], color=WATER_COLOR, linewidth=3)],
        labels=["re-swept: the block or an upwind neighbour block changed",
                "skipped: nothing it depends on changed",
                "coast (seeded edge)"],
        loc="upper center", ncol=3, fontsize=LABEL_FONTSIZE, frameon=False,
        bbox_to_anchor=(0.5, panel_bottom - 0.02),
    )

    for ax, letter, label in zip(panel_axes, "abcde", panel_labels):
        _figure_style.panel_letters(ax, letter, fontsize=LETTER_FONTSIZE, label=label)

    _figure_style.save_figure(fig, OUT_DIR / "eikonal_example_block_sweep.png")
    plt.close(fig)
    pct_str = ", ".join(f"{100 * records[k - 1]['frac_swept']:.0f}" for k in checkpoints)
    _figure_style.write_caption(OUT_DIR / "eikonal_example_block_sweep.png", [
        "Block skipping only re-sweeps a block when it or an upwind neighbour actually",
        f"changed last round, cutting block visits to {100 * frac.mean():.0f}% overall here while",
        "staying bit-identical to sweeping every block every round.",
        "",
        "Scenario: open land seeded along the coast (top edge, blue bar) with one spiral",
        "maze of impassable walls (friction 9999) inland - the same slow, direction-",
        f"reversing case as the rounds figure, at production block size scaled down ({FIG2_BLOCK}",
        "cells here vs. SWEEP_BLOCK_SIZE=64 in production). No CRS or uncertainty: a",
        "synthetic unprojected grid, deterministic solve.",
        "",
        "Colours: each cell is one sweep block (not one grid cell). Orange = re-swept that",
        "round; light grey = skipped. Not a magnitude scale - a 2-category, directly",
        "labelled split (see legend).",
        "",
        f"Panels: (a)-(e) which blocks were processed after round {{{', '.join(str(k) for k in checkpoints)}}}",
        f"of {n_rounds} to full convergence ({pct_str}% of block visits respectively) -",
        "fewer blocks re-swept as the solution settles.",
    ])
    print(f"Wrote {OUT_DIR / 'eikonal_example_block_sweep.png'}")


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
    make_figure_3()
