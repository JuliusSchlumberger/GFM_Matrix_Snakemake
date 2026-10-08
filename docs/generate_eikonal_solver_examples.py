"""Builds three small, stylized (synthetic, not real-tile) example figures for
docs/methods_02_flood_depth.md, run directly against the real production solver
code (src/eikonal.py, src/flood_model.py) - not hand-drawn illustrations.

Figure 1 (eikonal_example_rounds.png): why the inner loop needs more than
one round. A serpentine corridor (alternating-gap walls) forces the true
shortest path to reverse direction repeatedly, a known slow case for the
Fast Sweeping Method's fixed 4-direction sweep order. Shows the arrival
potential after 1, 2, 4 and 12 rounds, and the per-round max-change curve
against epsilon.

Figure 2 (eikonal_example_block_sweep.png): how block skipping saves work.
Uniform open land seeded along its top edge, with one larger Figure-1-style
spiral maze (impassable walls), solved with the real `_block_sweep` kernel:
a two-colour block grid per round (re-swept vs skipped) as the front winds
inward (asserted bit-identical to both the production solver and plain
dense sweeps).

Figure 3 (eikonal_example_obstacle_coupling.png): why the outer loop is
needed. A low-friction (easy-to-cross) but tall ridge separates the coast
from a low-lying inland basin. The raw solve lets a friction-cheap "shortcut"
across the ridge reach the basin with an illegitimately high potential,
flooding it even though the ridge is physically too tall to ever be
overtopped by the real boundary water level; obstacle coupling detects and
blocks this and re-solves.

Usage:
    python generate_eikonal_solver_examples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eikonal import _block_sweep, _dense_sweep, _ORTHANT_ORDER, solve_eikonal_dense  # noqa: E402
from flood_model import flood_depth_dense  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

WATER_COLOR = "#2a78d6"
LAND_COLOR = "#d8d8d4"
WALL_COLOR = "#4a4a46"
BLOCK_COLOR = "#d6572a"
NOT_REACHED_COLOR = "#eeeeee"
SWEEP_FILL_COLOR = "#f4d35e"  # panel (a) reached-cells fill - yellow, not blue, since blue reads as "flooded"
ATTENUATION_CMAP = "Blues_r"  # dark = close to seed/little attenuation, light = far/heavily attenuated


# ---------------------------------------------------------------------------
# Figure 1: rounds / epsilon
# ---------------------------------------------------------------------------

def build_spiral_friction(size: int = 27, n_rings: int = 3, gap_width: int = 9) -> np.ndarray:
    """Low-friction field with nested high-friction square rings, each with
    a single gap, the gap side alternating N/E/S/W ring to ring - forces the
    true shortest path to spiral inward, reversing direction repeatedly. A
    known slow case for Fast Sweeping's fixed 4-direction sweep order, since
    each round only relays information once along each sweep direction.
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


# Per sweep, in round order (1, 4, 3, 2): (example cell offset from seed,
# row-neighbour offset, column-neighbour offset) - verified against the real
# quadrant each orthant alone reaches from a center seed (see conversation),
# not just the neighbour table.
SWEEP_NEIGHBOR_INFO = [
    (1, "sweep 1", (2, 2), (1, 2), (2, 1)),
    (4, "sweep 2", (2, -2), (1, -2), (2, -1)),
    (3, "sweep 3", (-2, -2), (-1, -2), (-2, -1)),
    (2, "sweep 4", (-2, 2), (-1, 2), (-2, 1)),
]


def _draw_sweep_quadrant_diagram(
    ax, orthant: int, label: str, example_offset: tuple[int, int],
    neighbor_a_offset: tuple[int, int], neighbor_b_offset: tuple[int, int], size: int = 7,
) -> None:
    """One sweep, run alone from a fresh (unseeded) grid with a single seed
    at the center - shows exactly which cells that one sweep updates (real
    output of `_dense_sweep`, not illustrative). A single arrow from the
    seed toward one example cell shows the sweep's general direction; that
    cell's own two upwind neighbours are outlined too.
    """
    friction = np.ones((size, size), dtype=np.float32)
    cy, cx = size // 2, size // 2
    t = np.full((size + 1, size + 1), 99.0, dtype=np.float32)
    # Seed t[cy+1, cx+1], not t[cy, cx]: the displayed array is the cropped
    # `reached = t[1:, 1:]` (see module docstring - cell (r,c) reads vertex
    # (r+1,c+1)), so reached[cy, cx] == t[cy+1, cx+1]. Seeding t[cy, cx]
    # directly would put the true seeded cell one row/col up-left of where
    # the ★ marker below is actually drawn.
    t[cy + 1, cx + 1] = 0.0
    neg_two, eight, four = np.float32(-2.0), np.float32(8.0), np.float32(4.0)
    _dense_sweep(t, friction, orthant, neg_two, eight, four)
    reached = t[1:, 1:] < 99.0

    ax.imshow(np.ones((size, size)), cmap=ListedColormap([NOT_REACHED_COLOR]), vmin=0, vmax=1)
    ax.imshow(np.where(reached, 1, np.nan), cmap=ListedColormap([SWEEP_FILL_COLOR]), vmin=0, vmax=1)
    ax.scatter([cx], [cy], marker="*", s=170, color=BLOCK_COLOR, edgecolor="white", linewidth=0.8, zorder=5)

    ex_row, ex_col = cy + example_offset[0], cx + example_offset[1]
    ax.annotate(
        "", xy=(ex_col, ex_row), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color=BLOCK_COLOR, linewidth=2.0, mutation_scale=16,
                         shrinkA=14, shrinkB=22),
        zorder=6,
    )
    for row_off, col_off in (neighbor_a_offset, neighbor_b_offset):
        n_row, n_col = cy + row_off, cx + col_off
        ax.add_patch(plt.Rectangle((n_col - 0.5, n_row - 0.5), 1, 1, fill=False,
                                    edgecolor=BLOCK_COLOR, linewidth=1.5, linestyle=(0, (2, 1)), zorder=6))
    ax.add_patch(plt.Rectangle((ex_col - 0.5, ex_row - 0.5), 1, 1, fill=False, edgecolor=BLOCK_COLOR, linewidth=2.0, zorder=7))
    ax.text(ex_col, ex_row, "t", fontsize=11, ha="center", va="center", color=BLOCK_COLOR, weight="bold", zorder=8)

    ax.set_title(label, fontsize=10.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure_1() -> None:
    friction = build_spiral_friction()
    n_rows, n_cols = friction.shape
    seed_rows = np.array([0], dtype=np.int64)
    seed_cols = np.array([n_cols // 2], dtype=np.int64)
    seed_values = np.array([0.0])  # t=0 at the single source point

    # Snapshots at n rounds via the fixed sweep_budget mode (4 sweeps/round),
    # plus the real epsilon-based converged solve for comparison.
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

    fig = plt.figure(figsize=(12, 8.7))
    gs = fig.add_gridspec(
        3, 4, height_ratios=[1.9, 3, 1.3],
        hspace=0.18, wspace=0.04, top=0.95, bottom=0.06, left=0.05, right=0.93,
    )

    panel_a_axes = []
    for i, (orthant, label, ex_off, na_off, nb_off) in enumerate(SWEEP_NEIGHBOR_INFO):
        ax = fig.add_subplot(gs[0, i])
        _draw_sweep_quadrant_diagram(ax, orthant, label, ex_off, na_off, nb_off)
        panel_a_axes.append(ax)

    seed_row, seed_col = 0, n_cols // 2
    wall_mask = friction > 10
    converged = snapshots[round_checkpoints[-1]]
    raw_vmax = float(np.max(converged[converged < 90]))
    # Rescaled for display so the colorbar reads like a real flood attenuation
    # (0-2m), matching Figure 3's units - the maze's own friction units are
    # otherwise arbitrary (friction=1 open cells, 50 walls) and only
    # meaningful relative to each other, not as a real physical scale.
    DISPLAY_VMAX_M = 2.0
    panel_b_axes = []
    for i, n in enumerate(round_checkpoints):
        ax = fig.add_subplot(gs[1, i])
        unreached_mask = wall_mask | (snapshots[n] >= 90)
        arr = np.where(unreached_mask, np.nan, snapshots[n] / raw_vmax * DISPLAY_VMAX_M)
        ax.imshow(np.ones_like(arr), cmap=ListedColormap([NOT_REACHED_COLOR]), origin="upper")  # not-yet-reached
        im = ax.imshow(arr, cmap=ATTENUATION_CMAP, vmin=0, vmax=DISPLAY_VMAX_M, origin="upper")
        ax.imshow(np.where(wall_mask, 1, np.nan), cmap=ListedColormap([WALL_COLOR]), origin="upper")
        ax.scatter([seed_col], [seed_row], marker="*", s=180, color=BLOCK_COLOR,
                   edgecolor="white", linewidth=0.8, zorder=5)
        ax.set_title(f"after {n} round{'s' if n != 1 else ''}\n({4 * n} sweeps)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        panel_b_axes.append(ax)

    fig.colorbar(
        im, ax=panel_b_axes, fraction=0.046, pad=0.015, shrink=0.7, aspect=18,
        label="cumulative attenuation from seed (m)",
    )

    ax2 = fig.add_subplot(gs[2, :])
    rounds_x = np.arange(2, 2 + len(max_change_per_round))
    display_change = np.maximum(max_change_per_round, 1e-3)  # log-scale floor; 0 = fully converged
    ax2.plot(rounds_x, display_change, color="#2a78d6", linewidth=1.6, marker="o", markersize=3)
    ax2.axhline(epsilon, color=BLOCK_COLOR, linestyle="--", linewidth=1.2, label=f"epsilon = {epsilon}")
    ax2.set_yscale("log")
    ax2.set_xlabel("round")
    ax2.set_ylabel("max change")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Panel letters aligned on a common x (figure fraction), each at the top
    # of its own row (in figure fraction, from each row's axes position).
    label_x = 0.005
    for ax, letter in ((panel_a_axes[0], "(a)"), (panel_b_axes[0], "(b)"), (ax2, "(c)")):
        row_top = ax.get_position().y1
        fig.text(label_x, row_top, letter, fontsize=13, weight="bold", ha="left", va="top")

    fig.savefig(OUT_DIR / "eikonal_example_rounds.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'eikonal_example_rounds.png'}")
    print(f"  max_change by round 40: {max_change_per_round[-1]:.6g} (epsilon={epsilon})")


# ---------------------------------------------------------------------------
# Figure 3: obstacle coupling
# ---------------------------------------------------------------------------

def build_ridge_scenario(n_rows: int = 70, n_cols: int = 110) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ocean on the left, a tall-but-low-friction ridge band (rows 15-55)
    with open flanks above/below it (rows 0-14, 56-69), and a low-lying
    basin behind both.

    The ridge is too tall to ever legitimately flood, but its low friction
    makes it numerically the CHEAPEST route inland for interior rows -
    cheaper than the honest detour around it via the flanks - so a raw
    solve lets it inflate the basin's reported depth there. The flanks give
    the basin a second, legitimate connection to the sea, so the ordinary
    post-solve connectivity filter (unrelated to obstacle coupling) can't
    hide the effect by disconnecting it entirely - what changes with
    obstacle coupling is the basin's DEPTH behind the ridge, not whether it
    floods at all, matching the depth-difference character of the real
    (production-scale) validation this mirrors.

    Friction is used directly as cost-per-grid-cell (no distance term - see
    flood_depth_dense), so magnitudes are chosen relative to the 3.0m
    boundary level and domain size, not to any real Manning's n scale.
    """
    dem = np.full((n_rows, n_cols), 1.0, dtype=np.float32)
    mask = np.zeros((n_rows, n_cols), dtype=np.int8)
    friction = np.full((n_rows, n_cols), 0.01, dtype=np.float32)

    ocean_cols = 6
    mask[:, :ocean_cols] = 1  # OCEAN_CODE
    dem[:, :ocean_cols] = 0.0

    ridge_lo, ridge_hi = 35, 42
    ridge_row_lo, ridge_row_hi = 15, 55
    dem[ridge_row_lo:ridge_row_hi, ridge_lo:ridge_hi] = 8.0        # taller than the 3.0m boundary level
    friction[ridge_row_lo:ridge_row_hi, ridge_lo:ridge_hi] = 0.001  # deliberately LOW friction (bare rock)

    return dem, mask, friction


def make_figure_3() -> None:
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    ridge_lo, ridge_hi = 35, 42
    for ax, coupling, title in zip(axes[:2], (False, True), ("without obstacle coupling", "with obstacle coupling")):
        wd, _ = results[coupling]
        depth_display = np.where(mask == 1, np.nan, wd)
        ax.imshow(np.where(dem > 4, 1, np.nan), cmap=ListedColormap([WALL_COLOR]), origin="upper")
        im = ax.imshow(depth_display, cmap="Blues", vmin=0, vmax=3.0, origin="upper")
        ax.axvspan(ridge_lo - 0.5, ridge_hi - 0.5, color="none", edgecolor="#4a4a46", linewidth=1.2, linestyle=":")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02, label="flood depth (m)")

    ax = axes[2]
    wd_off, _ = results[False]
    wd_on, _ = results[True]
    depth_reduction = np.where(mask == 1, np.nan, wd_off - wd_on)
    ax.imshow(np.where(dem > 4, 1, np.nan), cmap=ListedColormap([WALL_COLOR]), origin="upper")
    im2 = ax.imshow(depth_reduction, cmap="Reds", vmin=0, vmax=float(np.nanmax(depth_reduction)), origin="upper")
    ax.axvspan(ridge_lo - 0.5, ridge_hi - 0.5, color="none", edgecolor="#4a4a46", linewidth=1.2, linestyle=":")
    ax.set_title("depth reduction from\nobstacle coupling", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im2, ax=ax, fraction=0.035, pad=0.02, label="depth reduction (m)")

    fig.suptitle(
        "Outer loop: a friction-cheap crossing of a physically-impassable ridge\n"
        "inflates depth behind it (interior rows) unless detected and blocked - the\n"
        "open flanks above/below the ridge are unaffected either way",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT_DIR / "eikonal_example_obstacle_coupling.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'eikonal_example_obstacle_coupling.png'}")


# ---------------------------------------------------------------------------
# Figure 2: block skipping
# ---------------------------------------------------------------------------

FIG2_BLOCK = 16  # illustration block size (production: SWEEP_BLOCK_SIZE=64 on ~5000x8000-cell tiles)


def _traced_block_solve(friction, seed_rows, seed_cols, seed_values, max_rounds: int):
    """`solve_eikonal_dense`'s own round loop on the real `_block_sweep`
    kernel (fully converged, epsilon=0), at the illustration block size,
    recording per round which blocks were actually processed (in at least
    one of its 4 sweeps) and which changed. Returns (t, per-round records).
    """
    m, n = friction.shape
    t = np.full((m + 1, n + 1), 99.0, dtype=friction.dtype)
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
            "changed": changed_at >= first_sweep,
            "frac_swept": n_swept / (4 * changed_at.size),
            "t": t[1:, 1:].copy(),
        })
        if max_change <= 0.0:
            break
    return t, records


def build_coastal_maze_scene(n_rows: int = 175, n_cols: int = 255, maze_size: int = 97) -> np.ndarray:
    """Uniform open land (settles within a few rounds) with one Figure-1-style
    spiral maze in the lower right, its walls impassable (friction 9999, the
    obstacle-coupling block value) - mimics a real tile, where most of the
    domain settles early and only a few winding areas keep the solve going.

    Sized so the (n_rows+1, n_cols+1) vertex grid is an exact multiple of
    FIG2_BLOCK: otherwise the extra vertex row/column forms a thin edge block
    with nothing to update in some sweep directions, which would read as
    "skipped" (e.g. only 93% processed in round 1) without being real skipping.
    """
    friction = np.full((n_rows, n_cols), 0.01, dtype=np.float32)
    maze = build_spiral_friction(size=maze_size, n_rings=10, gap_width=7)
    r0, c0 = n_rows - maze_size - 6, n_cols - maze_size - 8
    friction[r0:r0 + maze_size, c0:c0 + maze_size] = np.where(maze > 10, np.float32(9999.0), np.float32(0.01))
    return friction


def make_figure_2() -> None:
    friction = build_coastal_maze_scene()
    n_rows, n_cols = friction.shape
    # "Coastline": the whole top edge seeded at t=0.
    seed_rows = np.zeros(n_cols, dtype=np.int64)
    seed_cols = np.arange(n_cols, dtype=np.int64)
    seed_values = np.zeros(n_cols, dtype=np.float32)

    t_blk, records = _traced_block_solve(friction, seed_rows, seed_cols, seed_values, max_rounds=200)
    # Same result as the production solver (block size 64) and as plain dense sweeps - the whole point.
    t_prod = solve_eikonal_dense(friction, seed_rows, seed_cols, seed_values, epsilon=0.0, max_rounds=200)
    assert np.array_equal(t_blk, t_prod), "illustration block solve differs from production solve_eikonal_dense"
    t_dense = np.full_like(t_blk, 99.0)
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
    skip_color, swept_color = "#e4e4e0", BLOCK_COLOR
    block_cmap = ListedColormap([skip_color, swept_color])

    fig = plt.figure(figsize=(14, 3.1))
    gs = fig.add_gridspec(1, len(checkpoints), wspace=0.08, top=0.83, bottom=0.02, left=0.02, right=0.98)

    for i, k in enumerate(checkpoints):
        rec = records[k - 1]
        ax = fig.add_subplot(gs[0, i])
        # One square per block: re-swept this round (in at least one of its 4 sweeps) vs skipped.
        ax.pcolormesh(rec["processed"].astype(int), cmap=block_cmap, vmin=0, vmax=1,
                      edgecolors="white", linewidth=1.2)
        nbr, nbc = rec["processed"].shape
        ax.plot([0, nbc], [0, 0], color=WATER_COLOR, linewidth=5, solid_capstyle="butt", clip_on=False)
        ax.set_xlim(0, nbc)
        ax.set_ylim(nbr, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"round {k}\n{100 * rec['frac_swept']:.0f}% of block visits", fontsize=10.5)
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=swept_color),
                 plt.Rectangle((0, 0), 1, 1, color=skip_color),
                 plt.Line2D([0], [0], color=WATER_COLOR, linewidth=5)],
        labels=["re-swept: the block or an upwind neighbour block changed",
                "skipped: nothing it depends on changed",
                "coast (seeded edge)"],
        loc="upper center", ncol=3, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.0),
    )

    fig.savefig(OUT_DIR / "eikonal_example_block_sweep.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'eikonal_example_block_sweep.png'}")


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
    make_figure_3()
