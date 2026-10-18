"""Render the synthetic eikonal-solver figures for docs/methods_02_flood_depth.md.

Runs the production solver (src/eikonal.py, src/flood_model.py) on toy grids:
rounds/epsilon (spiral maze), block skipping, obstacle coupling (tall ridge).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from eikonal import _block_sweep, _dense_sweep, _ORTHANT_ORDER, solve_eikonal_dense  # noqa: E402
from flood_model import flood_depth_dense  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1]

WATER_COLOR = "#2a78d6"
LAND_COLOR = "#d8d8d4"
WALL_COLOR = "#4a4a46"
BLOCK_COLOR = "#d6572a"
NOT_REACHED_COLOR = "#eeeeee"
SWEEP_FILL_COLOR = "#f4d35e"  # not blue: blue reads as "flooded"
ATTENUATION_CMAP = "Blues_r"


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
    ax, orthant: int, label: str, example_offset: tuple[int, int],
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
    """Write eikonal_example_rounds.png: sweep quadrants, rounds, max change vs epsilon."""
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
    # Maze friction units are arbitrary (1 open, 50 walls); rescale to 0-2 m for display.
    DISPLAY_VMAX_M = 2.0
    panel_b_axes = []
    for i, n in enumerate(round_checkpoints):
        ax = fig.add_subplot(gs[1, i])
        unreached_mask = wall_mask | (snapshots[n] >= 90)
        arr = np.where(unreached_mask, np.nan, snapshots[n] / raw_vmax * DISPLAY_VMAX_M)
        ax.imshow(np.ones_like(arr), cmap=ListedColormap([NOT_REACHED_COLOR]), origin="upper")
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

    label_x = 0.005
    for ax, letter in ((panel_a_axes[0], "(a)"), (panel_b_axes[0], "(b)"), (ax2, "(c)")):
        row_top = ax.get_position().y1
        fig.text(label_x, row_top, letter, fontsize=13, weight="bold", ha="left", va="top")

    fig.savefig(OUT_DIR / "eikonal_example_rounds.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
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


FIG2_BLOCK = 16  # production uses SWEEP_BLOCK_SIZE=64


def _traced_block_solve(friction, seed_rows, seed_cols, seed_values, max_rounds: int):
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
    """Build open land with one spiral maze whose walls are impassable (friction 9999).

    The (n_rows+1, n_cols+1) vertex grid must be a multiple of FIG2_BLOCK, or a
    thin edge block reads as "skipped" without real skipping.
    """
    friction = np.full((n_rows, n_cols), 0.01, dtype=np.float32)
    maze = build_spiral_friction(size=maze_size, n_rings=10, gap_width=7)
    r0, c0 = n_rows - maze_size - 6, n_cols - maze_size - 8
    friction[r0:r0 + maze_size, c0:c0 + maze_size] = np.where(maze > 10, np.float32(9999.0), np.float32(0.01))
    return friction


def make_figure_2() -> None:
    """Write eikonal_example_block_sweep.png: re-swept vs skipped blocks per round."""
    friction = build_coastal_maze_scene()
    n_rows, n_cols = friction.shape
    seed_rows = np.zeros(n_cols, dtype=np.int64)
    seed_cols = np.arange(n_cols, dtype=np.int64)
    seed_values = np.zeros(n_cols, dtype=np.float32)

    t_blk, records = _traced_block_solve(friction, seed_rows, seed_cols, seed_values, max_rounds=200)
    # Block skipping must be bit-identical to the production solver and plain sweeps.
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
