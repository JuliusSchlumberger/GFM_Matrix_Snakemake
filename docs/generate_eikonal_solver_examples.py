"""Builds two small, stylized (synthetic, not real-tile) example figures for
docs/flood_depth_method.md, run directly against the real production solver
code (src/eikonal.py, src/flood_model.py) - not hand-drawn illustrations.

Figure 1 (eikonal_example_rounds.png): why the inner loop needs more than
one round. A serpentine corridor (alternating-gap walls) forces the true
shortest path to reverse direction repeatedly, a known slow case for the
Fast Sweeping Method's fixed 4-direction sweep order. Shows the arrival
potential after 1, 2, 4 and 12 rounds, and the per-round max-change curve
against epsilon.

Figure 2 (eikonal_example_obstacle_coupling.png): why the outer loop is
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
from eikonal import solve_eikonal_dense  # noqa: E402
from flood_model import flood_depth_dense  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent

WATER_COLOR = "#2a78d6"
LAND_COLOR = "#d8d8d4"
WALL_COLOR = "#4a4a46"
BLOCK_COLOR = "#d6572a"


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

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1.3], hspace=0.35, wspace=0.15)

    wall_mask = friction > 10
    converged = snapshots[round_checkpoints[-1]]
    vmax = float(np.max(converged[converged < 90]))
    for i, n in enumerate(round_checkpoints):
        ax = fig.add_subplot(gs[0, i])
        unreached_mask = wall_mask | (snapshots[n] >= 90)
        arr = np.where(unreached_mask, np.nan, snapshots[n])
        ax.imshow(np.ones_like(arr), cmap=ListedColormap(["#eeeeee"]), origin="upper")  # not-yet-reached
        im = ax.imshow(arr, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
        ax.imshow(np.where(wall_mask, 1, np.nan), cmap=ListedColormap([WALL_COLOR]), origin="upper")
        ax.set_title(f"after {n} round{'s' if n != 1 else ''}\n({4 * n} sweeps)", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("source (t=0) along top edge", fontsize=9)

    cbar_ax = fig.add_axes([0.92, 0.42, 0.015, 0.42])
    fig.colorbar(im, cax=cbar_ax, label="arrival potential t")

    ax2 = fig.add_subplot(gs[1, :])
    rounds_x = np.arange(2, 2 + len(max_change_per_round))
    display_change = np.maximum(max_change_per_round, 1e-3)  # log-scale floor; 0 = fully converged
    ax2.plot(rounds_x, display_change, color="#2a78d6", linewidth=1.6, marker="o", markersize=3)
    ax2.axhline(epsilon, color=BLOCK_COLOR, linestyle="--", linewidth=1.2, label=f"epsilon = {epsilon}")
    ax2.set_yscale("log")
    ax2.set_xlabel("round")
    ax2.set_ylabel("max change\nthat round (log)")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Inner loop: repeated rounds relax the solution toward convergence\n"
        "(serpentine corridor - a known slow case for a fixed 4-direction sweep order)",
        fontsize=13,
    )
    fig.savefig(OUT_DIR / "eikonal_example_rounds.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'eikonal_example_rounds.png'}")
    print(f"  max_change by round 40: {max_change_per_round[-1]:.6g} (epsilon={epsilon})")


# ---------------------------------------------------------------------------
# Figure 2: obstacle coupling
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


def make_figure_2() -> None:
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


if __name__ == "__main__":
    make_figure_1()
    make_figure_2()
