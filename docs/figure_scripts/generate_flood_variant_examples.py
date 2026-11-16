"""Render the flood-variant comparison figure for docs/methods_02_flood_depth.md.

Runs all four FLOOD_MODELS (src/flood_model.py) on one synthetic scenario built to
isolate what each added mechanism changes: a coast-connectivity check (the isolated
pit), path-following vs straight-line attenuation (the ridge detour), and the
duration correction's spatially-varying friction (event_volume.DurationCorrection).
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
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from event_volume import DurationCorrection  # noqa: E402
from flood_model import effective_dem, flood_depth_dense  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1]

WALL_COLOR = "#4a4a46"
PIT_BOX_COLOR = "#e8702a"
POINT_COLOR = "#8a3ffc"
BOX_COLOR = {"A": "#4a4a46", "B": "#2a78d6", "C": "#c2410c"}

SEED_LEVEL_M = 2.5
RATE_M_PER_CELL = 0.01  # straight_line's attenuation_m_per_m and eikonal's friction
DEPTH_BIN_M = 0.10  # isoline-style colour banding step

N_ROWS, N_COLS = 60, 160
PIT_COL_LO, PIT_COL_HI = 20, 40
BERM_COL_LO, BERM_COL_HI = 18, 42
PIT_ROW_HI = 22
BERM_ROW_HI = 24
RIDGE_COL_LO, RIDGE_COL_HI = 70, 77
RIDGE_ROW_LO = 9  # ridge occupies rows RIDGE_ROW_LO:N_ROWS; rows 0:RIDGE_ROW_LO are the gap

SAMPLE_POINTS = {"near": (50, 85), "far": (50, 150)}


def build_variant_scenario(n_rows: int = N_ROWS, n_cols: int = N_COLS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ocean, a below-level pit embanked out to the domain's top edge, and a ridge.

    The pit has no coastline anywhere in the domain, not merely none nearby - its berm
    reaches the top edge, so there is no gap left for a real flood path to reach it. The
    ridge blocks every direct route to the far side except around its own gap at the top,
    so the real connected path there is longer than the straight-line distance to it.
    """
    dem = np.full((n_rows, n_cols), 0.5, dtype=np.float32)
    mask = np.zeros((n_rows, n_cols), dtype=np.int8)
    friction = np.full((n_rows, n_cols), RATE_M_PER_CELL, dtype=np.float32)

    ocean_cols = 6
    mask[:, :ocean_cols] = 1
    dem[:, :ocean_cols] = 0.0

    dem[:BERM_ROW_HI, BERM_COL_LO:PIT_COL_LO] = 6.0
    dem[:BERM_ROW_HI, PIT_COL_HI:BERM_COL_HI] = 6.0
    dem[PIT_ROW_HI:BERM_ROW_HI, PIT_COL_LO:PIT_COL_HI] = 6.0
    dem[:PIT_ROW_HI, PIT_COL_LO:PIT_COL_HI] = -1.0

    dem[RIDGE_ROW_LO:n_rows, RIDGE_COL_LO:RIDGE_COL_HI] = 10.0

    return dem, mask, friction


def _seeds(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(n_rows, dtype=np.int64)
    cols = np.full(n_rows, 5, dtype=np.int64)
    values = np.full(n_rows, SEED_LEVEL_M)
    return rows, cols, values


def run_base_variants(dem: np.ndarray, mask: np.ndarray, friction: np.ndarray) -> dict[str, tuple[np.ndarray, dict]]:
    """Run the four FLOOD_MODELS, no duration correction."""
    transform = Affine.identity()
    seed_rows, seed_cols, seed_values = _seeds(dem.shape[0])
    common = dict(dem=dem, mask=mask, friction=friction, transform=transform,
                  seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
                  k=1, idw_power=1.0, resolution=1.0)
    results = {}
    for model in ("bathtub", "bathtub_connected"):
        results[model] = flood_depth_dense(**common, model=model)
    results["straight_line"] = flood_depth_dense(**common, model="straight_line", attenuation_m_per_m=RATE_M_PER_CELL)
    results["eikonal"] = flood_depth_dense(
        **common, model="eikonal", obstacle_coupling=True, max_rounds=60, max_outer_iterations=3,
        waterlevel_epsilon_m=0.03,
    )
    return results


def build_duration_correction(n_rows: int) -> DurationCorrection:
    """A single-station, single-peak synthetic storm tide; t_ref/alpha at production defaults."""
    n_steps = 48  # 24h at 30-min steps
    t = np.arange(n_steps)
    peak = n_steps // 2
    series = SEED_LEVEL_M * 0.5 * (1.0 + np.cos(2 * np.pi * (t - peak) / n_steps))
    station_lonlat = np.array([[5.0, n_rows / 2.0]])
    return DurationCorrection(
        station_lonlat=station_lonlat, series=series[None, :], dt_s=1800.0,
        t_ref_s=12 * 3600.0, alpha=1.0, max_multiplier=10.0, crs=None,
    )


def run_duration_correction(dem: np.ndarray, mask: np.ndarray, friction: np.ndarray,
                            base_eikonal_wd: np.ndarray) -> dict:
    """The duration-modified friction field, and the depth it gives the two friction-based models.

    `straight_line` has no spatial friction field to correct: its "updated" depth rescales its
    one attenuation_m_per_m by the domain-mean multiplier over the first-pass wet cells - the
    same scalar flood_model.py itself reports as diagnostics["mean_duration_multiplier"] for the
    eikonal pass. `eikonal`'s own updated depth instead goes through the production
    duration_correction path unchanged (flood_model.flood_depth_dense).
    """
    transform = Affine.identity()
    seed_rows, seed_cols, seed_values = _seeds(dem.shape[0])
    dc = build_duration_correction(dem.shape[0])

    sources = mask == 1
    sources[seed_rows, seed_cols] = True
    multiplier = dc.multiplier(base_eikonal_wd, effective_dem(dem, mask), sources, transform)
    modified_friction = friction * multiplier
    wet = base_eikonal_wd > 0
    mean_multiplier = float(multiplier[wet].mean()) if wet.any() else 1.0

    common = dict(dem=dem, mask=mask, friction=friction, transform=transform,
                  seed_rows=seed_rows, seed_cols=seed_cols, seed_values=seed_values,
                  k=1, idw_power=1.0, resolution=1.0)
    straight_line_wd, _ = flood_depth_dense(
        **common, model="straight_line", attenuation_m_per_m=RATE_M_PER_CELL * mean_multiplier,
    )
    eikonal_wd, eikonal_diag = flood_depth_dense(
        **common, model="eikonal", obstacle_coupling=True, max_rounds=60, max_outer_iterations=3,
        waterlevel_epsilon_m=0.03, duration_correction=dc,
    )
    return {
        "modified_friction": modified_friction, "mean_multiplier": mean_multiplier,
        "straight_line": straight_line_wd, "eikonal": eikonal_wd, "eikonal_diag": eikonal_diag,
    }


def _depth_cmap_norm(vmax: float) -> tuple[ListedColormap, BoundaryNorm]:
    n_bins = max(1, int(np.ceil(vmax / DEPTH_BIN_M)))
    bounds = np.arange(0, (n_bins + 1) * DEPTH_BIN_M, DEPTH_BIN_M)
    cmap = ListedColormap(plt.cm.Blues(np.linspace(0.15, 0.95, n_bins)))
    return cmap, BoundaryNorm(bounds, cmap.N)


def _draw_depth(ax, dem: np.ndarray, mask: np.ndarray, wd: np.ndarray, cmap, norm):
    depth = np.where(mask == 1, np.nan, wd)
    depth = np.where(depth <= 0, np.nan, depth)
    ax.imshow(np.ones_like(dem), cmap=ListedColormap(["#eeeeee"]), vmin=0, vmax=1)
    im = ax.imshow(depth, cmap=cmap, norm=norm)
    ax.add_patch(Rectangle((BERM_COL_LO, 0), BERM_COL_HI - BERM_COL_LO, BERM_ROW_HI,
                            fill=False, edgecolor=PIT_BOX_COLOR, linewidth=1.4))
    ax.text((BERM_COL_LO + BERM_COL_HI) / 2, BERM_ROW_HI + 3, f"pit: {wd[:PIT_ROW_HI, PIT_COL_LO:PIT_COL_HI].mean():.2f} m",
            fontsize=8.5, color=PIT_BOX_COLOR, ha="center", va="top", weight="bold")
    for r, c in SAMPLE_POINTS.values():
        ax.scatter([c], [r], marker="o", s=38, facecolor="white", edgecolor=POINT_COLOR, linewidth=1.4, zorder=5)
        ax.annotate(f"{wd[r, c]:.2f} m", xy=(c, r), xytext=(c, r - 6), fontsize=8.5, color=POINT_COLOR,
                    ha="center", weight="bold", zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _group_bbox(axes_list, pad: float = 0.018, pad_top: float = 0.01) -> tuple[float, float, float, float]:
    positions = [ax.get_position() for ax in axes_list]
    x0 = min(p.x0 for p in positions) - pad
    x1 = max(p.x1 for p in positions) + pad
    y0 = min(p.y0 for p in positions) - pad
    y1 = max(p.y1 for p in positions) + pad_top
    return x0, y0, x1 - x0, y1 - y0


def _draw_group_box(fig, axes_list, color: str, label: str, pad_top: float = 0.055) -> None:
    x0, y0, w, h = _group_bbox(axes_list, pad_top=pad_top)
    patch = FancyBboxPatch(
        (x0, y0), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        transform=fig.transFigure, figure=fig, fill=False, edgecolor=color, linewidth=2.0, zorder=10,
    )
    fig.add_artist(patch)
    fig.text(x0 + 0.014, y0 + h - 0.006, label, fontsize=11.5, color=color, weight="bold", ha="left", va="top")


def make_figure() -> None:
    dem, mask, friction = build_variant_scenario()
    base = run_base_variants(dem, mask, friction)
    dur = run_duration_correction(dem, mask, friction, base["eikonal"][0])

    pit_slice = (slice(0, PIT_ROW_HI), slice(PIT_COL_LO, PIT_COL_HI))
    print("=== base variants (no duration correction) ===")
    for name, (wd, diag) in base.items():
        pts = {label: float(wd[r, c]) for label, (r, c) in SAMPLE_POINTS.items()}
        print(f"  {name:18s} pit={wd[pit_slice].mean():.3f}  points={pts}  diag={diag}")

    print("=== duration correction ===")
    print(f"  mean friction multiplier over first-pass wet cells: {dur['mean_multiplier']:.3f}")
    for name in ("straight_line", "eikonal"):
        wd = dur[name]
        pts = {label: float(wd[r, c]) for label, (r, c) in SAMPLE_POINTS.items()}
        print(f"  updated {name:14s} points={pts}")
    print(f"  updated eikonal diag={dur['eikonal_diag']}")

    vmax_depth = max(
        max(float(np.nanmax(wd)) for wd, _ in base.values()),
        float(np.nanmax(dur["straight_line"])), float(np.nanmax(dur["eikonal"])),
    )
    depth_cmap, depth_norm = _depth_cmap_norm(vmax_depth)
    vmax_friction = float(dur["modified_friction"].max())
    friction_cmap, friction_norm = plt.cm.Oranges, plt.Normalize(vmin=0, vmax=vmax_friction)

    fig = plt.figure(figsize=(20, 8.6))
    gs1 = fig.add_gridspec(1, 6, wspace=0.1, top=0.88, bottom=0.52, left=0.02, right=0.885)
    gs2 = fig.add_gridspec(1, 6, wspace=0.1, top=0.42, bottom=0.04, left=0.02, right=0.885)
    ax_dem, ax_fric, ax_bt, ax_btc, ax_sl, ax_eik = (fig.add_subplot(gs1[0, i]) for i in range(6))
    ax_fric_mod = fig.add_subplot(gs2[0, 0:2])
    ax_sl_dc = fig.add_subplot(gs2[0, 2:4])
    ax_eik_dc = fig.add_subplot(gs2[0, 4:6])

    terrain = np.where(mask == 1, -1, dem)
    ax_dem.imshow(terrain, cmap="terrain", vmin=-2, vmax=10)
    ax_dem.add_patch(Rectangle((BERM_COL_LO, 0), BERM_COL_HI - BERM_COL_LO, BERM_ROW_HI,
                                fill=False, edgecolor=PIT_BOX_COLOR, linewidth=1.4))
    ax_dem.annotate(
        "gap", xy=(73, 4), xytext=(100, 24), fontsize=10, color=WALL_COLOR, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=WALL_COLOR, linewidth=1.3), annotation_clip=False,
    )
    ax_dem.set_title("elevation (m)", fontsize=10.5)
    ax_dem.set_xticks([])
    ax_dem.set_yticks([])

    im_fric = ax_fric.imshow(np.where(mask == 1, np.nan, friction), cmap=friction_cmap, norm=friction_norm)
    ax_fric.set_title(f"friction (uniform, {RATE_M_PER_CELL:g} / cell)", fontsize=10.5)
    ax_fric.set_xticks([])
    ax_fric.set_yticks([])

    labels = {
        "bathtub": "bathtub\n(IDW level, no connectivity)",
        "bathtub_connected": "bathtub_connected\n(+ coast-connectivity filter)",
        "straight_line": "straight_line\n(+ straight-line attenuation)",
        "eikonal": "eikonal\n(+ path-following attenuation)",
    }
    im_depth = None
    for ax, name in zip((ax_bt, ax_btc, ax_sl, ax_eik), ("bathtub", "bathtub_connected", "straight_line", "eikonal")):
        wd, _ = base[name]
        im_depth = _draw_depth(ax, dem, mask, wd, depth_cmap, depth_norm)
        ax.set_title(labels[name], fontsize=10.5)

    im_fric_mod = ax_fric_mod.imshow(np.where(mask == 1, np.nan, dur["modified_friction"]),
                                      cmap=friction_cmap, norm=friction_norm)
    ax_fric_mod.set_title("friction x duration multiplier", fontsize=10.5)
    ax_fric_mod.set_xticks([])
    ax_fric_mod.set_yticks([])

    _draw_depth(ax_sl_dc, dem, mask, dur["straight_line"], depth_cmap, depth_norm)
    ax_sl_dc.set_title("straight_line, duration-corrected\n(domain-mean multiplier)", fontsize=10.5)
    _draw_depth(ax_eik_dc, dem, mask, dur["eikonal"], depth_cmap, depth_norm)
    ax_eik_dc.set_title("eikonal, duration-corrected", fontsize=10.5)

    fig.colorbar(im_fric, ax=ax_fric, location="left", fraction=0.09, pad=0.06, label="friction\n(m/cell)")
    fig.colorbar(im_depth, ax=ax_eik, fraction=0.035, pad=0.02, label="flood depth (m)", extend="max")
    fig.colorbar(im_fric_mod, ax=ax_fric_mod, location="left", fraction=0.09, pad=0.06, label="friction\n(m/cell)")
    fig.colorbar(im_depth, ax=ax_eik_dc, fraction=0.07, pad=0.02, label="flood depth (m)", extend="max")

    _draw_group_box(fig, [ax_dem, ax_fric], BOX_COLOR["A"], "A - inputs", pad_top=0.065)
    _draw_group_box(fig, [ax_bt, ax_btc, ax_sl, ax_eik], BOX_COLOR["B"], "B - flood depth, no duration correction",
                     pad_top=0.065)
    _draw_group_box(fig, [ax_fric_mod, ax_sl_dc, ax_eik_dc], BOX_COLOR["C"],
                     "C - duration correction (friction + corrected depth)", pad_top=0.065)

    fig.suptitle(
        "Four FLOOD_MODELS variants on one scenario, with and without duration correction\n"
        "orange box = embanked pit (no coastline anywhere); circles = sample points at fixed locations",
        fontsize=12.5, y=0.995,
    )
    fig.savefig(OUT_DIR / "flood_variant_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'flood_variant_comparison.png'}")


if __name__ == "__main__":
    make_figure()
