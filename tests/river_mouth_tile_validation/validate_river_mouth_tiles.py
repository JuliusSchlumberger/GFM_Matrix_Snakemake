"""Validate tile_chunking.py's is_river_mouth 1x1deg TILE-level marking
(filter_floodable_tiles -> is_river_mouth_signature, OCEAN_FRAC_MIN=0.10,
RIVER_FRAC_MIN=0.02 at 500m coarse resolution - config.yml's
tile_generation.river_mouth_ocean_frac_min/river_mouth_river_frac_min).

This one is about the DISCRETE per-1x1deg-tile marking used to seed chunk
growth at delta mouths (tile_chunking.build_chunks) and compute_run_order's
river-mouth tie-break priority.

Two parts:
  A. Regression test - real known-delta tiles (already confirmed present
     in deltadtm_tiles_floodable.gpkg's is_river_mouth=True set earlier
     this session) must still be flagged True; synthetic mask arrays
     covering each threshold edge (pure ocean, pure land, coastal-no-
     river, river-below-threshold, river-but-no-ocean-access) must be
     flagged False. Run any time to catch a future regression without
     needing to re-inspect plots by eye.
  B. Zoomed diagnostic plot for EVERY tile currently flagged
     is_river_mouth=True in the real deltadtm_tiles_floodable.gpkg (79 as
     of 2026-08) - not just the handful spot-checked earlier - so every
     single detection can be visually audited. Land = grey, alpha=0.4,
     drawn behind everything else; ocean = light blue; river = teal; the
     specific detected 1x1deg tile's own boundary = red outline on top.

Usage:
    python validate_river_mouth_tiles.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

REPO_ROOT = Path(r"C:/Users/Schlu005/GFM")
sys.path.insert(0, str(REPO_ROOT / "snakemake_workflow" / "src"))

from tile_chunking import (  # noqa: E402
    _confirm_river_mouth_component,
    _mosaic_nearest_coarse,
    is_river_mouth_signature,
    river_mouth_fractions,
)
from tiles import _scan_mask_dir  # noqa: E402

MASK_DIR = Path("D:/GFM/inputs/DeltaDTM_masks")
# Small fixture holding just the known coordinates below (real
# is_river_mouth/ocean_frac/river_frac from tile_chunking._classify_tile_one
# against real DeltaDTM data), kept local to this test directory - was
# previously a whole-globe by-product of the retired deltadtm_coverage/
# prototype (deleted 2026-08 as dead code, which took this fixture down
# with it; regenerated 2026-08 scoped to just what this test needs).
FLOODABLE_TILES_PATH = Path(__file__).resolve().parent / "known_river_mouth_tiles.gpkg"
OUT_DIR = Path(__file__).resolve().parent / "plots"

OCEAN_CODE = 1
RIVER_CODE = 3
OCEAN_FRAC_MIN = 0.10
RIVER_FRAC_MIN = 0.02
COARSE_RESOLUTION_M = 500.0  # matches tile_generation's own marking resolution
ZOOM_SAMPLE_RESOLUTION_M = 150.0  # finer, for a visually detailed zoom (not used for the fraction test itself)
SPECKLE_BUFFER_DEG = 2.0  # matches config.yml's tile_generation.river_mouth_speckle_buffer_deg
MIN_COASTAL_COMPONENT_CELLS = 5000  # matches river_mouth_min_coastal_component_cells

# Real coordinates confirmed present in is_river_mouth=True earlier this
# session (see the global filter_floodable.py run + this conversation's own
# spot-checks) - the regression test asserts these are STILL flagged True.
# NOTE (2026-08): N31E120 and N32E119 were REMOVED from this list after
# visual + connected-component-based investigation showed they are false
# positives (isolated inland water, not real Yangtze-mouth coastline) - see
# KNOWN_FALSE_POSITIVE_COORDS below and the river_mouth_min_coastal_
# component_cells fix in tile_chunking.filter_floodable_tiles/
# _confirm_river_mouth_component.
KNOWN_RIVER_MOUTH_COORDS = [
    "N22E089", "N22E090",  # Ganges-Brahmaputra
    "N30E120", "N31E121",  # Yangtze
    "N73E126", "N73E127", "N73E128",  # Lena
    "N51E003",  # Rhine-Scheldt
    "N09E106",  # Mekong
    "S03W045", "N04W052",  # Amazon
    "N08W061",  # Orinoco
    "N60W163",  # Yukon-Kuskokwim
    "N16E094", "N16E095", "N16E096", "N16E097",  # Irrawaddy
]

# Confirmed false positives (2026-08): flagged by the naive per-tile
# ocean_frac/river_frac check, but shown (via zoomed plots + connected-
# component analysis over a wide buffered window) to be isolated inland
# water bodies DeltaDTM miscodes as `ocean_code`, nowhere near the real
# coast - largest connected ocean component within a 2deg buffer window
# maxed out at 1402 cells, vs >100k for every real delta checked. The
# river_mouth_min_coastal_component_cells fix must reject all of these.
KNOWN_FALSE_POSITIVE_COORDS = [
    "N30E114",  # inland Yangtze near Wuhan, ~500km from the coast
    "N31E118",  # inland Yangtze near Nanjing
    "N31E120",  # Suzhou/Kunshan aquaculture ponds, real coast a degree further east
    "N32E119",  # Gaoyou Lake, ~150km inland from the coast
    "N68W135",  # Arctic tundra thermokarst lakes, North Slope Alaska
]


# ---------------------------------------------------------------------------
# Part A - regression test
# ---------------------------------------------------------------------------

def _synthetic_mask(ocean_frac: float, river_frac: float, size: int = 100) -> np.ndarray:
    """A size x size mask array with exactly the requested ocean/river
    fractions (land elsewhere) - for exercising is_river_mouth_signature's
    threshold logic directly, without depending on any real coordinate.
    """
    n_ocean = round(ocean_frac * size * size)
    n_river = round(river_frac * size * size)
    flat = np.zeros(size * size, dtype=np.uint8)  # land
    flat[:n_ocean] = OCEAN_CODE
    flat[n_ocean:n_ocean + n_river] = RIVER_CODE
    rng = np.random.default_rng(0)
    rng.shuffle(flat)
    return flat.reshape(size, size)


def run_regression_test() -> None:
    print("=== Part A: regression test ===")

    # --- real known-good deltas: must still be flagged True ---
    tiles = gpd.read_file(FLOODABLE_TILES_PATH)
    flagged = set(tiles.loc[tiles["is_river_mouth"], "coord"])
    missing = [c for c in KNOWN_RIVER_MOUTH_COORDS if c not in flagged]
    assert not missing, f"REGRESSION: known river-mouth tiles no longer flagged: {missing}"
    print(f"PASS: all {len(KNOWN_RIVER_MOUTH_COORDS)} known real river-mouth tiles still flagged "
          f"is_river_mouth=True")

    # --- speckle re-check (_confirm_river_mouth_component), against real data ---
    mask_index = _scan_mask_dir(MASK_DIR)
    coord_lookup = {row.coord: (int(row.lat), int(row.lon)) for row in tiles.itertuples()}

    for coord in KNOWN_RIVER_MOUTH_COORDS:
        lat, lon = coord_lookup[coord]
        confirmed, largest = _confirm_river_mouth_component(
            lat, lon, mask_index, OCEAN_CODE, COARSE_RESOLUTION_M,
            SPECKLE_BUFFER_DEG, MIN_COASTAL_COMPONENT_CELLS,
        )
        assert confirmed, (
            f"REGRESSION: real river-mouth tile {coord} REJECTED by the speckle re-check "
            f"(largest connected component only {largest} cells)"
        )
    print(f"PASS: all {len(KNOWN_RIVER_MOUTH_COORDS)} known real river-mouth tiles CONFIRMED "
          f"by the speckle re-check (_confirm_river_mouth_component)")

    for coord in KNOWN_FALSE_POSITIVE_COORDS:
        lat, lon = coord_lookup[coord]
        confirmed, largest = _confirm_river_mouth_component(
            lat, lon, mask_index, OCEAN_CODE, COARSE_RESOLUTION_M,
            SPECKLE_BUFFER_DEG, MIN_COASTAL_COMPONENT_CELLS,
        )
        assert not confirmed, (
            f"REGRESSION: known false-positive tile {coord} was CONFIRMED by the speckle re-check "
            f"(largest connected component {largest} cells) - fix regressed"
        )
    print(f"PASS: all {len(KNOWN_FALSE_POSITIVE_COORDS)} known false-positive tiles correctly "
          f"REJECTED by the speckle re-check")

    # --- synthetic edge cases: must be flagged False ---
    synthetic_cases = [
        ("pure ocean (ocean=1.0, river=0.0)", 1.0, 0.0),
        ("pure land (ocean=0.0, river=0.0)", 0.0, 0.0),
        ("coastal, no river (ocean=0.5, river=0.0)", 0.5, 0.0),
        ("river below threshold (ocean=0.5, river=0.01)", 0.5, 0.01),
        ("river present but no ocean access (ocean=0.05, river=0.05)", 0.05, 0.05),
    ]
    for label, ocean_frac, river_frac in synthetic_cases:
        mask = _synthetic_mask(ocean_frac, river_frac)
        result = is_river_mouth_signature(mask, OCEAN_CODE, RIVER_CODE, OCEAN_FRAC_MIN, RIVER_FRAC_MIN)
        assert result is False, f"REGRESSION: '{label}' should NOT be flagged a river mouth, got True"
    print(f"PASS: all {len(synthetic_cases)} synthetic non-river-mouth edge cases correctly flagged False")

    # --- synthetic positive: clean case comfortably above both thresholds ---
    mask = _synthetic_mask(ocean_frac=0.15, river_frac=0.03)
    result = is_river_mouth_signature(mask, OCEAN_CODE, RIVER_CODE, OCEAN_FRAC_MIN, RIVER_FRAC_MIN)
    assert result is True, "REGRESSION: clean synthetic positive (ocean=0.15, river=0.03) should be flagged True"
    print("PASS: clean synthetic positive case (both fractions comfortably above threshold) correctly flagged True")
    print()


# ---------------------------------------------------------------------------
# Part B - zoomed diagnostic plot per currently-detected tile
# ---------------------------------------------------------------------------

def plot_tile(coord: str, lat: int, lon: int, mask_index: dict, buffer_deg: float = 0.75) -> None:
    bbox = (lon - buffer_deg, lat - buffer_deg, lon + 1 + buffer_deg, lat + 1 + buffer_deg)
    result = _mosaic_nearest_coarse(bbox, mask_index, None, ZOOM_SAMPLE_RESOLUTION_M)
    if result is None:
        print(f"  {coord}: no coverage in the zoom window - skipping plot")
        return
    mask_band, _dem, transform = result

    is_land = mask_band == 0
    is_ocean = mask_band == OCEAN_CODE
    is_river = mask_band == RIVER_CODE

    extent = (bbox[0], bbox[2], bbox[1], bbox[3])
    fig, ax = plt.subplots(figsize=(7, 7))

    # Land: grey, alpha 0.4, drawn first (behind everything else).
    land_rgba = np.zeros((*mask_band.shape, 4))
    land_rgba[is_land] = (0.5, 0.5, 0.5, 0.4)
    ax.imshow(land_rgba, extent=extent, origin="upper", zorder=1)

    # Ocean: light blue, opaque.
    ocean_rgba = np.zeros((*mask_band.shape, 4))
    ocean_rgba[is_ocean] = (0.68, 0.85, 0.90, 1.0)
    ax.imshow(ocean_rgba, extent=extent, origin="upper", zorder=2)

    # River: teal, opaque - distinct from both land and ocean.
    river_rgba = np.zeros((*mask_band.shape, 4))
    river_rgba[is_river] = (0.0, 0.70, 0.70, 1.0)
    ax.imshow(river_rgba, extent=extent, origin="upper", zorder=3)

    # The specific detected 1x1deg tile's own boundary, on top.
    ax.add_patch(Rectangle((lon, lat), 1, 1, fill=False, edgecolor="red", linewidth=2.2, zorder=4))

    ocean_frac, river_frac = river_mouth_fractions(mask_band, OCEAN_CODE, RIVER_CODE)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(f"{coord}  (tile ocean_frac={ocean_frac:.3f}, river_frac={river_frac:.3f})")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=(0.5, 0.5, 0.5, 0.4), label="land"),
        Rectangle((0, 0), 1, 1, facecolor=(0.68, 0.85, 0.90, 1.0), label="ocean"),
        Rectangle((0, 0), 1, 1, facecolor=(0.0, 0.70, 0.70, 1.0), label="river"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", linewidth=2, label="detected tile"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{coord}.png", dpi=140)
    plt.close(fig)


def generate_all_plots() -> None:
    print("=== Part B: zoomed plots for every currently-detected river-mouth tile ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles = gpd.read_file(FLOODABLE_TILES_PATH)
    mouths = tiles[tiles["is_river_mouth"]].sort_values("coord")
    mask_index = _scan_mask_dir(MASK_DIR)
    print(f"{len(mouths)} tiles to plot -> {OUT_DIR}")

    for i, row in enumerate(mouths.itertuples(), 1):
        plot_tile(row.coord, int(row.lat), int(row.lon), mask_index)
        if i % 10 == 0:
            print(f"  {i}/{len(mouths)} plotted")

    print(f"Done: {len(mouths)} plots written to {OUT_DIR}")


def main() -> None:
    run_regression_test()
    generate_all_plots()


if __name__ == "__main__":
    main()
