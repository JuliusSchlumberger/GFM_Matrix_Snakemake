"""Diagnostic figures comparing flood depths across tile overlap zones.

For up to 6 focal tiles that share actual flood-data overlap with at least one
neighbour (assessed via domain bbox from model_bbox.json), one PNG is written
per focal tile.  Each figure shows:
  - Whitesmoke land-polygon background (same as plot_merged_results).
  - Grey: cells where exactly one tile reports a positive flood depth.
  - Red shades: cells where two or more tiles both report a positive flood
    depth, coloured by the maximum depth difference across contributing tiles.
  - Coloured rectangle outlines: each tile's model domain bbox (model_bbox.json),
    one colour per tile, used to identify tile contributions without masking
    the flood data.

Focal tiles are selected so that (a) their domain bbox overlaps at least one
neighbour's domain bbox, (b) actual flood overlap (≥2 tiles with depth > 0
at the same cell) exists in the group, and (c) no tile is the focal tile more
than once.  Tiles already assigned to a group are deprioritised to favour
geographic diversity across panels.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import get_data_catalog  # noqa: E402
from merge import AQUEDUCT_NODATA, _bounds_intersect  # noqa: E402

pp_cfg = snakemake.config["postprocessing"]  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821

_RESOLUTION_DEG = pp_cfg["plots"]["overlap_diag_resolution_arcsec"] / 3600.0


def _output_shape(minx: float, miny: float, maxx: float, maxy: float) -> tuple[int, int]:
    """Return (out_h, out_w) for the target diagnostic resolution."""
    out_w = max(1, round((maxx - minx) / _RESOLUTION_DEG))
    out_h = max(1, round((maxy - miny) / _RESOLUTION_DEG))
    return out_h, out_w


tile_rasters = list(snakemake.input.waterdepth_tiles)  # noqa: F821
output_dir = Path(snakemake.output.diagnostics)  # noqa: F821

N_PANELS = 6
TILE_PALETTE = plt.get_cmap("tab10")

output_dir.mkdir(parents=True, exist_ok=True)

data_catalog = get_data_catalog()
coastlines_path = data_catalog.get_source("land_polygons").path


def _tile_id(path: str) -> str:
    return Path(path).parts[-3]


def _load_domain_bbox(path: str) -> list[float]:
    """Load model_bbox.json for a tile; fall back to raster bounds."""
    bbox_file = Path(path).parents[1] / "inputs" / "model_bbox.json"
    if bbox_file.exists():
        with open(bbox_file) as f:
            return json.load(f)
    with rasterio.open(path) as src:
        return list(src.bounds)


def _read_into_grid(path: str, bounds: tuple, out_h: int, out_w: int) -> np.ndarray:
    """Read a tile raster resampled (max) into a common output grid.

    Uses warp reproject with Resampling.max so that any non-zero flood depth
    in each output pixel's footprint is preserved — sparse flood cells are not
    silently dropped when the target resolution is much coarser than the
    native ~1.6"/px (~50 m) resolution.  (Resampling.max is only valid for
    warp operations, not for src.read with out_shape.)
    """
    minx, miny, maxx, maxy = bounds
    dst_transform = transform_from_bounds(minx, miny, maxx, maxy, out_w, out_h)
    dst = np.zeros((out_h, out_w), dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=src.crs,
            resampling=Resampling.max,
        )
    return dst


# ── collect domain bboxes ────────────────────────────────────────────────────
domain_bboxes: dict[str, list[float]] = {p: _load_domain_bbox(p) for p in tile_rasters}

# ── build candidate groups using domain bbox overlap ─────────────────────────
all_groups: list[tuple[str, list[str]]] = []
for path in tile_rasters:
    db = domain_bboxes[path]
    neighbours = [
        p for p in tile_rasters
        if p != path and _bounds_intersect(tuple(db), tuple(domain_bboxes[p]))
    ]
    if neighbours:
        all_groups.append((path, [path] + neighbours))

# ── select up to N_PANELS focal tiles ────────────────────────────────────────
# Conditions: (1) domain bbox overlap exists, (2) actual flood overlap exists.
# Cache (combined_bounds, out_h, out_w) to avoid re-deriving in the plot loop.
Selection = tuple[str, list[str], tuple[float, float, float, float], int, int]
selected: list[Selection] = []
used: set[str] = set()

for focal, all_paths in all_groups:
    if focal in used or len(selected) >= N_PANELS:
        continue

    blist = [domain_bboxes[p] for p in all_paths]
    minx = min(b[0] for b in blist)
    miny = min(b[1] for b in blist)
    maxx = max(b[2] for b in blist)
    maxy = max(b[3] for b in blist)
    combined_bounds = (minx, miny, maxx, maxy)

    out_h, out_w = _output_shape(minx, miny, maxx, maxy)

    # Quick flood-overlap check at plot resolution
    arrays = np.stack([_read_into_grid(p, combined_bounds, out_h, out_w) for p in all_paths])
    valid = (arrays > 0) & (arrays < AQUEDUCT_NODATA)
    if valid.sum(axis=0).max() < 2:
        continue  # no cell has ≥2 tiles with flood data — skip this focal tile

    selected.append((focal, all_paths, combined_bounds, out_h, out_w))
    used.update(all_paths)

if not selected:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, "No overlapping tiles with flood data found.",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_axis_off()
    fig.savefig(output_dir / "no_overlap.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    raise SystemExit(0)

# ── one PNG per focal tile ───────────────────────────────────────────────────
for panel_idx, (focal, all_paths, combined_bounds, out_h, out_w) in enumerate(selected):
    minx, miny, maxx, maxy = combined_bounds

    arrays = np.stack([_read_into_grid(p, combined_bounds, out_h, out_w) for p in all_paths])
    valid = (arrays > 0) & (arrays < AQUEDUCT_NODATA)
    count = valid.sum(axis=0)
    unique = count == 1
    overlap = count >= 2

    # ── RGBA composite ───────────────────────────────────────────────────
    img = np.zeros((out_h, out_w, 4), dtype=float)

    # Unique flood cells → medium grey
    img[unique, :3] = 0.55
    img[unique, 3] = 0.75

    # Overlap cells → Reds by max depth difference
    diff_max = 0.0
    if overlap.any():
        vals = np.where(valid, arrays.astype(float), np.nan)
        diff = np.where(overlap, np.nanmax(vals, axis=0) - np.nanmin(vals, axis=0), np.nan)
        diff_max = float(np.nanmax(diff))
        norm = np.where(overlap, diff / diff_max if diff_max > 0 else 0.0, 0.0)
        reds = plt.get_cmap("Reds")
        red_rgba = reds(np.clip(0.25 + 0.75 * norm, 0, 1))
        img[overlap] = red_rgba[overlap]

    # ── figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    # Land polygon background
    coastlines = gpd.read_file(coastlines_path, layer="land_polygons",
                               bbox=(minx, miny, maxx, maxy))
    if not coastlines.empty:
        coastlines.plot(ax=ax, color="whitesmoke", edgecolor="whitesmoke",
                        linewidth=0.5, zorder=0)

    extent = (minx, maxx, miny, maxy)
    ax.imshow(img, extent=extent, origin="upper", aspect="equal",
              interpolation="nearest", zorder=1)

    # Domain bbox outlines — one colour per tile
    for i, path in enumerate(all_paths):
        b = domain_bboxes[path]
        color = TILE_PALETTE(i % 10)
        lw = 2.2 if path == focal else 1.2
        rect = mpatches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            linewidth=lw, edgecolor=color, facecolor="none", zorder=2,
        )
        ax.add_patch(rect)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Tile overlap diagnostic — focal tile {_tile_id(focal)} — {waterlevel_name}\n"
        "Grey: unique flood extent  |  Red: max depth difference in overlap zone",
        fontsize=9,
    )

    # Legend
    handles = [
        mpatches.Patch(facecolor=(0.55, 0.55, 0.55, 0.75), edgecolor="none",
                       label="Unique flood (one tile)"),
        mpatches.Patch(facecolor=plt.get_cmap("Reds")(0.75), edgecolor="none",
                       label="Overlap zone (max diff)"),
    ]
    for i, path in enumerate(all_paths):
        label = f"Tile {_tile_id(path)}" + (" ★ focal" if path == focal else "")
        handles.append(mpatches.Patch(facecolor="none", edgecolor=TILE_PALETTE(i % 10),
                                      linewidth=1.5, label=label))
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.85, ncol=2)

    if diff_max > 0:
        sm = plt.cm.ScalarMappable(cmap="Reds", norm=plt.Normalize(0, diff_max))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Max depth difference (m)", shrink=0.55, pad=0.02)

    out_path = output_dir / f"{panel_idx + 1:02d}_tile_{_tile_id(focal)}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
