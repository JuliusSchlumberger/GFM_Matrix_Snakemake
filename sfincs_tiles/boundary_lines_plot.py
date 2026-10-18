"""Per-tile diagnostic figure of the SFINCS station boundary lines (split out of
build_station_boundary_lines.py, cleanup 2026-10-18; pure move)."""

from __future__ import annotations
import math
import sys
from pathlib import Path
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
from matplotlib.colors import ListedColormap
from rasterio.warp import transform as warp_transform
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import WATER_COLOR  # noqa: E402

from boundary_lines_geometry import (  # noqa: E402
    OCEAN_CODE,
    SEGMENT_KINDS,
    STATION_STYLE,
)


def _plot(tile_id, tile_dir, gpkg, mask, bounds, tile, out_path: Path, row: dict) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(18, 9.5))
    extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    land_sea = np.where(mask == OCEAN_CODE, 0, 1)
    layers = {name for name, _ in pyogrio.list_layers(gpkg)} if gpkg.exists() else set()
    segs = gpd.read_file(gpkg, layer="segments") if "segments" in layers else None
    st = gpd.read_file(gpkg, layer="stations") if "stations" in layers else None

    # (a) lines
    ax.imshow(land_sea, extent=extent, cmap=ListedColormap([WATER_COLOR, "#e4e4e0"]), vmin=0, vmax=1,
              interpolation="nearest")
    tile.boundary.plot(ax=ax, color="black", linewidth=1.0)
    ax.plot([], [], color="black", linewidth=1.0, label="tile")
    msk_path = tile_dir / "sfincs_model" / "sfincs.msk"
    if msk_path.exists():
        from sfincs_grid import read_sfincs_mask
        m2d, tr, crs = read_sfincs_mask(tile_dir / "sfincs_model")
        r, c = np.nonzero(m2d == 2)  # waterlevel boundary
        lon, lat = warp_transform(crs, "EPSG:4326", *(tr * (c + 0.5, r + 0.5)))
        ax.scatter(lon, lat, s=1.5, color="#1f4e8c", alpha=0.6, label="SFINCS boundary cells", zorder=3)
    if segs is not None:
        for kind, color in SEGMENT_KINDS.items():
            sel = segs[segs["kind"] == kind]
            if len(sel):
                sel.plot(ax=ax, color=color, linewidth=2.2, zorder=4)
                ax.plot([], [], color=color, linewidth=2.2, label=f"{kind} link (n={len(sel)})")
    if st is not None and "node_lon" in st:
        for x0, y0, x1, y1 in zip(st.geometry.x, st.geometry.y, st["node_lon"], st["node_lat"]):
            ax.plot([x0, x1], [y0, y1], color="#555555", linewidth=0.6, zorder=4)
    if st is not None:
        for status, (marker, color) in STATION_STYLE.items():
            sel = st[st["status"] == status] if "status" in st else (st if status == "used" else st.iloc[:0])
            if len(sel):
                ax.scatter(sel.geometry.x, sel.geometry.y, s=40 if marker == "o" else 55, marker=marker,
                           color=color, edgecolors="black", linewidths=0.6, zorder=5,
                           label=f"station: {status} (n={len(sel)})")
        for k, p in zip(st["station"], st.geometry):
            ax.annotate(str(k), (p.x, p.y), xytext=(4, 4), textcoords="offset points", fontsize=7, zorder=6)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    flags = [f"{k}={row[k]}" for k in ("n_parts", "n_open_ends", "n_flagged", "mode") if k in row]
    ax.set_title(f"(a) tile {tile_id}  boundary lines   " + "  ".join(flags), fontsize=10)
    ax.legend(loc="lower left", fontsize=8)

    # (b) active / inactive split of the tile
    tb = tile.total_bounds
    ax2.imshow(land_sea, extent=extent, cmap=ListedColormap([WATER_COLOR, "#e4e4e0"]), vmin=0, vmax=1,
               interpolation="nearest")
    handles = []
    for name, style, label in [
        ("inactive", dict(facecolor="#4d4d4d", alpha=0.55, edgecolor="none"), "inactive cells (ocean side)"),
        ("land_inactive", dict(facecolor="#d62728", alpha=0.85, edgecolor="none"), "CONFLICT: land inactive"),
        ("ocean_active", dict(facecolor="none", hatch="////", edgecolor="#ff7f0e", linewidth=0.0),
         "open sea left active (reaches tile edge)"),
    ]:
        if name in layers:
            gpd.read_file(gpkg, layer=name).plot(ax=ax2, **style, zorder=3)
            handles.append(plt.Rectangle((0, 0), 1, 1, **style, label=label))
    if segs is not None:
        segs.plot(ax=ax2, color="black", linewidth=1.6, zorder=4)
        handles.append(plt.Line2D([0], [0], color="black", linewidth=1.6, label="boundary line"))
    tile.boundary.plot(ax=ax2, color="black", linewidth=1.0, linestyle="--", zorder=4)
    ax2.set_xlim(tb[0], tb[2])
    ax2.set_ylim(tb[1], tb[3])
    verdict = "suitable" if row.get("suitable") else "NOT suitable (conflicts)"
    ax2.set_title(f"(b) tile domain: {verdict}   land inactive {row.get('land_inactive_pct', '-')} % of land, "
                  f"open sea active {row.get('ocean_active_pct', '-')} % of ocean", fontsize=10)
    if handles:
        ax2.legend(handles=handles, loc="lower left", fontsize=8)

    for a_ in (ax, ax2):
        a_.set_aspect(1 / math.cos(math.radians(0.5 * (bounds[1] + bounds[3]))))
        a_.set_xlabel("longitude (°E)")
        a_.set_ylabel("latitude (°N)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
