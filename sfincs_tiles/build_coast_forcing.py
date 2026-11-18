"""Coastline boundary for SFINCS (coast-boundary test, C1): boundary cells, seed levels, hydrographs.

`smoothness` checks, per tile, how far the forced levels of neighbouring boundary cells differ over
the event (a source of SFINCS instabilities), on the tile's native mask coarsened to roughly the
SFINCS grid (findings_coast_boundary.md, stage 0b).

Usage:
    python build_coast_forcing.py smoothness --config <cfg> --tile-ids 3758 891 1677 [--hydrograph-nc <nc>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
import yaml  # noqa: E402
from scipy import ndimage  # noqa: E402
from sklearn.neighbors import BallTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_station_boundary_lines as bsl  # noqa: E402
from build_line_forcing import COAST_HG_PEAK_HR, read_hydrographs  # noqa: E402
from calibration_paths import RETURN_PERIOD, WATERLEVEL_NAME  # noqa: E402
from connected_forcing import (  # noqa: E402
    SEARCH_BUFFER_DEG, SEARCH_MIN_SIZE_DEG, STATION_OCEAN_TOL_M, _components, _label_at,
)
from gfm_config import read_figures_root, read_root, storm_tide_hydrograph_path  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402
from run_calibration_models import (  # noqa: E402
    HYDRO_WINDOW_HR, NODATA_CODE, OCEAN_CODE, production_seeds, subgrid_coastline_mask,
)

SUBGRID_FACTOR = 4  # 30 m native/subgrid pixels per 120 m SFINCS cell
OCEAN_FRACTION_MIN = 0.5
MSK_INACTIVE, MSK_ACTIVE, MSK_BOUNDARY = 0, 1, 2
_STRUCTURE_8 = np.ones((3, 3), dtype=bool)
FORCING_MODES = ("nearest", "blend")
JUMP_THRESHOLDS_M = (0.05, 0.10, 0.25)


def coarsen_mask(mask: np.ndarray, factor: int, ocean_code: int = OCEAN_CODE) -> np.ndarray:
    """Coarsen a class mask by `factor`: ocean where >= OCEAN_FRACTION_MIN of the block is ocean.

    Other blocks take their most frequent non-ocean class. Trailing rows/columns that do not fill
    a block are dropped.
    """
    h, w = (mask.shape[0] // factor) * factor, (mask.shape[1] // factor) * factor
    blocks = mask[:h, :w].reshape(h // factor, factor, w // factor, factor).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(h // factor, w // factor, factor * factor).astype(np.int64)
    ocean = (blocks == ocean_code).mean(axis=2) >= OCEAN_FRACTION_MIN
    codes = np.unique(blocks)
    non_ocean = [c for c in codes if c != ocean_code]
    if not non_ocean:
        return np.full(ocean.shape, ocean_code, dtype=mask.dtype)
    counts = np.stack([(blocks == c).sum(axis=2) for c in non_ocean], axis=2)
    out = np.asarray(non_ocean)[counts.argmax(axis=2)]
    return np.where(ocean, ocean_code, out).astype(mask.dtype)


def coast_boundary_msk(mask: np.ndarray, river_code: int, ocean_code: int = OCEAN_CODE) -> np.ndarray:
    """Return SFINCS msk: coastline cells 2, the rest of edge-connected ocean 0, everything else 1.

    Coastline as the eikonal (`run_calibration_models.subgrid_coastline_mask`). Ocean bodies that do
    not reach the domain edge (unresolved lagoons) stay active.
    """
    coast = subgrid_coastline_mask(mask, ocean_code, river_code)
    ocean = mask == ocean_code
    labels, _ = ndimage.label(ocean, structure=_STRUCTURE_8)
    edge = np.zeros_like(ocean)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    edge |= ndimage.binary_dilation(mask == NODATA_CODE, structure=_STRUCTURE_8)
    edge_labels = np.unique(labels[edge & ocean])
    edge_ocean = np.isin(labels, edge_labels[edge_labels != 0])
    msk = np.full(mask.shape, MSK_ACTIVE, dtype=np.int8)
    msk[edge_ocean] = MSK_INACTIVE
    msk[coast] = MSK_BOUNDARY
    return msk


def connected_hydrograph_stations(cell_lonlat: np.ndarray, hg_xy: np.ndarray, tile: gpd.GeoDataFrame,
                                  root: Path, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (station indices (n, k), haversine distances (n, k) rad, unconnected flag) per cell.

    The k nearest hydrograph stations in the cell's own ocean body (global native mask over the
    production station search box, as connected_forcing.interior_points), nearest first; fewer
    stations in the body repeat the last one. A cell whose body holds no station takes the
    nearest stations overall and is flagged.
    """
    labels, ltr = _components(root, tile)
    r_px = int(np.ceil(STATION_OCEAN_TOL_M / (abs(ltr.e) * 110570.0)))
    b = bsl._search_box(tile, SEARCH_BUFFER_DEG, SEARCH_MIN_SIZE_DEG)
    inbox = np.flatnonzero((hg_xy[:, 0] >= b[0]) & (hg_xy[:, 0] <= b[2]) & (hg_xy[:, 1] >= b[1]) & (hg_xy[:, 1] <= b[3]))
    st_label = np.array([_label_at(labels, ltr, *hg_xy[j], r_px) for j in inbox])
    cell_label = np.array([_label_at(labels, ltr, lon, lat, 2 * SUBGRID_FACTOR) for lon, lat in cell_lonlat])
    idx = np.zeros((len(cell_lonlat), k), dtype=np.int64)
    dist = np.zeros((len(cell_lonlat), k))
    unconnected = np.zeros(len(cell_lonlat), dtype=bool)
    for lab in np.unique(cell_label):
        cells = np.flatnonzero(cell_label == lab)
        pool = inbox[st_label == lab] if lab else inbox[:0]
        if not len(pool):
            pool, unconnected[cells] = np.arange(len(hg_xy)), True
        kk = min(k, len(pool))
        tree = BallTree(np.radians(hg_xy[pool][:, ::-1]), metric="haversine")
        d, j = tree.query(np.radians(cell_lonlat[cells][:, ::-1]), k=kk)
        pad = k - kk
        idx[cells] = np.pad(pool[j], ((0, 0), (0, pad)), mode="edge")
        dist[cells] = np.pad(d, ((0, 0), (0, pad)), mode="edge")
    return idx, dist, unconnected


def window_series(hg: np.ndarray, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Return (elapsed h, series) over the simulated window (HYDRO_WINDOW_HR)."""
    elapsed = (times - times[0]).total_seconds().to_numpy() / 3600.0
    keep = (elapsed >= HYDRO_WINDOW_HR[0]) & (elapsed <= HYDRO_WINDOW_HR[1])
    return elapsed[keep], hg[:, keep]


def cell_forcing(series: np.ndarray, idx: np.ndarray, dist: np.ndarray, seed: np.ndarray, mode: str,
                 idw_power: float) -> np.ndarray:
    """Return each cell's forced level series (n, t), m: a station series shifted so its max is the seed.

    mode "nearest": the nearest station's series (plan step 5, as the eikonal duration correction).
    mode "blend": the inverse-distance blend (weights distance^-idw_power, as the seeds) of the
    k stations' series. Product peaks all sit at the same time (storm peak on the tidal high
    water at t = 0), so blending does not lower the peak.
    """
    if mode == "nearest":
        z = series[idx[:, 0]]
    elif mode == "blend":
        with np.errstate(divide="ignore"):
            w = dist ** -idw_power
        exact = ~np.isfinite(w)
        w = np.where(exact.any(axis=1, keepdims=True), exact.astype(float), w)
        w /= w.sum(axis=1, keepdims=True)
        z = np.einsum("nk,nkt->nt", w, series[idx])
    else:
        raise ValueError(f"mode must be nearest or blend, got {mode!r}")
    return z - z.max(axis=1, keepdims=True) + seed[:, None]


def adjacent_pairs(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return index pairs (i, j), i < j, of cells that touch (8-connected)."""
    pos = {(r, c): k for k, (r, c) in enumerate(zip(rows.tolist(), cols.tolist()))}
    a, b = [], []
    for k, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            j = pos.get((r + dr, c + dc))
            if j is not None:
                a.append(k)
                b.append(j)
    return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)


def level_jumps(z: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 20_000) -> tuple[np.ndarray, np.ndarray]:
    """Return per pair (max_t |z_a(t) - z_b(t)| in m, the time index where it occurs)."""
    jump = np.empty(len(a))
    at = np.empty(len(a), dtype=np.int64)
    for s in range(0, len(a), chunk):
        d = np.abs(z[a[s:s + chunk]] - z[b[s:s + chunk]])
        at[s:s + chunk] = d.argmax(axis=1)
        jump[s:s + chunk] = d.max(axis=1)
    return jump, at


def _jump_stats(prefix: str, jump: np.ndarray) -> dict:
    if not len(jump):
        return {f"{prefix}_max_m": 0.0}
    return {f"{prefix}_max_m": float(jump.max()), f"{prefix}_p99_m": float(np.percentile(jump, 99)),
            f"{prefix}_p90_m": float(np.percentile(jump, 90)),
            **{f"{prefix}_n_pairs_over_{round(t * 100)}cm": int((jump > t).sum()) for t in JUMP_THRESHOLDS_M}}


def tile_smoothness(tile_id: str, root: Path, config: dict, hg_xy: np.ndarray, series: np.ndarray,
                    elapsed_hr: np.ndarray) -> tuple[dict, pd.DataFrame, dict]:
    """Return (summary, adjacent-cell level jumps, map layers) for one tile's coarsened coastline.

    Jump of a pair of adjacent boundary cells: max over the window of the difference of their
    forced levels, unshifted. Reported for the nearest-station and the blended forcing, and for
    the seeds alone (the jump at the peak).
    """
    inputs = root / "model_outputs" / tile_id / "inputs"
    with retry_transient_io(rasterio.open, inputs / "mask.tif") as src:
        native, ntr, crs = src.read(1).astype(np.int16), src.transform, src.crs
    mask = coarsen_mask(native, SUBGRID_FACTOR)
    transform = ntr * ntr.scale(SUBGRID_FACTOR, SUBGRID_FACTOR)
    river_code = int(config["tile_generation"]["river_code"])
    flooding = config["simulation"]["flooding"]
    knn, idw_power = int(flooding["knn"]), float(flooding["idw_power"])
    boundaries = retry_transient_io(gpd.read_file, inputs / f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg")
    seeds = production_seeds(mask, transform, crs, boundaries, knn, river_code, idw_power)
    if seeds is None:
        raise ValueError(f"tile {tile_id}: no coastline cell on the coarsened mask")
    rows, cols, level = seeds
    lon, lat = rasterio.transform.xy(transform, rows, cols)
    cell_lonlat = np.column_stack([lon, lat])
    tile = gpd.read_file(inputs / "tile_geometry.gpkg").to_crs("EPSG:4326")
    idx, dist, unconnected = connected_hydrograph_stations(cell_lonlat, hg_xy, tile, root, knn)

    a, b = adjacent_pairs(rows, cols)
    seed_jump = np.abs(level[a] - level[b]).astype(np.float64)
    pairs = pd.DataFrame({"tile_id": tile_id, "cell_a": a, "cell_b": b, "lon": cell_lonlat[a, 0],
                          "lat": cell_lonlat[a, 1], "station_a": idx[a, 0], "station_b": idx[b, 0],
                          "seed_jump_m": seed_jump})
    layers = {"mask": mask, "transform": transform, "station": idx[:, 0], "level": level,
              "lon": cell_lonlat[:, 0], "lat": cell_lonlat[:, 1]}
    summary = {"tile_id": tile_id, "n_boundary_cells": len(rows), "n_adjacent_pairs": len(a),
               "n_hydrograph_stations": int(len(np.unique(idx[:, 0]))),
               "n_station_switches": int((idx[a, 0] != idx[b, 0]).sum()),
               "n_cells_unconnected_station": int(unconnected.sum()),
               "nearest_station_km_p95": float(np.percentile(dist[:, 0], 95) * 6371.0),
               "seed_m_min": float(level.min()), "seed_m_max": float(level.max()),
               **_jump_stats("seed_jump", seed_jump)}
    for mode in FORCING_MODES:
        z = cell_forcing(series, idx, dist, level.astype(np.float64), mode, idw_power)
        jump, at = level_jumps(z, a, b)
        pairs[f"{mode}_jump_m"], pairs[f"{mode}_jump_at_h"] = jump, elapsed_hr[at]
        cell = np.zeros(len(rows))
        np.maximum.at(cell, a, jump)
        np.maximum.at(cell, b, jump)
        layers[f"{mode}_jump"] = cell
        summary.update(_jump_stats(f"{mode}_jump", jump))
    return summary, pairs, layers


def _plot_tile(tile_id: str, layers: dict, out: Path, source: str) -> None:
    mask, tr = layers["mask"], layers["transform"]
    extent = (tr.c, tr.c + tr.a * mask.shape[1], tr.f + tr.e * mask.shape[0], tr.f)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), layout="constrained")
    land = np.where(mask == OCEAN_CODE, np.nan, 1.0)
    panels = [("seed level (m)", layers["level"], "viridis", None, None),
              ("nearest hydrograph station (id mod 20)", layers["station"] % 20, "tab20", 0, 19),
              ("max level jump to a neighbour, nearest (m)", layers["nearest_jump"], "magma_r", 0, 0.5),
              ("max level jump to a neighbour, blend (m)", layers["blend_jump"], "magma_r", 0, 0.5)]
    for ax, (title, values, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(land, extent=extent, cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
        order = np.argsort(values)  # largest on top
        sc = ax.scatter(layers["lon"][order], layers["lat"][order], c=values[order], s=2, cmap=cmap,
                        vmin=vmin, vmax=vmax, linewidths=0)
        fig.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(f"tile {tile_id}, stage 0b, {source}: {title}", fontsize=8)
        ax.set_aspect(1 / np.cos(np.radians(np.mean(layers["lat"]))))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def cmd_smoothness(args) -> None:
    """Write stage-0b smoothness tables and quick-look maps for the given tiles."""
    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root = read_root(config_path)
    nc = Path(args.hydrograph_nc) if args.hydrograph_nc else storm_tide_hydrograph_path(config_path)
    hg_lon, hg_lat, hg, times = read_hydrographs(nc)
    elapsed_hr, series = window_series(hg, times)
    hg_xy = np.column_stack([hg_lon, hg_lat])
    out = Path(args.out_dir) if args.out_dir else root / "calibration_coastbnd_SFINCS" / "stage0b"
    fig_dir = (out if args.out_dir
               else read_figures_root(config_path) / "sfincs_runs" / "calibration_coastbnd_SFINCS" / "stage0b")
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"hydrographs: {nc} ({len(hg_lon)} stations, window {HYDRO_WINDOW_HR} h, peak at {COAST_HG_PEAK_HR} h)")
    summaries, all_jumps = [], []
    for tid in args.tile_ids:
        summary, jumps, layers = tile_smoothness(str(tid), root, config, hg_xy, series, elapsed_hr)
        summaries.append(summary)
        all_jumps.append(jumps)
        _plot_tile(str(tid), layers, fig_dir / f"smoothness_{tid}_{nc.stem}.png", nc.stem)
        print(pd.Series(summary).to_string(), flush=True)
    pd.DataFrame(summaries).to_csv(out / f"smoothness_tiles_{nc.stem}.csv", index=False)
    pd.concat(all_jumps).to_csv(out / f"smoothness_cell_pairs_{nc.stem}.csv", index=False)
    print(f"wrote {out} and {fig_dir}")


def main() -> None:
    """Run the requested subcommand."""
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["smoothness"])
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--tile-ids", nargs="+", required=True)
    parser.add_argument("--hydrograph-nc", default=None, help="default: the config's storm_tide_source hydrographs")
    parser.add_argument("--out-dir", default=None, help="tables and figures here (default: <root>/calibration_coastbnd_SFINCS/stage0b, figures under figures_root)")
    args = parser.parse_args()
    {"smoothness": cmd_smoothness}[args.command](args)


if __name__ == "__main__":
    main()
