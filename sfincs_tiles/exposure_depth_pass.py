"""Stage-1 raster pass of the exposure/depth assessment: population-weighted agreement per tile.

For each tile, on SFINCS's 30 m subgrid over SFINCS-active land in the tile's Voronoi core (as the
dedup scoring): area- and population-weighted matched / model-only / SFINCS-only counts per
population-density class, exact depth-difference histograms per SFINCS depth class on both-wet
cells, and the depths in each model's own-only cells. Population: WorldPop counts, each subgrid
pixel weighted count x pixel area / cell area (production's equal share, average_pool_to_grid).
Reads the calibration directory; writes one npz + json per tile to --out-dir.

Usage:
    python exposure_depth_pass.py select --config <config.yml> --out-dir <dir>
    python exposure_depth_pass.py stage1 --config <config.yml> --out-dir <dir> [--fsf 12] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import decode_waterdepth_cm, run_path  # noqa: E402
from flood_agreement import WET_THRESHOLD_M  # noqa: E402
from gfm_config import read_root, resolve_catalog_path  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402
from sfincs_grid import active_on_subgrid, build_tile_tree, core_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exposure_grid import population_per_pixel  # noqa: E402

LAND_CODE = 0
DENSITY_EDGES = (10.0, 100.0, 1000.0)  # people per km2: classes < 10, 10-100, 100-1000, > 1000
DENSITY_LABELS = ("<10", "10-100", "100-1000", ">1000")
SFINCS_CLASS_EDGES = (0.3, 0.5, 1.0, 1.5, 2.0, 3.0)  # m, from WET_THRESHOLD_M; the last class > 3 m
DIFF_EDGES = np.round(np.arange(-3.0, 3.001, 0.01), 2)  # m, model - SFINCS, 1 cm bins (+ catch-alls)
DEPTH_EDGES = np.round(np.arange(0.0, 3.001, 0.1), 1)  # m, own-only cell depths (+ > 3 m)
N_PER_DENSITY_CLASS = 25
# Tiles kept in the subset if they exist: SF Bay, Severn estuary, Bay of Fundy (lon, lat).
NAMED_SITES = {"SF Bay": (-122.3, 37.8), "Severn": (-3.0, 51.4), "Fundy": (-64.5, 45.3)}


def legacy_raster(out_dir: Path, fsf: float) -> Path:
    """Return the eikonal raster of the earlier calibration layout (effective friction factor)."""
    return out_dir / f"eikonal_on_subgrid_waterdepth_RP100_SLR_0_fsf{fsf:g}.tif"


def _weighted_hist(values: np.ndarray, weights: np.ndarray, edges: np.ndarray) -> np.ndarray:
    full = np.concatenate(([-np.inf], edges, [np.inf]))
    return np.histogram(values, bins=full, weights=weights)[0]


def tile_pass(tile_dir: Path, fsf: float, pop_path: Path, tree, tile_ids, run_key: str | None = None) -> dict:
    """Compute one tile's stage-1 accumulators (dict of arrays and scalars)."""
    sfincs_dir = tile_dir / "sfincs_model"
    with retry_transient_io(rasterio.open, sfincs_dir / "hmax_subgrid.tif") as src:
        sf = src.read(1).astype(np.float32)
        nodata, transform, crs, shape = src.nodata, src.transform, src.crs, src.shape
    if nodata is not None and not np.isnan(nodata):
        sf[sf == nodata] = np.nan
    path = run_path(tile_dir / "outputs", run_key) if run_key else legacy_raster(tile_dir / "outputs", fsf)
    model, _, _, mshape = decode_waterdepth_cm(path)
    if mshape != shape:
        raise ValueError(f"{path.name} {mshape} does not match the subgrid {shape}")
    mog = np.empty(shape, dtype=np.float32)
    with retry_transient_io(rasterio.open, tile_dir / "inputs" / "mask.tif") as src:
        reproject(source=rasterio.band(src, 1), destination=mog, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest)
    score = ((mog == LAND_CODE) & active_on_subgrid(sfincs_dir, shape, transform)
             & core_mask(shape, transform, crs, tile_dir.name, tree, tile_ids))
    people, density = population_per_pixel(pop_path, shape, transform, crs)
    pixel_km2 = abs(transform.a * transform.e - transform.b * transform.d) / 1e6
    dclass = np.digitize(density, DENSITY_EDGES)

    sf_wet = score & np.isfinite(sf) & (sf > WET_THRESHOLD_M)
    m_wet = score & np.isfinite(model) & (model > WET_THRESHOLD_M)
    both, m_only, s_only = m_wet & sf_wet, m_wet & ~sf_wet, ~m_wet & sf_wet
    out = {"tile_id": tile_dir.name, "pixel_km2": pixel_km2,
           "pop_total": float(people[score].sum()), "area_km2": float(score.sum() * pixel_km2)}
    weights = {"area": np.full(shape, pixel_km2), "pop": people}
    for wname, w in weights.items():
        out[f"counts_{wname}"] = np.array([[w[sel & (dclass == c)].sum() for sel in (both, m_only, s_only)]
                                           for c in range(len(DENSITY_LABELS))])  # (density class, 3)
        d = (model - sf)[both]
        sclass = np.digitize(sf[both], SFINCS_CLASS_EDGES)
        out[f"diff_{wname}"] = np.stack([_weighted_hist(d[sclass == c], w[both][sclass == c], DIFF_EDGES)
                                         for c in range(len(SFINCS_CLASS_EDGES) + 1)])
        out[f"model_only_depth_{wname}"] = _weighted_hist(model[m_only], w[m_only], DEPTH_EDGES)
        out[f"sfincs_only_depth_{wname}"] = _weighted_hist(sf[s_only], w[s_only], DEPTH_EDGES)
    return out


def _pass_task(args) -> tuple[str, str | None]:
    tile_dir, out_dir, fsf, pop_path, tree, tile_ids = args
    try:
        res = tile_pass(tile_dir, fsf, pop_path, tree, tile_ids)
    except Exception as exc:  # one broken tile must not stop the pass; it is reported
        return tile_dir.name, f"{type(exc).__name__}: {exc}"
    arrays = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out_dir / f"{tile_dir.name}.npz", **arrays)
    (out_dir / f"{tile_dir.name}.json").write_text(
        json.dumps({k: v for k, v in res.items() if not isinstance(v, np.ndarray)}))
    return tile_dir.name, None


def select(args) -> None:
    """Stratify tiles with SFINCS flooding by population density; write subset_tiles.csv."""
    root = read_root(Path(args.config))
    base = root / args.base_dir_name
    pop_path = resolve_catalog_path(Path(args.catalog), root, "population")
    rows = []
    for p in sorted(base.glob("*/outputs/calibration_result.json")):
        r = json.loads(p.read_text())
        if r["sfincs"]["flooded_km2"] <= 0:
            continue
        with rasterio.open(p.parent.parent / "inputs" / "mask.tif") as m:
            b = transform_bounds(m.crs, "EPSG:4326", *m.bounds)
        with rasterio.open(pop_path) as src:
            win = from_bounds(*b, src.transform).round_offsets().round_lengths()
            c = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float64)
            c[(c == src.nodata) | (c < 0)] = 0.0
        land = r["domain"]["land_km2"]
        rows.append({"tile_id": int(r["tile_id"]), "lon": (b[0] + b[2]) / 2, "lat": (b[1] + b[3]) / 2,
                     "bounds": b, "pop": c.sum(), "land_km2": land,
                     "density": c.sum() / land if land > 0 else 0.0})
    tiles = pd.DataFrame(rows)
    tiles["density_class"] = [DENSITY_LABELS[i] for i in np.digitize(tiles.density, DENSITY_EDGES)]
    picked = [g.sample(min(len(g), N_PER_DENSITY_CLASS), random_state=0) for _, g in tiles.groupby("density_class")]
    subset = pd.concat(picked)
    for name, (lon, lat) in NAMED_SITES.items():
        hit = tiles[[b[0] <= lon <= b[2] and b[1] <= lat <= b[3] for b in tiles.bounds]]
        subset = pd.concat([subset, hit.assign(site=name)])
    subset = subset.drop_duplicates("tile_id").drop(columns="bounds")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tiles.drop(columns="bounds").to_csv(out / "tile_density.csv", index=False)
    subset.to_csv(out / "subset_tiles.csv", index=False)
    print(tiles.density_class.value_counts().to_string())
    print(f"{len(subset)} tile(s) in the subset -> {out / 'subset_tiles.csv'}")


def stage1(args) -> None:
    root = read_root(Path(args.config))
    base = root / args.base_dir_name
    pop_path = resolve_catalog_path(Path(args.catalog), root, "population")
    out = Path(args.out_dir)
    tiles_out = out / "tiles"
    tiles_out.mkdir(parents=True, exist_ok=True)
    ids = (pd.read_csv(out / "subset_tiles.csv").tile_id.astype(str).tolist() if not args.all_tiles
           else (base / "tile_ids.txt").read_text().split())
    todo = [t for t in ids if not (tiles_out / f"{t}.npz").exists()]
    tree, tile_ids = build_tile_tree(base / "tile_selection.gpkg")
    print(f"{len(todo)} of {len(ids)} tile(s) to pass", flush=True)
    jobs = [(base / t, tiles_out, args.fsf, pop_path, tree, tile_ids) for t in todo]
    failed = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, fut in enumerate(as_completed([pool.submit(_pass_task, j) for j in jobs]), 1):
            tid, err = fut.result()
            if err:
                failed.append(tid)
                print(f"  tile {tid} failed: {err}", flush=True)
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} done, {len(failed)} failed", flush=True)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["select", "stage1"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--catalog", default=str(repo / "snakemake_workflow" / "config" / "data_catalog_gfm.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fsf", type=float, default=12.0, help="effective friction factor of the earlier layout")
    parser.add_argument("--all-tiles", action="store_true", help="stage1: every tile instead of the subset")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    {"select": select, "stage1": stage1}[args.stage](args)


if __name__ == "__main__":
    main()
